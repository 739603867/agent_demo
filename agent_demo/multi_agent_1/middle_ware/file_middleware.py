import asyncio
import concurrent.futures
import mimetypes
import uuid
import warnings
from pathlib import Path
from typing import Annotated, NotRequired, Literal, cast, Any, Callable, Awaitable

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ResponseT, ModelResponse, ExtendedModelResponse
from langchain_core.messages import HumanMessage, ContentBlock, ToolMessage, AnyMessage, BaseMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import ToolRuntime
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT
from pydantic import BaseModel, Field

from agent_demo.multi_agent_1.backend.protocol import (
    BackendProtocol,
    EditResult,
    FileData,
    ReadResult,
    WriteResult, SandboxBackendProtocol, BACKEND_TYPES, execute_accepts_timeout,
)
from agent_demo.multi_agent_1.backend.state import StateBackend
from agent_demo.multi_agent_1.backend.utils import format_content_with_line_numbers, truncate_if_too_long, \
    validate_path, _get_file_type, check_empty_content, format_grep_matches, sanitize_tool_call_id
from agent_demo.multi_agent_1.utils.message_utils import append_to_system_message

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
GLOB_TIMEOUT = 20.0  # seconds
LINE_NUMBER_WIDTH = 6
DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 100

# read_file 截断消息模板
# {file_path} 会在运行时填充
READ_FILE_TRUNCATION_MSG = (
    "\n\n[Output was truncated due to size limits. "
    "The file content is very large. "
    "Consider reformatting the file to make it easier to navigate. "
    "For example, if this is JSON, use execute(command='jq . {file_path}') to pretty-print it with line breaks. "
    "For other formats, you can use appropriate formatting tools to split long lines.]"
)

# 用于截断计算时估算的每个 token 对应字符数。
# 使用每个 token 约 4 个字符作为保守估计（实际比例会因内容而变化）
# 这里会稍微高估，以避免原本可能装得下的内容被过早驱逐
NUM_CHARS_PER_TOKEN = 4

def _file_data_reducer(left: dict[str, FileData] | None, right: dict[str, FileData | None]) -> dict[str, FileData]:
    """Merge file updates with support for deletions.

    This reducer enables file deletion by treating `None` values in the right
    dictionary as deletion markers. It's designed to work with LangGraph's
    state management where annotated reducers control how state updates merge.

    Args:
        left: Existing files dictionary. May be `None` during initialization.
        right: New files dictionary to merge. Files with `None` values are
            treated as deletion markers and removed from the result.

    Returns:
        Merged dictionary where right overwrites left for matching keys,
        and `None` values in right trigger deletions.

    Example:
        ```python
        existing = {"/file1.txt": FileData(...), "/file2.txt": FileData(...)}
        updates = {"/file2.txt": None, "/file3.txt": FileData(...)}
        result = file_data_reducer(existing, updates)
        # Result: {"/file1.txt": FileData(...), "/file3.txt": FileData(...)}
        ```
    """
    if left is None:
        return {k: v for k, v in right.items() if v is not None}

    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result

class FilesystemState(AgentState):
    """State for the filesystem middleware."""

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """Files in the filesystem."""

class LsSchema(BaseModel):
    """Input schema for the `ls` tool."""

    path: str = Field(description="Absolute path to the directory to list. Must be absolute, not relative.")

class ReadFileSchema(BaseModel):
    """Input schema for the `read_file` tool."""

    file_path: str = Field(description="Absolute path to the file to read. Must be absolute, not relative.")
    offset: int = Field(
        default=DEFAULT_READ_OFFSET,
        description="Line number to start reading from (0-indexed). Use for pagination of large files.",
    )
    limit: int = Field(
        default=DEFAULT_READ_LIMIT,
        description="Maximum number of lines to read. Use for pagination of large files.",
    )
class WriteFileSchema(BaseModel):
    """Input schema for the `write_file` tool."""

    file_path: str = Field(description="Absolute path where the file should be created. Must be absolute, not relative.")
    content: str = Field(description="The text content to write to the file. This parameter is required.")


class EditFileSchema(BaseModel):
    """Input schema for the `edit_file` tool."""

    file_path: str = Field(description="Absolute path to the file to edit. Must be absolute, not relative.")
    old_string: str = Field(description="The exact text to find and replace. Must be unique in the file unless replace_all is True.")
    new_string: str = Field(description="The text to replace old_string with. Must be different from old_string.")
    replace_all: bool = Field(
        default=False,
        description="If True, replace all occurrences of old_string. If False (default), old_string must be unique.",
    )


class GlobSchema(BaseModel):
    """Input schema for the `glob` tool."""

    pattern: str = Field(description="Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md').")
    path: str = Field(default="/", description="Base directory to search from. Defaults to root '/'.")


class GrepSchema(BaseModel):
    """Input schema for the `grep` tool."""

    pattern: str = Field(description="Text pattern to search for (literal string, not regex).")
    path: str | None = Field(default=None, description="Directory to search in. Defaults to current working directory.")
    glob: str | None = Field(default=None, description="Glob pattern to filter which files to search (e.g., '*.py').")
    output_mode: Literal["files_with_matches", "content", "count"] = Field(
        default="files_with_matches",
        description="Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
    )


class ExecuteSchema(BaseModel):
    """Input schema for the `execute` tool."""

    command: str = Field(description="Shell command to execute in the sandbox environment.")
    timeout: int | None = Field(
        default=None,
        description="Optional timeout in seconds for this command. Overrides the default timeout. Use 0 for no-timeout execution on backends that support it.",
    )

LIST_FILES_TOOL_DESCRIPTION = """列出目录中的所有文件。

这个工具适合用来浏览文件系统，并找到接下来要读取或编辑的正确文件。
在使用 read_file 或 edit_file 之前，你几乎总是应该先使用这个工具。"""

READ_FILE_TOOL_DESCRIPTION = """从文件系统中读取文件。

假设这个工具可以读取所有文件。如果用户提供了某个文件路径，就假定这个路径是有效的。读取一个不存在的文件也是允许的，工具会返回错误。

用法：
- 默认会从文件开头开始，最多读取 100 行
- **处理大文件和探索代码库时尤其重要**：请使用 offset 和 limit 做分页，避免上下文溢出
  - 第一次查看：read_file(path, limit=100) 先看文件结构
  - 继续看后面部分：read_file(path, offset=100, limit=200) 读取接下来的 200 行
  - 只有在确实为了编辑需要时，才省略 limit（读取整个文件）
- 指定 offset 和 limit：read_file(path, offset=0, limit=100) 会读取前 100 行
- 结果会使用类似 cat -n 的格式返回，行号从 1 开始
- 超过 5,000 个字符的长行会被拆成多行，并带有续行标记（例如 5.1、5.2 等）。当你指定 limit 时，这些续行也会计入行数限制
- 你可以在一次响应里调用多个工具。通常更好的做法是成批预读多个可能有用的文件
- 如果你读取的文件存在但内容为空，你会收到一个 system reminder 警告，而不是文件内容
- 图片文件（`.png`、`.jpg`、`.jpeg`、`.gif`、`.webp`）会以多模态图片内容块的形式返回（见 https://docs.langchain.com/oss/python/langchain/messages#multimodal）。

处理图片任务时：
- 对 `.png/.jpg/.jpeg/.gif/.webp` 使用 `read_file(file_path=...)`
- 不要给图片使用 `offset`/`limit`（分页只适用于文本）
- 如果历史中的图片细节被压缩掉了，请再次对同一路径调用 `read_file`

- 在编辑文件之前，你应该始终先确保已经读过这个文件。"""

EDIT_FILE_TOOL_DESCRIPTION = """在文件中执行精确字符串替换。

用法：
- 编辑前必须先读取文件。如果你在没有先读取文件的情况下尝试编辑，这个工具会报错
- 编辑时要保留 read 输出里的原始缩进（tab / 空格）。old_string 和 new_string 中绝不能包含行号前缀
- 应始终优先编辑已有文件，而不是创建新文件
- 只有当用户明确要求时才可以使用 emoji。"""


WRITE_FILE_TOOL_DESCRIPTION = """向文件系统中新建文件并写入内容。

用法：
- write_file 工具会创建一个新文件
- 能编辑已有文件时，优先使用 edit_file，而不是新建文件
"""

GLOB_TOOL_DESCRIPTION = """查找匹配 glob 模式的文件。

支持标准 glob 模式：`*`（任意字符）、`**`（任意目录）、`?`（单个字符）。
返回所有匹配路径的绝对路径列表。

示例：
- `**/*.py` - 查找所有 Python 文件
- `*.txt` - 查找根目录下所有文本文件
- `/subdir/**/*.md` - 查找 /subdir 下所有 Markdown 文件"""

GREP_TOOL_DESCRIPTION = """在多个文件中搜索文本模式。

按字面文本搜索（不是正则），并根据 output_mode 返回匹配到的文件或内容。
括号、中括号、竖线等特殊字符都会按普通字符处理，而不是正则操作符。

示例：
- 搜索所有文件：`grep(pattern="TODO")`
- 只搜索 Python 文件：`grep(pattern="import", glob="*.py")`
- 显示匹配行：`grep(pattern="error", output_mode="content")`
- 搜索带特殊字符的代码：`grep(pattern="def __init__(self):")`"""

EXECUTE_TOOL_DESCRIPTION = """在隔离的沙箱环境中执行 shell 命令。

用法：
这个工具会在沙箱环境中执行给定命令，并正确处理安全与输出。
在执行命令前，请遵循下面这些步骤：
1. 目录检查：
   - 如果命令会创建新目录或文件，先使用 ls 工具确认父目录存在，并且位置正确
   - 例如，在运行 "mkdir foo/bar" 之前，先用 ls 检查 "foo" 是否存在且确实是目标父目录
2. 命令执行：
   - 对包含空格的文件路径，始终使用双引号包裹（例如：cd "path with spaces/file.txt"）
   - 正确引用示例：
     - cd "/Users/name/My Documents"（正确）
     - cd /Users/name/My Documents（错误，会失败）
     - python "/path/with spaces/script.py"（正确）
     - python /path/with spaces/script.py（错误，会失败）
   - 确认引用正确后再执行命令
   - 捕获命令输出
用法说明：
  - 命令会在隔离的沙箱环境中运行
  - 返回合并后的 stdout/stderr 输出以及退出码
  - 如果输出过大，可能会被截断
  - 对于长时间运行的命令，可以使用可选的 timeout 参数覆盖默认超时（例如 execute(command="make build", timeout=300)）
  - 对于支持无超时执行的后端，timeout=0 可能表示禁用超时
  - 非常重要：你必须避免使用 find、grep 这类搜索命令。请改用 grep、glob 工具进行搜索。你也必须避免使用 cat、head、tail 这类读取命令，而应使用 read_file 读取文件。
  - 当需要执行多条命令时，使用 ';' 或 '&&' 分隔。不要使用换行（引号内的换行除外）
    - 当命令前后有依赖关系时，使用 '&&'（例如 "mkdir dir && cd dir"）
    - 只有在你不在意前一条命令是否失败时，才使用 ';'
  - 尽量通过绝对路径并避免使用 cd，以在整个会话中保持当前工作目录稳定

示例：
  好的示例：
    - execute(command="pytest /foo/bar/tests")
    - execute(command="python /path/to/script.py")
    - execute(command="npm install && npm test")
    - execute(command="make build", timeout=300)

  不推荐的示例（避免这样做）：
    - execute(command="cd /foo/bar && pytest tests")  # 请改用绝对路径
    - execute(command="cat file.txt")  # 请改用 read_file 工具
    - execute(command="find . -name '*.py'")  # 请改用 glob 工具
    - execute(command="grep -r 'pattern' .")  # 请改用 grep 工具

注意：只有当后端支持执行能力（SandboxBackendProtocol）时，这个工具才可用。
如果不支持执行，这个工具会返回错误信息。"""

FILESYSTEM_SYSTEM_PROMPT = """## 遵循以下约定

- 编辑前先读取文件——先理解已有内容，再进行修改
- 模仿现有风格、命名约定和代码模式

## 文件系统工具 `ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`

你可以使用这些工具与文件系统交互。
所有文件路径都必须以 `/` 开头。请遵循各个工具的说明文档，在读取大文件时使用分页（offset/limit）。

- ls：列出目录中的文件（要求绝对路径）
- read_file：从文件系统读取文件
- write_file：向文件系统写入文件
- edit_file：编辑文件系统中的文件
- glob：查找匹配模式的文件（例如 `"**/*.py"`）
- grep：在文件中搜索文本

## 大型工具结果

当某个工具结果过大时，它可能不会直接内联返回，而是被转存到文件系统中。这种情况下，请使用 `read_file` 分块查看保存的结果；如果你需要在 `/large_tool_results/` 下搜索多个被转存的结果且不知道确切路径，也可以使用 `grep`。被转存的工具结果会保存在 `/large_tool_results/<tool_call_id>` 下。"""

EXECUTION_SYSTEM_PROMPT = """## `execute` 工具

你可以使用 `execute` 工具在沙箱环境中运行 shell 命令。
可用它来执行命令、脚本、测试、构建以及其他 shell 操作。

- execute：在沙箱中运行 shell 命令（返回输出和退出码）"""

def _supports_execution(backend: BackendProtocol) -> bool:
    """检查后端是否支持命令执行。

    对于 CompositeBackend，会检查默认后端是否支持执行。
    对于其他后端，则检查它们是否实现了 SandboxBackendProtocol。

    Args:
        backend: 要检查的后端。

    Returns:
        如果后端支持执行则返回 True，否则返回 False。
    """
    # 对于其他后端，直接使用 isinstance 检查
    return isinstance(backend, SandboxBackendProtocol)

# 应从大结果驱逐逻辑中排除的工具。
#
# 这个元组中的工具在结果超过 token 限制时，不应该把结果转存到文件系统。
# 不同工具被排除的原因不同：
#
# 1. 自带截断能力的工具（ls、glob、grep）：
#    这些工具在输出过大时会自己做截断。出现这种情况时，通常意味着查询条件过宽，
#    更需要缩小搜索范围，而不是保留完整结果。此时保留下来的截断结果往往更像噪声，
#    更合适的做法是提示 LLM 收窄搜索条件。
#
# 2. 截断行为有问题的工具（read_file）：
#    read_file 的难点在于，失败场景可能是单行特别长
#    （比如 jsonl 文件中，每一行都包含很长的 payload）。如果我们去截断 read_file 的结果，
#    agent 可能又会继续用 read_file 重新读取这个已经被截断的文件，但这并没有帮助。
#
# 3. 永远不会超限的工具（edit_file、write_file）：
#    这些工具只会返回非常简短的确认信息，通常不可能大到超过 token 限制，
#    所以没必要对它们做这类检查。
TOOLS_EXCLUDED_FROM_EVICTION = (
    "ls",
    "glob",
    "grep",
    "read_file",
    "edit_file",
    "write_file",
)

TOO_LARGE_TOOL_MSG = """工具结果过大，此次工具调用 {tool_call_id} 的结果已保存到文件系统中的这个路径：{file_path}

你可以使用 read_file 工具从文件系统中读取这个结果，但要注意每次只读取一部分。

你可以在 read_file 调用中指定 offset 和 limit。例如，如果要读取前 100 行，可以使用 offset=0、limit=100。

下面是一个预览，展示了结果开头和结尾的内容（形如 `... [N lines truncated] ...` 的行表示中间有内容被省略）：

{content_sample}
"""

TOO_LARGE_HUMAN_MSG = """消息内容过大，已保存到文件系统中的这个路径：{file_path}

你可以使用带分页参数（offset 和 limit）的 read_file 工具读取完整内容。

下面是一个预览，展示了内容开头和结尾的部分：

{content_sample}
"""

def _build_evicted_human_content(
    message: HumanMessage,
    replacement_text: str,
) -> str | list[ContentBlock]:
    """为被转存的 HumanMessage 构造替换内容，并保留非文本块。

    如果原内容是普通字符串，就直接返回 replacement_text。
    如果原内容是包含多种 block 的列表（例如文本 + 图片），
    就把所有文本 block 替换成一个新的文本 block，内容为 replacement_text，
    同时保留所有非文本 block。

    Args:
        message: 原始的、将被转存的 HumanMessage。
        replacement_text: 截断提示和预览文本。

    Returns:
        替换后的内容：可能是字符串，也可能是内容块列表。
    """
    if isinstance(message.content, str):
        return replacement_text
    media_blocks = [block for block in message.content_blocks if block["type"] != "text"]
    if not media_blocks:
        return replacement_text
    return [cast("ContentBlock", {"type": "text", "text": replacement_text}), *media_blocks]

def _build_truncated_human_message(message: HumanMessage, file_path: str) -> HumanMessage:
    """为模型请求构造一个截断后的 HumanMessage。

    这个函数会根据 state 中仍然保留的完整内容生成预览，
    并返回一个更轻量的替换消息给模型使用。
    它只做字符串处理，不涉及后端 I/O。

    Args:
        message: 原始 HumanMessage（完整内容仍保留在 state 中）。
        file_path: 内容被转存到后端后的路径。

    Returns:
        一个新的 HumanMessage，内容已截断，但 `id` 保持不变。
    """
    content_str = _extract_text_from_message(message)
    content_sample = _create_content_preview(content_str)
    replacement_text = TOO_LARGE_HUMAN_MSG.format(
        file_path=file_path,
        content_sample=content_sample,
    )
    evicted = _build_evicted_human_content(message, replacement_text)
    return message.model_copy(update={"content": evicted})

def _extract_text_from_message(message: BaseMessage) -> str:
    """使用消息的 `content_blocks` 属性提取文本。

    这个函数会把所有文本内容块拼接起来，并忽略非文本块
    （例如图片、音频等），避免二进制负载把大小估算放大。

    Args:
        message: 要提取文本的 BaseMessage。

    Returns:
        拼接后的文本内容；如果需要，也可回退为字符串化内容。
    """
    texts = [block["text"] for block in message.content_blocks if block["type"] == "text"]
    return "\n".join(texts)

def _create_content_preview(content_str: str, *, head_lines: int = 5, tail_lines: int = 5) -> str:
    """创建内容预览，展示开头和结尾，并在中间加入截断标记。

    Args:
        content_str: 要生成预览的完整内容字符串。
        head_lines: 开头要展示的行数。
        tail_lines: 结尾要展示的行数。

    Returns:
        带行号的格式化预览字符串。
    """
    lines = content_str.splitlines()

    if len(lines) <= head_lines + tail_lines:
        # If file is small enough, show all lines
        preview_lines = [line[:1000] for line in lines]
        return format_content_with_line_numbers(preview_lines, start_line=1)

    # Show head and tail with truncation marker
    head = [line[:1000] for line in lines[:head_lines]]
    tail = [line[:1000] for line in lines[-tail_lines:]]

    head_sample = format_content_with_line_numbers(head, start_line=1)
    truncation_notice = f"\n... [{len(lines) - head_lines - tail_lines} lines truncated] ...\n"
    tail_sample = format_content_with_line_numbers(tail, start_line=len(lines) - tail_lines + 1)

    return head_sample + truncation_notice + tail_sample

def _build_evicted_content(message: ToolMessage, replacement_text: str) -> str | list[ContentBlock]:
    """为被转存的消息构造替换内容，并保留非文本块。

    如果原内容是普通字符串，就直接返回 replacement_text。
    如果原内容是包含多种 block 的列表（例如文本 + 图片），
    就把所有文本 block 替换成一个新的文本 block，内容为 replacement_text，
    同时保留所有非文本 block。

    Args:
        message: 原始的、将被转存的 ToolMessage。
        replacement_text: 截断提示和预览文本。

    Returns:
        替换后的内容：可能是字符串，也可能是内容块列表。
    """
    if isinstance(message.content, str):
        return replacement_text
    media_blocks = [block for block in message.content_blocks if block["type"] != "text"]
    if not media_blocks:
        # All content is text, so a plain string replacement is sufficient.
        return replacement_text
    return [cast("ContentBlock", {"type": "text", "text": replacement_text}), *media_blocks]

class FilesystemMiddleware(AgentMiddleware[FilesystemState, ContextT, ResponseT]):
    """为 agent 提供文件系统工具，以及可选的执行工具的中间件。

    这个中间件会为 agent 添加文件系统工具：`ls`、`read_file`、`write_file`、
    `edit_file`、`glob` 和 `grep`。

    文件可以存储在任何实现了 `BackendProtocol` 的后端中。

    如果后端实现了 `SandboxBackendProtocol`，还会额外添加 `execute` 工具，
    用于执行 shell 命令。

    这个中间件还会在工具结果过大时，自动把结果转存到文件系统中，
    以避免上下文窗口被塞满。

    Args:
        backend: 用于文件存储以及可选执行能力的后端。

            如果不提供，默认使用 `StateBackend`（即把文件临时存到 agent state 中）。

            如果需要持久化存储或混合存储，可以使用带自定义路由的 `CompositeBackend`。

            如果需要执行能力，请使用实现了 `SandboxBackendProtocol` 的后端。
        system_prompt: 可选的自定义 system prompt 覆盖值。
        custom_tool_descriptions: 可选的自定义工具描述覆盖值。
        tool_token_limit_before_evict: 在把工具结果转存到文件系统前允许的 token 上限。

            超过后，会使用配置好的后端把结果写入文件系统，
            并用一个截断预览和文件引用来替换原结果。

    Example:
        ```python
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.backends import StateBackend, StoreBackend, CompositeBackend
        from langchain.agents import create_agent

        # 仅临时存储（默认，不支持执行）
        agent = create_agent(middleware=[FilesystemMiddleware()])

        # 使用混合存储（临时 + 持久化 /memories/）
        backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})
        agent = create_agent(middleware=[FilesystemMiddleware(backend=backend)])

        # 使用沙箱后端（支持执行）
        from my_sandbox import DockerSandboxBackend

        sandbox = DockerSandboxBackend(container_id="my-container")
        agent = create_agent(middleware=[FilesystemMiddleware(backend=sandbox)])
        ```
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
        human_message_token_limit_before_evict: int | None = 50000,
        max_execute_timeout: int = 3600,
    ) -> None:
        """初始化文件系统中间件。

        Args:
            backend: 用于文件存储和可选执行能力的后端，或一个工厂可调用对象。
                如果不提供，默认使用 StateBackend。
            system_prompt: 可选的自定义 system prompt 覆盖值。
            custom_tool_descriptions: 可选的自定义工具描述覆盖值。
            tool_token_limit_before_evict: 在把工具结果转存到文件系统前允许的 token 上限。
            human_message_token_limit_before_evict: 在把 HumanMessage 转存到文件系统前允许的 token 上限。
            max_execute_timeout: execute 工具里每条命令可覆盖的 timeout 最大秒数。

                默认值为 3600 秒（1 小时）。任何超过这个值的单条命令 timeout
                都会被拒绝，并返回错误信息。

        Raises:
            ValueError: 当 `max_execute_timeout` 不是正数时抛出。
        """
        if max_execute_timeout <= 0:
            msg = f"max_execute_timeout must be positive, got {max_execute_timeout}"
            raise ValueError(msg)
        # Use provided backend or default to StateBackend factory
        self.backend = backend if backend is not None else (StateBackend)

        # Store configuration (private - internal implementation details)
        self._custom_system_prompt = system_prompt
        self._custom_tool_descriptions = custom_tool_descriptions or {}
        self._tool_token_limit_before_evict = tool_token_limit_before_evict
        self._human_message_token_limit_before_evict = human_message_token_limit_before_evict
        self._max_execute_timeout = max_execute_timeout

        self.tools = [
            self._create_ls_tool(),
            self._create_read_file_tool(),
            self._create_write_file_tool(),
            self._create_edit_file_tool(),
            self._create_glob_tool(),
            self._create_grep_tool(),
            self._create_execute_tool(),
        ]

    def _get_backend(self, runtime: ToolRuntime[Any, Any]) -> BackendProtocol:
        """从后端实例或后端工厂中解析出真正的后端对象。

        Args:
            runtime: 工具运行时上下文。

        Returns:
            解析后的后端实例。
        """
        if callable(self.backend):
            return self.backend(runtime)  # ty: ignore[call-top-callable]
        return self.backend

    def _create_ls_tool(self) -> BaseTool:
        """创建 ls（列文件）工具。"""
        tool_description = self._custom_tool_descriptions.get("ls") or LIST_FILES_TOOL_DESCRIPTION

        def sync_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        ) -> str:
            """ls 工具的同步包装。"""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            ls_result = resolved_backend.ls(validated_path)
            if ls_result.error:
                return f"Error: {ls_result.error}"
            infos = ls_result.entries or []
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        async def async_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        ) -> str:
            """ls 工具的异步包装。"""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            ls_result = await resolved_backend.als(validated_path)
            if ls_result.error:
                return f"Error: {ls_result.error}"
            infos = ls_result.entries or []
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        return StructuredTool.from_function(
            name="ls",
            description=tool_description,
            func=sync_ls,
            coroutine=async_ls,
            infer_schema=False,
            args_schema=LsSchema,
        )

    def _create_read_file_tool(self) -> BaseTool:  # noqa: C901
        """创建 read_file 工具。"""
        tool_description = self._custom_tool_descriptions.get("read_file") or READ_FILE_TOOL_DESCRIPTION
        token_limit = self._tool_token_limit_before_evict

        def _truncate(content: str, file_path: str, limit: int) -> str:
            lines = content.splitlines(keepends=True)
            if len(lines) > limit:
                lines = lines[:limit]
                content = "".join(lines)

            if token_limit and len(content) >= NUM_CHARS_PER_TOKEN * token_limit:
                truncation_msg = READ_FILE_TRUNCATION_MSG.format(file_path=file_path)
                max_content_length = NUM_CHARS_PER_TOKEN * token_limit - len(truncation_msg)
                content = content[:max_content_length] + truncation_msg

            return content

        def _handle_read_result(
            read_result: ReadResult | str,
            validated_path: str,
            tool_call_id: str | None,
            offset: int,
            limit: int,
        ) -> ToolMessage | str:
            if isinstance(read_result, str):
                warnings.warn(
                    "Returning a plain `str` from `backend.read()` is deprecated. "
                    "Return a `ReadResult` instead. Returning `str` will not be "
                    "supported in a future version.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                # Legacy backends already format with line numbers
                return _truncate(read_result, validated_path, limit)

            if read_result.error:
                return f"Error: {read_result.error}"

            if read_result.file_data is None:
                return f"Error: no data returned for '{validated_path}'"

            file_type = _get_file_type(validated_path)
            content = read_result.file_data["content"]

            if file_type != "text":
                mime_type = mimetypes.guess_type("file" + Path(validated_path).suffix)[0] or "application/octet-stream"
                return ToolMessage(
                    content_blocks=cast("list[ContentBlock]", [{"type": file_type, "base64": content, "mime_type": mime_type}]),
                    name="read_file",
                    tool_call_id=tool_call_id,
                    additional_kwargs={"read_file_path": validated_path, "read_file_media_type": mime_type},
                )

            empty_msg = check_empty_content(content)
            if empty_msg:
                return empty_msg

            content = format_content_with_line_numbers(content, start_line=offset + 1)
            # We apply truncation again after formatting content as continuation lines
            # can increase line count
            return _truncate(content, validated_path, limit)

        def sync_read_file(
            file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
            offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = DEFAULT_READ_OFFSET,
            limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = DEFAULT_READ_LIMIT,
        ) -> ToolMessage | str:
            """read_file 工具的同步包装。"""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            read_result = resolved_backend.read(validated_path, offset=offset, limit=limit)
            return _handle_read_result(read_result, validated_path, runtime.tool_call_id, offset, limit)

        async def async_read_file(
            file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
            offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = DEFAULT_READ_OFFSET,
            limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = DEFAULT_READ_LIMIT,
        ) -> ToolMessage | str:
            """read_file 工具的异步包装。"""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            read_result = await resolved_backend.a_read(validated_path, offset=offset, limit=limit)
            return _handle_read_result(read_result, validated_path, runtime.tool_call_id, offset, limit)

        return StructuredTool.from_function(
            name="read_file",
            description=tool_description,
            func=sync_read_file,
            coroutine=async_read_file,
            infer_schema=False,
            args_schema=ReadFileSchema,
        )

    def _create_write_file_tool(self) -> BaseTool:
        """创建 write_file 工具。"""
        tool_description = self._custom_tool_descriptions.get("write_file") or WRITE_FILE_TOOL_DESCRIPTION

        def sync_write_file(
            file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
            content: Annotated[str, "The text content to write to the file. This parameter is required."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> Command | str:
            """write_file 工具的同步包装。"""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            res: WriteResult = resolved_backend.write(validated_path, content)
            if res.error:
                return res.error
            # If backend returns state update, wrap into Command with ToolMessage
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Updated file {res.path}",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Updated file {res.path}"

        async def async_write_file(
            file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
            content: Annotated[str, "The text content to write to the file. This parameter is required."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> Command | str:
            """write_file 工具的异步包装。"""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            res: WriteResult = await resolved_backend.a_write(validated_path, content)
            if res.error:
                return res.error
            # If backend returns state update, wrap into Command with ToolMessage
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Updated file {res.path}",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Updated file {res.path}"

        return StructuredTool.from_function(
            name="write_file",
            description=tool_description,
            func=sync_write_file,
            coroutine=async_write_file,
            infer_schema=False,
            args_schema=WriteFileSchema,
        )

    def _create_edit_file_tool(self) -> BaseTool:
        """创建 edit_file 工具。"""
        tool_description = self._custom_tool_descriptions.get("edit_file") or EDIT_FILE_TOOL_DESCRIPTION

        def sync_edit_file(
            file_path: Annotated[str, "Absolute path to the file to edit. Must be absolute, not relative."],
            old_string: Annotated[str, "The exact text to find and replace. Must be unique in the file unless replace_all is True."],
            new_string: Annotated[str, "The text to replace old_string with. Must be different from old_string."],
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: Annotated[bool, "If True, replace all occurrences of old_string. If False (default), old_string must be unique."] = False,
        ) -> Command | str:
            """edit_file 工具的同步包装。"""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            res: EditResult = resolved_backend.edit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return res.error
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

        async def async_edit_file(
            file_path: Annotated[str, "Absolute path to the file to edit. Must be absolute, not relative."],
            old_string: Annotated[str, "The exact text to find and replace. Must be unique in the file unless replace_all is True."],
            new_string: Annotated[str, "The text to replace old_string with. Must be different from old_string."],
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: Annotated[bool, "If True, replace all occurrences of old_string. If False (default), old_string must be unique."] = False,
        ) -> Command | str:
            """edit_file 工具的异步包装。"""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            res: EditResult = await resolved_backend.a_edit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return res.error
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

        return StructuredTool.from_function(
            name="edit_file",
            description=tool_description,
            func=sync_edit_file,
            coroutine=async_edit_file,
            infer_schema=False,
            args_schema=EditFileSchema,
        )

    def _create_glob_tool(self) -> BaseTool:
        """创建 glob 工具。"""
        tool_description = self._custom_tool_descriptions.get("glob") or GLOB_TOOL_DESCRIPTION

        def sync_glob(
            pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
        ) -> str:
            """glob 工具的同步包装。"""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(resolved_backend.glob, pattern, path=validated_path)
                try:
                    glob_result = future.result(timeout=GLOB_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    return f"Error: glob timed out after {GLOB_TIMEOUT}s. Try a more specific pattern or a narrower path."
            if glob_result.error:
                return f"Error: {glob_result.error}"
            infos = glob_result.matches or []
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        async def async_glob(
            pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
        ) -> str:
            """glob 工具的异步包装。"""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            try:
                glob_result = await asyncio.wait_for(
                    resolved_backend.a_glob(pattern, path=validated_path),
                    timeout=GLOB_TIMEOUT,
                )
            except TimeoutError:
                return f"Error: glob timed out after {GLOB_TIMEOUT}s. Try a more specific pattern or a narrower path."
            if glob_result.error:
                return f"Error: {glob_result.error}"
            infos = glob_result.matches or []
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        return StructuredTool.from_function(
            name="glob",
            description=tool_description,
            func=sync_glob,
            coroutine=async_glob,
            infer_schema=False,
            args_schema=GlobSchema,
        )

    def _create_grep_tool(self) -> BaseTool:
        """创建 grep 工具。"""
        tool_description = self._custom_tool_descriptions.get("grep") or GREP_TOOL_DESCRIPTION

        def sync_grep(
            pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str | None, "Directory to search in. Defaults to current working directory."] = None,
            glob: Annotated[str | None, "Glob pattern to filter which files to search (e.g., '*.py')."] = None,
            output_mode: Annotated[
                Literal["files_with_matches", "content", "count"],
                "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
            ] = "files_with_matches",
        ) -> str:
            """grep 工具的同步包装。"""
            resolved_backend = self._get_backend(runtime)
            grep_result = resolved_backend.grep(pattern, path=path, glob=glob)
            if grep_result.error:
                return grep_result.error
            matches = grep_result.matches or []
            formatted = format_grep_matches(matches, output_mode)
            return truncate_if_too_long(formatted)

        async def async_grep(
            pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str | None, "Directory to search in. Defaults to current working directory."] = None,
            glob: Annotated[str | None, "Glob pattern to filter which files to search (e.g., '*.py')."] = None,
            output_mode: Annotated[
                Literal["files_with_matches", "content", "count"],
                "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
            ] = "files_with_matches",
        ) -> str:
            """grep 工具的异步包装。"""
            resolved_backend = self._get_backend(runtime)
            grep_result = await resolved_backend.agrep(pattern, path=path, glob=glob)
            if grep_result.error:
                return grep_result.error
            matches = grep_result.matches or []
            formatted = format_grep_matches(matches, output_mode)
            return truncate_if_too_long(formatted)

        return StructuredTool.from_function(
            name="grep",
            description=tool_description,
            func=sync_grep,
            coroutine=async_grep,
            infer_schema=False,
            args_schema=GrepSchema,
        )

    def _create_execute_tool(self) -> BaseTool:  # noqa: C901
        """创建用于沙箱命令执行的 execute 工具。"""
        tool_description = self._custom_tool_descriptions.get("execute") or EXECUTE_TOOL_DESCRIPTION

        def sync_execute(  # noqa: PLR0911 - early returns for distinct error conditions
            command: Annotated[str, "Shell command to execute in the sandbox environment."],
            runtime: ToolRuntime[None, FilesystemState],
            timeout: Annotated[
                int | None,
                "Optional timeout in seconds for this command. Overrides the default timeout. Use 0 for no-timeout execution on backends that support it.",
            ] = None,
        ) -> str:
            """execute 工具的同步包装。"""
            if timeout is not None:
                if timeout < 0:
                    return f"Error: timeout must be non-negative, got {timeout}."
                if timeout > self._max_execute_timeout:
                    return f"Error: timeout {timeout}s exceeds maximum allowed ({self._max_execute_timeout}s)."

            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not _supports_execution(resolved_backend):
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution (SandboxBackendProtocol). "
                    "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                )

            # Safe cast: _supports_execution validates that execute()/aexecute() exist
            # (either SandboxBackendProtocol or CompositeBackend with sandbox default)
            executable = cast("SandboxBackendProtocol", resolved_backend)
            if timeout is not None and not execute_accepts_timeout(type(executable)):
                return (
                    "Error: This sandbox backend does not support per-command "
                    "timeout overrides. Update your sandbox package to the "
                    "latest version, or omit the timeout parameter."
                )
            try:
                result = executable.execute(command, timeout=timeout) if timeout is not None else executable.execute(command)
            except NotImplementedError as e:
                # Handle case where execute() exists but raises NotImplementedError
                return f"Error: Execution not available. {e}"
            except ValueError as e:
                return f"Error: Invalid parameter. {e}"

            # Format output for LLM consumption
            parts = [result.output]

            if result.exit_code is not None:
                status = "succeeded" if result.exit_code == 0 else "failed"
                parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

            if result.truncated:
                parts.append("\n[Output was truncated due to size limits]")

            return "".join(parts)

        async def async_execute(  # noqa: PLR0911 - early returns for distinct error conditions
            command: Annotated[str, "Shell command to execute in the sandbox environment."],
            runtime: ToolRuntime[None, FilesystemState],
            # ASYNC109 - timeout is a semantic parameter forwarded to the
            # backend's implementation, not an asyncio.timeout() contract.
            timeout: Annotated[  # noqa: ASYNC109
                int | None,
                "Optional timeout in seconds for this command. Overrides the default timeout. Use 0 for no-timeout execution on backends that support it.",
            ] = None,
        ) -> str:
            """execute 工具的异步包装。"""
            if timeout is not None:
                if timeout < 0:
                    return f"Error: timeout must be non-negative, got {timeout}."
                if timeout > self._max_execute_timeout:
                    return f"Error: timeout {timeout}s exceeds maximum allowed ({self._max_execute_timeout}s)."

            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not _supports_execution(resolved_backend):
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution (SandboxBackendProtocol). "
                    "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                )

            # Safe cast: _supports_execution validates that execute()/aexecute() exist
            executable = cast("SandboxBackendProtocol", resolved_backend)
            if timeout is not None and not execute_accepts_timeout(type(executable)):
                return (
                    "Error: This sandbox backend does not support per-command "
                    "timeout overrides. Update your sandbox package to the "
                    "latest version, or omit the timeout parameter."
                )
            try:
                result = await executable.aexecute(command, timeout=timeout) if timeout is not None else await executable.aexecute(command)
            except NotImplementedError as e:
                # Handle case where execute() exists but raises NotImplementedError
                return f"Error: Execution not available. {e}"
            except ValueError as e:
                return f"Error: Invalid parameter. {e}"

            # Format output for LLM consumption
            parts = [result.output]

            if result.exit_code is not None:
                status = "succeeded" if result.exit_code == 0 else "failed"
                parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

            if result.truncated:
                parts.append("\n[Output was truncated due to size limits]")

            return "".join(parts)

        return StructuredTool.from_function(
            name="execute",
            description=tool_description,
            func=sync_execute,
            coroutine=async_execute,
            infer_schema=False,
            args_schema=ExecuteSchema,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse:
        """更新 system prompt、过滤工具，并转存过大的 HumanMessage。

        除了 system prompt 和工具过滤逻辑之外，这个方法还负责处理过大的 HumanMessage：

        1. 任何已经在 `additional_kwargs` 中带有 `lc_evicted_to` 标记的消息，
           都会在模型请求中被替换成一个截断预览（state 中的原内容不变）。
        2. 如果最新一条消息是一个还未打标记、并且超过转存阈值的 HumanMessage，
           它的内容会被写入后端，并通过 `ExtendedModelResponse` 在 state 中打上标记。

        Args:
            request: 当前正在处理的模型请求。
            handler: 接收修改后请求的处理函数。

        Returns:
            模型响应，或者一个带有 state 更新的 `ExtendedModelResponse`，
            用来标记新近被转存的消息。
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)  # ty: ignore[invalid-argument-type]
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts).strip()

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        eviction_result = self._evict_and_truncate_messages(request)
        if eviction_result is not None:
            messages, state_command = eviction_result
            request = request.override(messages=messages)
            response = handler(request)
            if state_command is not None:
                return ExtendedModelResponse(model_response=response, command=state_command)
            return response

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse:
        """（异步）更新 system prompt，并根据后端能力过滤工具。

        同时也会把过大的 HumanMessage 转存到文件系统。
        完整说明可参考 `wrap_model_call`。

        Args:
            request: 当前正在处理的模型请求。
            handler: 接收修改后请求的处理函数。

        Returns:
            处理函数返回的模型响应，或者一个带有 state 更新的 `ExtendedModelResponse`，
            用来标记新近被转存的消息。
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)  # ty: ignore[invalid-argument-type]
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts).strip()

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        eviction_result = await self._aevict_and_truncate_messages(request)
        if eviction_result is not None:
            messages, state_command = eviction_result
            request = request.override(messages=messages)
            response = await handler(request)
            if state_command is not None:
                return ExtendedModelResponse(model_response=response, command=state_command)
            return response

        return await handler(request)

    def _process_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        """处理过大的 ToolMessage：把内容转存到文件系统。

        Args:
            message: 需要被转存的大 ToolMessage。
            resolved_backend: 用来写入内容的文件系统后端。

        Returns:
            一个二元组 (processed_message, files_update)：
            - processed_message: 新的 ToolMessage，内容已截断并带有文件引用
            - files_update: 需要应用到 state 的文件更新字典；如果转存失败则为 None

        Note:
            会从所有文本内容块中提取文本并拼接，用于大小判断和转存。
            非文本块（图片、音频等）会在替换消息里保留下来，避免多模态上下文丢失。
            模型可以通过读取后端中保存的文件，恢复完整文本。
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, None

        content_str = _extract_text_from_message(message)

        # Check if content exceeds eviction threshold
        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, None

        # Write content to filesystem
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = resolved_backend.write(file_path, content_str)
        if result.error:
            return message, None

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        evicted = _build_evicted_content(message, replacement_text)
        processed_message = ToolMessage(
            content=cast("str | list[str | dict]", evicted),
            tool_call_id=message.tool_call_id,
            name=message.name,
            id=message.id,
            artifact=message.artifact,
            status=message.status,
            additional_kwargs=dict(message.additional_kwargs),
            response_metadata=dict(message.response_metadata),
        )
        return processed_message, result.files_update

    async def _aprocess_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        """`_process_large_message` 的异步版本。

        使用异步后端方法，避免在异步上下文中调用同步接口。
        完整说明请参考 `_process_large_message`。
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, None

        content_str = _extract_text_from_message(message)

        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, None

        # Write content to filesystem using async method
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = await resolved_backend.awrite(file_path, content_str)
        if result.error:
            return message, None

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        evicted = _build_evicted_content(message, replacement_text)
        processed_message = ToolMessage(
            content=cast("str | list[str | dict]", evicted),
            tool_call_id=message.tool_call_id,
            name=message.name,
            id=message.id,
            artifact=message.artifact,
            status=message.status,
            additional_kwargs=dict(message.additional_kwargs),
            response_metadata=dict(message.response_metadata),
        )
        return processed_message, result.files_update

    def _get_backend_from_runtime(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> BackendProtocol:
        """从裸 `Runtime` 中解析后端。

        这个函数会基于 `Runtime` 构造一个 `ToolRuntime`，
        以满足后端工厂接口的要求。它主要用于 `before_agent` 这类 hook，
        因为这些 hook 拿到的是 `Runtime`，而不是 `ToolRuntime`。

        Args:
            state: 当前 agent state。
            runtime: 运行时上下文。

        Returns:
            解析后的后端实例。
        """
        if not callable(self.backend):
            return self.backend
        config = cast("RunnableConfig", getattr(runtime, "config", {}))
        tool_runtime = ToolRuntime(
            state=state,
            context=runtime.context,
            stream_writer=runtime.stream_writer,
            store=runtime.store,
            config=config,
            tool_call_id=None,
        )
        return self.backend(tool_runtime)  # ty: ignore[call-top-callable, invalid-argument-type]

    def _check_eviction_needed(
        self,
        messages: list[AnyMessage],
    ) -> tuple[bool, bool]:
        """检查是否需要进行消息转存处理。

        Args:
            messages: 要检查的消息列表。

        Returns:
            一个二元组 (has_tagged, new_eviction_needed)。
        """
        if not self._human_message_token_limit_before_evict:
            return False, False

        threshold = NUM_CHARS_PER_TOKEN * self._human_message_token_limit_before_evict
        has_tagged = any(isinstance(msg, HumanMessage) and msg.additional_kwargs.get("lc_evicted_to") for msg in messages)
        new_eviction_needed = False
        if messages and isinstance(messages[-1], HumanMessage):
            last = messages[-1]
            if not last.additional_kwargs.get("lc_evicted_to") and len(_extract_text_from_message(last)) > threshold:
                new_eviction_needed = True
        return has_tagged, new_eviction_needed

    @staticmethod
    def _apply_eviction_and_truncate(
        messages: list[AnyMessage],
        write_result: WriteResult | None,
        file_path: str | None,
    ) -> tuple[list[AnyMessage], Command | None]:
        """给新转存的消息打标记，并截断所有已打标记的消息。

        Args:
            messages: 消息列表（如果写入成功，可能会被修改）。
            write_result: 后端写入结果；如果没有尝试新的转存，则为 `None`。
            file_path: 内容被写入到的路径。

        Returns:
            一个二元组 (processed_messages, state_command)。
        """
        state_command: Command | None = None

        if write_result is not None and file_path is not None and not write_result.error:
            last = messages[-1]
            tagged = last.model_copy(
                update={
                    "additional_kwargs": {
                        **last.additional_kwargs,
                        "lc_evicted_to": file_path,
                    }
                }
            )
            update: dict[str, Any] = {"messages": [tagged]}
            if write_result.files_update is not None:
                update["files"] = write_result.files_update
            state_command = Command(update=update)
            messages = [*messages[:-1], tagged]

        processed: list[AnyMessage] = []
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.additional_kwargs.get("lc_evicted_to"):
                processed.append(_build_truncated_human_message(msg, msg.additional_kwargs["lc_evicted_to"]))
            else:
                processed.append(msg)

        return processed, state_command

    def _evict_and_truncate_messages(
        self,
        request: ModelRequest[ContextT],
    ) -> tuple[list[AnyMessage], Command | None] | None:
        """转存新的超大 HumanMessage，并截断所有已打标记的消息。

        如果没有任何消息需要处理，就返回 `None`（快速路径）。
        否则返回 `(processed_messages, command)`：其中 `command` 用来在 state 中
        标记新近被转存的消息；如果只是截断之前已经打标记的消息，则 `command` 为 `None`。

        Args:
            request: 当前正在处理的模型请求。

        Returns:
            如果发生了处理，则返回 (messages, command)；否则返回 `None`。
        """
        messages = list(request.messages)
        has_tagged, new_eviction_needed = self._check_eviction_needed(messages)
        if not has_tagged and not new_eviction_needed:
            return None

        write_result: WriteResult | None = None
        file_path: str | None = None
        if new_eviction_needed:
            backend = self._get_backend_from_runtime(request.state, request.runtime)
            file_path = f"/conversation_history/{uuid.uuid4()}.md"
            write_result = backend.write(file_path, _extract_text_from_message(messages[-1]))

        return self._apply_eviction_and_truncate(messages, write_result, file_path)

    async def _aevict_and_truncate_messages(
        self,
        request: ModelRequest[ContextT],
    ) -> tuple[list[AnyMessage], Command | None] | None:
        """`_evict_and_truncate_messages` 的异步版本。

        Args:
            request: 当前正在处理的模型请求。

        Returns:
            如果发生了处理，则返回 (messages, command)；否则返回 `None`。
        """
        messages = list(request.messages)
        has_tagged, new_eviction_needed = self._check_eviction_needed(messages)
        if not has_tagged and not new_eviction_needed:
            return None

        write_result: WriteResult | None = None
        file_path: str | None = None
        if new_eviction_needed:
            backend = self._get_backend_from_runtime(request.state, request.runtime)
            file_path = f"/conversation_history/{uuid.uuid4()}.md"
            write_result = await backend.awrite(file_path, _extract_text_from_message(messages[-1]))

        return self._apply_eviction_and_truncate(messages, write_result, file_path)

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """在工具结果写入 state 之前，拦截并处理过大的结果。

        Args:
            tool_result: 可能需要被转存的工具结果（ToolMessage 或 Command）。
            runtime: 提供文件系统后端访问能力的工具运行时。

        Returns:
            如果结果足够小，就原样返回；否则返回一个 Command，
            其中包含已写入文件系统的内容，以及一个截断后的消息。

        Note:
            这个函数既能处理单个 ToolMessage，也能处理包含多条消息的 Command。
            过大的内容会被自动转存到文件系统，以避免上下文窗口溢出。
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, files_update = self._process_large_message(
                tool_result,
                resolved_backend,
            )
            return (
                Command(
                    update={
                        "files": files_update,
                        "messages": [processed_message],
                    }
                )
                if files_update is not None
                else processed_message
            )

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            accumulated_file_updates = dict(update.get("files", {}))
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, files_update = self._process_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
                if files_update is not None:
                    accumulated_file_updates.update(files_update)
            return Command(update={**update, "messages": processed_messages, "files": accumulated_file_updates})
        msg = f"Unreachable code reached in _intercept_large_tool_result: for tool_result of type {type(tool_result)}"
        raise AssertionError(msg)

    async def _aintercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """`_intercept_large_tool_result` 的异步版本。

        使用异步后端方法，避免在异步上下文中调用同步接口。
        完整说明请参考 `_intercept_large_tool_result`。
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, files_update = await self._aprocess_large_message(
                tool_result,
                resolved_backend,
            )
            return (
                Command(
                    update={
                        "files": files_update,
                        "messages": [processed_message],
                    }
                )
                if files_update is not None
                else processed_message
            )

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            accumulated_file_updates = dict(update.get("files", {}))
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, files_update = await self._aprocess_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
                if files_update is not None:
                    accumulated_file_updates.update(files_update)
            return Command(update={**update, "messages": processed_messages, "files": accumulated_file_updates})
        msg = f"Unreachable code reached in _aintercept_large_tool_result: for tool_result of type {type(tool_result)}"
        raise AssertionError(msg)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """检查工具调用结果的大小，过大时就转存到文件系统。

        Args:
            request: 当前正在处理的工具调用请求。
            handler: 接收修改后请求的处理函数。

        Returns:
            原始 ToolMessage，或者一个在 state 中携带 ToolResult 的伪工具消息。
        """
        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return handler(request)

        tool_result = handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """（异步）检查工具调用结果的大小，过大时就转存到文件系统。

        Args:
            request: 当前正在处理的工具调用请求。
            handler: 接收修改后请求的处理函数。

        Returns:
            原始 ToolMessage，或者一个在 state 中携带 ToolResult 的伪工具消息。
        """
        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return await handler(request)

        tool_result = await handler(request)
        return await self._aintercept_large_tool_result(tool_result, request.runtime)