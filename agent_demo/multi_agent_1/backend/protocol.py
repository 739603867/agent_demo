import abc
import asyncio
import inspect
import logging
from functools import lru_cache
from typing import Literal, TypedDict, NotRequired, Any, TypeAlias, Callable

from dataclasses import dataclass

from langgraph.prebuilt import ToolRuntime

logger = logging.getLogger(__name__)

FileOperationError = Literal[
    "file_not_found",
    "permission_denied",
    "is_directory",
    "invalid_path",
]


class FileInfo(TypedDict):
    path: str
    is_dir: NotRequired[bool]
    size: NotRequired[int]
    modified_at: NotRequired[str]


class GrepMatch(TypedDict):
    path: str
    line: str
    text: str


class FileData(TypedDict):
    content: str
    encoding: str
    created_at: NotRequired[str]
    modified_at: NotRequired[str]


@dataclass
class ReadResult():
    """后端 read 操作的结果。

        Attributes:
            error: 失败时的错误信息，成功时为 None。
            file_data: 成功时的 FileData 字典，失败时为 None。
    """
    error: str | None = None
    file_data: FileData | None = None


@dataclass
class WriteResult():
    """后端 write 操作的结果。

        Attributes:
            error: 失败时的错误信息，成功时为 None。
            path: 成功时为已写入文件的绝对路径，失败时为 None。
            files_update: checkpoint 后端使用的状态更新字典；外部存储为 None。
                Checkpoint 后端会填充 `{file_path: file_data}` 以更新 LangGraph state。
                外部后端则设为 None（因为已持久化到磁盘 / S3 / 数据库等）。

        Examples:
            >>> # Checkpoint storage
            >>> WriteResult(path="/f.txt", files_update={"/f.txt": {...}})
            >>> # External storage
            >>> WriteResult(path="/f.txt", files_update=None)
            >>> # Error
            >>> WriteResult(error="File exists")
        """
    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None


@dataclass
class EditResult():
    """后端 edit 操作的结果。

        Attributes:
            error: 失败时的错误信息，成功时为 None。
            path: 成功时为已编辑文件的绝对路径，失败时为 None。
            files_update: checkpoint 后端使用的状态更新字典；外部存储为 None。
                Checkpoint 后端会填充 `{file_path: file_data}` 以更新 LangGraph state。
                外部后端则设为 None（因为已持久化到磁盘 / S3 / 数据库等）。
            occurrences: 成功时为完成的替换次数，失败时为 None。

        Examples:
            >>> # Checkpoint storage
            >>> EditResult(path="/f.txt", files_update={"/f.txt": {...}}, occurrences=1)
            >>> # External storage
            >>> EditResult(path="/f.txt", files_update=None, occurrences=2)
            >>> # Error
            >>> EditResult(error="File not found")
        """
    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None
    occurrences: int | None = None


@dataclass
class LsResult:
    """后端 ls 操作的结果。

    Attributes:
        error: 失败时的错误信息，成功时为 None。
        entries: 成功时为文件信息字典列表，失败时为 None。
    """

    error: str | None = None
    entries: list["FileInfo"] | None = None


@dataclass
class GrepResult:
    error: str | None = None
    matches: list["GrepMatch"] | None = None


@dataclass
class GlobResult:
    """后端 glob 操作的结果。

    Attributes:
        error: 失败时的错误信息，成功时为 None。
        matches: 成功时为匹配到的文件信息字典列表，失败时为 None。
    """

    error: str | None = None
    matches: list["FileInfo"] | None = None

@dataclass
class ExecuteResponse:
    """代码执行结果。

    为 LLM 消费场景优化的简化 schema。
    """

    output: str
    """执行命令后的 stdout 与 stderr 合并输出。"""

    exit_code: int | None = None
    """进程退出码。

    0 表示成功，非 0 表示失败。
    """

    truncated: bool = False
    """输出是否因后端限制而被截断。"""

@dataclass
class FileDownloadResponse:
    """单个文件下载操作的结果。

    该响应被设计为支持批量操作中的部分成功。

    对于特定可恢复场景，错误会尽量标准化为 `FileOperationError` 字面量，
    以适配 LLM 参与文件操作的用例。

    Examples:
        >>> # Success
        >>> FileDownloadResponse(path="/app/config.json", content=b"{...}", error=None)
        >>> # Failure
        >>> FileDownloadResponse(path="/wrong/path.txt", content=None, error="file_not_found")
    """

    path: str
    """请求的文件路径。

    该字段便于在处理批量结果时做关联，也方便生成清晰的错误信息。"""

    content: bytes | None = None
    """成功时为文件字节内容，失败时为 `None`。"""

    error: FileOperationError | None = None
    """已知场景下为 `FileOperationError` 字面量，否则为后端自定义错误字符串。

    成功时为 `None`。
    """


@dataclass
class FileUploadResponse:
    """单个文件上传操作的结果。

    该响应被设计为支持批量操作中的部分成功。

    对于特定可恢复场景，错误会尽量标准化为 `FileOperationError` 字面量，
    以适配 LLM 参与文件操作的用例。

    Examples:
        >>> # Success
        >>> FileUploadResponse(path="/app/data.txt", error=None)
        >>> # Failure
        >>> FileUploadResponse(path="/readonly/file.txt", error="permission_denied")
    """

    path: str
    """请求的文件路径。

    该字段便于在处理批量结果时做关联，也便于输出清晰的错误信息。
    """

    error: FileOperationError | None = None
    """已知场景下为 `FileOperationError` 字面量，否则为后端自定义错误字符串。

    成功时为 `None`。
    """


# 不使用 @abstractmethod，以避免破坏那些只实现了部分方法的子类
class BackendProtocol(abc.ABC):
    r"""可插拔内存后端的统一协议。

       后端可以将文件存储在不同位置（state、文件系统、数据库等），
       并为文件操作提供统一接口。

       所有文件数据都表示为如下结构的字典::

           {
               "content": str,  # 文本内容（utf-8）或 base64 编码的二进制
               "encoding": str,  # 文本为 "utf-8"，二进制为 "base64"
               "created_at": str,  # ISO 格式时间戳
               "modified_at": str,  # ISO 格式时间戳
           }
    """

    def ls(self, path: str) -> LsResult:
        raise NotImplementedError

    async def als(self, path: str) -> LsResult:
        """`ls` 的异步版本。"""
        return await asyncio.to_thread(self.ls, path)

    def read(self, file_path: str, offset: int, limit: int = 2000) -> ReadResult:
        """读取带行号语义的文件内容。

               Args:
                   file_path: 要读取的文件绝对路径，必须以 `/` 开头。
                   offset: 起始读取的行号（0 索引），默认 0。
                   limit: 最多读取的行数，默认 2000。

               Returns:
                   ReadResult，成功时包含读取结果，失败时包含错误信息。
        """
        raise NotImplementedError

    async def a_read(self, file_path: str, offset: int, limit: int = 2000) -> ReadResult:
        """read 的异步版本。"""
        return asyncio.to_thread(self.read, file_path, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        """向文件系统中的新文件写入内容；若文件已存在则报错。

               Args:
                   file_path: 要创建的文件绝对路径。

                       必须以 `/` 开头。
                   content: 要写入文件的字符串内容。

               Returns:
                   WriteResult。
        """
        raise NotImplementedError

    async def a_write(self, file_path: str, content: str) -> WriteResult:
        """
        write 的异步版本
        """
        return asyncio.to_thread(self.write, file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool) -> EditResult:
        """在现有文件中执行精确字符串替换。

                Args:
                    file_path: 要编辑的文件绝对路径，必须以 `'/'` 开头。
                    old_string: 要查找并替换的精确字符串。

                        必须连同空白和缩进完全匹配。
                    new_string: 用来替换 old_string 的字符串。

                        必须与 old_string 不同。
                    replace_all: 若为 True，则替换全部出现位置。

                        若为 False（默认），则 `old_string` 在文件中必须唯一，
                        否则编辑失败。

                Returns:
                    EditResult。
        """
        raise NotImplementedError

    async def a_edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool) -> EditResult:
        return asyncio.to_thread(self.edit, file_path, old_string, new_string, replace_all)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> "GrepResult":
        """在文件中搜索字面文本模式。

        Args:
            pattern: 要搜索的字面字符串（不是正则）。

                会在文件内容中执行精确子串匹配。

                示例：`"TODO"` 会匹配所有包含 `"TODO"` 的行。

            path: 可选的搜索目录路径。

                如果为 None，则搜索当前工作目录。

                示例：`'/workspace/src'`

            glob: 可选的 glob 模式，用于筛选要搜索的文件。

                它作用于文件名/路径，而不是内容。

                支持标准 glob 通配符：

                - `*` 匹配文件名中的任意字符
                - `**` 递归匹配任意目录
                - `?` 匹配单个字符
                - `[abc]` 匹配集合中的单个字符

        Examples:
            - `'*.py'` - 只搜索 Python 文件
            - `'**/*.txt'` - 递归搜索所有 `.txt` 文件
            - `'src/**/*.js'` - 搜索 src/ 下的 JS 文件
            - `'test[0-9].txt'` - 搜索 `test0.txt`、`test1.txt` 等

        Returns:
            带匹配结果或错误信息的 `GrepResult`。
        """
        raise NotImplementedError

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> "GrepResult":
        """`grep` 的异步版本。"""
        return await asyncio.to_thread(self.grep, pattern, path, glob)

    def glob(self, pattern: str, path: str = "/") -> "GlobResult":
        """查找匹配 glob 模式的文件。

        Args:
            pattern: 用于匹配文件路径的 glob 模式。

                支持标准 glob 语法：

                - `*` 匹配文件名/目录名中的任意字符
                - `**` 递归匹配任意目录
                - `?` 匹配单个字符
                - `[abc]` 匹配集合中的单个字符

            path: 搜索起始目录。

                默认值：`'/'`（根目录）。

                该模式会相对于此路径进行匹配。

        Returns:
            包含匹配文件或错误信息的 GlobResult。
        """
        raise NotImplementedError

    async def a_glob(self, pattern: str, path: str = "/") -> "GlobResult":
        """`glob` 的异步版本。"""
        return await asyncio.to_thread(self.glob, pattern, path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """向沙箱上传多个文件。

        该 API 既适合开发者直接调用，也适合通过自定义工具暴露给 LLM 使用。

        Args:
            files: 要上传的 `(path, content)` 元组列表。

        Returns:
            与输入文件一一对应的 FileUploadResponse 对象列表。

                返回顺序与输入顺序一致（`response[i]` 对应 `files[i]`）。

                可通过 error 字段判断每个文件的成功或失败。

        Examples:
            ```python
            responses = sandbox.upload_files(
                [
                    ("/app/config.json", b"{...}"),
                    ("/app/data.txt", b"content"),
                ]
            )
            ```
        """
        raise NotImplementedError

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """upload_files 的异步版本。"""
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """从沙箱下载多个文件。

        该 API 既适合开发者直接调用，也适合通过自定义工具暴露给 LLM 使用。

        Args:
            paths: 要下载的文件路径列表。

        Returns:
            与输入路径一一对应的 `FileDownloadResponse` 对象列表。

                返回顺序与输入顺序一致（`response[i]` 对应 `paths[i]`）。

                可通过 error 字段判断每个文件的成功或失败。
        """
        raise NotImplementedError

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """download_files 的异步版本。"""
        return await asyncio.to_thread(self.download_files, paths)

class SandboxBackendProtocol(BackendProtocol):
    """在 `BackendProtocol` 基础上增加 shell 命令执行能力。

    用于运行在隔离环境中的后端（容器、虚拟机、远程主机等）。

    新增 `execute()` / `aexecute()` 和 `id` 属性。

    参考 `BaseSandbox`：它通过委派给 `execute()` 实现了全部继承来的文件操作。
    """

    @property
    def id(self) -> str:
        """沙箱后端实例的唯一标识符。"""
        raise NotImplementedError

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """在沙箱环境中执行 shell 命令。

        这是为 LLM 消费场景优化后的简化接口。

        Args:
            command: 要执行的完整 shell 命令字符串。
            timeout: 等待命令完成的最大秒数。

                若为 None，则使用后端的默认超时。

                为保证跨后端行为一致，调用方应提供非负整数。
                某些支持“无超时执行”的后端中，0 可能表示禁用超时。

        Returns:
            带有合并输出、退出码和截断标志的 `ExecuteResponse`。
        """
        raise NotImplementedError

    async def aexecute(
        self,
        command: str,
        *,
        # ASYNC109 - timeout 是一个语义参数，会被转发给同步实现，
        # 不是 asyncio.timeout() 的契约。
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> ExecuteResponse:
        """execute 的异步版本。"""
        # 中间件层会先校验 timeout 支持情况，因此这里的保护
        # 只针对绕过中间件直接调用的场景。
        if timeout is not None and execute_accepts_timeout(type(self)):
            return await asyncio.to_thread(self.execute, command, timeout=timeout)
        return await asyncio.to_thread(self.execute, command)


@lru_cache(maxsize=128)
def execute_accepts_timeout(cls: type[SandboxBackendProtocol]) -> bool:
    """检查某个后端类的 `execute` 是否接受 `timeout` 关键字参数。

    较旧的后端包没有限制其 SDK 依赖下界，因此可能不支持
    后来在 `SandboxBackendProtocol` 中新增的 `timeout` 关键字。

    结果会按类缓存，以避免重复做签名反射带来的开销。
    """
    try:
        sig = inspect.signature(cls.execute)
    except (ValueError, TypeError):
        logger.warning(
            "Could not inspect signature of %s.execute; assuming timeout is not supported. This may indicate a backend packaging issue.",
            cls.__qualname__,
            exc_info=True,
        )
        return False
    else:
        return "timeout" in sig.parameters


BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]
BACKEND_TYPES = BackendProtocol | BackendFactory
