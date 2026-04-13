import datetime
import os
import re
import warnings
from pathlib import PurePosixPath, Path
from datetime import UTC, datetime
from typing import Literal, Any, overload, Sequence
from agent_demo.multi_agent_1.backend.protocol import FileData, FileInfo as _FileInfo, GrepMatch as _GrepMatch, \
    GrepResult, ReadResult
import wcmatch.glob as wcglob

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"

FileType = Literal["text", "image", "audio", "video", "file"]
"""按扩展名分类得到的文件类型。"""

_EXTENSION_TO_FILE_TYPE: dict[str, FileType] = {
    # 图片（https://ai.google.dev/gemini-api/docs/image-understanding）
    ".png": "image",
    ".jpeg": "image",
    ".jpg": "image",
    ".webp": "image",
    ".heic": "image",
    ".heif": "image",
    # 视频（https://ai.google.dev/gemini-api/docs/video-understanding）
    ".mp4": "video",
    ".mpeg": "video",
    ".mov": "video",
    ".avi": "video",
    ".flv": "video",
    ".mpg": "video",
    ".webm": "video",
    ".wmv": "video",
    ".3gpp": "video",
    # 音频（https://ai.google.dev/gemini-api/docs/audio）
    ".wav": "audio",
    ".mp3": "audio",
    ".aiff": "audio",
    ".aac": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    # 其他文件
    ".pdf": "file",
    ".ppt": "file",
    ".pptx": "file",
}
"""非文本文件的扩展名到类型映射。

来源于 Google 多模态 API 支持的格式：

- 图片：https://ai.google.dev/gemini-api/docs/image-understanding
- 视频：https://ai.google.dev/gemini-api/docs/video-understanding
- 音频：https://ai.google.dev/gemini-api/docs/audio
"""

MAX_LINE_LENGTH = 5000
LINE_NUMBER_WIDTH = 6
TOOL_RESULT_TOKEN_LIMIT = 20000  # Same threshold as eviction
TRUNCATION_GUIDANCE = "... [results truncated, try being more specific with your parameters]"

# 为了向后兼容而重新导出 protocol 类型
FileInfo = _FileInfo
GrepMatch = _GrepMatch


def _normalize_content(file_data: FileData) -> str:
    """将 file_data 中的内容规范化为普通字符串。

    这里是针对旧版 `list[str]` 文件格式进行向后兼容转换的唯一入口。
    新代码会将 `content` 存储为普通 `str`；旧数据仍可能保存为按行组成的列表。

    Args:
        file_data: 带有 `content` 键的 FileData 字典。

    Returns:
        规范化后的单个字符串内容。
    """
    content = file_data["content"]
    if isinstance(content, list):
        warnings.warn(
            "FileData with list[str] content is deprecated. Content should be stored as a plain str.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "\n".join(content)
    return content


def sanitize_tool_call_id(tool_call_id: str) -> str:
    r"""清理 tool_call_id，防止路径穿越和分隔符问题。

    会将危险字符（`.`, `/`, `\`）替换为下划线。
    """
    return tool_call_id.replace(".", "_").replace("/", "_").replace("\\", "_")


def format_content_with_line_numbers(
        content: str | list[str],
        start_line: int = 1,
) -> str:
    """按行号格式化文件内容（类似 `cat -n`）。

    对超过 MAX_LINE_LENGTH 的长行会切分为多段，并使用续行标记（如 5.1、5.2）。

    Args:
        content: 字符串形式的文件内容，或按行组成的列表。
        start_line: 起始行号（默认 1）。

    Returns:
        带行号和续行标记的格式化文本。
    """
    if isinstance(content, str):
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
    else:
        lines = content

    result_lines = []
    for i, line in enumerate(lines):
        line_num = i + start_line

        if len(line) <= MAX_LINE_LENGTH:
            result_lines.append(f"{line_num:{LINE_NUMBER_WIDTH}d}\t{line}")
        else:
            # 将长行拆分为多个分块，并带上续行标记
            num_chunks = (len(line) + MAX_LINE_LENGTH - 1) // MAX_LINE_LENGTH
            for chunk_idx in range(num_chunks):
                start = chunk_idx * MAX_LINE_LENGTH
                end = min(start + MAX_LINE_LENGTH, len(line))
                chunk = line[start:end]
                if chunk_idx == 0:
                    # 第一块使用正常的行号
                    result_lines.append(f"{line_num:{LINE_NUMBER_WIDTH}d}\t{chunk}")
                else:
                    # 后续分块使用小数形式标记（例如 5.1、5.2）
                    continuation_marker = f"{line_num}.{chunk_idx}"
                    result_lines.append(f"{continuation_marker:>{LINE_NUMBER_WIDTH}}\t{chunk}")

    return "\n".join(result_lines)


def check_empty_content(content: str) -> str | None:
    """检查内容是否为空，并返回对应的提示信息。

    Args:
        content: 要检查的内容。

    Returns:
        若内容为空则返回警告信息，否则返回 None。
    """
    if not content or content.strip() == "":
        return EMPTY_CONTENT_WARNING
    return None


def _get_file_type(path: str) -> FileType:
    """根据扩展名对文件进行分类。

    Args:
        path: 要分类的文件路径。

    Returns:
        `"text"`、`"image"`、`"audio"`、`"video"` 或 `"file"` 之一。
        对未识别的扩展名默认返回 `"text"`。
    """
    return _EXTENSION_TO_FILE_TYPE.get(PurePosixPath(path).suffix.lower(), "text")


def _to_legacy_file_data(file_data: FileData) -> dict[str, Any]:
    r"""将 FileData 字典转换为旧版（v1）存储格式。

    v1 格式会将内容存储为 `list[str]`（按 `\n` 拆分行），
    并省略 `encoding` 字段。当后端使用 `file_format="v1"` 时，
    应使用此函数以保持与依赖 `list[str]` 内容格式的旧消费者兼容。

    Args:
        file_data: 现代（v2）格式的 FileData，具有 `content: str` 和 `encoding`。

    Returns:
        其中 `content` 为 `list[str]` 的字典，并带有 `created_at` /
        `modified_at` 时间戳；不包含 `encoding` 键。
    """
    content = file_data["content"]
    result: dict[str, Any] = {
        "content": content.split("\n"),
    }
    if "created_at" in file_data:
        result["created_at"] = file_data["created_at"]
    if "modified_at" in file_data:
        result["modified_at"] = file_data["modified_at"]
    return result


def file_data_to_string(file_data: FileData) -> str:
    """将 FileData 转换为普通字符串内容。

    Args:
        file_data: 带有 `content` 键的 FileData 字典。

    Returns:
        单个字符串形式的内容。
    """
    return _normalize_content(file_data)


def create_file_data(
        content: str,
        created_at: str | None = None,
        encoding: str = "utf-8",
) -> FileData:
    """创建带时间戳的 FileData 对象。

    Args:
        content: 字符串形式的文件内容（纯文本或 base64 编码的二进制）。
        created_at: 可选的创建时间戳（ISO 格式）。
        encoding: 内容编码，`"utf-8"` 表示文本，`"base64"` 表示二进制。

    Returns:
        带 content、encoding 和时间戳的 FileData 字典。
    """
    now = datetime.now(UTC).isoformat()

    return {
        "content": content,
        "encoding": encoding,
        "created_at": created_at or now,
        "modified_at": now,
    }


def update_file_data(file_data: FileData, content: str) -> FileData:
    """用新内容更新 FileData，并保留创建时间戳。

    Args:
        file_data: 现有的 FileData 字典。
        content: 新的字符串内容。

    Returns:
        更新后的 FileData 字典。
    """
    now = datetime.now(UTC).isoformat()

    result = FileData(
        content=content,
        encoding=file_data.get("encoding", "utf-8"),
    )
    if "created_at" in file_data:
        result["created_at"] = file_data["created_at"]
    result["modified_at"] = now
    return result


def slice_read_response(
        file_data: FileData,
        offset: int,
        limit: int,
) -> str | ReadResult:
    """按请求的行范围切片文件数据，但不做格式化。

    返回请求窗口内的原始文本。行号格式化会在下游中间件层完成。

    Args:
        file_data: FileData 字典。
        offset: 行偏移（0 索引）。
        limit: 最多返回的行数。

    Returns:
        成功时返回切片后的原始内容字符串；如果 offset 超出文件长度，
        则返回带 `error` 的 `ReadResult`。
    """
    content = file_data_to_string(file_data)

    if not content or content.strip() == "":
        return content

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return ReadResult(error=f"Line offset {offset} exceeds file length ({len(lines)} lines)")

    selected_lines = lines[start_idx:end_idx]
    return "\n".join(selected_lines)


def format_read_response(
        file_data: FileData,
        offset: int,
        limit: int,
) -> str:
    """将文件数据格式化为带行号的读取响应。

    .. deprecated::
        请改用 `slice_read_response`，并单独调用
        `format_content_with_line_numbers`。

    Args:
        file_data: FileData 字典。
        offset: 行偏移（0 索引）。
        limit: 最多读取的行数。

    Returns:
        格式化后的内容，或错误信息。
    """
    content = file_data_to_string(file_data)
    empty_msg = check_empty_content(content)
    if empty_msg:
        return empty_msg

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    selected_lines = lines[start_idx:end_idx]
    return format_content_with_line_numbers(selected_lines, start_line=start_idx + 1)


def perform_string_replacement(
        content: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
) -> tuple[str, int] | str:
    """执行字符串替换，并校验出现次数。

    Args:
        content: 原始内容。
        old_string: 要被替换的字符串。
        new_string: 替换后的字符串。
        replace_all: 是否替换全部出现位置。

    Returns:
        成功时返回 `(new_content, occurrences)`；失败时返回错误信息字符串。
    """
    occurrences = content.count(old_string)

    if occurrences == 0:
        return f"Error: String not found in file: '{old_string}'"

    if occurrences > 1 and not replace_all:
        return (
            f"Error: String '{old_string}' appears {occurrences} times in file. "
            f"Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
        )

    new_content = content.replace(old_string, new_string)
    return new_content, occurrences


@overload
def truncate_if_too_long(result: list[str]) -> list[str]: ...


@overload
def truncate_if_too_long(result: str) -> str: ...


def truncate_if_too_long(result: list[str] | str) -> list[str] | str:
    """当列表或字符串结果超过 token 限制时进行截断（粗略按 4 个字符约等于 1 个 token 估算）。"""
    if isinstance(result, list):
        total_chars = sum(len(item) for item in result)
        if total_chars > TOOL_RESULT_TOKEN_LIMIT * 4:
            return result[: len(result) * TOOL_RESULT_TOKEN_LIMIT * 4 // total_chars] + [
                TRUNCATION_GUIDANCE]  # noqa: RUF005  # Concatenation preferred for clarity
        return result
    # 字符串
    if len(result) > TOOL_RESULT_TOKEN_LIMIT * 4:
        return result[: TOOL_RESULT_TOKEN_LIMIT * 4] + "\n" + TRUNCATION_GUIDANCE
    return result


def validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    r"""为安全目的校验并规范化文件路径。

    通过阻止目录穿越并统一路径格式，确保路径可安全使用。
    所有路径都会被规范为使用正斜杠，并以 `/` 开头。

    该函数面向虚拟文件系统路径设计，会拒绝 Windows 绝对路径
    （例如 `C:/...`、`F:/...`），以保持一致性并避免路径格式歧义。

    Args:
        path: 要校验并规范化的路径。
        allowed_prefixes: 可选的允许路径前缀列表。

            如果提供，规范化后的路径必须以其中之一开头。

    Returns:
        以 `/` 开头、使用正斜杠的规范化路径。

    Raises:
        ValueError: 当路径包含穿越序列（`..` 或 `~`）、是 Windows 绝对路径，
            或在指定了 `allowed_prefixes` 时未以前缀之一开头。

    Example:
        ```python
        validate_path("foo/bar")  # Returns: "/foo/bar"
        validate_path("/./foo//bar")  # Returns: "/foo/bar"
        validate_path("../etc/passwd")  # Raises ValueError
        validate_path(r"C:\\Users\\file.txt")  # Raises ValueError
        validate_path("/data/file.txt", allowed_prefixes=["/data/"])  # OK
        validate_path("/etc/file.txt", allowed_prefixes=["/data/"])  # Raises ValueError
        ```
    """
    # 检查是否存在作为路径片段的穿越序列（而不是普通子串），以避免
    # 错误拒绝诸如 "foo..bar.txt" 这样的合法文件名
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if ".." in parts or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # 拒绝 Windows 绝对路径（例如 C:\...、D:/...）
    if re.match(r"^[a-zA-Z]:", path):
        msg = f"Windows absolute paths are not supported: {path}. Please use virtual paths starting with / (e.g., /workspace/file.txt)"
        raise ValueError(msg)

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    # 纵深防御：确认 normpath 没有产生路径穿越
    if ".." in normalized.split("/"):
        msg = f"Path traversal detected after normalization: {path} -> {normalized}"
        raise ValueError(msg)

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


def _normalize_path(path: str | None) -> str:
    """将路径规范化为标准形式。

    会将路径转换为以 `/` 开头的绝对形式，去除尾部斜杠
    （根路径除外），并校验路径非空。

    Args:
        path: 要规范化的路径（None 默认为 `"/"`）。

    Returns:
        以 `/` 开头的规范化路径（除根路径外不带尾部斜杠）。

    Raises:
        ValueError: 当路径无效（去空白后为空字符串）时。

    Example:
        _normalize_path(None) -> "/"
        _normalize_path("/dir/") -> "/dir"
        _normalize_path("dir") -> "/dir"
        _normalize_path("/") -> "/"
    """
    path = path or "/"
    if not path or path.strip() == "":
        msg = "Path cannot be empty"
        raise ValueError(msg)

    normalized = path if path.startswith("/") else "/" + path

    # 只有根路径应保留尾部斜杠
    if normalized != "/" and normalized.endswith("/"):
        normalized = normalized.rstrip("/")

    return normalized


def _filter_files_by_path(files: dict[str, Any], normalized_path: str) -> dict[str, Any]:
    """按规范化路径过滤文件字典，同时处理精确文件匹配和目录前缀匹配。

    要求传入的是 `_normalize_path` 返回的规范化路径
    （除根路径外不带尾部斜杠）。

    Args:
        files: 从文件路径映射到文件数据的字典。
        normalized_path: `_normalize_path` 返回的规范化路径（例如 `/`、`/dir`、`/dir/file`）。

    Returns:
        与该路径匹配的过滤后文件字典。

    Example:
        files = {"/dir/file": {...}, "/dir/other": {...}}
        _filter_files_by_path(files, "/dir/file")  # Returns {"/dir/file": {...}}
        _filter_files_by_path(files, "/dir")       # Returns both files
    """
    # 检查路径是否精确匹配某个文件
    if normalized_path in files:
        return {normalized_path: files[normalized_path]}

    # 否则按目录前缀处理
    if normalized_path == "/":
        # 根目录：匹配所有以 / 开头的文件
        return {fp: fd for fp, fd in files.items() if fp.startswith("/")}
    # 非根目录：补上尾部斜杠做前缀匹配
    dir_prefix = normalized_path + "/"
    return {fp: fd for fp, fd in files.items() if fp.startswith(dir_prefix)}


def _glob_search_files(
        files: dict[str, Any],
        pattern: str,
        path: str = "/",
) -> str:
    r"""在文件字典中搜索与 glob 模式匹配的路径。

    Args:
        files: 文件路径到 FileData 的映射字典。
        pattern: glob 模式（例如 `"*.py"`、`"**/*.ts"`）。
        path: 搜索起始的基础路径。

    Returns:
        以换行分隔的文件路径字符串，按 modified_at 从近到远排序。
        若无匹配，返回 `"No files found"`。

    Example:
        ```python
        files = {"/src/main.py": FileData(...), "/test.py": FileData(...)}
        _glob_search_files(files, "*.py", "/")
        # Returns: "/test.py\n/src/main.py"（按 modified_at 排序）
        ```
    """
    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return "No files found"

    filtered = _filter_files_by_path(files, normalized_path)

    # 遵循标准 glob 语义：
    # - 不含路径分隔符的模式（如 `*.py`）只会匹配相对于 `path`
    #   的当前目录（非递归）。
    # - 如需递归匹配，必须显式使用 `**`。
    # 匹配是相对路径进行的，因此先去掉 pattern 前导的 `/`。
    effective_pattern = pattern.lstrip("/")

    matches = []
    for file_path, file_data in filtered.items():
        # 计算用于 glob 匹配的相对路径
        # 如果 normalized_path 是 "/dir"，我们希望 "/dir/file.txt" -> "file.txt"
        # 如果 normalized_path 是 "/dir/file.txt"（精确文件），我们希望得到 "file.txt"
        if normalized_path == "/":
            relative = file_path[1:]  # 去掉前导斜杠
        elif file_path == normalized_path:
            # 精确文件匹配：只使用文件名
            relative = file_path.split("/")[-1]
        else:
            # 目录前缀匹配：去掉目录路径部分
            relative = file_path[len(normalized_path) + 1:]  # +1 表示斜杠

        if wcglob.globmatch(relative, effective_pattern, flags=wcglob.BRACE | wcglob.GLOBSTAR):
            matches.append((file_path, file_data["modified_at"]))

    matches.sort(key=lambda x: x[1], reverse=True)

    if not matches:
        return "No files found"

    return "\n".join(fp for fp, _ in matches)


def _format_grep_results(
        results: dict[str, list[tuple[int, str]]],
        output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """根据输出模式格式化 grep 搜索结果。

    Args:
        results: 从文件路径映射到 `(line_num, line_content)` 列表的字典。
        output_mode: 输出格式，可选 `"files_with_matches"`、`"content"` 或 `"count"`。

    Returns:
        格式化后的字符串输出。
    """
    if output_mode == "files_with_matches":
        return "\n".join(sorted(results.keys()))
    if output_mode == "count":
        lines = []
        for file_path in sorted(results.keys()):
            count = len(results[file_path])
            lines.append(f"{file_path}: {count}")
        return "\n".join(lines)
    lines = []
    for file_path in sorted(results.keys()):
        lines.append(f"{file_path}:")
        for line_num, line in results[file_path]:
            lines.append(f"  {line_num}: {line}")
    return "\n".join(lines)


def _grep_search_files(
        files: dict[str, Any],
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
) -> str:
    r"""在文件内容中搜索正则模式。

    Args:
        files: 文件路径到 FileData 的映射字典。
        pattern: 要搜索的正则表达式模式。
        path: 搜索起始的基础路径。
        glob: 可选的 glob 模式，用于筛选文件（如 `"*.py"`）。
        output_mode: 输出格式，可选 `"files_with_matches"`、`"content"` 或 `"count"`。

    Returns:
        格式化后的搜索结果；若无结果，返回 `"No matches found"`。

    Example:
        ```python
        files = {"/file.py": FileData(content="import os\nprint('hi')", ...)}
        _grep_search_files(files, "import", "/")
        # Returns: "/file.py"（当 output_mode="files_with_matches" 时）
        ```
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return "No matches found"

    filtered = _filter_files_by_path(files, normalized_path)

    if glob:
        filtered = {fp: fd for fp, fd in filtered.items() if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)}

    results: dict[str, list[tuple[int, str]]] = {}
    for file_path, file_data in filtered.items():
        content_str = _normalize_content(file_data)
        for line_num, line in enumerate(content_str.split("\n"), 1):
            if regex.search(line):
                if file_path not in results:
                    results[file_path] = []
                results[file_path].append((line_num, line))

    if not results:
        return "No matches found"
    return _format_grep_results(results, output_mode)


# -------- 供组合逻辑使用的结构化辅助函数 --------


def grep_matches_from_files(
        files: dict[str, Any],
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
) -> GrepResult:
    """从内存中的文件映射返回结构化 grep 匹配结果。

    执行的是字面文本搜索（不是正则）。

    成功时返回带 matches 的 GrepResult。
    这里刻意不抛异常，以保持后端在工具上下文中不抛出，并保留面向用户的错误信息行为。
    """
    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return GrepResult(matches=[])

    filtered = _filter_files_by_path(files, normalized_path)

    if glob:
        filtered = {fp: fd for fp, fd in filtered.items() if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)}

    matches: list[GrepMatch] = []
    for file_path, file_data in filtered.items():
        content_str = _normalize_content(file_data)
        for line_num, line in enumerate(content_str.split("\n"), 1):
            if pattern in line:  # 简单子串搜索，用于字面匹配
                matches.append({"path": file_path, "line": int(line_num), "text": line})
    return GrepResult(matches=matches)


def build_grep_results_dict(matches: list[GrepMatch]) -> dict[str, list[tuple[int, str]]]:
    """将结构化匹配结果分组为 formatter 使用的旧版字典形式。"""
    grouped: dict[str, list[tuple[int, str]]] = {}
    for m in matches:
        grouped.setdefault(m["path"], []).append((m["line"], m["text"]))
    return grouped


def format_grep_matches(
        matches: list[GrepMatch],
        output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """使用现有格式化逻辑来格式化结构化 grep 匹配结果。"""
    if not matches:
        return "No matches found"
    return _format_grep_results(build_grep_results_dict(matches), output_mode)
