"""`StateBackend`：将文件存储在 LangGraph agent state 中（临时）。"""

import base64
from typing import TYPE_CHECKING, Any

from agent_demo.multi_agent_1.backend.protocol import (
    BackendProtocol,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from agent_demo.multi_agent_1.backend.utils import (
    _get_file_type,
    _glob_search_files,
    _to_legacy_file_data,
    create_file_data,
    file_data_to_string,
    grep_matches_from_files,
    perform_string_replacement,
    slice_read_response,
    update_file_data,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime


class StateBackend(BackendProtocol):
    """将文件存储在 agent state 中的后端（临时）。

    使用 LangGraph 的状态管理与 checkpoint 机制。文件会在同一会话线程内持久存在，
    但不会跨线程保留。状态会在每个 agent 步骤后自动 checkpoint。

    特殊处理：由于 LangGraph state 必须通过 Command 对象更新
    （而不是直接修改），因此这些操作返回的是 Command 对象而非 None。
    这一点通过 `uses_state=True` 标志体现。
    """

    def __init__(
        self,
        runtime: "ToolRuntime",
    ) -> None:
        r"""使用 runtime 初始化 StateBackend。

        Args:
            runtime: 提供 store 访问和配置的 `ToolRuntime` 实例。
        """
        self.runtime = runtime

    def _prepare_for_storage(self, file_data: FileData) -> dict[str, Any]:
        """将 FileData 转换为 state 存储所使用的格式。

        """
        return {**file_data}

    def ls(self, path: str) -> LsResult:
        """以非递归方式列出指定目录中的文件和目录。

        Args:
            path: 目录的绝对路径。

        Returns:
            返回当前目录下直接子项的 FileInfo 风格字典列表。
            目录路径会带有结尾的 `/`，并且 `is_dir=True`。
        """
        files = self.runtime.state.get("files", {})
        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        # 将路径规范化为带尾部斜杠的形式，以便正确匹配前缀
        normalized_path = path if path.endswith("/") else path + "/"

        for k, fd in files.items():
            # 检查文件是否位于指定目录或其子目录中
            if not k.startswith(normalized_path):
                continue

            # 获取该目录之后的相对路径
            relative = k[len(normalized_path) :]

            # 如果相对路径里包含 `/`，说明它位于子目录中
            if "/" in relative:
                # 提取直属子目录名称
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
                continue

            # 这是当前目录中的直接文件
            # 向后兼容：计算大小时处理旧版的 list[str] 内容
            raw = fd.get("content", "")
            size = len("\n".join(raw)) if isinstance(raw, list) else len(raw)
            infos.append(
                {
                    "path": k,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", ""),
                }
            )

        # 将目录加入结果列表
        infos.extend(FileInfo(path=subdir, is_dir=True, size=0, modified_at="") for subdir in sorted(subdirs))

        infos.sort(key=lambda x: x.get("path", ""))
        return LsResult(entries=infos)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """读取请求行范围内的文件内容。

        Args:
            file_path: 文件的绝对路径。
            offset: 起始读取的行偏移（0 索引）。
            limit: 最多读取的行数。

        Returns:
            返回指定窗口内原始（未格式化）的内容的 ReadResult。
            行号格式化由中间件完成。
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)

        if file_data is None:
            return ReadResult(error=f"File '{file_path}' not found")

        if _get_file_type(file_path) != "text":
            return ReadResult(file_data=file_data)

        sliced = slice_read_response(file_data, offset, limit)
        if isinstance(sliced, ReadResult):
            return sliced
        sliced_fd = FileData(
            content=sliced,
            encoding=file_data.get("encoding", "utf-8"),
        )
        if "created_at" in file_data:
            sliced_fd["created_at"] = file_data["created_at"]
        if "modified_at" in file_data:
            sliced_fd["modified_at"] = file_data["modified_at"]
        return ReadResult(file_data=sliced_fd)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """创建一个带内容的新文件。

        返回带有 `files_update` 的 WriteResult，用于更新 LangGraph state。
        """
        files = self.runtime.state.get("files", {})

        if file_path in files:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        new_file_data = create_file_data(content)
        return WriteResult(path=file_path, files_update={file_path: self._prepare_for_storage(new_file_data)})

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """通过替换字符串出现位置来编辑文件。

        返回带有 `files_update` 和出现次数的 EditResult。
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)

        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        return EditResult(path=file_path, files_update={file_path: self._prepare_for_storage(new_file_data)}, occurrences=int(occurrences))

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """在 state 文件中按字面文本模式搜索。"""
        files = self.runtime.state.get("files", {})
        return grep_matches_from_files(files, pattern, path if path is not None else "/", glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """获取匹配 glob 模式的文件的 FileInfo。"""
        files = self.runtime.state.get("files", {})
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return GlobResult(matches=[])
        paths = result.split("\n")
        infos: list[FileInfo] = []
        for p in paths:
            fd = files.get(p)
            if fd:
                # 向后兼容：计算大小时处理旧版的 list[str] 内容
                raw = fd.get("content", "")
                size = len("\n".join(raw)) if isinstance(raw, list) else len(raw)
            else:
                size = 0
            infos.append(
                {
                    "path": p,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", "") if fd else "",
                }
            )
        return GlobResult(matches=infos)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """向 state 中上传多个文件。

        Args:
            files: 要上传的 `(path, content)` 元组列表

        Returns:
            与输入文件一一对应的 FileUploadResponse 对象列表
        """
        msg = (
            "StateBackend does not support upload_files yet. You can upload files "
            "directly by passing them in invoke if you're storing files in the memory."
        )
        raise NotImplementedError(msg)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """从 state 中下载多个文件。

        Args:
            paths: 要下载的文件路径列表

        Returns:
            与输入路径一一对应的 FileDownloadResponse 对象列表
        """
        state_files = self.runtime.state.get("files", {})
        responses: list[FileDownloadResponse] = []

        for path in paths:
            file_data = state_files.get(path)

            if file_data is None:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue

            content_str = file_data_to_string(file_data)

            encoding = file_data.get("encoding", "utf-8")
            content_bytes = content_str.encode("utf-8") if encoding == "utf-8" else base64.standard_b64decode(content_str)
            responses.append(FileDownloadResponse(path=path, content=content_bytes, error=None))

        return responses
