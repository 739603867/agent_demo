import base64
import json
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

import wcmatch.glob as wcglob

from agent_demo.multi_agent_1.backend.protocol import BackendProtocol, EditResult, WriteResult, ReadResult, LsResult, \
    FileInfo, FileData, GrepResult, GrepMatch, GlobResult, ExecuteResponse, FileDownloadResponse, FileUploadResponse
from agent_demo.multi_agent_1.backend.utils import _get_file_type, check_empty_content, perform_string_replacement

logger = logging.getLogger(__name__)

FileOperationError = Literal[
    "file_not_found",
    "permission_denied",
    "is_directory",
    "invalid_path",
]

class FileSystemProtocol(BackendProtocol):
    """直接通过文件系统读写文件的后端。

        文件使用它们在文件系统中的真实路径进行访问。相对路径会基于当前工作目录来解析。
        内容以纯文本形式读写，元数据（时间戳）来自文件系统的 stat 信息。

        !!! warning "安全警告"

            这个后端会授予 agent 直接读写文件系统的能力。请谨慎使用，并且只在合适的环境中使用。

            **适合的使用场景：**

            - 本地开发 CLI（如编码助手、开发工具）
            - CI/CD 流水线（见下方安全注意事项）

            **不适合的使用场景：**

            - Web 服务器或 HTTP API —— 此时应改用 `StateBackend`、`StoreBackend` 或
                `SandboxBackend`

            **安全风险：**

            - Agent 可以读取任何有权限访问的文件，包括敏感信息（API key、凭证、`.env` 文件）
            - 如果再结合网络工具，可能通过 SSRF 攻击把这些敏感信息泄露出去
            - 对文件的修改是永久性的、不可逆的

            **推荐的保护措施：**

            1. 启用 Human-in-the-Loop（HITL）中间件，对敏感操作进行人工审核
            2. 不要让敏感文件出现在可访问的文件路径中（尤其是在 CI/CD 里）
            3. 在生产环境中，优先使用 `StateBackend`、`StoreBackend` 或 `SandboxBackend`

            通常我们希望这个后端搭配 Human-in-the-Loop（HITL）中间件一起使用；
            如果需要运行不受信任的工作负载，也应放在正确隔离的沙箱环境中。

            !!! note

                `virtual_mode=True` 的主要用途是提供“虚拟路径语义”（例如配合
                `CompositeBackend` 使用）。它也能通过阻止路径穿越（`..`、`~`）以及
                阻止访问 `root_dir` 之外的绝对路径，提供一定的基于路径的保护。
                但它**并不**提供真正的沙箱能力，也不提供进程隔离。
                默认值 `virtual_mode=False` 即使设置了 `root_dir`，也**不具备安全性保证**。
    """

    def __init__(self, root_dir: str, virtual_mod: bool, max_file_size_mb: int = 10) -> None:
        """初始化文件系统后端。

                Args:
                    root_dir: 文件操作的可选根目录。

                        默认为当前工作目录。

                        - 当 `virtual_mode=False`（默认）时：它只影响相对路径如何解析。
                        - 当 `virtual_mode=True` 时：它会作为文件系统操作的虚拟根目录。

                    virtual_mode: 是否启用虚拟路径模式。

                        **主要用途：** 当与 `CompositeBackend` 一起使用时，提供稳定的、与具体后端无关的路径语义。
                        `CompositeBackend` 会去掉路由前缀，并把规范化后的路径转发给目标后端。

                        当为 `True` 时，所有路径都会被当成以 `root_dir` 为锚点的虚拟路径。
                        会阻止路径穿越（`..`、`~`），并确保解析后的路径始终留在 `root_dir` 内。

                        当为 `False`（默认）时，绝对路径会原样使用，相对路径会基于 `root_dir` 解析。
                        这种模式无法阻止 agent 访问 `root_dir` 之外的路径。

                        - 绝对路径（例如 `/etc/passwd`）会完全绕过 `root_dir`
                        - 带有 `..` 的相对路径可以逃出 `root_dir`
                        - agent 将拥有不受限制的文件系统访问能力

                    max_file_size_mb: 某些操作（如 grep 的 Python 回退搜索）允许处理的最大文件大小，单位 MB。

                        超过这个大小的文件在搜索时会被跳过。默认是 10 MB。
        """
        super().__init__()
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self.virtual_mode = virtual_mod
        self.max_file_size_md = max_file_size_mb * 1024 * 1024

    def _resolve_path(self, key: str) -> Path:
        """在带安全检查的情况下解析文件路径。

                当 `virtual_mode=True` 时，会把传入路径视为位于 `self.cwd` 下的“虚拟绝对路径”，
                禁止路径穿越（`..`、`~`），并确保最终解析后的路径不会逃出根目录。

                当 `virtual_mode=False` 时，保留旧行为：绝对路径可直接使用；相对路径会基于 cwd 解析。

                Args:
                    key: 文件路径（可以是绝对路径、相对路径，或者在 `virtual_mode=True` 下的虚拟路径）。

                Returns:
                    解析后的绝对 `Path` 对象。

                Raises:
                    ValueError: 当在 `virtual_mode` 下检测到路径穿越，或解析后的路径跑到了根目录之外时抛出。
        """
        if self.virtual_mode:
            vpath = key if key.startswith("/") else "/" + key
            if ".." in vpath or vpath.startswith("~"):
                msg = "Path traversal not allowed"
                raise ValueError(msg)
            full = (self.cwd / vpath.lstrip("/")).resolve()
            try:
                full.relative_to(self.cwd)
            except ValueError:
                msg = f"Path:{full} outside root directory: {self.cwd}"
                raise ValueError(msg) from None
            return full

        path = Path(key)
        if path.is_absolute():
            return path
        return (self.cwd / path).resolve()

    def _to_virtual_path(self, path: Path) -> str:
        """把文件系统路径转换成相对于 cwd 的虚拟路径。

            Args:
                path: 要转换的文件系统路径。

            Returns:
                以前导 `/` 开头、使用正斜杠的相对路径字符串。

            Raises:
                ValueError: 如果路径位于 cwd 之外。
                OSError: 如果路径无法解析（例如损坏的符号链接、权限不足）。
        """
        return "/" + path.resolve().relative_to(self.cwd).as_posix()

    def ls(self, path: str) -> LsResult:
        """列出指定目录中的文件和目录（非递归）。

            Args:
                path: 要列出文件的绝对目录路径。

            Returns:
                返回当前目录下直接文件和目录的 `FileInfo` 风格字典列表。
                目录路径会带有结尾的 `/`，并且 `is_dir=True`。
        """
        dir_path = self._resolve_path(path)
        if not dir_path.exists() or not dir_path.is_dir():
            return LsResult(entries=[])

        results: list[FileInfo] = []

        # 把 cwd 转成字符串，便于后面比较
        cwd_str = str(self.cwd)
        if not cwd_str.endswith("/"):
            cwd_str += "/"
            # 只列出当前目录的直属子项（不递归）
            try:
                for child_path in dir_path.iterdir():
                    try:
                        is_file = child_path.is_file()
                        is_dir = child_path.is_dir()
                    except OSError:
                        continue

                    abs_path = str(child_path)

                    if not self.virtual_mode:
                        # 非虚拟模式：使用绝对路径
                        if is_file:
                            try:
                                st = child_path.stat()
                                results.append(
                                    {
                                        "path": abs_path,
                                        "is_dir": False,
                                        "size": int(st.st_size),
                                        "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                                        # noqa: DTZ006  # Local filesystem timestamps don't need timezone
                                    }
                                )
                            except OSError:
                                results.append({"path": abs_path, "is_dir": False})
                        elif is_dir:
                            try:
                                st = child_path.stat()
                                results.append(
                                    {
                                        "path": abs_path + "/",
                                        "is_dir": True,
                                        "size": 0,
                                        "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                                        # noqa: DTZ006  # Local filesystem timestamps don't need timezone
                                    }
                                )
                            except OSError:
                                results.append({"path": abs_path + "/", "is_dir": True})
                    else:
                        # 虚拟模式：使用 Path 去掉 cwd 前缀，以兼容不同平台
                        try:
                            virt_path = self._to_virtual_path(child_path)
                        except ValueError:
                            logger.debug("Skipping path outside root: %s", child_path)
                            continue
                        except OSError:
                            logger.warning("Could not resolve path: %s", child_path, exc_info=True)
                            continue

                        if is_file:
                            try:
                                st = child_path.stat()
                                results.append(
                                    {
                                        "path": virt_path,
                                        "is_dir": False,
                                        "size": int(st.st_size),
                                        "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                                        # noqa: DTZ006  # Local filesystem timestamps don't need timezone
                                    }
                                )
                            except OSError:
                                results.append({"path": virt_path, "is_dir": False})
                        elif is_dir:
                            try:
                                st = child_path.stat()
                                results.append(
                                    {
                                        "path": virt_path + "/",
                                        "is_dir": True,
                                        "size": 0,
                                        "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                                        # noqa: DTZ006  # Local filesystem timestamps don't need timezone
                                    }
                                )
                            except OSError:
                                results.append({"path": virt_path + "/", "is_dir": True})
            except (OSError, PermissionError):
                pass

            # 按路径排序，保证返回顺序稳定
            results.sort(key=lambda x: x.get("path", ""))
            return LsResult(entries=results)

    def read(self, file_path: str, offset: int, limit: int = 2000) -> ReadResult:
        """读取指定行范围内的文件内容。

            Args:
                file_path: 绝对或相对文件路径。
                offset: 从第几行开始读取（0 下标）。
                limit: 最多读取多少行。

            Returns:
                返回 `ReadResult`，其中包含所请求窗口范围内的原始（未格式化）内容。
                行号格式化由中间件负责处理。
        """
        resolved_path = self._resolve_path(file_path)

        if not resolved_path.exists() or not resolved_path.is_file():
            return ReadResult(error=f"File '{file_path}' not found")

        try:
            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            if _get_file_type(file_path) != "text":
                with os.fdopen(fd, "rb") as f:
                    raw = f.read()
                encoded = base64.standard_b64encode(raw).decode("ascii")
                return ReadResult(file_data=FileData(content=encoded, encoding="base64"))

            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            empty_msg = check_empty_content(content)
            if empty_msg:
                return ReadResult(file_data=FileData(content=empty_msg, encoding="utf-8"))

            lines = content.splitlines()
            start_idx = offset
            end_idx = min(start_idx + limit, len(lines))

            if start_idx >= len(lines):
                return ReadResult(error=f"Line offset {offset} exceeds file length ({len(lines)} lines)")

            selected_lines = lines[start_idx:end_idx]
            return ReadResult(file_data=FileData(content="\n".join(selected_lines), encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as e:
            return ReadResult(error=f"Error reading file '{file_path}': {e}")

    def write(self, file_path: str, content: str) -> WriteResult:
        """创建一个新文件并写入内容。

               Args:
                   file_path: 新文件要创建到的路径。
                   content: 要写入文件的文本内容。

               Returns:
                   成功时返回带路径信息的 `WriteResult`；
                   如果文件已存在或写入失败，则返回错误信息。
                   对于外部存储，`files_update=None`。
        """
        resolved_path = self._resolve_path(file_path)

        if resolved_path.exists():
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        try:
            # 如果父目录不存在，就先创建
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            # 优先使用 O_NOFOLLOW，避免通过符号链接进行写入
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)

            return WriteResult(path=file_path, files_update=None)
        except (OSError, UnicodeEncodeError) as e:
            return WriteResult(error=f"写入文件 '{file_path}' 时出错：{e}")

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool) -> EditResult:
        """通过替换字符串内容来编辑文件。

                Args:
                    file_path: 要编辑的文件路径。
                    old_string: 要查找并替换的原文本。
                    new_string: 替换后的新文本。
                    replace_all: 如果为 `True`，替换所有匹配项；如果为 `False`（默认），只有当匹配项恰好出现一次时才替换。

                Returns:
                    成功时返回带路径和替换次数的 `EditResult`；
                    如果文件不存在或替换失败，则返回错误信息。
                    对于外部存储，`files_update=None`。
        """

        resolved_path = self._resolve_path(file_path)

        if not resolved_path.exists() or not resolved_path.is_file():
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            # 安全地读取文件
            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            # 统一 old_string / new_string 中的换行符格式，使其和上面以文本模式读取出来的内容一致。
            # Python 的“通用换行符”机制（newline=None 时的默认行为）
            # 会在读取时把 \r\n 和单独的 \r 都转换成 \n。
            # 如果调用方是通过二进制模式读取内容（例如 download_files）得到字符串，
            # 那它传进来的内容可能还带着 \r\n 或 \r，
            # 这样就会和当前文件中已经规范成 \n 的内容匹配不上。
            old_string = old_string.replace("\r\n", "\n").replace("\r", "\n")
            new_string = new_string.replace("\r\n", "\n").replace("\r", "\n")

            result = perform_string_replacement(content, old_string, new_string, replace_all)

            if isinstance(result, str):
                return EditResult(error=result)

            new_content, occurrences = result

            # 安全地写回文件
            flags = os.O_WRONLY | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)

            return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return EditResult(error=f"编辑文件 '{file_path}' 时出错：{e}")

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> "GrepResult":
        """在文件中搜索字面量文本模式。

               如果系统中有 ripgrep，就优先使用它；否则回退到 Python 实现的搜索。

               Args:
                   pattern: 要搜索的字面字符串（不是正则）。
                   path: 要搜索的目录或文件路径。默认是当前目录。
                   glob: 可选的 glob 模式，用于过滤要搜索的文件。

               Returns:
                   返回包含匹配结果或错误信息的 `GrepResult`。
               """
        # 解析基础搜索路径
        try:
            base_full = self._resolve_path(path or ".")
        except ValueError:
            return GrepResult(matches=[])

        if not base_full.exists():
            return GrepResult(matches=[])

        # 先尝试使用 ripgrep（`-F` 表示按字面量字符串搜索）
        results = self._ripgrep_search(pattern, base_full, glob)
        if results is None:
            # Python 回退方案里需要先转义 pattern，才能实现字面量搜索
            results = self._python_search(re.escape(pattern), base_full, glob)

        matches: list[GrepMatch] = []
        for fpath, items in results.items():
            for line_num, line_text in items:
                matches.append({"path": fpath, "line": int(line_num), "text": line_text})
        return GrepResult(matches=matches)

    def _ripgrep_search(self, pattern: str, base_full: Path, include_glob: str | None) -> dict[str, list[
        tuple[int, str]]] | None:  # noqa: C901  # 为了日志把 except 子句拆开了
        """使用 ripgrep 的固定字符串（字面量）模式进行搜索。

        Args:
            pattern: 要搜索的字面字符串（未转义）。
            base_full: 已解析好的基础搜索路径。
            include_glob: 可选的 glob 模式，用于过滤文件。

        Returns:
            返回一个字典：key 是文件路径，value 是 `(行号, 行文本)` 元组列表。
            如果 ripgrep 不可用或超时，则返回 `None`。
        """
        cmd = ["rg", "--json", "-F"]  # -F 表示启用固定字符串（字面量）搜索模式
        if include_glob:
            cmd.extend(["--glob", include_glob])
        cmd.extend(["--", pattern, str(base_full)])

        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        results: dict[str, list[tuple[int, str]]] = {}
        for line in proc.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "match":
                continue
            pdata = data.get("data", {})
            ftext = pdata.get("path", {}).get("text")
            if not ftext:
                continue
            p = Path(ftext)
            if self.virtual_mode:
                try:
                    virt = self._to_virtual_path(p)
                except ValueError:
                    logger.debug("Skipping grep result outside root: %s", p)
                    continue
                except OSError:
                    logger.warning("Could not resolve grep result path: %s", p, exc_info=True)
                    continue
            else:
                virt = str(p)
            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue
            results.setdefault(virt, []).append((int(ln), lt))

        return results

    def _python_search(self, pattern: str, base_full: Path, include_glob: str | None) -> dict[
        str, list[tuple[int, str]]]:  # noqa: C901, PLR0912
        """当 ripgrep 不可用时，使用 Python 进行回退搜索。

        会递归搜索文件，并遵守 `max_file_size_bytes` 的大小限制。

        Args:
            pattern: 经过 `re.escape` 处理后的正则模式，用于实现字面量搜索。
            base_full: 已解析好的基础搜索路径。
            include_glob: 可选的 glob 模式，用于按文件名过滤。

        Returns:
            返回一个字典：key 是文件路径，value 是 `(行号, 行文本)` 元组列表。
        """
        # 提前把转义后的 pattern 编译一次，提高循环中的执行效率
        regex = re.compile(pattern)

        results: dict[str, list[tuple[int, str]]] = {}
        root = base_full if base_full.is_dir() else base_full.parent

        for fp in root.rglob("*"):
            try:
                if not fp.is_file():
                    continue
            except (PermissionError, OSError):
                continue
            if include_glob:
                rel_path = str(fp.relative_to(root))
                if not wcglob.globmatch(rel_path, include_glob, flags=wcglob.BRACE | wcglob.GLOBSTAR):
                    continue
            try:
                if fp.stat().st_size > self.max_file_size_bytes:
                    continue
            except OSError:
                continue
            try:
                content = fp.read_text()
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    if self.virtual_mode:
                        try:
                            virt_path = self._to_virtual_path(fp)
                        except ValueError:
                            logger.debug("Skipping grep result outside root: %s", fp)
                            continue
                        except OSError:
                            logger.warning("Could not resolve grep result path: %s", fp, exc_info=True)
                            continue
                    else:
                        virt_path = str(fp)
                    results.setdefault(virt_path, []).append((line_num, line))

        return results

    def glob(self, pattern: str, path: str = "/") -> "GlobResult":
        """查找匹配 glob 模式的文件。

                Args:
                    pattern: 用来匹配文件的 glob 模式（例如 `'*.py'`、`'**/*.txt'`）。
                    path: 搜索起始目录。默认是根目录 `/`。

                Returns:
                    返回包含匹配文件或错误信息的 `GlobResult`。
        """
        if pattern.startswith("/"):
            pattern = pattern.lstrip("/")

        if self.virtual_mode and ".." in Path(pattern).parts:
            msg = "Path traversal not allowed in glob pattern"
            raise ValueError(msg)

        search_path = self.cwd if path == "/" else self._resolve_path(path)
        if not search_path.exists() or not search_path.is_dir():
            return GlobResult(matches=[])

        results: list[FileInfo] = []
        try:
            # 使用递归 glob，以便像测试预期的那样匹配子目录中的文件
            for matched_path in search_path.rglob(pattern):
                try:
                    is_file = matched_path.is_file()
                except (PermissionError, OSError):
                    continue
                if not is_file:
                    continue
                if self.virtual_mode:
                    try:
                        matched_path.resolve().relative_to(self.cwd)
                    except ValueError:
                        continue
                abs_path = str(matched_path)
                if not self.virtual_mode:
                    try:
                        st = matched_path.stat()
                        results.append(
                            {
                                "path": abs_path,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                                # noqa: DTZ006  # Local filesystem timestamps don't need timezone
                            }
                        )
                    except OSError:
                        results.append({"path": abs_path, "is_dir": False})
                else:
                    # 虚拟模式：使用 Path 处理路径，以兼容不同平台
                    try:
                        virt = self._to_virtual_path(matched_path)
                    except ValueError:
                        logger.debug("Skipping glob result outside root: %s", matched_path)
                        continue
                    except OSError:
                        logger.warning("Could not resolve glob result path: %s", matched_path, exc_info=True)
                        continue
                    try:
                        st = matched_path.stat()
                        results.append(
                            {
                                "path": virt,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                                # noqa: DTZ006  # Local filesystem timestamps don't need timezone
                            }
                        )
                    except OSError:
                        results.append({"path": virt, "is_dir": False})
        except (OSError, ValueError):
            pass

        results.sort(key=lambda x: x.get("path", ""))
        return GlobResult(matches=results)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """把多个文件上传到文件系统中。

        Args:
            files: `(path, content)` 元组列表，其中 `content` 是 bytes。

        Returns:
            返回 `FileUploadResponse` 列表，每个输入文件对应一个响应。
            响应顺序与输入顺序一致。
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                resolved_path = self._resolve_path(path)

                # Create parent directories if needed
                resolved_path.parent.mkdir(parents=True, exist_ok=True)

                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(resolved_path, flags, 0o644)
                with os.fdopen(fd, "wb") as f:
                    f.write(content)

                responses.append(FileUploadResponse(path=path, error=None))
            except Exception as exc:
                error = _map_exception_to_standard_error(exc)
                if error is None:
                    raise
                responses.append(FileUploadResponse(path=path, error=error))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """从文件系统中下载多个文件。

        Args:
            paths: 要下载的文件路径列表。

        Returns:
            返回 `FileDownloadResponse` 列表，每个输入路径对应一个响应。
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                resolved_path = self._resolve_path(path)
                # Use flags to optionally prevent symlink following if supported by the OS
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as f:
                    content = f.read()
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except Exception as exc:
                error = _map_exception_to_standard_error(exc)
                if error is None:
                    raise
                responses.append(FileDownloadResponse(path=path, content=None, error=error))
        return responses

def _map_exception_to_standard_error(exc: Exception) -> FileOperationError | None:
    """把捕获到的异常映射成标准化的 `FileOperationError` 错误码。

    分类只基于异常类型本身（标准库异常继承体系）。
    如果某个异常无法仅通过类型识别，就返回 `None`，
    由调用方决定是重新抛出异常，还是退回到 `str(exc)` 这种字符串错误信息。

    Args:
        exc: 要分类的异常对象。

    Returns:
        返回一个 `FileOperationError` 字面量；如果无法识别，则返回 `None`。
    """
    if isinstance(exc, FileNotFoundError):
        return "file_not_found"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, IsADirectoryError):
        return "is_directory"
    if isinstance(exc, (NotADirectoryError, FileExistsError)):
        return "invalid_path"
    if isinstance(exc, ValueError):
        return "invalid_path"
    return None
