# -*- coding: utf-8 -*-
"""
标题：L02-07（稳定版）tools.py —— Range Read + Hunks Patch + 不崩溃的工具契约
执行代码：
  由 main.py 导入并 init_tools 后使用
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain_core.tools import tool


class _Config:
    """运行时配置与证据/状态（由 main.py 初始化 + 每轮更新）。"""

    workdir: Optional[Path] = None
    write_policy: str = "ask"
    todo_file: Optional[Path] = None

    # 本轮是否允许写（由 main.py 每轮根据用户输入设置）
    allow_write_in_turn: bool = False
    last_user_text: str = ""

    # tests 证据
    last_test_rc: Optional[int] = None
    last_test_cmd: Optional[str] = None
    last_test_stdout: str = ""
    last_test_stderr: str = ""

    # 变更证据
    changed_files: List[str] = []
    last_diffs: Dict[str, str] = {}

    # 补丁状态：拒绝/已应用
    rejected_patch_ids: set[str] = set()
    applied_patch_ids: set[str] = set()

    allowed_cmd_prefixes: tuple[str, ...] = (
        "python -m pytest",
        "pytest",
        "python ",
    )


CFG = _Config()


# --------------------------
# 初始化与回合意图
# --------------------------


def ensure_toy_repo(repo_dir: Path) -> None:
    """确保 toy_repo 存在并包含最小示例文件。"""
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "tests").mkdir(parents=True, exist_ok=True)

    calc_py = repo_dir / "calc.py"
    test_py = repo_dir / "tests" / "test_calc.py"

    if not calc_py.exists():
        calc_py.write_text(
            textwrap.dedent(
                """\
                # calc.py
                def add(a: int, b: int) -> int:
                    # BUG: 这里故意写错，让测试失败
                    return a - b
                """
            ),
            encoding="utf-8",
        )

    if not test_py.exists():
        test_py.write_text(
            textwrap.dedent(
                """\
                # tests/test_calc.py
                from calc import add

                def test_add_basic():
                    assert add(2, 3) == 5
                """
            ),
            encoding="utf-8",
        )


def init_tools(workdir: Path, write_policy: str) -> None:
    """初始化工具全局配置（workdir 强制绝对路径）。"""
    wd = workdir.resolve()
    CFG.workdir = wd
    CFG.write_policy = write_policy
    CFG.todo_file = (wd / ".agent_todo.json").resolve()

    CFG.allow_write_in_turn = False
    CFG.last_user_text = ""

    CFG.last_test_rc = None
    CFG.last_test_cmd = None
    CFG.last_test_stdout = ""
    CFG.last_test_stderr = ""

    CFG.changed_files = []
    CFG.last_diffs = {}

    CFG.rejected_patch_ids = set()
    CFG.applied_patch_ids = set()

    ensure_toy_repo(wd)


def set_turn_context(user_text: str) -> None:
    """
    每轮由 main.py 调用：根据“用户明确意图”决定是否允许写。
    目标：避免 review 类指令被 LLM 误判为要改代码而触发写工具。
    """
    CFG.last_user_text = user_text or ""
    s = (user_text or "").lower()

    # 明确“只读/审阅”倾向（不允许写）
    readonly_markers = [
        "review",
        "查看",
        "浏览",
        "列出",
        "结构",
        "目录",
        "有哪些文件",
        "tree",
        "清单",
        "overview",
        "审阅",
    ]

    # 明确“要改动”倾向（允许写）
    write_markers = [
        "修复",
        "修改",
        "改一下",
        "添加",
        "新增",
        "实现",
        "重构",
        "删除",
        "更新",
        "fix",
        "bug",
        "patch",
        "apply",
        "写入",
    ]

    if any(m in s for m in write_markers):
        CFG.allow_write_in_turn = True
        return

    if any(m in s for m in readonly_markers):
        CFG.allow_write_in_turn = False
        return

    # 默认：不允许写（更安全、更符合“review 不应改代码”）
    CFG.allow_write_in_turn = False


def _require_cfg() -> None:
    if CFG.workdir is None or CFG.todo_file is None:
        raise RuntimeError(
            "tools not initialized: call init_tools(workdir, policy) first"
        )


# --------------------------
# 路径契约：不抛崩溃型异常（返回可读错误）
# --------------------------


def _strip_workdir_prefix(s: str) -> str:
    assert CFG.workdir is not None
    wd_name = CFG.workdir.name.replace("\\", "/")
    prefix = wd_name + "/"
    return s[len(prefix) :] if s.startswith(prefix) else s


def normalize_dir(user_dir: Optional[str]) -> tuple[bool, str]:
    """
    目录契约：允许 None/"" -> "."
    返回：(ok, rel_path_or_error_message)
    """
    _require_cfg()
    assert CFG.workdir is not None

    s = (user_dir or "").strip().replace("\\", "/")
    if s == "":
        s = "."
    if s.startswith("./"):
        s = s[2:]
    s = _strip_workdir_prefix(s)

    p = Path(s)
    if p.is_absolute() or (len(s) >= 2 and s[1] == ":" and s[0].isalpha()):
        return False, "absolute path is not allowed"

    candidate = (CFG.workdir / s).resolve()
    if not str(candidate).startswith(str(CFG.workdir)):
        return False, "path escapes workdir"

    return True, candidate.relative_to(CFG.workdir).as_posix()


def normalize_path(user_path: Optional[str]) -> tuple[bool, str]:
    """
    文件路径契约：允许缺参/空参，但返回可恢复错误（不让 Pydantic/异常把程序打崩）
    返回：(ok, rel_path_or_error_message)
    """
    _require_cfg()
    assert CFG.workdir is not None

    s = (user_path or "").strip().replace("\\", "/")
    if s == "":
        return False, "path is empty"
    if s.startswith("./"):
        s = s[2:]
    s = _strip_workdir_prefix(s)

    p = Path(s)
    if p.is_absolute() or (len(s) >= 2 and s[1] == ":" and s[0].isalpha()):
        return False, "absolute path is not allowed"

    candidate = (CFG.workdir / s).resolve()
    if not str(candidate).startswith(str(CFG.workdir)):
        return False, "path escapes workdir"

    return True, candidate.relative_to(CFG.workdir).as_posix()


def _abs(rel_norm: str) -> Path:
    _require_cfg()
    assert CFG.workdir is not None
    return (CFG.workdir / rel_norm).resolve()


def _unified_diff(old: str, new: str, filename: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )
    return "".join(diff)


def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def _patch_id(file_rel: str, old_text: str, payload: str) -> str:
    """patch_id 绑定旧内容摘要 + 补丁意图，避免文件变化后误判重复。"""
    h = hashlib.sha256()
    h.update(file_rel.encode("utf-8"))
    h.update(b"\nold=" + _sha256_text(old_text).encode("utf-8"))
    h.update(b"\npayload=" + payload.encode("utf-8"))
    return h.hexdigest()[:16]


def _json(status: str, **kwargs: Any) -> str:
    obj = {"status": status, **kwargs}
    return json.dumps(obj, ensure_ascii=False)


# --------------------------
# Tools
# --------------------------


@tool("repo_tree")
def repo_tree(
    dir: Optional[str] = ".", recursive: bool = True, max_results: int = 200
) -> str:
    """列出工作区文件结构（用于 review/overview；不会修改任何文件）。"""
    _require_cfg()
    ok, rel = normalize_dir(dir)
    if not ok:
        return f"(error) {rel}"

    root = _abs(rel)
    if not root.exists():
        return f"(error) dir not found: {rel}"
    if not root.is_dir():
        return f"(error) not a dir: {rel}"

    out: List[str] = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file():
                out.append(p.relative_to(CFG.workdir).as_posix())  # type: ignore[arg-type]
                if len(out) >= max_results:
                    break
    else:
        for p in root.iterdir():
            if p.is_file():
                out.append(p.relative_to(CFG.workdir).as_posix())  # type: ignore[arg-type]
                if len(out) >= max_results:
                    break
    return "\n".join(out) if out else "(empty)"


@tool("todoread")
def todoread() -> str:
    """读取任务清单（todo）。若不存在则返回 (todo is empty)。"""
    _require_cfg()
    assert CFG.todo_file is not None
    if not CFG.todo_file.exists():
        return "(todo is empty)"
    return CFG.todo_file.read_text(encoding="utf-8", errors="replace")


@tool("todowrite")
def todowrite(items: Optional[List[str]] = None) -> str:
    """写入任务清单（todo），覆盖写入。items 建议 3~7 条。"""
    _require_cfg()
    assert CFG.todo_file is not None
    if not items:
        return "(error) items is empty"
    CFG.todo_file.write_text(
        json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return f"[OK] todo saved: {len(items)} items"


@tool("list_files")
def list_files(
    dir: Optional[str] = ".", recursive: bool = False, max_results: int = 200
) -> str:
    """列出目录下文件（相对 workdir 路径；dir 允许 ''/None，语义等价 '.'）。"""
    _require_cfg()
    ok, rel = normalize_dir(dir)
    if not ok:
        return f"(error) {rel}"

    root = _abs(rel)
    if not root.exists():
        return f"(error) dir not found: {rel}"
    if not root.is_dir():
        return f"(error) not a dir: {rel}"

    out: List[str] = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file():
                out.append(p.relative_to(CFG.workdir).as_posix())  # type: ignore[arg-type]
                if len(out) >= max_results:
                    break
    else:
        for p in root.iterdir():
            if p.is_file():
                out.append(p.relative_to(CFG.workdir).as_posix())  # type: ignore[arg-type]
                if len(out) >= max_results:
                    break
    return "\n".join(out) if out else "(empty)"


@tool("grep")
def grep(
    pattern: Optional[str] = None, path: Optional[str] = ".", max_results: int = 50
) -> str:
    """搜索文本（简单包含匹配）。返回：file:line:content。path 允许 ''/None。"""
    _require_cfg()
    if not pattern:
        return "(error) pattern is empty"

    ok, rel = normalize_dir(path)
    if not ok:
        return f"(error) {rel}"

    target = _abs(rel)
    if not target.exists():
        return f"(error) path not found: {rel}"

    results: List[str] = []

    def scan_file(fp: Path) -> None:
        text = fp.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                results.append(f"{fp.relative_to(CFG.workdir).as_posix()}:{i}:{line}")  # type: ignore[arg-type]
                if len(results) >= max_results:
                    return

    if target.is_file():
        scan_file(target)
    else:
        for fp in target.rglob("*"):
            if fp.is_file():
                scan_file(fp)
                if len(results) >= max_results:
                    break

    return "\n".join(results) if results else "(no matches)"


@tool("read_file")
def read_file(path: Optional[str] = "") -> str:
    """读取文件全文（path 为空会返回可恢复错误，不会抛崩溃异常）。"""
    _require_cfg()
    ok, rel = normalize_path(path)
    if not ok:
        return f"(error) {rel}"

    p = _abs(rel)
    if not p.exists():
        return f"(error) file not found: {rel}"
    return p.read_text(encoding="utf-8", errors="replace")


@tool("read_file_range")
def read_file_range(
    path: Optional[str] = "", start_line: int = 1, end_line: int = 200
) -> str:
    """
    局部读取（稳定语义）：
    - 行号从 1 开始
    - end_line 若超过总行数，会自动截断到 total_lines（不再抛越界导致退出）
    """
    _require_cfg()
    ok, rel = normalize_path(path)
    if not ok:
        return f"(error) {rel}"

    p = _abs(rel)
    if not p.exists():
        return f"(error) file not found: {rel}"

    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)

    if total == 0:
        return f"file={rel}\ntotal_lines=0\n(range is empty)"

    # 归一化行号（不崩溃）
    if start_line < 1:
        start_line = 1
    if end_line < 1:
        end_line = total
    if start_line > total:
        return (
            f"file={rel}\ntotal_lines={total}\nshowing_lines=(empty)\n(range is empty)"
        )

    if end_line > total:
        end_line = total

    if start_line > end_line:
        start_line, end_line = end_line, start_line

    seg = []
    for ln in range(start_line, end_line + 1):
        seg.append(f"{ln:>4} | {lines[ln - 1]}")
    header = f"file={rel}\ntotal_lines={total}\nshowing_lines={start_line}-{end_line}\n"
    return header + "\n".join(seg)


def _write_gate_or_reason() -> Optional[str]:
    """写入门控：用户本轮未明确要求修改时，禁止写工具直接落盘/弹窗。"""
    if CFG.allow_write_in_turn:
        return None
    return "WRITE_NOT_REQUESTED: 用户本轮未明确提出修改需求（例如只是 review/查看目录）。请先询问用户是否要改代码。"


@tool("apply_hunks")
def apply_hunks(
    file_path: Optional[str] = "", hunks: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    局部补丁（最小修改）。
    - hunks 缺失不会触发 Pydantic missing 崩溃；会返回结构化错误
    - 若本轮未明确允许写，会返回 WRITE_NOT_REQUESTED（不弹确认）
    """
    _require_cfg()

    gate = _write_gate_or_reason()
    if gate is not None:
        return _json("WRITE_NOT_REQUESTED", message=gate)

    ok, rel = normalize_path(file_path)
    if not ok:
        return _json("ERROR", message=rel)

    p = _abs(rel)
    if not p.exists():
        return _json("ERROR", message=f"file not found: {rel}")

    if not hunks:
        return _json(
            "ERROR",
            message="hunks is empty; you must provide hunks with start_line/end_line/replacement",
        )

    old_text = p.read_text(encoding="utf-8", errors="replace")
    has_trailing_nl = old_text.endswith("\n")
    old_lines = old_text.splitlines()
    total = len(old_lines)

    # 校验并标准化 hunks（不抛崩溃型异常，返回结构化错误）
    normalized: List[Dict[str, Any]] = []
    for h in hunks:
        if not isinstance(h, dict):
            return _json("ERROR", message="each hunk must be a dict")
        if "start_line" not in h or "end_line" not in h or "replacement" not in h:
            return _json(
                "ERROR", message="hunk must include start_line, end_line, replacement"
            )

        sl = h["start_line"]
        el = h["end_line"]
        rep = h["replacement"]

        if not isinstance(sl, int) or not isinstance(el, int):
            return _json("ERROR", message="start_line/end_line must be int")
        if not isinstance(rep, str):
            return _json("ERROR", message="replacement must be str")
        if sl < 1:
            return _json("ERROR", message="start_line must be >= 1")
        if el < sl - 1:
            return _json("ERROR", message="end_line must be >= start_line-1")

        # 插入允许 sl==total+1；替换/删除 el 不能超过 total
        if sl > total + 1:
            return _json(
                "ERROR",
                message=f"start_line out of range: {sl} > total_lines+1 {total+1}",
            )
        if el >= sl and el > total:
            return _json(
                "ERROR", message=f"end_line out of range: {el} > total_lines {total}"
            )

        normalized.append({"start_line": sl, "end_line": el, "replacement": rep})

    payload = json.dumps(
        {
            "mode": "hunks",
            "hunks": sorted(normalized, key=lambda x: (x["start_line"], x["end_line"])),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    pid = _patch_id(rel, old_text, payload)

    if pid in CFG.applied_patch_ids:
        return _json("ALREADY_APPLIED", patch_id=pid, file=rel)

    if pid in CFG.rejected_patch_ids:
        return _json(
            "REJECTED_REPEAT",
            patch_id=pid,
            file=rel,
            next="ASK_USER",
            message="该补丁已被用户拒绝过；不要重复尝试相同修改，应询问用户如何处理。",
        )

    # 检查区间不重叠（返回结构化错误）
    slices: List[Dict[str, Any]] = []
    for h in normalized:
        s = h["start_line"] - 1
        if h["end_line"] == h["start_line"] - 1:
            e = s  # insertion
        else:
            e = h["end_line"]  # inclusive -> exclusive
        slices.append({"s": s, "e": e, "replacement": h["replacement"]})

    slices_sorted = sorted(slices, key=lambda x: (x["s"], x["e"]), reverse=True)
    prev_s = None
    for slc in slices_sorted:
        if prev_s is None:
            prev_s = slc["s"]
            continue
        if slc["e"] > prev_s:
            return _json("ERROR", message="overlapping hunks are not allowed")
        prev_s = slc["s"]

    new_lines = old_lines[:]
    for slc in slices_sorted:
        rep_lines = slc["replacement"].splitlines() if slc["replacement"] != "" else []
        new_lines[slc["s"] : slc["e"]] = rep_lines

    new_text = "\n".join(new_lines)
    if has_trailing_nl:
        new_text += "\n"

    diff = _unified_diff(old_text, new_text, rel)

    # 写入策略
    if CFG.write_policy == "deny":
        return _json(
            "DENY_POLICY",
            patch_id=pid,
            file=rel,
            next="ASK_USER",
            message="当前策略禁止写入；请询问用户是否切换策略。",
            diff=diff if diff.strip() else "(无 diff)",
        )

    if CFG.write_policy == "ask":
        print("\n========== [WRITE PREVIEW / 需要确认] ==========")
        print(diff if diff.strip() else "(无 diff)")
        print("===============================================")
        ans = input("是否写入该补丁？(y/n) ").strip().lower()
        if ans not in ("y", "yes"):
            CFG.rejected_patch_ids.add(pid)
            return _json(
                "USER_REJECTED",
                patch_id=pid,
                file=rel,
                next="ASK_USER",
                message="用户拒绝了该补丁；不要重复尝试相同修改，应询问用户希望如何处理。",
                diff=diff if diff.strip() else "(无 diff)",
            )

    # 落盘
    p.write_text(new_text, encoding="utf-8")
    CFG.applied_patch_ids.add(pid)

    if rel not in CFG.changed_files:
        CFG.changed_files.append(rel)
    CFG.last_diffs[rel] = diff if diff.strip() else "(无 diff)"

    return _json(
        "APPLIED", patch_id=pid, file=rel, diff=diff if diff.strip() else "(无 diff)"
    )


@tool("bash")
def bash(command: Optional[str] = "") -> str:
    """执行命令（前缀白名单）。pytest 结果会记录用于 submit。"""
    _require_cfg()
    assert CFG.workdir is not None

    cmd = (command or "").strip()
    if not cmd:
        return "(error) command is empty"
    if not any(cmd.startswith(pfx) for pfx in CFG.allowed_cmd_prefixes):
        return f"(error) command not allowed: {cmd!r}"

    proc = subprocess.run(
        cmd,
        cwd=str(CFG.workdir),
        shell=True,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )

    if cmd.startswith("python -m pytest") or cmd.startswith("pytest"):
        CFG.last_test_cmd = cmd
        CFG.last_test_rc = proc.returncode
        CFG.last_test_stdout = proc.stdout
        CFG.last_test_stderr = proc.stderr

    return (
        f"returncode={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
    )


@tool("submit")
def submit(
    final: Optional[str] = "",
    evidence: Optional[List[str]] = None,
    task_type: Literal["tests", "general"] = "general",
) -> str:
    """
    完成闸门（不抛崩溃异常；用 REJECT 返回）。
    - tests：必须 pytest rc==0
    - evidence：必须非空；如果发生过修改，必须包含变更说明
    """
    _require_cfg()
    final = (final or "").strip()
    if not final:
        return _json("REJECT", reason="final is empty")

    if not evidence:
        return _json("REJECT", reason="evidence is empty")

    if task_type == "tests":
        if CFG.last_test_rc is None:
            return _json("REJECT", reason="tests 任务未检测到 pytest 运行证据。")
        if CFG.last_test_rc != 0:
            return _json("REJECT", reason=f"pytest 未通过（rc={CFG.last_test_rc}）。")

    if len(CFG.changed_files) > 0:
        if not any(
            ("改" in e or "diff" in e or "changed" in e or "变更" in e or "行号" in e)
            for e in evidence
        ):
            return _json(
                "REJECT", reason="发生过文件修改，但 evidence 未包含变更说明。"
            )

    return _json("ACCEPT", reason="ok")
