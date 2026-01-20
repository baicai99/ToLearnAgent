# -*- coding: utf-8 -*-
"""
标题：L02-08 tools.py —— TaskType + Evidence 模板 + write_file（创建文件稳定）+ 工具不崩溃契约
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
    workdir: Optional[Path] = None
    write_policy: str = "ask"
    todo_file: Optional[Path] = None

    # 每轮上下文（由 main.py 设置）
    allow_write_in_turn: bool = False
    last_user_text: str = ""

    # 证据记录（自动追加，供 submit 检查）
    evidence_log: List[str] = []

    # tests 证据
    last_test_rc: Optional[int] = None
    last_test_cmd: Optional[str] = None
    last_test_stdout: str = ""
    last_test_stderr: str = ""

    # 变更证据
    changed_files: List[str] = []
    created_files: List[str] = []
    last_diffs: Dict[str, str] = {}

    # 补丁状态：拒绝/已应用（用来阻止重复尝试）
    rejected_patch_ids: set[str] = set()
    applied_patch_ids: set[str] = set()

    allowed_cmd_prefixes: tuple[str, ...] = (
        "python -m pytest",
        "pytest",
        "python ",
    )


CFG = _Config()


def _json(status: str, **kwargs: Any) -> str:
    return json.dumps({"status": status, **kwargs}, ensure_ascii=False)


def ensure_toy_repo(repo_dir: Path) -> None:
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
    wd = workdir.resolve()
    CFG.workdir = wd
    CFG.write_policy = write_policy
    CFG.todo_file = (wd / ".agent_todo.json").resolve()

    CFG.allow_write_in_turn = False
    CFG.last_user_text = ""

    CFG.evidence_log = []

    CFG.last_test_rc = None
    CFG.last_test_cmd = None
    CFG.last_test_stdout = ""
    CFG.last_test_stderr = ""

    CFG.changed_files = []
    CFG.created_files = []
    CFG.last_diffs = {}

    CFG.rejected_patch_ids = set()
    CFG.applied_patch_ids = set()

    ensure_toy_repo(wd)


def set_turn_context(user_text: str) -> None:
    """
    本轮是否允许写：默认 False；只有用户明确要求“创建/修改/修复/新增/实现”等才 True。
    """
    CFG.last_user_text = user_text or ""
    s = (user_text or "").lower()

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
        "创建",
        "新建",
        "生成",
        "命名",
        "create",
        "add file",
        "write file",
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
    CFG.allow_write_in_turn = False


def _require_cfg() -> None:
    if CFG.workdir is None:
        raise RuntimeError(
            "tools not initialized: call init_tools(workdir, policy) first"
        )


def _strip_workdir_prefix(s: str) -> str:
    assert CFG.workdir is not None
    wd_name = CFG.workdir.name.replace("\\", "/")
    prefix = wd_name + "/"
    return s[len(prefix) :] if s.startswith(prefix) else s


def normalize_dir(user_dir: Optional[str]) -> tuple[bool, str]:
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
    h = hashlib.sha256()
    h.update(file_rel.encode("utf-8"))
    h.update(b"\nold=" + _sha256_text(old_text).encode("utf-8"))
    h.update(b"\npayload=" + payload.encode("utf-8"))
    return h.hexdigest()[:16]


def _write_gate_or_reason() -> Optional[str]:
    if CFG.allow_write_in_turn:
        return None
    return "WRITE_NOT_REQUESTED: 用户本轮未明确提出修改/创建需求（例如只是 review）。请先询问用户是否要改/建文件。"


def _log_evidence(line: str) -> None:
    CFG.evidence_log.append(line)


@tool("evidence_read")
def evidence_read(max_items: int = 50) -> str:
    """读取当前已记录的 evidence（最多 max_items 条，后进先出）。"""
    _require_cfg()
    items = CFG.evidence_log[-max_items:]
    if not items:
        return "(evidence is empty)"
    return "\n".join(items)


@tool("repo_tree")
def repo_tree(
    dir: Optional[str] = ".", recursive: bool = True, max_results: int = 200
) -> str:
    """列出工作区文件结构（用于 review）。"""
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

    res = "\n".join(out) if out else "(empty)"
    _log_evidence(f"[tree] files_count={len(out)} dir={rel}")
    return res


@tool("list_files")
def list_files(
    dir: Optional[str] = ".", recursive: bool = False, max_results: int = 200
) -> str:
    """列出目录文件（只读；dir 允许 ''/None）。"""
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

    _log_evidence(
        f"[list_files] files_count={len(out)} dir={rel} recursive={recursive}"
    )
    return "\n".join(out) if out else "(empty)"


@tool("grep")
def grep(
    pattern: Optional[str] = None, path: Optional[str] = ".", max_results: int = 50
) -> str:
    """搜索文本（只读；简单包含匹配）。"""
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

    _log_evidence(f"[grep] pattern={pattern!r} hits={len(results)} path={rel}")
    return "\n".join(results) if results else "(no matches)"


@tool("read_file_range")
def read_file_range(
    path: Optional[str] = "", start_line: int = 1, end_line: int = 200
) -> str:
    """
    局部读取：end_line 超界会自动截断，不抛异常导致程序退出。
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
    _log_evidence(
        f"[read_range] file={rel} lines={start_line}-{end_line} total={total}"
    )
    return (
        f"file={rel}\ntotal_lines={total}\nshowing_lines={start_line}-{end_line}\n"
        + "\n".join(seg)
    )


@tool("write_file")
def write_file(file_path: Optional[str] = "", content: Optional[str] = "") -> str:
    """
    创建/覆盖文件（用于新增文件或整段生成）。
    强约束：仅当用户本轮明确要求创建/修改时才允许写入（由 set_turn_context 控制）。
    """
    _require_cfg()
    gate = _write_gate_or_reason()
    if gate is not None:
        return _json("WRITE_NOT_REQUESTED", message=gate)

    ok, rel = normalize_path(file_path)
    if not ok:
        return _json("ERROR", message=rel)

    if content is None:
        content = ""

    p = _abs(rel)
    existed = p.exists()
    old_text = p.read_text(encoding="utf-8", errors="replace") if existed else ""
    new_text = content

    # 统一换行：保持以 \n 结尾（利于 diff 与执行）
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"

    payload = json.dumps(
        {"mode": "write_file", "content": new_text}, ensure_ascii=False, sort_keys=True
    )
    pid = _patch_id(rel, old_text, payload)

    if pid in CFG.applied_patch_ids:
        return _json("ALREADY_APPLIED", patch_id=pid, file=rel)

    diff = _unified_diff(old_text, new_text, rel)

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
        ans = input("是否写入该文件？(y/n) ").strip().lower()
        if ans not in ("y", "yes"):
            CFG.rejected_patch_ids.add(pid)
            return _json(
                "USER_REJECTED",
                patch_id=pid,
                file=rel,
                next="ASK_USER",
                message="用户拒绝了写入；不要重复尝试同一写入，应询问用户希望如何处理。",
            )

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new_text, encoding="utf-8")

    CFG.applied_patch_ids.add(pid)
    if existed:
        if rel not in CFG.changed_files:
            CFG.changed_files.append(rel)
    else:
        if rel not in CFG.created_files:
            CFG.created_files.append(rel)

    CFG.last_diffs[rel] = diff if diff.strip() else "(无 diff)"
    _log_evidence(
        f"[write_file] file={rel} existed={existed} lines={len(new_text.splitlines())}"
    )

    return _json(
        "APPLIED", patch_id=pid, file=rel, diff=diff if diff.strip() else "(无 diff)"
    )


@tool("apply_hunks")
def apply_hunks(file_path: Optional[str] = "", hunks: Any = None) -> str:
    """
    局部补丁（最小修改）。
    关键：hunks 使用 Any，避免 Pydantic 因类型错误直接崩溃。
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

    if (
        not isinstance(hunks, list)
        or len(hunks) == 0
        or not all(isinstance(x, dict) for x in hunks)
    ):
        return _json(
            "ERROR",
            message="hunks must be a non-empty list of dict: "
            '{"start_line": int, "end_line": int, "replacement": str}. '
            "For creating a new file, use write_file instead.",
        )

    old_text = p.read_text(encoding="utf-8", errors="replace")
    has_trailing_nl = old_text.endswith("\n")
    old_lines = old_text.splitlines()
    total = len(old_lines)

    normalized: List[Dict[str, Any]] = []
    for h in hunks:
        if "start_line" not in h or "end_line" not in h or "replacement" not in h:
            return _json(
                "ERROR",
                message="each hunk must include start_line, end_line, replacement",
            )
        sl = h["start_line"]
        el = h["end_line"]
        rep = h["replacement"]
        if (
            not isinstance(sl, int)
            or not isinstance(el, int)
            or not isinstance(rep, str)
        ):
            return _json(
                "ERROR",
                message="start_line/end_line must be int; replacement must be str",
            )
        if sl < 1:
            return _json("ERROR", message="start_line must be >= 1")
        if el < sl - 1:
            return _json("ERROR", message="end_line must be >= start_line-1")
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

    # 不允许重叠区间（降序应用）
    slices: List[Dict[str, Any]] = []
    for h in normalized:
        s = h["start_line"] - 1
        e = s if h["end_line"] == h["start_line"] - 1 else h["end_line"]
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
        rep_lines = slc["replacement"].splitlines() if slc["replacement"] else []
        new_lines[slc["s"] : slc["e"]] = rep_lines

    new_text = "\n".join(new_lines)
    if has_trailing_nl:
        new_text += "\n"

    diff = _unified_diff(old_text, new_text, rel)

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
                message="用户拒绝了该补丁；不要重复尝试同一修改，应询问用户希望如何处理。",
            )

    p.write_text(new_text, encoding="utf-8")

    CFG.applied_patch_ids.add(pid)
    if rel not in CFG.changed_files:
        CFG.changed_files.append(rel)
    CFG.last_diffs[rel] = diff if diff.strip() else "(无 diff)"
    _log_evidence(f"[apply_hunks] file={rel} hunks={len(normalized)}")

    return _json(
        "APPLIED", patch_id=pid, file=rel, diff=diff if diff.strip() else "(无 diff)"
    )


@tool("bash")
def bash(command: Optional[str] = "") -> str:
    """在工作区执行命令（前缀白名单）。pytest 结果会记录用于 submit。"""
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
        _log_evidence(f"[pytest] cmd={cmd!r} rc={proc.returncode}")

    return (
        f"returncode={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
    )


@tool("submit")
def submit(
    final: Optional[str] = "",
    evidence: Optional[List[str]] = None,
    task_type: Literal["review", "create", "tests", "implement"] = "implement",
) -> str:
    """
    L02-08：任务类型协议 + 最小证据模板

    - review：必须“只读”（不能有 changed/created）；evidence 至少 1 条，且应包含 tree/list_files/grep/read_range 之一
    - create：必须 created_files 非空；evidence 需包含 created 文件名
    - tests：必须 pytest rc==0；evidence 需包含 pytest rc 与变更/定位信息
    - implement：必须 changed_files 或 created_files 非空；evidence 需包含变更说明
    """
    _require_cfg()
    final = (final or "").strip()
    if not final:
        return _json("REJECT", reason="final is empty")
    if not evidence:
        return _json("REJECT", reason="evidence is empty")

    # review 不允许写入
    if task_type == "review":
        if CFG.changed_files or CFG.created_files:
            return _json(
                "REJECT", reason="review 任务不应产生写入（changed/created 非空）。"
            )
        ok_marker = any(
            (
                "[tree]" in e
                or "[list_files]" in e
                or "[grep]" in e
                or "[read_range]" in e
            )
            for e in CFG.evidence_log
        )
        if not ok_marker:
            return _json(
                "REJECT",
                reason="review 任务缺少只读证据（tree/list_files/grep/read_range）。",
            )
        return _json("ACCEPT", reason="ok")

    if task_type == "create":
        if not CFG.created_files:
            return _json(
                "REJECT", reason="create 任务未检测到新建文件（created_files 为空）。"
            )
        # evidence 必须包含文件名
        if not any(any(f in ev for f in CFG.created_files) for ev in evidence):
            return _json(
                "REJECT",
                reason=f"create 任务 evidence 未包含新建文件名：{CFG.created_files}",
            )
        return _json("ACCEPT", reason="ok")

    if task_type == "tests":
        if CFG.last_test_rc is None:
            return _json("REJECT", reason="tests 任务未检测到 pytest 运行证据。")
        if CFG.last_test_rc != 0:
            return _json("REJECT", reason=f"pytest 未通过（rc={CFG.last_test_rc}）。")
        # 还要至少有一次变更或明确定位证据
        has_change = bool(CFG.changed_files or CFG.created_files)
        has_locate = any(
            ("[grep]" in e or "[read_range]" in e) for e in CFG.evidence_log
        )
        if not (has_change and has_locate):
            return _json(
                "REJECT",
                reason="tests 任务 evidence 不足：需要定位证据（grep/read_range）+ 变更（changed/created）。",
            )
        return _json("ACCEPT", reason="ok")

    # implement
    if not (CFG.changed_files or CFG.created_files):
        return _json(
            "REJECT", reason="implement 任务未检测到任何写入（changed/created 为空）。"
        )
    if not any(
        (
            "改" in e
            or "diff" in e
            or "变更" in e
            or "changed" in e
            or "write_file" in e
            or "apply_hunks" in e
        )
        for e in evidence
    ):
        return _json("REJECT", reason="implement 任务 evidence 未包含变更说明。")

    return _json("ACCEPT", reason="ok")
