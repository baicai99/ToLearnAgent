# -*- coding: utf-8 -*-
"""
标题：L02-08F core_runtime.py —— 运行时：路径边界/工具实现/写入审批预览
执行代码：
  由 tools.py / main.py 导入使用，不直接运行
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


TaskType = Literal["chat", "review", "create", "tests", "implement"]
PatchKind = Literal["write_file", "apply_hunks"]


@dataclass
class PendingPatch:
    patch_id: str
    file: str  # rel posix
    existed: bool
    kind: PatchKind
    new_text: str
    diff: str


@dataclass
class RuntimeState:
    workdir: Optional[Path] = None
    write_policy: str = "ask"  # allow/ask/deny

    task_type: TaskType = "chat"
    last_user_text: str = ""

    evidence_log: List[str] = field(default_factory=list)

    last_test_rc: Optional[int] = None
    last_test_cmd: Optional[str] = None
    last_test_stdout: str = ""
    last_test_stderr: str = ""

    changed_files: List[str] = field(default_factory=list)
    created_files: List[str] = field(default_factory=list)
    last_diffs: Dict[str, str] = field(default_factory=dict)

    rejected_patch_ids: set[str] = field(default_factory=set)
    applied_patch_ids: set[str] = field(default_factory=set)

    pending_patches: Dict[str, PendingPatch] = field(default_factory=dict)

    allowed_cmd_prefixes: Tuple[str, ...] = (
        "python -m pytest",
        "pytest",
        "python ",
    )


STATE = RuntimeState()


def _json(status: str, **kwargs: Any) -> str:
    return json.dumps({"status": status, **kwargs}, ensure_ascii=False)


def _require_init() -> None:
    if STATE.workdir is None:
        raise RuntimeError(
            "runtime not initialized: call init_runtime(workdir, policy) first"
        )


def _log_evidence(line: str) -> None:
    STATE.evidence_log.append(line)


def init_runtime(workdir: Path, write_policy: str) -> None:
    wd = workdir.resolve()
    STATE.workdir = wd
    STATE.write_policy = write_policy

    STATE.task_type = "chat"
    STATE.last_user_text = ""

    STATE.evidence_log = []

    STATE.last_test_rc = None
    STATE.last_test_cmd = None
    STATE.last_test_stdout = ""
    STATE.last_test_stderr = ""

    STATE.changed_files = []
    STATE.created_files = []
    STATE.last_diffs = {}

    STATE.rejected_patch_ids = set()
    STATE.applied_patch_ids = set()

    STATE.pending_patches = {}

    _ensure_toy_repo(wd)


def set_turn_context(user_text: str) -> TaskType:
    _require_init()
    s = (user_text or "").strip()
    sl = s.lower()
    STATE.last_user_text = s

    greet = [
        "你好",
        "hi",
        "hello",
        "在吗",
        "谢谢",
        "你是谁",
        "咋样",
        "早上好",
        "晚上好",
    ]
    if len(sl) <= 12 and any(w in sl for w in greet):
        STATE.task_type = "chat"
        return STATE.task_type

    review_markers = [
        "review",
        "查看",
        "浏览",
        "列出",
        "结构",
        "目录",
        "tree",
        "overview",
        "审阅",
        "有什么",
    ]
    create_markers = [
        "创建",
        "新建",
        "生成",
        "命名",
        "create",
        "new task file",
        "new file",
        "write file",
    ]
    tests_markers = ["pytest", "测试", "test", "让测试通过", "通过测试"]
    implement_markers = [
        "修复",
        "修改",
        "添加",
        "新增",
        "实现",
        "重构",
        "更新",
        "fix",
        "bug",
        "patch",
    ]

    if any(m in sl for m in create_markers):
        STATE.task_type = "create"
    elif any(m in sl for m in tests_markers):
        STATE.task_type = "tests"
    elif any(m in sl for m in implement_markers):
        STATE.task_type = "implement"
    elif any(m in sl for m in review_markers):
        STATE.task_type = "review"
    else:
        STATE.task_type = "review"

    return STATE.task_type


def _strip_workdir_prefix(s: str) -> str:
    assert STATE.workdir is not None
    wd_name = STATE.workdir.name.replace("\\", "/")
    prefix = wd_name + "/"
    return s[len(prefix) :] if s.startswith(prefix) else s


# [L02-08F CHANGE] 空字符串目录 -> "."（这是“契约级修复”，不是补丁兜底）
def normalize_dir(user_dir: Optional[str]) -> Tuple[bool, str]:
    _require_init()
    assert STATE.workdir is not None

    s = (user_dir or "").strip().replace("\\", "/")
    if s == "":
        s = "."
    if s.startswith("./"):
        s = s[2:]
    s = _strip_workdir_prefix(s)

    p = Path(s)
    if p.is_absolute() or (len(s) >= 2 and s[1] == ":" and s[0].isalpha()):
        return False, "absolute path is not allowed"

    candidate = (STATE.workdir / s).resolve()
    if not str(candidate).startswith(str(STATE.workdir)):
        return False, "path escapes workdir"

    return True, candidate.relative_to(STATE.workdir).as_posix()


def normalize_path(user_path: Optional[str]) -> Tuple[bool, str]:
    _require_init()
    assert STATE.workdir is not None

    s = (user_path or "").strip().replace("\\", "/")
    if s == "":
        return False, "path is empty"
    if s.startswith("./"):
        s = s[2:]
    s = _strip_workdir_prefix(s)

    p = Path(s)
    if p.is_absolute() or (len(s) >= 2 and s[1] == ":" and s[0].isalpha()):
        return False, "absolute path is not allowed"

    candidate = (STATE.workdir / s).resolve()
    if not str(candidate).startswith(str(STATE.workdir)):
        return False, "path escapes workdir"

    return True, candidate.relative_to(STATE.workdir).as_posix()


def _abs(rel_norm: str) -> Path:
    _require_init()
    assert STATE.workdir is not None
    return (STATE.workdir / rel_norm).resolve()


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


def write_gate() -> Optional[str]:
    if STATE.task_type in ("chat", "review"):
        return f"WRITE_NOT_ALLOWED: 当前任务类型为 {STATE.task_type}（只读/聊天）。如需修改/创建文件，请明确说明。"
    return None


def repo_tree_impl(
    dir: Optional[str] = ".", recursive: Any = True, max_results: Any = 200
) -> str:
    _require_init()
    ok, rel = normalize_dir(dir)
    if not ok:
        return f"(error) {rel}"

    root = _abs(rel)
    if not root.exists():
        return f"(error) dir not found: {rel}"
    if not root.is_dir():
        return f"(error) not a dir: {rel}"

    rec = bool(recursive)
    mr = (
        int(max_results)
        if isinstance(max_results, (int, float, str)) and str(max_results).isdigit()
        else 200
    )

    out: List[str] = []
    if rec:
        for p in root.rglob("*"):
            if p.is_file():
                out.append(p.relative_to(STATE.workdir).as_posix())  # type: ignore[arg-type]
                if len(out) >= mr:
                    break
    else:
        for p in root.iterdir():
            if p.is_file():
                out.append(p.relative_to(STATE.workdir).as_posix())  # type: ignore[arg-type]
                if len(out) >= mr:
                    break

    _log_evidence(f"[tree] dir={rel} recursive={rec} files={len(out)}")
    return "\n".join(out) if out else "(empty)"


def list_files_impl(
    dir: Optional[str] = ".", recursive: Any = False, max_results: Any = 200
) -> str:
    return repo_tree_impl(dir=dir, recursive=recursive, max_results=max_results)


def grep_impl(
    pattern: Optional[str], path: Optional[str] = ".", max_results: Any = 50
) -> str:
    _require_init()
    if not pattern:
        return "(error) pattern is empty"

    ok, rel = normalize_dir(path)
    if not ok:
        return f"(error) {rel}"

    target = _abs(rel)
    if not target.exists():
        return f"(error) path not found: {rel}"

    mr = (
        int(max_results)
        if isinstance(max_results, (int, float, str)) and str(max_results).isdigit()
        else 50
    )

    results: List[str] = []

    def scan_file(fp: Path) -> None:
        text = fp.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                results.append(f"{fp.relative_to(STATE.workdir).as_posix()}:{i}:{line}")  # type: ignore[arg-type]
                if len(results) >= mr:
                    return

    if target.is_file():
        scan_file(target)
    else:
        for fp in target.rglob("*"):
            if fp.is_file():
                scan_file(fp)
                if len(results) >= mr:
                    break

    _log_evidence(f"[grep] pattern={pattern!r} hits={len(results)} path={rel}")
    return "\n".join(results) if results else "(no matches)"


# [L02-08F CHANGE] end_line 越界 -> 截断到 total（契约级行为，避免崩溃）
def read_file_range_impl(
    path: Optional[str], start_line: Any = 1, end_line: Any = 200
) -> str:
    _require_init()
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

    def to_int(x: Any, default: int) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, str) and x.strip().isdigit():
            return int(x.strip())
        return default

    sl = to_int(start_line, 1)
    el = to_int(end_line, 200)

    if sl < 1:
        sl = 1
    if el < 1:
        el = total
    if sl > total:
        return (
            f"file={rel}\ntotal_lines={total}\nshowing_lines=(empty)\n(range is empty)"
        )
    if el > total:
        el = total
    if sl > el:
        sl, el = el, sl

    seg = [f"{ln:>4} | {lines[ln - 1]}" for ln in range(sl, el + 1)]
    _log_evidence(f"[read_range] file={rel} lines={sl}-{el} total={total}")
    return f"file={rel}\ntotal_lines={total}\nshowing_lines={sl}-{el}\n" + "\n".join(
        seg
    )


def write_file_impl(file_path: Optional[str], content: Optional[str]) -> str:
    _require_init()
    gate = write_gate()
    if gate is not None:
        return _json("WRITE_NOT_ALLOWED", message=gate)

    ok, rel = normalize_path(file_path)
    if not ok:
        return _json("ERROR", message=rel)

    p = _abs(rel)
    existed = p.exists()
    old_text = p.read_text(encoding="utf-8", errors="replace") if existed else ""
    new_text = content or ""
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"

    payload = json.dumps(
        {"mode": "write_file", "content": new_text}, ensure_ascii=False, sort_keys=True
    )
    pid = _patch_id(rel, old_text, payload)
    diff = _unified_diff(old_text, new_text, rel)

    if pid in STATE.applied_patch_ids:
        return _json("ALREADY_APPLIED", patch_id=pid, file=rel)
    if pid in STATE.rejected_patch_ids:
        return _json(
            "REJECTED_REPEAT",
            patch_id=pid,
            file=rel,
            next="ASK_USER",
            message="该写入已被用户拒绝过；不要重复尝试，应询问用户如何处理。",
        )

    if STATE.write_policy == "deny":
        return _json(
            "DENY_POLICY",
            patch_id=pid,
            file=rel,
            next="ASK_USER",
            message="当前策略禁止写入；请询问用户是否切换策略。",
            diff=diff if diff.strip() else "(无 diff)",
        )

    if STATE.write_policy == "ask":
        plan = PendingPatch(
            patch_id=pid,
            file=rel,
            existed=existed,
            kind="write_file",
            new_text=new_text,
            diff=diff if diff.strip() else "(无 diff)",
        )
        STATE.pending_patches[pid] = plan
        _log_evidence(f"[preview_write_file] file={rel} patch_id={pid}")
        return _json(
            "PREVIEW",
            patch_id=pid,
            file=rel,
            diff=plan.diff,
            message="已生成写入预览，等待图层审批（interrupt）。",
        )

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new_text, encoding="utf-8")
    STATE.applied_patch_ids.add(pid)
    if existed:
        if rel not in STATE.changed_files:
            STATE.changed_files.append(rel)
    else:
        if rel not in STATE.created_files:
            STATE.created_files.append(rel)
    STATE.last_diffs[rel] = diff if diff.strip() else "(无 diff)"
    _log_evidence(
        f"[write_file] file={rel} existed={existed} lines={len(new_text.splitlines())}"
    )
    return _json("APPLIED", patch_id=pid, file=rel, diff=STATE.last_diffs[rel])


def apply_hunks_impl(file_path: Optional[str], hunks: Any) -> str:
    _require_init()
    gate = write_gate()
    if gate is not None:
        return _json("WRITE_NOT_ALLOWED", message=gate)

    ok, rel = normalize_path(file_path)
    if not ok:
        return _json("ERROR", message=rel)

    p = _abs(rel)
    if not p.exists():
        return _json(
            "ERROR", message=f"file not found: {rel}（创建新文件请用 write_file）"
        )

    if (
        not isinstance(hunks, list)
        or len(hunks) == 0
        or not all(isinstance(x, dict) for x in hunks)
    ):
        return _json(
            "ERROR",
            message="hunks 必须是非空 list[dict]，每项包含 start_line/end_line/replacement。",
        )

    old_text = p.read_text(encoding="utf-8", errors="replace")
    has_trailing_nl = old_text.endswith("\n")
    old_lines = old_text.splitlines()
    total = len(old_lines)

    normalized: List[Dict[str, Any]] = []
    for h in hunks:
        if "start_line" not in h or "end_line" not in h or "replacement" not in h:
            return _json(
                "ERROR", message="每个 hunk 必须包含 start_line/end_line/replacement"
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
                message="start_line/end_line 必须是 int；replacement 必须是 str",
            )
        if sl < 1:
            return _json("ERROR", message="start_line 必须 >= 1")
        if el < sl - 1:
            return _json("ERROR", message="end_line 必须 >= start_line-1")
        if sl > total + 1:
            return _json(
                "ERROR", message=f"start_line 越界：{sl} > total_lines+1 {total+1}"
            )
        if el >= sl and el > total:
            return _json("ERROR", message=f"end_line 越界：{el} > total_lines {total}")

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

    if pid in STATE.applied_patch_ids:
        return _json("ALREADY_APPLIED", patch_id=pid, file=rel)
    if pid in STATE.rejected_patch_ids:
        return _json(
            "REJECTED_REPEAT",
            patch_id=pid,
            file=rel,
            next="ASK_USER",
            message="该补丁已被用户拒绝过；不要重复尝试，应询问用户如何处理。",
        )

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
            return _json("ERROR", message="hunks 区间重叠：不允许 overlapping hunks")
        prev_s = slc["s"]

    new_lines = old_lines[:]
    for slc in slices_sorted:
        rep_lines = slc["replacement"].splitlines() if slc["replacement"] else []
        new_lines[slc["s"] : slc["e"]] = rep_lines

    new_text = "\n".join(new_lines)
    if has_trailing_nl:
        new_text += "\n"

    diff = _unified_diff(old_text, new_text, rel)

    if STATE.write_policy == "deny":
        return _json(
            "DENY_POLICY",
            patch_id=pid,
            file=rel,
            next="ASK_USER",
            message="当前策略禁止写入；请询问用户是否切换策略。",
            diff=diff if diff.strip() else "(无 diff)",
        )

    if STATE.write_policy == "ask":
        plan = PendingPatch(
            patch_id=pid,
            file=rel,
            existed=True,
            kind="apply_hunks",
            new_text=new_text,
            diff=diff if diff.strip() else "(无 diff)",
        )
        STATE.pending_patches[pid] = plan
        _log_evidence(
            f"[preview_apply_hunks] file={rel} patch_id={pid} hunks={len(normalized)}"
        )
        return _json(
            "PREVIEW",
            patch_id=pid,
            file=rel,
            diff=plan.diff,
            message="已生成补丁预览，等待图层审批（interrupt）。",
        )

    p.write_text(new_text, encoding="utf-8")
    STATE.applied_patch_ids.add(pid)
    if rel not in STATE.changed_files:
        STATE.changed_files.append(rel)
    STATE.last_diffs[rel] = diff if diff.strip() else "(无 diff)"
    _log_evidence(f"[apply_hunks] file={rel} hunks={len(normalized)}")
    return _json("APPLIED", patch_id=pid, file=rel, diff=STATE.last_diffs[rel])


def commit_patch_impl(patch_id: Optional[str]) -> str:
    _require_init()
    pid = (patch_id or "").strip()
    if not pid:
        return _json("ERROR", message="patch_id is empty")

    if pid in STATE.rejected_patch_ids:
        return _json(
            "REJECTED_REPEAT",
            patch_id=pid,
            next="ASK_USER",
            message="该补丁已被用户拒绝过；不会再次提交。",
        )

    plan = STATE.pending_patches.get(pid)
    if plan is None:
        return _json("ERROR", message=f"pending patch not found: {pid}")

    p = _abs(plan.file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(plan.new_text, encoding="utf-8")

    STATE.pending_patches.pop(pid, None)
    STATE.applied_patch_ids.add(pid)

    if plan.existed:
        if plan.file not in STATE.changed_files:
            STATE.changed_files.append(plan.file)
    else:
        if plan.file not in STATE.created_files:
            STATE.created_files.append(plan.file)

    STATE.last_diffs[plan.file] = plan.diff
    _log_evidence(f"[commit_patch] patch_id={pid} file={plan.file} kind={plan.kind}")
    return _json("APPLIED", patch_id=pid, file=plan.file, diff=plan.diff)


def reject_patch_impl(patch_id: Optional[str]) -> str:
    _require_init()
    pid = (patch_id or "").strip()
    if not pid:
        return _json("ERROR", message="patch_id is empty")

    plan = STATE.pending_patches.pop(pid, None)
    STATE.rejected_patch_ids.add(pid)
    _log_evidence(
        f"[reject_patch] patch_id={pid} file={(plan.file if plan else '(unknown)')}"
    )
    return _json(
        "USER_REJECTED",
        patch_id=pid,
        file=(plan.file if plan else None),
        message="用户拒绝补丁；不要重复尝试，应停止并询问用户如何处理。",
    )


def bash_impl(command: Optional[str]) -> str:
    _require_init()
    assert STATE.workdir is not None

    cmd = (command or "").strip()
    if not cmd:
        return "(error) command is empty"
    if not any(cmd.startswith(pfx) for pfx in STATE.allowed_cmd_prefixes):
        return f"(error) command not allowed: {cmd!r}"

    proc = subprocess.run(
        cmd,
        cwd=str(STATE.workdir),
        shell=True,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )

    if cmd.startswith("python -m pytest") or cmd.startswith("pytest"):
        STATE.last_test_cmd = cmd
        STATE.last_test_rc = proc.returncode
        STATE.last_test_stdout = proc.stdout
        STATE.last_test_stderr = proc.stderr
        _log_evidence(f"[pytest] cmd={cmd!r} rc={proc.returncode}")

    return (
        f"returncode={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
    )


def evidence_read_impl(max_items: Any = 50) -> str:
    _require_init()
    n = 50
    if isinstance(max_items, int):
        n = max(1, min(200, max_items))
    items = STATE.evidence_log[-n:]
    return "\n".join(items) if items else "(evidence is empty)"


def todowrite_impl(items: Any) -> str:
    _require_init()
    assert STATE.workdir is not None

    if (
        not isinstance(items, list)
        or len(items) == 0
        or not all(isinstance(x, str) for x in items)
    ):
        return "(error) items must be a non-empty list[str]"

    todo_file = (STATE.workdir / ".agent_todo.json").resolve()
    todo_file.write_text(
        json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log_evidence(f"[todo] saved items={len(items)}")
    return f"[OK] todo saved: {len(items)} items"


def todoread_impl() -> str:
    _require_init()
    assert STATE.workdir is not None
    todo_file = (STATE.workdir / ".agent_todo.json").resolve()
    if not todo_file.exists():
        return "(todo is empty)"
    return todo_file.read_text(encoding="utf-8", errors="replace")


def _ensure_toy_repo(repo_dir: Path) -> None:
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
