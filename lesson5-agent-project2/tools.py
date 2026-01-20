# -*- coding: utf-8 -*-
"""
标题：L02-06 tools.py —— @tool 工具集 + 证据驱动 submit 闸门（Stop Rule）
执行代码：
  由 main.py 导入并初始化（不要单独运行）
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any

from langchain_core.tools import tool


class _Config:
    """运行时配置与可验证证据缓存（由 main.py 初始化）。"""

    workdir: Optional[Path] = None
    write_policy: str = "ask"
    todo_file: Optional[Path] = None

    # 可验证证据（供 submit 检查）
    last_test_rc: Optional[int] = None
    last_test_cmd: Optional[str] = None
    last_test_stdout: str = ""
    last_test_stderr: str = ""

    changed_files: List[str] = []
    last_diffs: Dict[str, str] = {}

    allowed_cmd_prefixes: tuple[str, ...] = (
        "python -m pytest",
        "pytest",
        "python ",
    )


CFG = _Config()


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
    """
    初始化工具全局配置。
    - workdir 强制绝对路径（避免相对/绝对混用导致崩溃）
    """
    wd = workdir.resolve()
    CFG.workdir = wd
    CFG.write_policy = write_policy
    CFG.todo_file = (wd / ".agent_todo.json").resolve()

    # 重置证据缓存（新会话开始）
    CFG.last_test_rc = None
    CFG.last_test_cmd = None
    CFG.last_test_stdout = ""
    CFG.last_test_stderr = ""
    CFG.changed_files = []
    CFG.last_diffs = {}

    ensure_toy_repo(wd)


def _require_cfg() -> None:
    if CFG.workdir is None or CFG.todo_file is None:
        raise RuntimeError("tools not initialized: call init_tools(workdir, policy) first")


def _normalize_rel_path(user_path: str) -> str:
    """
    将工具入参规范化为“相对 workdir 的 posix 路径”，并做越界校验。
    - 禁止绝对路径
    - 剥离 ./ 与 workdir 前缀（例如 toy_repo/calc.py）
    """
    _require_cfg()
    assert CFG.workdir is not None

    s = (user_path or "").strip().replace("\\", "/")
    if not s:
        raise ValueError("path is empty")

    p = Path(s)
    if p.is_absolute() or (len(s) >= 2 and s[1] == ":" and s[0].isalpha()):
        raise ValueError("absolute path is not allowed")

    if s.startswith("./"):
        s = s[2:]

    wd_name = CFG.workdir.name.replace("\\", "/")
    prefix = wd_name + "/"
    if s.startswith(prefix):
        s = s[len(prefix):]

    candidate = (CFG.workdir / s).resolve()
    if not str(candidate).startswith(str(CFG.workdir)):
        raise ValueError("path escapes workdir")

    return candidate.relative_to(CFG.workdir).as_posix()


def _abs_path(rel_norm: str) -> Path:
    _require_cfg()
    assert CFG.workdir is not None
    return (CFG.workdir / rel_norm).resolve()


def _unified_diff(old: str, new: str, filename: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )
    return "".join(diff)


@tool("todoread")
def todoread() -> str:
    """读取任务清单（todo）。若不存在则返回 (todo is empty)。"""
    _require_cfg()
    assert CFG.todo_file is not None
    if not CFG.todo_file.exists():
        return "(todo is empty)"
    return CFG.todo_file.read_text(encoding="utf-8", errors="replace")


@tool("todowrite")
def todowrite(items: List[str]) -> str:
    """写入任务清单（todo），覆盖写入。items 为字符串列表（3~7条为宜）。"""
    _require_cfg()
    assert CFG.todo_file is not None
    if not isinstance(items, list):
        raise ValueError("items must be a list of strings")
    CFG.todo_file.write_text(
        json.dumps({"items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return f"[OK] todo saved: {len(items)} items"


@tool("list_files")
def list_files(dir: str = ".", recursive: bool = False, max_results: int = 200) -> str:
    """列出目录下文件（返回相对 workdir 的路径，每行一个）。"""
    _require_cfg()
    # 当 LLM 传入空字符串时，使用默认值 "."
    rel = _normalize_rel_path(dir if dir else ".")
    root = _abs_path(rel)
    if not root.exists():
        raise FileNotFoundError(f"dir not found: {rel}")
    if not root.is_dir():
        raise ValueError(f"not a dir: {rel}")

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
    return "\n".join(out)


@tool("grep")
def grep(pattern: str, path: str = ".", max_results: int = 50) -> str:
    """在工作区内搜索文本（简单包含匹配）。返回：file:line:content。"""
    _require_cfg()
    if not pattern:
        raise ValueError("pattern is empty")

    rel = _normalize_rel_path(path)
    target = _abs_path(rel)
    if not target.exists():
        raise FileNotFoundError(f"path not found: {rel}")

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
def read_file(path: str) -> str:
    """读取文件全文（path 为相对 workdir）。"""
    _require_cfg()
    rel = _normalize_rel_path(path)
    p = _abs_path(rel)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {rel}")
    return p.read_text(encoding="utf-8", errors="replace")


@tool("apply_patch")
def apply_patch(file_path: str, new_content: str) -> str:
    """
    覆盖写入文件（简化版 patch）。写入受 policy 控制：allow/ask/deny。
    重要：此工具会记录 changed_files 与 diff，供 submit 作为可验证证据引用。
    """
    _require_cfg()
    rel = _normalize_rel_path(file_path)
    p = _abs_path(rel)

    old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    diff = _unified_diff(old, new_content, rel)

    if CFG.write_policy == "deny":
        return "[DENY] 当前策略禁止写入。以下为建议 diff（未落盘）：\n" + (diff if diff.strip() else "(无 diff)")

    if CFG.write_policy == "ask":
        print("\n========== [WRITE PREVIEW / 需要确认] ==========")
        print(diff if diff.strip() else "(无 diff)")
        print("===============================================")
        ans = input("是否写入该补丁？(y/n) ").strip().lower()
        if ans not in ("y", "yes"):
            return (
                "[ASK->NO] 用户拒绝了此次写入，补丁未落盘。\n"
                "【重要】请不要重复尝试同样的修改！你应该：\n"
                "1. 停止当前操作\n"
                "2. 询问用户希望如何处理（例如：是否需要不同的修改方案？或者跳过此文件？）\n"
                "以下是被拒绝的 diff：\n" + (diff if diff.strip() else "(无 diff)")
            )

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new_content, encoding="utf-8")

    if rel not in CFG.changed_files:
        CFG.changed_files.append(rel)
    CFG.last_diffs[rel] = diff if diff.strip() else "(无 diff)"

    return f"[OK] 已写入：{rel}\nDiff:\n{diff if diff.strip() else '(无 diff)'}"


@tool("bash")
def bash(command: str) -> str:
    """
    在工作区执行命令（命令受白名单前缀限制）。
    若执行的是 pytest，将记录 last_test_rc/stdout/stderr 作为证据。
    """
    _require_cfg()
    assert CFG.workdir is not None

    cmd = (command or "").strip()
    if not cmd:
        raise ValueError("command is empty")

    if not any(cmd.startswith(pfx) for pfx in CFG.allowed_cmd_prefixes):
        raise ValueError(f"command not allowed: {cmd!r}")

    proc = subprocess.run(
        cmd,
        cwd=str(CFG.workdir),
        shell=True,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )

    # 记录“可验证证据”
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
    final: str,
    evidence: List[str],
    task_type: Literal["tests", "general"] = "general",
) -> str:
    """
    提交“完成”。这是证据驱动的 Stop Rule 闸门：
    - task_type="tests"：必须满足最近一次 pytest returncode==0 才 ACCEPT
    - evidence：要求列出你用来支撑 final 的证据要点（例如：pytest 通过、修改了哪些文件、grep 命中等）

    返回：
      JSON 字符串：{"status":"ACCEPT"|"REJECT","reason": "..."}
    """
    _require_cfg()

    if not final:
        raise ValueError("final is empty")
    if not isinstance(evidence, list) or len(evidence) == 0:
        raise ValueError("evidence must be a non-empty list of strings")

    # tests 任务的硬门槛：pytest 必须通过
    if task_type == "tests":
        if CFG.last_test_rc is None:    
            return json.dumps(
                {"status": "REJECT", "reason": "tests 任务未检测到 pytest 运行证据（请先 bash 运行 python -m pytest -q）。"},
                ensure_ascii=False,
            )
        if CFG.last_test_rc != 0:
            return json.dumps(
                {"status": "REJECT", "reason": f"pytest 未通过（returncode={CFG.last_test_rc}），不能完成。请修复后再跑 pytest。"},
                ensure_ascii=False,
            )

    # 额外的基本一致性：如果写过文件但 evidence 没提到，拒绝（防止“无证据总结”）
    if len(CFG.changed_files) > 0:
        changed_hint = "changed_files=" + ",".join(CFG.changed_files)
        if not any("changed" in e or "diff" in e or "修改" in e or "变更" in e for e in evidence):
            return json.dumps(
                {"status": "REJECT", "reason": f"你修改了文件但 evidence 未包含变更说明（建议包含：{changed_hint}）。"},
                ensure_ascii=False,
            )

    return json.dumps({"status": "ACCEPT", "reason": "ok"}, ensure_ascii=False)