# -*- coding: utf-8 -*-
"""
标题：L02-01 tools.py —— 所有 @tool 都集中在这里（路径绝对化、写入三态、命令白名单、todo）
执行代码：
  被 main.py 导入后使用（不要单独运行）
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import List

from langchain_core.tools import tool


# -------------------------
# 运行时配置：由 main.py 初始化
# -------------------------

class _Config:
    workdir: Path | None = None
    write_policy: str = "ask"
    todo_file: Path | None = None

    # Windows 下 shell=True 默认走 cmd.exe；ls/cat 不一定可用
    allowed_cmd_prefixes: tuple[str, ...] = (
        "python -m pytest",
        "pytest",
        "python ",
        "dir",
        "type ",
        "ls",
        "cat ",
    )

CFG = _Config()


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
    """
    由 main.py 调用，设置全局配置。
    - workdir 强制为绝对路径
    """
    wd = workdir.resolve()
    CFG.workdir = wd
    CFG.write_policy = write_policy
    CFG.todo_file = (wd / ".agent_todo.json").resolve()
    ensure_toy_repo(wd)


def _require_cfg() -> None:
    if CFG.workdir is None or CFG.todo_file is None:
        raise RuntimeError("tools not initialized: call init_tools(workdir, policy) first")


def _normalize_rel_path(user_path: str) -> str:
    """
    规范化为“相对 workdir”的 posix 路径字符串，并做越界校验。
    - 禁止绝对路径
    - 剥离 ./ 与 workdir 名称前缀（例如 toy_repo/calc.py）
    """
    _require_cfg()
    assert CFG.workdir is not None

    s = (user_path or "").strip().replace("\\", "/")
    if not s:
        raise ValueError("path is empty")

    # 禁止绝对路径（Windows: C:/..., Unix: /...）
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
    _require_cfg()
    assert CFG.todo_file is not None
    if not CFG.todo_file.exists():
        return "(todo is empty)"
    return CFG.todo_file.read_text(encoding="utf-8", errors="replace")


@tool("todowrite")
def todowrite(items: List[str]) -> str:
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
    _require_cfg()
    rel = _normalize_rel_path(dir)
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
    _require_cfg()
    rel = _normalize_rel_path(path)
    p = _abs_path(rel)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {rel}")
    return p.read_text(encoding="utf-8", errors="replace")


@tool("apply_patch")
def apply_patch(file_path: str, new_content: str) -> str:
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
            return "[ASK->NO] 你拒绝了写入。以下 diff 未落盘：\n" + (diff if diff.strip() else "(无 diff)")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new_content, encoding="utf-8")
    return f"[OK] 已写入：{rel}\nDiff:\n{diff if diff.strip() else '(无 diff)'}"


@tool("bash")
def bash(command: str) -> str:
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
    return (
        f"returncode={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
    )