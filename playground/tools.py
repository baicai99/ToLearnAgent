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

def _require_cfg()->None:
    if CFG.workdir is None or CFG.todo_file is None:
        raise RuntimeError("")

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