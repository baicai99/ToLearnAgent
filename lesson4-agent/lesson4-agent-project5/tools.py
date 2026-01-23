# -*- coding: utf-8 -*-
"""
标题：L02-08A tools.py —— @tool 薄封装（短文件）
执行代码：
  由 main.py 导入并 bind_tools 使用
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

import core_runtime as cr


def init_tools(workdir: str, write_policy: str) -> None:
    cr.init_runtime(workdir=cr.Path(workdir), write_policy=write_policy)  # type: ignore[attr-defined]


def set_turn_context(user_text: str) -> str:
    return cr.set_turn_context(user_text)


@tool("repo_tree")
def repo_tree(
    dir: Optional[str] = ".", recursive: Any = True, max_results: Any = 200
) -> str:
    """列出工作区文件结构（只读，用于 review）。"""
    return cr.repo_tree_impl(dir=dir, recursive=recursive, max_results=max_results)


@tool("list_files")
def list_files(
    dir: Optional[str] = ".", recursive: Any = False, max_results: Any = 200
) -> str:
    """列出目录文件（只读）。"""
    return cr.list_files_impl(dir=dir, recursive=recursive, max_results=max_results)


@tool("grep")
def grep(
    pattern: Optional[str] = None, path: Optional[str] = ".", max_results: Any = 50
) -> str:
    """搜索文本（只读）。"""
    return cr.grep_impl(pattern=pattern, path=path, max_results=max_results)


@tool("read_file_range")
def read_file_range(
    path: Optional[str] = None, start_line: Any = 1, end_line: Any = 200
) -> str:
    """局部读取（越界会自动截断，不会抛异常导致退出）。"""
    return cr.read_file_range_impl(path=path, start_line=start_line, end_line=end_line)


@tool("write_file")
def write_file(file_path: Optional[str] = None, content: Optional[str] = None) -> str:
    """创建/覆盖文件（create/tests/implement 允许；review 禁止）。"""
    return cr.write_file_impl(file_path=file_path, content=content)


@tool("apply_hunks")
def apply_hunks(file_path: Optional[str] = None, hunks: Any = None) -> str:
    """局部补丁（hunks 必须是 list[dict]；review 禁止；创建文件请用 write_file）。"""
    return cr.apply_hunks_impl(file_path=file_path, hunks=hunks)


@tool("bash")
def bash(command: Optional[str] = None) -> str:
    """执行命令（仅允许 python/pytest 前缀）。"""
    return cr.bash_impl(command=command)


@tool("evidence_read")
def evidence_read(max_items: Any = 50) -> str:
    """读取当前 evidence（用于写 submit.evidence）。"""
    return cr.evidence_read_impl(max_items=max_items)


@tool("todowrite")
def todowrite(items: Any = None) -> str:
    """写入 todo（items: 非空 list[str]）。"""
    return cr.todowrite_impl(items)


@tool("todoread")
def todoread() -> str:
    """读取 todo。"""
    return cr.todoread_impl()


@tool("submit")
def submit(
    final: Optional[str] = None, evidence: Any = None, task_type: Optional[str] = None
) -> str:
    """完成闸门（按 task_type 校验：review/create/tests/implement）。"""
    return cr.submit_impl(final=final, evidence=evidence, task_type=task_type)
