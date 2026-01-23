# -*- coding: utf-8 -*-
"""
标题：L02-08G tools.py —— @tool 封装 + 新增 run_pytest
执行代码：
  由 main.py 导入并 bind_tools 使用
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool
import core_runtime as cr


def init_tools(workdir: str, write_policy: str) -> None:
    cr.init_runtime(Path(workdir), write_policy)


def set_turn_context(user_text: str) -> str:
    return cr.set_turn_context(user_text)


@tool("repo_tree")
def repo_tree(
    dir: Optional[str] = ".", recursive: Any = True, max_results: Any = 200
) -> str:
    """列出工作区文件结构（只读）。"""
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
    """局部读取（越界自动截断到文件末尾）。"""
    return cr.read_file_range_impl(path=path, start_line=start_line, end_line=end_line)


@tool("write_file")
def write_file(file_path: Optional[str] = None, content: Optional[str] = None) -> str:
    """创建/覆盖文件：ask 策略只 PREVIEW（等待图层审批）；allow 直接写。"""
    return cr.write_file_impl(file_path=file_path, content=content)


@tool("apply_hunks")
def apply_hunks(file_path: Optional[str] = None, hunks: Any = None) -> str:
    """局部补丁：hunks 为 list[dict]；ask 策略 PREVIEW+审批；allow 直接写。"""
    return cr.apply_hunks_impl(file_path=file_path, hunks=hunks)


@tool("commit_patch")
def commit_patch(patch_id: Optional[str] = None) -> str:
    """提交一个待审批补丁（patch_id 来自 PREVIEW）。"""
    return cr.commit_patch_impl(patch_id)


@tool("reject_patch")
def reject_patch(patch_id: Optional[str] = None) -> str:
    """拒绝一个待审批补丁（记录拒绝，避免重复尝试）。"""
    return cr.reject_patch_impl(patch_id)


# ===================== [L02-08G NEW] run_pytest 工具 =====================
@tool("run_pytest")
def run_pytest(path: str = "", args: str = "-q") -> str:
    """运行 pytest（统一用 python -m pytest），返回结构化 JSON（rc/cmd/summary）。"""
    return cr.run_pytest_impl(path=path, args=args)


# ========================================================================


@tool("evidence_read")
def evidence_read(max_items: Any = 50) -> str:
    """读取 evidence 日志（教学回放）。"""
    return cr.evidence_read_impl(max_items=max_items)


@tool("todowrite")
def todowrite(items: Any = None) -> str:
    """写入 todo（items: 非空 list[str]）。"""
    return cr.todowrite_impl(items)


@tool("todoread")
def todoread() -> str:
    """读取 todo。"""
    return cr.todoread_impl()
