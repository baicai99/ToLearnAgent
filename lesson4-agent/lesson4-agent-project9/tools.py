# -*- coding: utf-8 -*-
"""
标题：L02-08D tools.py —— @tool 薄封装（所有参数尽量 Optional，避免 pydantic missing field 崩溃）
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
    """局部读取（越界自动截断，不抛异常）。"""
    return cr.read_file_range_impl(path=path, start_line=start_line, end_line=end_line)


@tool("write_file")
def write_file(file_path: Optional[str] = None, content: Optional[str] = None) -> str:
    """创建/覆盖文件：ask 策略只返回 PREVIEW（等待图层审批）；allow 直接写。"""
    return cr.write_file_impl(file_path=file_path, content=content)


@tool("apply_hunks")
def apply_hunks(file_path: Optional[str] = None, hunks: Any = None) -> str:
    """局部补丁：ask 策略只返回 PREVIEW（等待图层审批）；allow 直接写。"""
    return cr.apply_hunks_impl(file_path=file_path, hunks=hunks)


@tool("commit_patch")
def commit_patch(patch_id: Optional[str] = None) -> str:
    """提交一个待审批补丁（patch_id 来自 PREVIEW 的返回）。"""
    return cr.commit_patch_impl(patch_id)


@tool("reject_patch")
def reject_patch(patch_id: Optional[str] = None) -> str:
    """拒绝一个待审批补丁（记录拒绝，避免重复尝试同补丁）。"""
    return cr.reject_patch_impl(patch_id)


@tool("bash")
def bash(command: Optional[str] = None) -> str:
    """执行命令（仅允许 python/pytest 前缀）。"""
    return cr.bash_impl(command=command)


@tool("evidence_read")
def evidence_read(max_items: Any = 50) -> str:
    """读取 evidence 日志（用于生成 submit.evidence）。"""
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
    """完成闸门（create/tests/implement 用于验收；chat/review 不鼓励调用）。"""
    return cr.submit_impl(final=final, evidence=evidence, task_type=task_type)
