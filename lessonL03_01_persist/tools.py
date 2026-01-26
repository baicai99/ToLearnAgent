# -*- coding: utf-8 -*-
"""
标题：L03-01 工具层（LangChain @tool）
执行：由 graph_agent.py 调用

[L03 NEW]
- 工具全部使用 langchain_core.tools.tool 装饰器
- 每个工具必须有 docstring（避免你之前遇到的 ValueError: missing docstring）
- 写入工具只“生成补丁计划”，不直接落盘（审批/策略在 graph 里做）
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from .core_runtime import (
    Runtime,
    unified_diff,
    apply_hunks_to_text,
    compute_base_sha,
    fingerprint_patch,
    safe_preview_for_terminal,
)


def init_runtime(workdir) -> Runtime:
    return Runtime(workdir=workdir)


@tool("list_files")
def list_files(dir_path: str = ".") -> str:
    """列出工作区（toy_repo）内的所有文件（递归）。dir_path 为空时等价于 "."。"""
    rt: Runtime = list_files._rt  # type: ignore[attr-defined]
    files = rt.list_files(dir_path)
    return json.dumps(
        {"dir": dir_path or ".", "count": len(files), "files": files},
        ensure_ascii=False,
    )


@tool("read_file_range")
def read_file_range(path: str, start_line: int = 1, end_line: int = 200) -> str:
    """读取文件指定行范围（start_line/end_line 会自动 clamp，不会因越界崩溃）。"""
    rt: Runtime = read_file_range._rt  # type: ignore[attr-defined]
    data = rt.read_file_range(path, start_line, end_line)
    return json.dumps(data, ensure_ascii=False)


@tool("grep")
def grep(pattern: str, path: str = ".", max_results: int = 50) -> str:
    """在工作区中进行简单子串搜索，返回匹配行列表。"""
    rt: Runtime = grep._rt  # type: ignore[attr-defined]
    hits = rt.grep(pattern, path, max_results)
    return json.dumps(
        {"pattern": pattern, "path": path, "count": len(hits), "hits": hits},
        ensure_ascii=False,
    )


@tool("bash")
def bash(cmd: str, timeout_sec: int = 20) -> str:
    """在工作区内执行命令（cwd=toy_repo），返回 stdout/stderr/returncode。"""
    import subprocess

    rt: Runtime = bash._rt  # type: ignore[attr-defined]
    # 强制在 workdir 执行
    p = subprocess.run(
        cmd,
        shell=True,
        cwd=str(rt.workdir),
        capture_output=True,
        text=True,
        timeout=int(timeout_sec),
    )
    return json.dumps(
        {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
        },
        ensure_ascii=False,
    )


@tool("propose_hunks")
def propose_hunks(file_path: str, hunks: List[Dict[str, Any]]) -> str:
    """
    生成“补丁计划”（不写入）：根据 hunks 计算 diff，并返回 patch_id/base_sha/fingerprint/diff。
    hunks 每项：start_line, end_line, replace_with
    """
    rt: Runtime = propose_hunks._rt  # type: ignore[attr-defined]
    p = rt.abs_path(file_path)

    old = ""
    if p.exists():
        old = p.read_text(encoding="utf-8", errors="replace")

    base_sha = compute_base_sha(old)
    new = apply_hunks_to_text(old, hunks)
    diff_text = unified_diff(old, new, file_path)
    patch_id = str(uuid.uuid4())
    fp = fingerprint_patch(file_path, diff_text)

    return json.dumps(
        {
            "status": "PENDING",
            "patch_id": patch_id,
            "file_path": file_path,
            "base_sha": base_sha,
            "fingerprint": fp,
            "diff": safe_preview_for_terminal(diff_text),
            "hunks": hunks,  # commit 时需要（但 diff 不一定可逆）
        },
        ensure_ascii=False,
    )


@tool("propose_write_file")
def propose_write_file(file_path: str, content: str) -> str:
    """
    生成“写文件补丁计划”（不写入）：用新 content 覆盖文件并产出 diff。
    """
    rt: Runtime = propose_write_file._rt  # type: ignore[attr-defined]
    p = rt.abs_path(file_path)

    old = ""
    if p.exists():
        old = p.read_text(encoding="utf-8", errors="replace")

    base_sha = compute_base_sha(old)
    diff_text = unified_diff(old, content, file_path)
    patch_id = str(uuid.uuid4())
    fp = fingerprint_patch(file_path, diff_text)

    return json.dumps(
        {
            "status": "PENDING",
            "patch_id": patch_id,
            "file_path": file_path,
            "base_sha": base_sha,
            "fingerprint": fp,
            "diff": safe_preview_for_terminal(diff_text),
            "mode": "write_file",
            "content": content,
        },
        ensure_ascii=False,
    )


def attach_runtime_to_tools(rt: Runtime, tool_fns: List[Any]) -> None:
    """
    给 @tool 函数挂 runtime（避免全局变量污染）。
    """
    for fn in tool_fns:
        setattr(fn, "_rt", rt)
