# -*- coding: utf-8 -*-
"""
标题：L03-04 工具层（LangChain @tool）
执行：python -m lessonL03_04_commands_modes.main
"""

from __future__ import annotations

import json
import subprocess
import uuid
from typing import Any, Dict, List

from langchain_core.tools import tool

from .core_runtime import (
    Runtime,
    unified_diff,
    apply_hunks_to_text,
    compute_base_sha,
    fingerprint_patch,
    safe_preview,
)


def attach_runtime(rt: Runtime, tool_fns: List[Any]) -> None:
    for fn in tool_fns:
        setattr(fn, "_rt", rt)


@tool("list_files")
def list_files(dir_path: str = ".") -> str:
    """列出工作区文件（递归）。dir_path 为空时等价 "."。"""
    rt: Runtime = list_files._rt  # type: ignore[attr-defined]
    files = rt.list_files(dir_path)
    return json.dumps({"dir": dir_path or ".", "count": len(files), "files": files}, ensure_ascii=False)


@tool("read_file_range")
def read_file_range(path: str, start_line: int = 1, end_line: int = 200) -> str:
    """读取文件行区间（自动 clamp，不会越界崩溃）。"""
    rt: Runtime = read_file_range._rt  # type: ignore[attr-defined]
    data = rt.read_file_range(path, start_line, end_line)
    return json.dumps(data, ensure_ascii=False)


@tool("grep")
def grep(pattern: str, path: str = ".", max_results: int = 50) -> str:
    """在工作区中做简单子串搜索，返回命中列表。"""
    rt: Runtime = grep._rt  # type: ignore[attr-defined]
    hits = rt.grep(pattern, path, max_results)
    return json.dumps({"pattern": pattern, "path": path, "count": len(hits), "hits": hits}, ensure_ascii=False)


@tool("bash")
def bash(cmd: str, timeout_sec: int = 20) -> str:
    """在工作区 cwd=toy_repo 执行命令，返回 stdout/stderr/returncode。"""
    rt: Runtime = bash._rt  # type: ignore[attr-defined]
    p = subprocess.run(
        cmd,
        shell=True,
        cwd=str(rt.workdir),
        capture_output=True,
        text=True,
        timeout=int(timeout_sec),
    )
    return json.dumps({"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}, ensure_ascii=False)


@tool("propose_hunks")
def propose_hunks(file_path: str, hunks: List[Dict[str, Any]]) -> str:
    """生成补丁计划（hunks 模式，不写入）：返回 patch_id/base_sha/fingerprint/diff/hunks。"""
    rt: Runtime = propose_hunks._rt  # type: ignore[attr-defined]
    p = rt.abs_path(file_path)

    old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    base_sha = compute_base_sha(old)
    new = apply_hunks_to_text(old, hunks)

    diff_text = unified_diff(old, new, file_path)
    plan = {
        "status": "PENDING",
        "patch_id": str(uuid.uuid4()),
        "file_path": file_path,
        "base_sha": base_sha,
        "fingerprint": fingerprint_patch(file_path, diff_text),
        "diff": safe_preview(diff_text),
        "mode": "hunks",
        "hunks": hunks,
    }
    return json.dumps(plan, ensure_ascii=False)


@tool("propose_write_file")
def propose_write_file(file_path: str, content: str) -> str:
    """生成补丁计划（整文件覆盖，不写入）：返回 patch_id/base_sha/fingerprint/diff/content。"""
    rt: Runtime = propose_write_file._rt  # type: ignore[attr-defined]
    p = rt.abs_path(file_path)

    old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    base_sha = compute_base_sha(old)

    diff_text = unified_diff(old, content, file_path)
    plan = {
        "status": "PENDING",
        "patch_id": str(uuid.uuid4()),
        "file_path": file_path,
        "base_sha": base_sha,
        "fingerprint": fingerprint_patch(file_path, diff_text),
        "diff": safe_preview(diff_text),
        "mode": "write_file",
        "content": content,
    }
    return json.dumps(plan, ensure_ascii=False)
