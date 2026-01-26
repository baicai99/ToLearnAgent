# -*- coding: utf-8 -*-
"""
标题：L03-01 核心运行时（沙箱文件系统 + 补丁计划）
执行：由 tools.py / graph_agent.py 调用

[L03 NEW]
- 统一的 Path 沙箱：所有文件操作必须在 WORKDIR 内
- list_files 接受空字符串 => 等价 "."
- read_file_range 自动 clamp，不再抛 end_line 越界错误
- 写入不在 tool 内直接落盘：先产出 PatchPlan，再由“审批/策略节点”决定 commit
- PatchPlan 带 base_sha：防止审批期间文件漂移导致误写（工程化必备）
"""

from __future__ import annotations

import hashlib
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _norm_dir(d: str | None) -> str:
    if d is None:
        return "."
    d = d.strip()
    return d if d else "."


def _safe_rel_path(path: str) -> str:
    p = (path or "").strip().replace("\\", "/")
    if not p:
        raise ValueError("path is empty")
    if p.startswith("/") or p.startswith("../") or "/../" in p:
        raise ValueError(f"invalid path (must be relative, no ..): {path}")
    return p


@dataclass(frozen=True)
class Runtime:
    workdir: Path

    def abs_path(self, rel: str) -> Path:
        rel_norm = _safe_rel_path(rel)
        p = (self.workdir / rel_norm).resolve()
        # 强约束：必须在 workdir 内
        if self.workdir not in p.parents and p != self.workdir:
            raise ValueError(f"path escapes workdir: {rel}")
        return p

    def list_files(self, dir_path: str | None = ".") -> List[str]:
        d = _norm_dir(dir_path)
        base = self.abs_path(d) if d != "." else self.workdir
        if not base.exists():
            raise FileNotFoundError(f"dir not found: {d}")
        if not base.is_dir():
            raise NotADirectoryError(f"not a directory: {d}")

        out: List[str] = []
        for p in base.rglob("*"):
            if p.is_file():
                out.append(p.relative_to(self.workdir).as_posix())
        out.sort()
        return out

    def read_file_range(
        self, path: str, start_line: int = 1, end_line: int = 200
    ) -> Dict[str, Any]:
        p = self.abs_path(path)
        if not p.exists():
            raise FileNotFoundError(f"file not found: {path}")
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)

        # [L03 NEW] clamp：永不因 end_line 越界而抛错
        sl = max(1, int(start_line))
        el = max(1, int(end_line))
        sl = min(sl, total if total > 0 else 1)
        el = min(el, total if total > 0 else 1)

        if total == 0:
            snippet = ""
        else:
            snippet = "\n".join(lines[sl - 1 : el])

        return {
            "path": path,
            "total_lines": total,
            "start_line": sl,
            "end_line": el,
            "text": snippet,
        }

    def grep(
        self, pattern: str, path: str | None = ".", max_results: int = 50
    ) -> List[Dict[str, Any]]:
        pat = (pattern or "").strip()
        if not pat:
            raise ValueError("pattern is empty")
        d = _norm_dir(path)
        base = self.abs_path(d) if d != "." else self.workdir
        if not base.exists():
            raise FileNotFoundError(f"path not found: {d}")

        results: List[Dict[str, Any]] = []
        for fp in base.rglob("*"):
            if not fp.is_file():
                continue
            rel = fp.relative_to(self.workdir).as_posix()
            try:
                txt = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(txt.splitlines(), start=1):
                if pat in line:
                    results.append({"file": rel, "line": i, "text": line})
                    if len(results) >= int(max_results):
                        return results
        return results

    def write_text(self, path: str, content: str) -> None:
        p = self.abs_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def unified_diff(old: str, new: str, file_path: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return "".join(diff)


def apply_hunks_to_text(old_text: str, hunks: List[Dict[str, Any]]) -> str:
    """
    hunks: 每个元素：
      - start_line: 1-based
      - end_line: 1-based inclusive（允许 start=end=0 表示在文件开头插入）
      - replace_with: str
    """
    lines = old_text.splitlines(keepends=True)

    # 为了确定性：按 start_line 升序应用，避免重叠
    hunks_sorted = sorted(
        hunks, key=lambda x: (int(x.get("start_line", 0)), int(x.get("end_line", 0)))
    )

    for h in hunks_sorted:
        sl = int(h.get("start_line", 0))
        el = int(h.get("end_line", 0))
        rep = h.get("replace_with", "")
        if sl == 0 and el == 0:
            # 插入到开头
            insert_lines = rep.splitlines(keepends=True)
            lines = insert_lines + lines
            continue

        if sl < 1 or el < 1 or el < sl:
            raise ValueError(f"invalid hunk range: start_line={sl}, end_line={el}")

        # clamp 到当前文本长度（但这里若超界，认为 hunk 生成错误，直接报错更利于学习）
        if sl > len(lines) + 1:
            raise ValueError(f"hunk start_line out of range: {sl} > {len(lines) + 1}")
        if el > len(lines):
            raise ValueError(f"hunk end_line out of range: {el} > {len(lines)}")

        rep_lines = rep.splitlines(keepends=True)
        # Python 切片：sl-1 到 el-1 inclusive => [:el]
        lines = lines[: sl - 1] + rep_lines + lines[el:]

    return "".join(lines)


def compute_base_sha(text: str) -> str:
    return _sha256_text(text)


def fingerprint_patch(file_path: str, diff_text: str) -> str:
    return _sha256_text(file_path + "\n" + diff_text)


def safe_preview_for_terminal(diff_text: str, max_chars: int = 6000) -> str:
    if len(diff_text) <= max_chars:
        return diff_text
    return diff_text[:max_chars] + "\n...(diff truncated)...\n"
