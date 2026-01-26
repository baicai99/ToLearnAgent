# -*- coding: utf-8 -*-
"""
执行代码：python -m lessonL03_02_safe_persist.main
标题：L03-02 核心运行时（沙箱路径 + diff + hunks）

[L03-02 NEW]
- list_files：dir_path 为空时等价 "."
- read_file_range：自动 clamp，不再出现 end_line 越界崩溃
- PatchPlan：带 base_sha + fingerprint，支持拒绝记忆与文件漂移检测
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


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

        sl = max(1, int(start_line))
        el = max(1, int(end_line))

        if total == 0:
            return {
                "path": path,
                "total_lines": 0,
                "start_line": 1,
                "end_line": 1,
                "text": "",
            }

        sl = min(sl, total)
        el = min(el, total)

        snippet = "\n".join(lines[sl - 1 : el])
        return {
            "path": path,
            "total_lines": total,
            "start_line": sl,
            "end_line": el,
            "text": snippet,
        }

    def write_text(self, path: str, content: str) -> None:
        p = self.abs_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def unified_diff(old: str, new: str, file_path: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines, fromfile=f"a/{file_path}", tofile=f"b/{file_path}"
    )
    return "".join(diff)


def apply_hunks_to_text(old_text: str, hunks: List[Dict[str, Any]]) -> str:
    lines = old_text.splitlines(keepends=True)

    hunks_sorted = sorted(
        hunks, key=lambda x: (int(x.get("start_line", 0)), int(x.get("end_line", 0)))
    )
    for h in hunks_sorted:
        sl = int(h.get("start_line", 0))
        el = int(h.get("end_line", 0))
        rep = h.get("replace_with", "")
        if sl == 0 and el == 0:
            lines = rep.splitlines(keepends=True) + lines
            continue
        if sl < 1 or el < 1 or el < sl:
            raise ValueError(f"invalid hunk range: start_line={sl}, end_line={el}")
        if sl > len(lines) + 1:
            raise ValueError(f"hunk start_line out of range: {sl} > {len(lines) + 1}")
        if el > len(lines):
            raise ValueError(f"hunk end_line out of range: {el} > {len(lines)}")

        rep_lines = rep.splitlines(keepends=True)
        lines = lines[: sl - 1] + rep_lines + lines[el:]

    return "".join(lines)


def compute_base_sha(text: str) -> str:
    return _sha256_text(text)


def fingerprint_patch(file_path: str, diff_text: str) -> str:
    return _sha256_text(file_path + "\n" + diff_text)


def safe_preview(diff_text: str, max_chars: int = 6000) -> str:
    if len(diff_text) <= max_chars:
        return diff_text
    return diff_text[:max_chars] + "\n...(diff truncated)...\n"
