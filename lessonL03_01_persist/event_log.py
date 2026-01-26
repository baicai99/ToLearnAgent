# -*- coding: utf-8 -*-
"""
标题：L03-01 事件日志（JSONL）
执行：由 main.py 自动写入 .l03_events.jsonl

[L03 NEW]
- 事件日志是“工程化 Agent”的核心：可复盘、可回放、可调试（对齐 opencode 的可观测性思路）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EventLogger:
    path: Path

    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        rec = {
            "ts": _utc_iso(),
            "type": event_type,
            "data": data,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
