# -*- coding: utf-8 -*-
"""
执行代码：python -m lessonL03_02_safe_persist.main
标题：L03-02 事件日志（JSONL）

[L03-02 NEW]
- 用 JSONL 记录：gate/llm/tool/approval/commit 等关键事件，便于复盘与对齐 opencode 风格的可观测性
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
        rec = {"ts": _utc_iso(), "type": event_type, "data": data}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
