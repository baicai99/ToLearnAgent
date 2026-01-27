# -*- coding: utf-8 -*-
"""
标题：L03-05 配置与 .env 加载
执行：python -m lessonL03_05_command_templates.main
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Literal

WritePolicy = Literal["allow", "ask", "deny"]
AgentMode = Literal["plan", "build"]

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

WORKDIR = (PROJECT_ROOT / "toy_repo").resolve()

SESSION_FILE = PROJECT_ROOT / ".l03_05_session.json"
CHECKPOINT_DB = PROJECT_ROOT / ".l03_05_checkpoints.sqlite"
EVENT_LOG = PROJECT_ROOT / ".l03_05_events.jsonl"

DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_WRITE_POLICY: WritePolicy = "ask"
DEFAULT_MODE: AgentMode = "build"


def _parse_env_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k:
            data[k] = v
    return data


def load_env_from_root() -> Tuple[str, str]:
    env_path = PROJECT_ROOT / ".env"
    kv = _parse_env_file(env_path)

    if "OPENAI_API_KEY" in kv and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = kv["OPENAI_API_KEY"]

    if "OPENAI_BASE_URL" in kv and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = kv["OPENAI_BASE_URL"]

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()

    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY（请在根目录 .env 或环境变量中设置）")

    return api_key, base_url
