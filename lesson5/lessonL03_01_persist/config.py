# -*- coding: utf-8 -*-
"""
标题：L03-01 配置与 .env 加载
执行：python lessonL03_01_persist/main.py

[L03 NEW]
- 不依赖 python-dotenv：自己实现最小 .env 解析（可控、可追溯）
- 统一给 ChatOpenAI 提供 api_key/base_url
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent  # 假设 lessonL03_01_persist/ 在项目根目录下
WORKDIR = (PROJECT_ROOT / "toy_repo").resolve()

SESSION_FILE = PROJECT_ROOT / ".l03_session.json"
CHECKPOINT_DB = PROJECT_ROOT / ".l03_checkpoints.sqlite"
EVENT_LOG = PROJECT_ROOT / ".l03_events.jsonl"


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
    """
    从根目录 .env 读取 OPENAI_API_KEY / OPENAI_BASE_URL，并注入到环境变量。
    返回：(api_key, base_url)

    约束：
    - 如果环境变量已设置，则不覆盖（避免你在 shell 里临时切换）
    """
    env_path = PROJECT_ROOT / ".env"
    kv = _parse_env_file(env_path)

    if "OPENAI_API_KEY" in kv and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = kv["OPENAI_API_KEY"]

    # OpenAI / LangChain 习惯用 OPENAI_BASE_URL
    if "OPENAI_BASE_URL" in kv and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = kv["OPENAI_BASE_URL"]

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()

    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY（请在根目录 .env 或环境变量中设置）")
    if not base_url:
        # 允许为空：代表使用默认 OpenAI base url
        base_url = ""

    return api_key, base_url
