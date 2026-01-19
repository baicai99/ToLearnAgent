# 00_config.py
# 目标：学会“读取环境变量 + 解析命令行参数 + 脱敏打印”
# 运行：
#   python 00_config.py --input "hello"
# 可选环境变量：
#   OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENAI_BASE_URL / BOCHA_API_KEY / LC_MODEL

import os
import argparse
import json
from datetime import datetime, timezone


def mask_secret(s: str) -> str:
    """脱敏：只保留后4位"""
    if not s:
        return ""
    if len(s) <= 4:
        return "*" * len(s)
    return "*" * (len(s) - 4) + s[-4:]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入文本")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", ""), help="模型标识（可选）")
    parser.add_argument("--dry-run", action="store_true", help="不调用外部服务，仅演示配置")
    args = parser.parse_args()

    cfg = {
        "OPENAI_API_KEY": mask_secret(os.getenv("OPENAI_API_KEY", "").strip()),
        "ANTHROPIC_API_KEY": mask_secret(os.getenv("ANTHROPIC_API_KEY", "").strip()),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "").strip(),
        "BOCHA_API_KEY": mask_secret(os.getenv("BOCHA_API_KEY", "").strip()),
        "LC_MODEL": args.model or os.getenv("LC_MODEL", "").strip(),
    }

    record = {
        "ts": now_iso(),
        "input": args.input,
        "dry_run": args.dry_run,
        "config_snapshot": cfg,
    }

    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
