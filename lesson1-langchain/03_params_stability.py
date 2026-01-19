# 03_params_stability.py
# 目标：感受 temperature / max_tokens / timeout / max_retries 对输出与稳定性的影响
# 运行：
#   python 03_params_stability.py --input "给我三个不同角度解释什么是上下文爆炸" --runs 3 --temperature 0.8
#
# 参考：Models 文档列出常用参数（temperature/max_tokens/timeout/max_retries）。:contentReference[oaicite:15]{index=15}

import os
import argparse
from langchain.chat_models import init_chat_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-4.1"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max_tokens", type=int, default=400)
    parser.add_argument("--max_retries", type=int, default=2)
    args = parser.parse_args()

    model = init_chat_model(
        args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )

    for i in range(args.runs):
        resp = model.invoke(args.input)
        print(f"\n--- run {i+1} ---")
        print(resp.content)


if __name__ == "__main__":
    main()
