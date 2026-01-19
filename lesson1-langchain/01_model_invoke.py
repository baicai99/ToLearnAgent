# 01_model_invoke.py
# 目标：用 LangChain 最小化调用一次模型（invoke）
# 运行：
#   python 01_model_invoke.py --input "用一句话解释 LangChain 是什么"
# 环境变量：
#   OPENAI_API_KEY 或 ANTHROPIC_API_KEY
# 可选：
#   LC_MODEL（例如 "gpt-4.1" 或 "claude-sonnet-4-5-20250929"）
#
# 参考：LangChain Models 文档推荐 init_chat_model + invoke 的最简方式。:contentReference[oaicite:11]{index=11}

import os
import argparse
from langchain.chat_models import init_chat_model  # 官方推荐入口 :contentReference[oaicite:12]{index=12}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-4.1"))
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max_tokens", type=int, default=500)
    args = parser.parse_args()

    model = init_chat_model(
        args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )

    resp = model.invoke(args.input)
    # resp 是 AIMessage，常用字段 resp.content
    print(resp.content)


if __name__ == "__main__":
    main()
