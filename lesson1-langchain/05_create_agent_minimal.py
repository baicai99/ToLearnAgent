# 05_create_agent_minimal.py
# 目标：用 create_agent 创建最小 Agent（1 个工具 + 1 次 invoke）
# 运行：
#   python 05_create_agent_minimal.py --question "what is the weather in sf"
#
# 参考：LangChain Quickstart 的“Build a basic agent”就是 create_agent + tools + system_prompt。:contentReference[oaicite:18]{index=18}

import os
import argparse
from langchain.agents import create_agent  # 你指定的入口 :contentReference[oaicite:19]{index=19}


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="what is the weather in sf")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "claude-sonnet-4-5-20250929"))
    args = parser.parse_args()

    agent = create_agent(
        model=args.model,  # 也可以传 model 实例；这里先用最简单的字符串方式 :contentReference[oaicite:20]{index=20}
        tools=[get_weather],
        system_prompt="You are a helpful assistant.",
    )

    resp = agent.invoke({"messages": [{"role": "user", "content": args.question}]})
    # resp 通常是 dict，包含 messages 等信息；这里直接打印，先感受结构
    print(resp)


if __name__ == "__main__":
    main()
