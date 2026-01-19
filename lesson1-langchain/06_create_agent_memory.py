# 06_create_agent_memory.py
# 目标：让 agent 具备“短期记忆”（同一 thread_id 的多次调用能共享状态）
# 运行：
#   python 06_create_agent_memory.py
#
# 参考：Quickstart 展示 InMemorySaver + thread_id 的用法。:contentReference[oaicite:21]{index=21}

import os
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  # Quickstart 示例 :contentReference[oaicite:22]{index=22}


def remember(name: str) -> str:
    """A trivial tool just to demonstrate tool availability."""
    return f"记住了：{name}"


def main():
    model_id = os.getenv("LC_MODEL", "claude-sonnet-4-5-20250929")

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model_id,
        tools=[remember],
        system_prompt="你是一个中文助手，会记住用户在同一对话线程里说过的话。",
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "demo-thread-1"}}

    r1 = agent.invoke({"messages": [{"role": "user", "content": "我叫白菜，请记住。"}]}, config=config)
    print("Round1:", r1)

    r2 = agent.invoke({"messages": [{"role": "user", "content": "我刚才叫什么？"}]}, config=config)
    print("Round2:", r2)


if __name__ == "__main__":
    main()
