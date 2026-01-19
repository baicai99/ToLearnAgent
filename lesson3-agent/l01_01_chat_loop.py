# -*- coding: utf-8 -*-
"""
Micro-lesson 01: 终端对话窗口 + 消息结构（System/Human/AI）

目标：
- 在终端输入一句话，模型回复一句话。
- 让你看清楚：messages 如何累积上下文（System/Human/AI 三种消息）。
- 本课不引入工具，不引入 LangGraph，不谈 agent，只打基础。

运行：
  pip install -U langchain langchain-openai
  export OPENAI_API_KEY="..."
  python l01_01_chat_loop.py --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage


SYSTEM_PROMPT = (
    "你是一个教学型助手。回答时：\n"
    "1) 先用一句话复述用户问题（便于确认理解）；\n"
    "2) 再用2-6句给出回答；\n"
    "3) 不要输出与问题无关的内容。\n"
)


def print_messages(messages: List[BaseMessage], max_chars: int = 220) -> None:
    """
    教学用：打印当前 messages 的角色与内容摘要。
    """
    print("\n--- messages（上下文快照）---")
    for i, m in enumerate(messages):
        role = m.__class__.__name__.replace("Message", "")
        content = (m.content or "").replace("\n", "\\n")
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        print(f"[{i:02d}] {role}: {content}")
    print("----------------------------\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-lesson 01: minimal chat loop")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI 模型名")
    parser.add_argument("--temperature", type=float, default=0.2, help="温度（越低越稳定）")
    parser.add_argument("--show-messages", action="store_true", help="每轮都打印 messages 快照")
    args = parser.parse_args()

    # 初始化 LLM（依赖环境变量 OPENAI_API_KEY / OPENAI_BASE_URL）
    llm = ChatOpenAI(model=args.model, temperature=args.temperature)

    # messages 是“会话状态”的核心：它就是你后面做 agent 的最小记忆单元
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    print("已启动终端对话。输入 exit/quit 退出。")
    if args.show_messages:
        print_messages(messages)

    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not user_text:
            continue

        # 1) 把用户输入作为 HumanMessage 追加到上下文
        messages.append(HumanMessage(content=user_text))

        if args.show_messages:
            print("\n[Debug] 发送给模型之前的上下文：")
            print_messages(messages)

        # 2) 调用模型
        ai_msg = llm.invoke(messages)

        # 3) 把模型回复作为 AIMessage 追加到上下文
        messages.append(AIMessage(content=ai_msg.content))

        # 4) 输出到终端
        print(f"AI> {ai_msg.content}")

        if args.show_messages:
            print("\n[Debug] 模型回复后，上下文变成：")
            print_messages(messages)


if __name__ == "__main__":
    main()
