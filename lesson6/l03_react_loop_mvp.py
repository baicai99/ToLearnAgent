import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# =========================
# L03: ReAct 内循环（MVP）
# =========================
# 关键变化：
# - L02：最多执行一轮工具 -> 再 ask 模型一次 -> 结束
# - L03：在同一轮用户输入中，反复：
#         invoke -> (tool_calls?) -> 执行工具并回填 ToolMessage -> invoke -> ...
#       直到 tool_calls 为空，输出最终回答


SYSTEM_PROMPT = """你是一个低配版的 coding agent（对话层）。
规则：
- 需要计算时必须调用工具（add/mul）。
- 工具结果回填后，再给出下一步或最终回答。
- 如果不需要工具，就直接回答。
"""


def build_llm() -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
    return llm


@tool
def add(a: float, b: float) -> float:
    """加法工具"""
    return a + b


@tool
def mul(a: float, b: float) -> float:
    """乘法工具"""
    return a * b


def main():
    llm = build_llm()
    llm = llm.bind_tools([add, mul])

    tools = {"add": add, "mul": mul}

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        user_text = input("User> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        messages.append(HumanMessage(content=user_text))

        # =========================
        # ReAct 内循环开始
        # =========================
        while True:
            resp = llm.invoke(messages)
            messages.append(resp)

            tool_calls = resp.tool_calls
            if not tool_calls:
                # 没有 tool_calls -> 这是“最终回答”或“本轮不需要工具”
                print("AI> " + (resp.content or ""))
                break

            # 教学用最小可观测性：让你看见每一轮模型要调用哪些工具
            print(f"[debug] tool_calls={tool_calls}")

            # 重要：本轮把模型返回的所有 tool_calls 都执行一遍
            for call in tool_calls:
                name = call["name"]
                args = call["args"]
                call_id = call["id"]

                # 不做复杂兜底：假设 name 一定在 tools 里（否则直接报错，利于学习）
                result = tools[name].invoke(args)

                # 工具执行也打印一下（最小 debug）
                print(f"[debug] tool_result name={name} args={args} result={result}")

                # [协议关键] ToolMessage 必须带 tool_call_id 对应上
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=call_id,
                    )
                )
            # 执行完所有 tool_calls 后，不 break
            # 继续 while True 再 invoke，让模型基于工具结果做下一步（可能继续要工具，也可能给最终回答）


if __name__ == "__main__":
    main()
