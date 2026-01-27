import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# =========================
# L02: Tool 一次调用闭环（MVP）
# =========================
# 本课新增：
# 1) @tool：声明工具
# 2) bind_tools：把工具注册给模型，否则通常不会出现 tool_calls
# 3) ToolMessage(tool_call_id=...)：把工具执行结果回填给模型（协议关键点）


SYSTEM_PROMPT = """你是一个低配版的 coding agent（对话层）。
规则：
- 能用工具就用工具，尤其是涉及计算时。
- 你会在需要时发起 tool_calls。
- 工具结果返回后，你要给出最终回答。
"""


@tool
def add(a: float, b: float) -> float:
    """
    [第一次出现] @tool 工具函数
    - 函数签名就是工具入参结构
    - 返回值会作为工具输出（我们会转成字符串回填给 ToolMessage）
    """
    return a + b


def build_llm() -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)

    # [第一次出现 / 易错点] bind_tools：不绑定工具，模型大概率不会产生 tool_calls
    llm = llm.bind_tools([add])
    return llm


def init_messages() -> list:
    return [SystemMessage(content=SYSTEM_PROMPT)]


def handle_command(user_input: str, messages: list) -> bool:
    text = user_input.strip()

    if text.lower() in {"exit", "quit"}:
        raise SystemExit(0)

    if text == ":reset":
        messages.clear()
        messages.extend(init_messages())
        print("AI> context cleared")
        return True

    if text == ":history":
        last = messages[-2:]
        last_types = [type(m).__name__ for m in last]
        print(f"len={len(messages)} last_types={last_types}")
        return True

    return False


def main():
    llm = build_llm()
    messages = init_messages()

    while True:
        user_text = input("User> ").strip()
        if not user_text:
            continue

        if handle_command(user_text, messages):
            continue

        # 1) 用户消息入历史
        messages.append(HumanMessage(content=user_text))

        # 2) 第一次调用：模型要么直接回答，要么请求调用工具
        resp = llm.invoke(messages)
        messages.append(resp)

        # [第一次出现] tool_calls：模型请求调用工具的结构化列表
        tool_calls = resp.tool_calls
        if not tool_calls:
            print("AI> " + (resp.content or ""))
            continue

        # 本课只做“一次闭环”：如果模型叫了工具，我们只执行第一条 tool_call
        call = tool_calls[0]
        name = call["name"]
        args = call["args"]
        call_id = call["id"]

        # [教学用最小可观测性] 打印 tool_calls 结构（方便你熟悉字段）
        print(f"[debug] tool_call name={name} args={args}")

        # 3) 执行工具（这里我们只有 add 一个工具，所以直接调用 add）
        # 注意：invoke 需要 dict 参数，args 正好是 dict
        result = add.invoke(args)

        # 4) 回填 ToolMessage（协议关键点：tool_call_id 必须与 call["id"] 对上）
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call_id,
            )
        )

        # 5) 第二次调用：模型看到工具结果后给最终回答
        final = llm.invoke(messages)
        messages.append(final)
        print("AI> " + (final.content or ""))


if __name__ == "__main__":
    main()
