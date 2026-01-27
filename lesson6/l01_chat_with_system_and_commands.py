import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# =========================
# L01: 对话脚手架 + System + 命令路由（MVP）
# =========================
# 本课新增两点：
# 1) SystemMessage：把“角色/规则”固定在对话最前面（非常像 opencode 的全局规则）
# 2) handle_command：把 :reset/:history/exit 做成最小路由，减少 main 里 if-else 噪声


SYSTEM_PROMPT = """你是一个低配版的 coding agent（对话层）。
规则：
- 回答要直接、可执行、少废话。
- 当用户的请求不明确时，先问 1 个最关键的澄清问题。
- 你不会假装已经执行了任何工具/命令；你只能提出下一步要做什么。
- 如果用户只是闲聊，就正常闲聊；如果是编码/调试，就用工程化语言回答。
"""


def build_llm() -> ChatOpenAI:
    """构造 LLM（L01 仍不涉及工具，不 bind_tools）。"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")

    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


def init_messages() -> list:
    """
    [第一次出现] init_messages：初始化对话历史。
    关键点：SystemMessage 通常应该放在 messages 的最前面。
    之后每轮只 append HumanMessage/AIMessage/ToolMessage，不要重复塞 system。
    """
    return [SystemMessage(content=SYSTEM_PROMPT)]


def show_history_brief(messages: list) -> None:
    """
    [第一次出现] :history 的“最小可读输出”
    - 不打印内容（太长、太乱）
    - 只打印总长度 + 最后两条消息类型
    """
    last = messages[-2:]  # Python 切片天然容错：不足两条就返回更短
    last_types = [type(m).__name__ for m in last]
    print(f"len={len(messages)} last_types={last_types}")


def handle_command(user_text: str, messages: list) -> bool:
    """
    最小命令路由。
    返回值含义：
    - True  : 这次输入被当作“命令”处理了，主循环应 continue（不要调用模型）
    - False : 不是命令，主循环应把它当普通对话发送给模型
    """
    text = user_text.strip()

    # 退出（交互必需，不算兜底）
    if text.lower() in {"exit", "quit"}:
        raise SystemExit(0)

    if text == ":reset":
        messages.clear()
        # 关键：清空后要把 system prompt 重新放回去，否则模型就“无规则”
        messages.extend(init_messages())
        print("AI> context cleared")
        return True

    if text == ":history":
        show_history_brief(messages)
        return True

    return False


def main():
    llm = build_llm()
    messages = init_messages()

    while True:
        user_text = input("User> ").strip()
        if not user_text:
            continue

        # 命令优先处理
        if handle_command(user_text, messages):
            continue

        messages.append(HumanMessage(content=user_text))
        resp = llm.invoke(messages)
        print("AI> " + (resp.content or ""))
        messages.append(resp)


if __name__ == "__main__":
    main()
