import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# =========================
# L00: 最小对话脚手架（MVP）
# =========================
# 你要记住的“骨架”只有三件事：
# 1) load_dotenv + getenv 读取配置
# 2) messages: 用 list 维护历史
# 3) while 循环：HumanMessage -> llm.invoke(messages) -> append(resp)


def build_llm() -> ChatOpenAI:
    """
    [第一次出现] build_llm：把“配置读取”和“模型初始化”固定成模板函数。
    以后加 tool、加 graph，都不动这块或只做小改动。
    """
    # [第一次出现] load_dotenv：用库从 .env 加载环境变量（不要手写解析）
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")  # 可为空
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    # [最低限度硬失败] 没 key 就直接报错：让你明确环境必须正确
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")

    # [第一次出现] ChatOpenAI：LangChain 的 OpenAI 兼容模型封装
    # 你目前只要知道：invoke(messages) -> 返回一个 AIMessage（resp）
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


def main():
    llm = build_llm()

    # [第一次出现] messages：对话历史（非常关键）
    # 规则（背下来）：
    # - 用户每说一句：messages.append(HumanMessage(...))
    # - 模型每回一句：messages.append(resp)  # resp 本身就是 AIMessage
    messages = []

    while True:
        user_text = input("User> ").strip()
        if not user_text:
            continue

        # [交互必需] 最小退出命令，不算“兜底”
        if user_text.lower() in {"exit", "quit"}:
            break

        # [第一次出现] HumanMessage：把用户文本包装成“标准消息对象”
        messages.append(HumanMessage(content=user_text))

        # [关键] 把整个历史喂给模型，模型才能在多轮里“接住上下文”
        resp = llm.invoke(messages)

        # [输出] resp.content 是模型的自然语言回复
        print("AI> " + (resp.content or ""))

        # [易错点] 直接 append(resp)
        # 不要写成 AIMessage(content=resp) 或 messages.append(resp.content)
        # 否则历史结构会坏掉，后面引入 tool/react 会更难 debug
        messages.append(resp)


if __name__ == "__main__":
    main()
