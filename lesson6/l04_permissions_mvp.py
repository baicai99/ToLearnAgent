import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# =========================
# L04: 权限三态（MVP）
# =========================
# 新增：
# - PERMISSIONS：每个工具一个模式：allow / ask / deny
# - check_permission：工具执行前统一过门
# - 命令：
#   - :perm            查看权限
#   - :perm add allow  设置权限
#   - :reset           清空上下文（保留 system）


SYSTEM_PROMPT = """你是一个低配版的 coding agent（对话层）。
规则：
- 遇到计算请优先调用工具（add/mul）。
- 工具结果回填后，再给出最终回答。
"""

# [第一次出现] 权限表：工具名 -> 模式
# - allow：直接执行
# - ask  ：执行前询问用户 y/n
# - deny ：拒绝执行
PERMISSIONS = {
    "add": "allow",
    "mul": "ask",
}


def build_llm() -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")

    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


@tool
def add(a: float, b: float) -> float:
    return a + b


@tool
def mul(a: float, b: float) -> float:
    return a * b


def init_messages() -> list:
    return [SystemMessage(content=SYSTEM_PROMPT)]


def show_permissions() -> None:
    items = " ".join([f"{k}={v}" for k, v in PERMISSIONS.items()])
    print(f"AI> permissions: {items}")


def set_permission(tool_name: str, mode: str) -> None:
    # MVP：不做复杂兜底；mode 只允许这三种，否则直接报错
    if mode not in {"allow", "ask", "deny"}:
        raise ValueError("mode must be one of: allow / ask / deny")
    PERMISSIONS[tool_name] = mode
    print(f"AI> permission set: {tool_name}={mode}")


def check_permission(tool_name: str, args: dict) -> bool:
    """
    工具执行“过门”：
    - allow -> True
    - deny  -> False
    - ask   -> 询问用户
    """
    mode = PERMISSIONS.get(tool_name, "ask")  # 默认 ask（更安全）
    if mode == "allow":
        return True
    if mode == "deny":
        print(f"AI> blocked by permission: {tool_name}=deny")
        return False

    # mode == "ask"
    # [第一次出现] 交互式权限询问：模拟 opencode 的 ask 行为
    answer = (
        input(f"Permission required: run {tool_name}{args}? (y/n) ").strip().lower()
    )
    return answer in {"y", "yes"}


def handle_command(user_input: str, messages: list) -> bool:
    text = user_input.strip()

    if text.lower() in {"exit", "quit"}:
        raise SystemExit(0)

    if text == ":reset":
        messages.clear()
        messages.extend(init_messages())
        print("AI> context cleared")
        return True

    if text == ":perm":
        show_permissions()
        return True

    # :perm <tool> <mode>
    if text.startswith(":perm "):
        parts = text.split()
        # 期望：3 段
        # MVP：不做复杂兜底，格式不对就抛错
        _, tool_name, mode = parts
        set_permission(tool_name, mode)
        return True

    return False


def main():
    llm = build_llm().bind_tools([add, mul])
    tools = {"add": add, "mul": mul}
    messages = init_messages()

    while True:
        user_text = input("User> ").strip()
        if not user_text:
            continue

        if handle_command(user_text, messages):
            continue

        messages.append(HumanMessage(content=user_text))

        # ===== ReAct 内循环 =====
        while True:
            resp = llm.invoke(messages)
            messages.append(resp)

            tool_calls = resp.tool_calls
            if not tool_calls:
                print("AI> " + (resp.content or ""))
                break

            for call in tool_calls:
                name = call["name"]
                args = call["args"]
                call_id = call["id"]

                # [L04 核心] 工具执行前先过权限门
                if not check_permission(name, args):
                    # 被拒绝：这里 MVP 处理方式是“把拒绝信息回填给模型”
                    # 这样模型能继续给出解释/替代方案，而不是死在这里。
                    messages.append(
                        ToolMessage(
                            content="BLOCKED_BY_PERMISSION",
                            tool_call_id=call_id,
                        )
                    )
                    continue

                result = tools[name].invoke(args)

                # 固定格式日志（你上一课的作业点）
                a = args.get("a")
                b = args.get("b")
                print(f"[tool] {name} a={a} b={b} -> {result}")

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=call_id,
                    )
                )
            # 执行完本轮 tool_calls，继续内循环，再 invoke 让模型决定下一步


if __name__ == "__main__":
    main()
