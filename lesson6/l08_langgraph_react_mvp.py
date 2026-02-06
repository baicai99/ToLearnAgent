import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# =========================
# L08: LangGraph 版 ReAct（MVP / 单文件）
# =========================
# 本课只学会一件事：
# - 把 while 循环改成图：
#   START -> chat -> (tools_condition ? tools : END)
#                tools -> chat
#
# 你要牢牢记住的 LangGraph 三件套：
# 1) StateGraph(MessagesState)
# 2) add_node + add_edge + add_conditional_edges
# 3) compile() 得到 graph，然后用 graph.invoke({"messages": [...]})

PROJECT_ROOT = Path.cwd().resolve()

SYSTEM_PROMPT = """你是一个低配版 coding agent（Build）。
规则：
- 查文件/读内容/搜关键字：优先用只读工具。
- 修改文件：必须用写工具（write_file/patch_file），不要假装写入成功。
- 工具结果回填后再回答。
"""

# 读默认 allow；写默认 ask（本课 ask 实现放进写工具里）
PERMISSIONS = {
    "list_files": "allow",
    "read_file": "allow",
    "grep_file": "allow",
    "write_file": "ask",
    "patch_file": "ask",
}


def build_llm() -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


# =========================
# Tools（路径约束写在 tool 内部，低抽象）
# =========================


@tool
def list_files(pattern: str = "**/*", root: str = ".", max_items: int = 200) -> str:
    rp = Path(root)
    if rp.is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    root_path = (PROJECT_ROOT / rp).resolve()
    if PROJECT_ROOT not in root_path.parents and root_path != PROJECT_ROOT:
        raise ValueError("Path escapes project root.")

    items = []
    for p in root_path.glob(pattern):
        if p.is_file():
            items.append(str(p.relative_to(PROJECT_ROOT)))
    items.sort()
    return "\n".join(items[:max_items])


@tool
def read_file(path: str, max_chars: int = 4000) -> str:
    p = Path(path)
    if p.is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    file_path = (PROJECT_ROOT / p).resolve()
    if PROJECT_ROOT not in file_path.parents and file_path != PROJECT_ROOT:
        raise ValueError("Path escapes project root.")
    return file_path.read_text(encoding="utf-8")[:max_chars]


@tool
def grep_file(path: str, pattern: str, max_hits: int = 50) -> str:
    p = Path(path)
    if p.is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    file_path = (PROJECT_ROOT / p).resolve()
    if PROJECT_ROOT not in file_path.parents and file_path != PROJECT_ROOT:
        raise ValueError("Path escapes project root.")

    lines = file_path.read_text(encoding="utf-8").splitlines()
    regex = re.compile(pattern)

    hits = []
    for i, line in enumerate(lines, start=1):
        if regex.search(line):
            hits.append(f"{i}: {line}")
            if len(hits) >= max_hits:
                break
    return "\n".join(hits) if hits else "(no matches)"


@tool
def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    # 写权限 ask 放在 tool 内（为兼容 ToolNode，不额外造执行层抽象）
    if PERMISSIONS.get("write_file", "ask") == "deny":
        return "BLOCKED_BY_PERMISSION"

    if PERMISSIONS.get("write_file", "ask") == "ask":
        # 不泄露全文，只显示 path + chars（你之前的要求）
        ans = (
            input(
                f"Permission required: write_file(path={path}, chars={len(content)}) [y/n] "
            )
            .strip()
            .lower()
        )
        if ans not in {"y", "yes"}:
            return "BLOCKED_BY_PERMISSION"

    p = Path(path)
    if p.is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    file_path = (PROJECT_ROOT / p).resolve()
    if PROJECT_ROOT not in file_path.parents and file_path != PROJECT_ROOT:
        raise ValueError("Path escapes project root.")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding=encoding)
    return f"WROTE path={file_path.relative_to(PROJECT_ROOT)} chars={len(content)}"


@tool
def patch_file(path: str, find: str, replace: str, count: int = 1) -> str:
    if PERMISSIONS.get("patch_file", "ask") == "deny":
        return "BLOCKED_BY_PERMISSION"

    if PERMISSIONS.get("patch_file", "ask") == "ask":
        ans = (
            input(f"Permission required: patch_file(path={path}) [y/n] ")
            .strip()
            .lower()
        )
        if ans not in {"y", "yes"}:
            return "BLOCKED_BY_PERMISSION"

    p = Path(path)
    if p.is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    file_path = (PROJECT_ROOT / p).resolve()
    if PROJECT_ROOT not in file_path.parents and file_path != PROJECT_ROOT:
        raise ValueError("Path escapes project root.")

    text = file_path.read_text(encoding="utf-8")
    new_text = text.replace(find, replace, count)
    if new_text == text:
        return "NO_CHANGE (find not found)"

    file_path.write_text(new_text, encoding="utf-8")
    return f"PATCHED path={file_path.relative_to(PROJECT_ROOT)} count={count}"


def make_graph():
    llm = build_llm().bind_tools(
        [list_files, read_file, grep_file, write_file, patch_file]
    )

    # [低频点] LangGraph 节点函数：输入 state，返回 state 的增量更新（这里更新 messages）
    def chat(state: MessagesState):
        # state["messages"] 是一串 langchain 的 message 对象
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    tools_node = ToolNode([list_files, read_file, grep_file, write_file, patch_file])

    workflow = StateGraph(MessagesState)

    workflow.add_node("chat", chat)
    workflow.add_node("tools", tools_node)

    workflow.add_edge(START, "chat")

    # [低频点] tools_condition：检查 chat 的输出是否包含 tool_calls
    # - 有 tool_calls -> 路由到 "tools"
    # - 没有 -> END
    workflow.add_conditional_edges(
        "chat",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # 工具执行后回到 chat，形成 agent loop
    workflow.add_edge("tools", "chat")

    return workflow.compile()


def main():
    graph = make_graph()

    # 为了保持“单文件 + 你熟悉的输入循环”，我们外层仍用 input()。
    # 但注意：真正的 ReAct 循环已经在 graph 里了。
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        user_text = input("User> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break
        if user_text == ":reset":
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            print("AI> context cleared")
            continue
        if user_text == ":perm":
            print("AI> permissions:", PERMISSIONS)
            continue

        messages.append(HumanMessage(content=user_text))

        # 关键：把当前 messages 作为 state 传进图
        out = graph.invoke({"messages": messages})

        # 图执行完成后，拿回更新后的 messages
        messages = out["messages"]

        # 最后一个 message 就是最终 AI 回复（因为走到 END 时 chat 输出无 tool_calls）
        print("AI> " + (messages[-1].content or ""))


if __name__ == "__main__":
    main()
