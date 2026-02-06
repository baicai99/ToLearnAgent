import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# =========================
# L09: LangGraph 线程记忆（MVP / 单文件 / 低抽象）
# =========================
# 本课目标：
# 1) 用 checkpointer + thread_id 实现“同一 thread 自动续聊”
# 2) 外层不再手动维护 messages（每次只传入本轮 HumanMessage）
# 3) 固化规范：每个 @tool 必须有 docstring（否则可能 ValueError）
#
# 约定：
# - 读工具默认 allow
# - 写工具默认 ask（权限询问放在写工具内部，兼容 ToolNode）

PROJECT_ROOT = Path.cwd().resolve()

SYSTEM_PROMPT = """你是一个低配版 coding agent（Build）。
规则：
- 查文件/读内容/搜关键字：优先用只读工具。
- 修改文件：必须用写工具（write_file/patch_file），不要假装写入成功。
- 工具结果回填后再回答。
"""

PERMISSIONS = {
    "list_files": "allow",
    "read_file": "allow",
    "grep_file": "allow",
    "write_file": "ask",
    "patch_file": "ask",
}


def build_llm() -> ChatOpenAI:
    """初始化模型：dotenv + getenv（缺 key 直接硬失败）。"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


# =========================
# Tools（注意：每个 tool 必须有 docstring）
# =========================

@tool
def list_files(pattern: str = "**/*", root: str = ".", max_items: int = 200) -> str:
    """列出项目内文件（只列文件，不列目录）。"""
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
    """读取文件内容（最多返回 max_chars 字符）。"""
    p = Path(path)
    if p.is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    file_path = (PROJECT_ROOT / p).resolve()
    if PROJECT_ROOT not in file_path.parents and file_path != PROJECT_ROOT:
        raise ValueError("Path escapes project root.")
    return file_path.read_text(encoding="utf-8")[:max_chars]


@tool
def grep_file(path: str, pattern: str, max_hits: int = 50) -> str:
    """在单个文件内用正则搜索，返回“行号:内容”。"""
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
    """写入/覆盖文件（写工具，默认 ask 权限）。"""
    # 权限询问放进 tool 内：为了直接复用 ToolNode，不在外层造抽象
    mode = PERMISSIONS.get("write_file", "ask")
    if mode == "deny":
        return "BLOCKED_BY_PERMISSION"
    if mode == "ask":
        ans = input(f"Permission required: write_file(path={path}, chars={len(content)}) [y/n] ").strip().lower()
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
    """在文件内做最小替换（默认替换 1 次）。"""
    mode = PERMISSIONS.get("patch_file", "ask")
    if mode == "deny":
        return "BLOCKED_BY_PERMISSION"
    if mode == "ask":
        ans = input(f"Permission required: patch_file(path={path}) [y/n] ").strip().lower()
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
    """构建 LangGraph：START -> chat -> (tools? tools : END); tools -> chat"""
    llm = build_llm().bind_tools([list_files, read_file, grep_file, write_file, patch_file])

    def chat(state: MessagesState):
        # 关键点：SystemMessage 不落盘到 state；每次调用时临时 prepend
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        resp = llm.invoke(msgs)
        return {"messages": [resp]}

    tools_node = ToolNode([list_files, read_file, grep_file, write_file, patch_file])

    g = StateGraph(MessagesState)
    g.add_node("chat", chat)
    g.add_node("tools", tools_node)

    g.add_edge(START, "chat")
    g.add_conditional_edges("chat", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "chat")

    # 本课新增：checkpointer
    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)


def main():
    graph = make_graph()

    # thread_id：同一个 thread_id 会自动续聊；换 thread_id 就是新会话
    thread_id = "default"

    while True:
        user_text = input(f"User({thread_id})> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        # 最小命令：只做线程切换（本课核心）
        if user_text.startswith(":thread "):
            # :thread abc
            _, tid = user_text.split(maxsplit=1)
            thread_id = tid.strip()
            print(f"AI> thread_id set to {thread_id}")
            continue

        if user_text == ":perm":
            print("AI> permissions:", PERMISSIONS)
            continue

        # 关键：不再维护 messages 变量
        # 只把“本轮新消息”交给图；历史由 checkpointer 根据 thread_id 自动加载/追加
        config = {"configurable": {"thread_id": thread_id}}
        out = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)

        # out["messages"] 是完整历史（从 checkpointer 加载 + 本轮更新）
        print("AI> " + (out["messages"][-1].content or ""))


if __name__ == "__main__":
    main()
