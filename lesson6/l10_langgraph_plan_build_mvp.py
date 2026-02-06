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
# L10: LangGraph Plan/Build 双代理（MVP / 单文件 / 低抽象）
# =========================
# 本课新增点（只记 3 个）：
# 1) 两张图：plan_graph / build_graph（同样的结构，system 不同）
# 2) 权限差异：Plan(写 deny) vs Build(写 ask)
# 3) thread_id 隔离：thread_id 自动加后缀（:plan / :build）
#
# 命令：
# - :mode plan | build | auto
# - :thread <id>
# - :perm
# - :reset   （仅重置当前 mode 的线程历史：通过换 thread_id 实现）


PROJECT_ROOT = Path.cwd().resolve()

PLAN_SYSTEM = """你是 Plan 代理（只读规划）。
规则：
- 你可以调用只读工具（list_files/read_file/grep_file）来获取事实。
- 你不允许修改任何文件：不要调用 write_file/patch_file（会被拒绝）。
- 输出必须包含：目标、计划步骤、需要用户确认的点。
- 如果用户明确要直接改文件：提示切换到 build（:mode build）。
"""

BUILD_SYSTEM = """你是 Build 代理（可执行）。
规则：
- 查文件/看内容/搜关键字：优先用只读工具。
- 修改文件：必须用写工具（write_file/patch_file），不要假装写入成功。
- 写工具执行前会由工具内部询问权限（ask），你只要正常发起 tool_calls。
- 工具结果回填后再回答，并总结改动（改了哪个文件、做了什么）。
"""

# 权限：读 allow；写在 plan=deny / build=ask
PLAN_PERMS = {
    "list_files": "allow",
    "read_file": "allow",
    "grep_file": "allow",
    "write_file": "deny",
    "patch_file": "deny",
}

BUILD_PERMS = {
    "list_files": "allow",
    "read_file": "allow",
    "grep_file": "allow",
    "write_file": "ask",
    "patch_file": "ask",
}

# 当前运行模式（默认 auto）
MODE = "auto"
THREAD_ID = "default"

# 为了让工具内部拿到“当前模式的权限”，这里用一个全局指针
# ToolNode 调用工具时，会读取 CURRENT_PERMS。
CURRENT_PERMS = BUILD_PERMS  # 默认给个值，运行时会切换


def build_llm() -> ChatOpenAI:
    """dotenv + getenv 初始化模型（缺 key 硬失败）。"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


# =========================
# Tools（每个 @tool 必须有 docstring）
# 路径约束写在 tool 内部（低抽象）
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
    """写入/覆盖文件（写工具：plan=deny，build=ask）。"""
    mode = CURRENT_PERMS.get("write_file", "ask")
    if mode == "deny":
        return "BLOCKED_BY_PERMISSION"
    if mode == "ask":
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
    """在文件内做最小替换（默认替换 1 次）。"""
    mode = CURRENT_PERMS.get("patch_file", "ask")
    if mode == "deny":
        return "BLOCKED_BY_PERMISSION"
    if mode == "ask":
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


def make_graph(system_prompt: str):
    """构建一张 LangGraph：START -> chat -> (tools? tools : END); tools -> chat"""
    llm = build_llm().bind_tools(
        [list_files, read_file, grep_file, write_file, patch_file]
    )

    def chat(state: MessagesState):
        # MVP：system 不落盘，临时 prepend
        msgs = [SystemMessage(content=system_prompt)] + state["messages"]
        resp = llm.invoke(msgs)
        return {"messages": [resp]}

    tools_node = ToolNode([list_files, read_file, grep_file, write_file, patch_file])

    g = StateGraph(MessagesState)
    g.add_node("chat", chat)
    g.add_node("tools", tools_node)

    g.add_edge(START, "chat")
    g.add_conditional_edges("chat", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "chat")

    return g


def auto_route(user_text: str) -> str:
    """
    自动路由（MVP）：可控、可改。
    - 只要出现“改/写/修复/创建/替换”等信号，走 build
    - 否则走 plan
    """
    t = user_text.lower()
    signals = [
        "修改",
        "替换",
        "patch",
        "写入",
        "创建",
        "生成代码",
        "实现",
        "重构",
        "修复",
        "bug",
        "fix",
    ]
    return "build" if any(s in t for s in signals) else "plan"


def main():
    global MODE, THREAD_ID, CURRENT_PERMS

    # 同一个 checkpointer 让多个 thread_id 都能续聊（内存版）
    checkpointer = InMemorySaver()

    # 两张图共享同一个 checkpointer（重点：靠 thread_id 区分会话）
    plan_graph = make_graph(PLAN_SYSTEM).compile(checkpointer=checkpointer)
    build_graph = make_graph(BUILD_SYSTEM).compile(checkpointer=checkpointer)

    while True:
        user_text = input(f"User({MODE}/{THREAD_ID})> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        # ===== 命令（最小集）=====
        if user_text.startswith(":mode "):
            _, m = user_text.split(maxsplit=1)
            m = m.strip().lower()
            if m not in {"plan", "build", "auto"}:
                print("AI> invalid mode, use: :mode plan | :mode build | :mode auto")
                continue
            MODE = m
            print(f"AI> mode set to {MODE}")
            continue

        if user_text.startswith(":thread "):
            _, tid = user_text.split(maxsplit=1)
            THREAD_ID = tid.strip()
            print(f"AI> thread_id set to {THREAD_ID}")
            continue

        if user_text == ":perm":
            print("AI> PLAN_PERMS:", PLAN_PERMS)
            print("AI> BUILD_PERMS:", BUILD_PERMS)
            continue

        if user_text == ":reset":
            # MVP：通过“换一个 thread_id”实现 reset（不引入额外 API）
            THREAD_ID = f"{THREAD_ID}_reset"
            print(f"AI> thread reset, new thread_id={THREAD_ID}")
            continue

        # ===== 本轮选择 plan/build =====
        use = auto_route(user_text) if MODE == "auto" else MODE

        if use == "plan":
            CURRENT_PERMS = PLAN_PERMS
            graph = plan_graph
            # 关键：线程隔离（同一用户 thread 下，plan/build 分开存）
            tid = f"{THREAD_ID}:plan"
        else:
            CURRENT_PERMS = BUILD_PERMS
            graph = build_graph
            tid = f"{THREAD_ID}:build"

        config = {"configurable": {"thread_id": tid}}
        out = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]}, config=config
        )

        print("AI> " + (out["messages"][-1].content or ""))


if __name__ == "__main__":
    main()
