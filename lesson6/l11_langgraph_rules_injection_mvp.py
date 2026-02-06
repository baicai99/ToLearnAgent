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
# L12: Skills 按需注入（MVP / 单文件 / 低抽象）
# =========================
# 新增点（本课只记这些）：
# 1) skills/ 目录：技能以 Markdown 文件存在
# 2) 两个工具：
#    - list_skills() -> 列出可用技能名
#    - load_skill(name) -> 读取技能内容，并写入“当前 thread 的技能缓存”
# 3) chat 节点每次调用模型前：注入 Rules + 已加载 Skills（prepend 到 prompt）
#
# 规则优先级：System(Build/Plan) > Rules(AGENTS/CLAUDE) > Skills(loaded) > history messages

PROJECT_ROOT = Path.cwd().resolve()
SKILLS_DIR = PROJECT_ROOT / "skills"

PLAN_SYSTEM = """你是 Plan 代理（只读规划）。
规则：
- 你可以调用只读工具获取事实。
- 你不允许修改任何文件：不要调用 write_file/patch_file（会被拒绝）。
- 输出必须包含：目标、计划步骤、需要用户确认的点。
- 如果你需要某项“工作方法/规范/流程”，优先尝试加载技能（list_skills/load_skill）。
"""

BUILD_SYSTEM = """你是 Build 代理（可执行）。
规则：
- 查文件/看内容/搜关键字：优先用只读工具。
- 修改文件：必须用写工具（write_file/patch_file），不要假装写入成功。
- 工具结果回填后再回答，并总结改动。
- 如果你需要某项“工作方法/规范/流程”，优先尝试加载技能（list_skills/load_skill）。
"""

PLAN_PERMS = {
    "list_files": "allow",
    "read_file": "allow",
    "grep_file": "allow",
    "write_file": "deny",
    "patch_file": "deny",
    # skills 工具默认 allow（它只是读本地 md）
    "list_skills": "allow",
    "load_skill": "allow",
}

BUILD_PERMS = {
    "list_files": "allow",
    "read_file": "allow",
    "grep_file": "allow",
    "write_file": "ask",
    "patch_file": "ask",
    "list_skills": "allow",
    "load_skill": "allow",
}

MODE = "auto"
THREAD_ID = "default"

# 重要：ToolNode 调用工具时，我们用全局变量给工具提供“当前权限/当前 thread”
CURRENT_PERMS = BUILD_PERMS
CURRENT_TID = "default:build"

# 本课核心：thread 级技能缓存（不进 messages 历史）
# key = tid（包含 :plan/:build 后缀）
# value = dict{name -> content}
SKILL_CACHE: dict[str, dict[str, str]] = {}


def build_llm() -> ChatOpenAI:
    """dotenv + getenv 初始化模型（缺 key 直接硬失败）。"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


def _find_first_upwards(start_dir: Path, filenames: list[str]) -> Path | None:
    """向上遍历目录找规则文件（MVP）。"""
    cur = start_dir.resolve()
    while True:
        for name in filenames:
            p = cur / name
            if p.is_file():
                return p
        if cur.parent == cur:
            return None
        cur = cur.parent


def load_rules_text(cwd: Path) -> tuple[str, list[str]]:
    """
    读取规则文本（MVP 子集）：
    1) 本地就近：向上找 AGENTS.md（优先）否则 CLAUDE.md
    2) 全局：~/.config/opencode/AGENTS.md
    3) 全局：~/.claude/CLAUDE.md
    """
    parts: list[str] = []
    sources: list[str] = []

    local = _find_first_upwards(cwd, ["AGENTS.md", "CLAUDE.md"])
    if local:
        parts.append(local.read_text(encoding="utf-8"))
        sources.append(f"local:{local}")

    global_agents = Path.home() / ".config" / "opencode" / "AGENTS.md"
    if global_agents.is_file():
        parts.append(global_agents.read_text(encoding="utf-8"))
        sources.append(f"global-opencode:{global_agents}")

    global_claude = Path.home() / ".claude" / "CLAUDE.md"
    if global_claude.is_file():
        parts.append(global_claude.read_text(encoding="utf-8"))
        sources.append(f"global-claude:{global_claude}")

    text = "\n\n".join([p.strip() for p in parts if p.strip()])
    if not text:
        return "", []
    wrapped = "以下是项目/个人规则（必须遵守）：\n\n" + text
    return wrapped, sources


# =========================
# Tools：文件只读 / 写入 / 规则 / 技能
# 注意：每个 @tool 必须有 docstring
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
    """写入/覆盖文件（plan=deny，build=ask）。"""
    mode = CURRENT_PERMS.get("write_file", "ask")
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
    mode = CURRENT_PERMS.get("patch_file", "ask")
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


@tool
def list_skills() -> str:
    """列出 skills/ 目录下可用技能名（不含扩展名）。"""
    if not SKILLS_DIR.exists():
        return "(no skills dir)"

    items = []
    for p in SKILLS_DIR.glob("*.md"):
        if p.is_file():
            items.append(p.stem)
    items.sort()
    return "\n".join(items) if items else "(no skills)"


@tool
def load_skill(name: str, max_chars: int = 8000) -> str:
    """
    加载一个技能，并写入“当前 thread 的技能缓存”。
    返回：LOADED / NOT_FOUND / TOO_LARGE 的结果字符串（可验证）。
    """
    # 权限（默认 allow）：这里保留入口以便你后续做 deny/ask
    mode = CURRENT_PERMS.get("load_skill", "allow")
    if mode == "deny":
        return "BLOCKED_BY_PERMISSION"
    if mode == "ask":
        ans = input(f"Permission required: load_skill(name={name}) [y/n] ").strip().lower()
        if ans not in {"y", "yes"}:
            return "BLOCKED_BY_PERMISSION"

    path = SKILLS_DIR / f"{name}.md"
    if not path.is_file():
        return "NOT_FOUND"

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return "EMPTY"

    # MVP：防止一次塞太多（不是兜底，是输出控制）
    text = text[:max_chars]

    # 写入当前 thread 的 cache（本课关键 side-effect）
    bucket = SKILL_CACHE.setdefault(CURRENT_TID, {})
    bucket[name] = text
    return f"LOADED name={name} chars={len(text)}"


def _render_loaded_skills(tid: str) -> str:
    """把当前 thread 已加载技能拼成一个可注入的文本块。"""
    bucket = SKILL_CACHE.get(tid, {})
    if not bucket:
        return ""
    # 固定格式，避免模型把它当普通聊天
    lines = ["以下是已加载技能（按需加载，必须遵守其约束与流程）："]
    for name, content in bucket.items():
        lines.append(f"\n## SKILL: {name}\n{content}")
    return "\n".join(lines)


def make_graph(system_prompt: str):
    """LangGraph：START -> chat -> (tools? tools : END); tools -> chat"""
    llm = build_llm().bind_tools([
        list_files, read_file, grep_file, write_file, patch_file,
        list_skills, load_skill
    ])

    def chat(state: MessagesState):
        # 1) Rules（每轮读取，MVP；后续作业做缓存）
        rules_text, _ = load_rules_text(Path.cwd())

        # 2) Skills（从 thread cache 注入）
        skills_text = _render_loaded_skills(CURRENT_TID)

        msgs = [SystemMessage(content=system_prompt)]
        if rules_text:
            msgs.append(SystemMessage(content=rules_text))
        if skills_text:
            msgs.append(SystemMessage(content=skills_text))
        msgs += state["messages"]

        resp = llm.invoke(msgs)
        return {"messages": [resp]}

    tools_node = ToolNode([
        list_files, read_file, grep_file, write_file, patch_file,
        list_skills, load_skill
    ])

    g = StateGraph(MessagesState)
    g.add_node("chat", chat)
    g.add_node("tools", tools_node)
    g.add_edge(START, "chat")
    g.add_conditional_edges("chat", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "chat")
    return g


def auto_route(user_text: str) -> str:
    """自动路由（MVP）：出现“改/写/修复/创建/替换”等信号 -> build，否则 plan。"""
    t = user_text.lower()
    signals = ["修改", "替换", "patch", "写入", "创建", "生成代码", "实现", "重构", "修复", "bug", "fix"]
    return "build" if any(s in t for s in signals) else "plan"


def main():
    global MODE, THREAD_ID, CURRENT_PERMS, CURRENT_TID

    checkpointer = InMemorySaver()
    plan_graph = make_graph(PLAN_SYSTEM).compile(checkpointer=checkpointer)
    build_graph = make_graph(BUILD_SYSTEM).compile(checkpointer=checkpointer)

    while True:
        user_text = input(f"User({MODE}/{THREAD_ID})> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        # --- commands（最小集）---
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

        if user_text == ":skills":
            # 手动查看可用技能 + 当前 thread 已加载技能
            print("AI> available skills:")
            print(list_skills())
            print("\nAI> loaded skills in current thread:")
            print("\n".join(SKILL_CACHE.get(CURRENT_TID, {}).keys()) or "(none)")
            continue

        if user_text.startswith(":load "):
            # 手动加载技能（便于你学习/测试，不依赖模型决定）
            _, name = user_text.split(maxsplit=1)
            print("AI> " + load_skill(name.strip()))
            continue

        if user_text == ":reset":
            THREAD_ID = f"{THREAD_ID}_reset"
            print(f"AI> thread reset, new thread_id={THREAD_ID}")
            continue

        # --- route ---
        use = auto_route(user_text) if MODE == "auto" else MODE
        if use == "plan":
            CURRENT_PERMS = PLAN_PERMS
            graph = plan_graph
            CURRENT_TID = f"{THREAD_ID}:plan"
        else:
            CURRENT_PERMS = BUILD_PERMS
            graph = build_graph
            CURRENT_TID = f"{THREAD_ID}:build"

        config = {"configurable": {"thread_id": CURRENT_TID}}
        out = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)
        print("AI> " + (out["messages"][-1].content or ""))


if __name__ == "__main__":
    main()
