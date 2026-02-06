import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# =========================
# L13: opencode 风格 skill 工具（渐进式披露）MVP
# =========================
# [NEW] 本课只新增：skill(name) 工具 + 工具描述里包含 <available_skills> 清单
# - 阶段 A：模型先看到可用技能清单（name + description）
# - 阶段 B：模型需要时调用 skill(name) 获取 SKILL.md 正文

PROJECT_ROOT = Path.cwd().resolve()

SYSTEM_PROMPT = """你是一个 coding agent（MVP）。
规则：
- 当你需要某种“固定流程/规范/工作方法”时，优先使用 skill 工具。
- 你必须在需要时调用 skill(name) 获取技能正文，不要凭空编造技能内容。
"""

# =========================
# [NEW] 约定：技能目录（MVP 只实现一种路径）
# .opencode/skills/<name>/SKILL.md
# =========================
SKILLS_ROOT = PROJECT_ROOT / ".opencode" / "skills"


def build_llm() -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


# =========================
# [NEW] 扫描 skills，并构造“可用技能清单”（阶段 A）
# - 只追求一个解法：只支持 YAML frontmatter 里一行一个键
# - 只解析 name/description（MVP）
# =========================
skills_index = []  # list[dict{name, description, path}]

if SKILLS_ROOT.exists():
    for skill_dir in sorted(SKILLS_ROOT.glob("*")):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue

        text = skill_md.read_text(encoding="utf-8")

        # YAML frontmatter：要求以 --- 开始，并在下一个 --- 结束（MVP）
        if not text.startswith("---"):
            continue

        # 取 frontmatter 段
        parts = text.split("---", 2)
        if len(parts) < 3:
            continue
        front = parts[1].strip()

        name = ""
        desc = ""
        for line in front.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k == "name":
                name = v
            if k == "description":
                desc = v

        if name and desc:
            skills_index.append(
                {"name": name, "description": desc, "path": str(skill_md)}
            )

# [NEW] 把 “可用技能清单”塞进 tool 的 description（opencode 渐进式披露关键）
available_skills_block = "<available_skills>\n"
for item in skills_index:
    available_skills_block += f"- {item['name']}: {item['description']}\n"
available_skills_block += "</available_skills>"


@tool(
    "skill",
    description=(
        "加载指定技能的 SKILL.md 正文，并返回给模型使用。\n\n"
        "你应该在需要某种固定流程/规范时调用此工具。\n\n" + available_skills_block
    ),
)
def skill(name: str, max_chars: int = 8000) -> str:
    """按 name 加载技能正文（来自 .opencode/skills/<name>/SKILL.md）。"""
    # 只追求一个解法：线性查找（MVP）
    target_path = None
    for item in skills_index:
        if item["name"] == name:
            target_path = item["path"]
            break
    if not target_path:
        return "NOT_FOUND"

    text = Path(target_path).read_text(encoding="utf-8")

    # 返回正文：去掉 frontmatter（MVP：用 split 找第三段）
    parts = text.split("---", 2)
    body = parts[2].strip() if len(parts) >= 3 else text.strip()
    return body[:max_chars]


def make_graph():
    llm = build_llm().bind_tools([skill])

    def chat(state: MessagesState):
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        resp = llm.invoke(msgs)
        return {"messages": [resp]}

    g = StateGraph(MessagesState)
    g.add_node("chat", chat)
    g.add_node("tools", ToolNode([skill]))

    g.add_edge(START, "chat")
    g.add_conditional_edges("chat", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "chat")

    return g.compile(checkpointer=InMemorySaver())


def main():
    graph = make_graph()

    # 单线程 MVP（你后面可以自己扩展 thread_id；本课先不引入）
    thread_id = "default"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_text = input("User> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        out = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]}, config=config
        )
        print("AI> " + (out["messages"][-1].content or ""))


if __name__ == "__main__":
    main()
