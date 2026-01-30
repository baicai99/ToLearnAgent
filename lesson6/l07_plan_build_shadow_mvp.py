import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# =========================
# L07: Plan/Build 影子双代理（MVP / 低抽象 / 单文件）
# =========================
# 本课新增（你要记住的点很少）：
# 1) 两套历史：plan_history / build_history（各自有自己的 SystemMessage）
# 2) 两套权限：PLAN_PERMS（写 deny） vs BUILD_PERMS（写 ask）
#
# 你可以用命令切换：
# - :mode plan
# - :mode build
# - :mode auto   （默认：按简单规则自动判定）
#
# MVP 原则：
# - 不做复杂封装
# - 不引入 agents/pydantic/langgraph
# - 只保留必要“边界定义”：路径不能绝对、不能逃逸项目根目录

PROJECT_ROOT = Path.cwd().resolve()

PLAN_SYSTEM = """你是 Plan 代理（只读规划）。
规则：
- 你可以调用只读工具（list_files/read_file/grep_file）来获取事实。
- 你不允许修改任何文件：不要调用 write_file/patch_file。
- 输出必须包含：
  1) 目标（你理解的需求）
  2) 计划步骤（清晰、可执行）
  3) 需要用户确认的点（尤其是“要不要实际改文件”）
- 如果用户明确要求你直接改文件：请提示用户切换到 build（:mode build）。
"""

BUILD_SYSTEM = """你是 Build 代理（可执行）。
规则：
- 查文件/看内容/搜关键字：优先用只读工具。
- 修改文件：必须用写工具（write_file/patch_file），不要假装写入成功。
- 每次写工具执行前，会由宿主程序询问权限（ask），你只需要正常发起 tool_calls。
- 工具结果回填后再回答，最后总结你做了哪些改动（改了哪个文件、改了什么）。
"""

# 权限策略（更像 opencode 的味道）
# - Plan：读 allow，写 deny（强制只读）
PLAN_PERMS = {
    "list_files": "allow",
    "read_file": "allow",
    "grep_file": "allow",
    "write_file": "deny",
    "patch_file": "deny",
}

# - Build：读 allow，写 ask（每次写都问你）
BUILD_PERMS = {
    "list_files": "allow",
    "read_file": "allow",
    "grep_file": "allow",
    "write_file": "ask",
    "patch_file": "ask",
}

# 当前模式：auto / plan / build
MODE = "auto"


def build_llm() -> ChatOpenAI:
    # dotenv 用库（python-dotenv），不手写
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


# =========================
# Tools（只读 + 写入）
# 注意：路径约束写在 tool 内部（低抽象）
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


def auto_mode(user_text: str) -> str:
    """
    [第一次出现/低频点] 自动模式判定（MVP）
    - 这里只做“非常粗”的规则：你后面作业会改进它
    - 不追求智能，只追求可控 + 能被你理解并改写
    """
    t = user_text.lower()

    # 明确“要改文件/写代码”的意图 -> build
    build_signals = ["修改", "替换", "patch", "写入", "创建文件", "生成代码", "实现", "加功能", "重构", "修复bug", "修复", "改一下"]
    if any(s in t for s in build_signals):
        return "build"

    # 其它默认 plan（先规划/先问清楚）
    return "plan"


def main():
    global MODE

    llm = build_llm().bind_tools([list_files, read_file, grep_file, write_file, patch_file])

    tools = {
        "list_files": list_files,
        "read_file": read_file,
        "grep_file": grep_file,
        "write_file": write_file,
        "patch_file": patch_file,
    }

    # 两套历史：更像 opencode 的 Plan/Build 分离
    plan_history = [SystemMessage(content=PLAN_SYSTEM)]
    build_history = [SystemMessage(content=BUILD_SYSTEM)]

    while True:
        user_text = input("User> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        # ===== 命令（最小集）=====
        if user_text == ":reset":
            plan_history.clear()
            build_history.clear()
            plan_history.append(SystemMessage(content=PLAN_SYSTEM))
            build_history.append(SystemMessage(content=BUILD_SYSTEM))
            print("AI> context cleared (plan/build both)")
            continue

        if user_text == ":mode":
            print(f"AI> mode={MODE} (use :mode plan | :mode build | :mode auto)")
            continue

        if user_text.startswith(":mode "):
            # :mode plan/build/auto
            _, m = user_text.split(maxsplit=1)
            m = m.strip().lower()
            if m not in {"plan", "build", "auto"}:
                print("AI> invalid mode, use: :mode plan | :mode build | :mode auto")
                continue
            MODE = m
            print(f"AI> mode set to {MODE}")
            continue

        if user_text == ":perm":
            print("AI> PLAN_PERMS:", PLAN_PERMS)
            print("AI> BUILD_PERMS:", BUILD_PERMS)
            continue

        if user_text == ":root":
            print(f"AI> project_root={PROJECT_ROOT}")
            continue

        # ===== 选择本轮使用 plan 还是 build =====
        if MODE == "auto":
            use_mode = auto_mode(user_text)
        else:
            use_mode = MODE

        if use_mode == "plan":
            history = plan_history
            perms = PLAN_PERMS
            print("AI> (mode=plan)")
        else:
            history = build_history
            perms = BUILD_PERMS
            print("AI> (mode=build)")

        history.append(HumanMessage(content=user_text))

        # ===== ReAct 内循环（协议硬要求）=====
        while True:
            resp = llm.invoke(history)
            history.append(resp)

            if not resp.tool_calls:
                print("AI> " + (resp.content or ""))
                break

            for call in resp.tool_calls:
                name = call["name"]
                args = call["args"]
                call_id = call["id"]

                # ===== 权限 gating：写在执行处，不抽函数 =====
                mode = perms.get(name, "ask")

                if mode == "deny":
                    # 关键：拒绝也要回填 ToolMessage，让模型知道“被拒绝了”，从而调整策略
                    history.append(ToolMessage(content="BLOCKED_BY_PERMISSION", tool_call_id=call_id))
                    continue

                if mode == "ask":
                    # 写工具 ask：不要把 content 全文打印出来，只给摘要（path + chars）
                    path_preview = args.get("path")
                    content = args.get("content", "")
                    chars = len(content) if isinstance(content, str) else 0

                    if name == "write_file":
                        prompt = f"Permission required: write_file(path={path_preview}, chars={chars}) [y/n] "
                    elif name == "patch_file":
                        prompt = f"Permission required: patch_file(path={path_preview}) [y/n] "
                    else:
                        args_preview = ", ".join([f"{k}={v}" for k, v in args.items()])
                        prompt = f"Permission required: {name}({args_preview}) [y/n] "

                    ans = input(prompt).strip().lower()
                    if ans not in {"y", "yes"}:
                        history.append(ToolMessage(content="BLOCKED_BY_PERMISSION", tool_call_id=call_id))
                        continue

                result = tools[name].invoke(args)
                print(f"[tool] {name} -> {str(result)[:160]}")
                history.append(ToolMessage(content=str(result), tool_call_id=call_id))


if __name__ == "__main__":
    main()
