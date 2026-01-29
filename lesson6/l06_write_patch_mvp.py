import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# =========================
# L06（低抽象版）：写入工具 + 权限 ask（MVP）
# =========================
# 你本课要记住的“新增骨架”只有两件事：
# 1) 新增写工具：write_file / patch_file（都在 tool 内部做路径约束）
# 2) 在执行工具前做权限 gating（写工具默认 ask）
#
# 约定：
# - 单文件
# - 不引入 pydantic / agents
# - 不做复杂兜底（不写一堆 try/except）
# - 路径安全：禁止绝对路径；禁止 ../ 逃逸项目根目录（这是边界定义，不算兜底）


SYSTEM_PROMPT = """你是一个低配版的 coding agent（对话层）。
规则：
- 查文件/读内容/搜关键字：优先用只读工具（list_files/read_file/grep_file）。
- 修改文件：必须用写工具（write_file/patch_file），不要假装写入成功。
- 工具结果回填后再回答，不要凭空猜文件内容。
"""

# 项目根目录：所有文件工具都只能访问该目录下的相对路径
PROJECT_ROOT = Path.cwd().resolve()

# 权限三态（本课只用到 allow/ask；deny 也支持）
# 读：默认 allow；写：默认 ask（更像 opencode）
PERMISSIONS = {
    "list_files": "allow",
    "read_file": "allow",
    "grep_file": "allow",
    "write_file": "ask",
    "patch_file": "ask",
}


def build_llm() -> ChatOpenAI:
    """
    [第一次出现/低频点] build_llm
    - dotenv 读取 .env（库优先）
    - getenv 取出 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
    - 缺 key 直接硬失败（不兜底）
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")

    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


# =========================
# 只读工具（最小集）
# =========================

@tool
def list_files(pattern: str = "**/*", root: str = ".", max_items: int = 200) -> str:
    """
    列出文件（只列文件，不列目录）
    参数：
    - root: 起始目录（相对路径）
    - pattern: glob pattern
    - max_items: 最多输出多少条（防刷屏）
    """
    # 路径约束写在 tool 内（不抽函数）
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
    """
    读取文件（截断）
    """
    p = Path(path)
    if p.is_absolute():
        raise ValueError("Absolute paths are not allowed.")

    file_path = (PROJECT_ROOT / p).resolve()
    if PROJECT_ROOT not in file_path.parents and file_path != PROJECT_ROOT:
        raise ValueError("Path escapes project root.")

    return file_path.read_text(encoding="utf-8")[:max_chars]


@tool
def grep_file(path: str, pattern: str, max_hits: int = 50) -> str:
    """
    单文件 grep（返回 行号:内容）
    """
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


# =========================
# 写工具（本课新增）
# =========================

@tool
def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """
    写入/覆盖文件（写工具）
    关键点：
    - 写入前也要做路径约束（禁止绝对路径/逃逸）
    - 父目录不存在就创建（mkdir parents=True）
    返回：
    - 写入路径（相对 PROJECT_ROOT）+ 写入字符数（便于可验证）
    """
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
    """
    最小 patch（默认只替换 1 次）
    - find: 要找的子串
    - replace: 替换成的子串
    - count: 替换次数（默认 1）
    返回：
    - NO_CHANGE 或 PATCHED + 次数（可验证）
    """
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


def main():
    llm = build_llm().bind_tools([list_files, read_file, grep_file, write_file, patch_file])

    # 工具路由表：tool_calls 里给的是 name，需要映射到对应 tool
    tools = {
        "list_files": list_files,
        "read_file": read_file,
        "grep_file": grep_file,
        "write_file": write_file,
        "patch_file": patch_file,
    }

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        user_text = input("User> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        # 命令：最小集（不搞复杂命令系统）
        if user_text == ":perm":
            print("AI> permissions:", PERMISSIONS)
            continue
        if user_text == ":reset":
            messages.clear()
            messages.append(SystemMessage(content=SYSTEM_PROMPT))
            print("AI> context cleared")
            continue

        messages.append(HumanMessage(content=user_text))

        # ===== ReAct 内循环（协议硬要求：不能塞进 tool）=====
        while True:
            resp = llm.invoke(messages)
            messages.append(resp)

            if not resp.tool_calls:
                print("AI> " + (resp.content or ""))
                break

            for call in resp.tool_calls:
                name = call["name"]
                args = call["args"]
                call_id = call["id"]

                # ===== 权限 gating（写在执行处，不抽函数）=====
                mode = PERMISSIONS.get(name, "ask")

                if mode == "deny":
                    messages.append(ToolMessage(content="BLOCKED_BY_PERMISSION", tool_call_id=call_id))
                    continue

                if mode == "ask":
                    # 作业点：写工具询问不要泄露 content 全文，只显示关键摘要
                    # 这里做最小摘要：path + chars（如果有 content）
                    path_preview = args.get("path")
                    content = args.get("content", "")
                    chars = len(content) if isinstance(content, str) else 0

                    # 如果不是写工具（比如你把某读工具也设成 ask），用通用预览
                    if name == "write_file":
                        prompt = f"Permission required: write_file(path={path_preview}, chars={chars}) [y/n] "
                    elif name == "patch_file":
                        prompt = f"Permission required: patch_file(path={path_preview}) [y/n] "
                    else:
                        args_preview = ", ".join([f"{k}={v}" for k, v in args.items()])
                        prompt = f"Permission required: {name}({args_preview}) [y/n] "

                    ans = input(prompt).strip().lower()
                    if ans not in {"y", "yes"}:
                        messages.append(ToolMessage(content="BLOCKED_BY_PERMISSION", tool_call_id=call_id))
                        continue

                # 执行工具
                result = tools[name].invoke(args)

                # 最小可观测性：固定格式输出（不要刷屏）
                print(f"[tool] {name} -> {str(result)[:160]}")

                # 回填 ToolMessage（协议硬要求：tool_call_id 必须对应）
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id))


if __name__ == "__main__":
    main()
