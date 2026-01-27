import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# =========================
# L05: 只读文件工具 + 权限三态（MVP）
# =========================
# 新增工具（只读）：
# - list_files(root=".", pattern="**/*")     列出文件
# - glob_files(pattern="**/*.py", root=".") 按 glob 找文件
# - read_file(path, max_chars=4000)         读文件（截断）
# - grep_file(path, pattern)                在单文件里 grep（带行号）
#
# 注意：
# - 仍然是 ReAct 内循环：invoke -> tool_calls -> 执行工具 -> ToolMessage 回填 -> 继续 invoke
# - 权限三态仍然适用：allow / ask / deny
# - 只读工具默认 allow（更符合 opencode 的常见策略：读通常放开，写才 ask/deny）


SYSTEM_PROMPT = """你是一个低配版的 coding agent（对话层）。
规则：
- 当用户问“项目里有什么文件/某文件内容/哪里出现某关键字”时，优先使用只读工具（list_files/glob_files/read_file/grep_file）。
- 工具结果回填后，再基于结果回答，不要凭空猜文件内容。
- 涉及计算可用 add/mul。
- 如果不需要工具就直接回答。
"""

# 约束：只允许访问该 root 下的相对路径（最小安全约束，不是兜底）
PROJECT_ROOT = Path.cwd().resolve()

# 权限表：工具名 -> allow/ask/deny
PERMISSIONS = {
    # 计算工具
    "add": "allow",
    "mul": "allow",
    # 只读文件工具（默认放开更友好）
    "list_files": "allow",
    "glob_files": "allow",
    "read_file": "allow",
    "grep_file": "allow",
}

DEFAULT_PERMISSION = "ask"


def build_llm() -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env / environment")

    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


def init_messages() -> list:
    return [SystemMessage(content=SYSTEM_PROMPT)]


def show_history_brief(messages: list) -> None:
    last = messages[-2:]
    last_types = [type(m).__name__ for m in last]
    print(f"len={len(messages)} last_types={last_types}")


def show_permissions() -> None:
    items = " ".join([f"{k}={v}" for k, v in sorted(PERMISSIONS.items())])
    print(f"AI> permissions: {items}")


def set_permission(tool_name: str, mode: str) -> None:
    if mode not in {"allow", "ask", "deny"}:
        raise ValueError("mode must be one of: allow / ask / deny")
    PERMISSIONS[tool_name] = mode
    print(f"AI> permission set: {tool_name}={mode}")


def check_permission(tool_name: str, args: dict) -> bool:
    mode = PERMISSIONS.get(tool_name, DEFAULT_PERMISSION)
    if mode == "allow":
        return True
    if mode == "deny":
        print(f"AI> blocked by permission: {tool_name}=deny")
        return False

    # ask
    # 为了像 opencode：把 args 变成更可读的 “tool(a=..., b=...)”
    args_preview = ", ".join([f"{k}={v}" for k, v in args.items()])
    answer = (
        input(f"Permission required: {tool_name}({args_preview}) [y/n] ")
        .strip()
        .lower()
    )
    return answer in {"y", "yes"}


def resolve_under_root(user_path: str) -> Path:
    """
    [第一次出现/低频点] 路径解析与最小约束：
    - 只允许相对路径
    - 解析后必须仍在 PROJECT_ROOT 内
    这不是“兜底”，是工具的安全边界定义（后面写文件工具更必须）。
    """
    p = Path(user_path)

    # 禁止绝对路径（避免用户/模型读到系统其他地方）
    if p.is_absolute():
        raise ValueError(
            "Absolute paths are not allowed. Use a relative path under project root."
        )

    resolved = (PROJECT_ROOT / p).resolve()
    if PROJECT_ROOT not in resolved.parents and resolved != PROJECT_ROOT:
        raise ValueError(
            "Path escapes project root. Use a relative path under project root."
        )
    return resolved


# ====== 计算工具（沿用）======


@tool
def add(a: float, b: float) -> float:
    return a + b


@tool
def mul(a: float, b: float) -> float:
    return a * b


# ====== 只读文件工具（新增）======


@tool
def list_files(root: str = ".", pattern: str = "**/*", max_items: int = 200) -> str:
    """
    列出文件（相对 PROJECT_ROOT 的某个子目录）
    - root: 起始目录（相对路径）
    - pattern: glob pattern（默认递归列出全部）
    - max_items: 最多返回多少条（避免刷屏；属于输出控制，不是兜底）
    """
    root_path = resolve_under_root(root)

    # 只列文件，不列目录
    items = []
    for p in root_path.glob(pattern):
        if p.is_file():
            rel = p.relative_to(PROJECT_ROOT)
            items.append(str(rel))

    items.sort()
    return "\n".join(items[:max_items])


@tool
def glob_files(pattern: str = "**/*.py", root: str = ".", max_items: int = 200) -> str:
    """
    按 pattern 找文件（等价于更专用的 list_files）
    """
    return list_files(root=root, pattern=pattern, max_items=max_items)


@tool
def read_file(path: str, max_chars: int = 4000) -> str:
    """
    读取文件内容（截断到 max_chars）
    - path: 相对路径
    """
    file_path = resolve_under_root(path)
    text = file_path.read_text(encoding="utf-8")
    return text[:max_chars]


@tool
def grep_file(path: str, pattern: str, max_hits: int = 50) -> str:
    """
    在单个文件中 grep（返回“行号:内容”）
    - pattern: 正则表达式（用 Python re）
    """
    file_path = resolve_under_root(path)
    text = file_path.read_text(encoding="utf-8").splitlines()

    regex = re.compile(pattern)
    hits = []
    for i, line in enumerate(text, start=1):
        if regex.search(line):
            hits.append(f"{i}: {line}")
            if len(hits) >= max_hits:
                break

    return "\n".join(hits) if hits else "(no matches)"


def handle_command(user_input: str, messages: list) -> bool:
    text = user_input.strip()

    if text.lower() in {"exit", "quit"}:
        raise SystemExit(0)

    if text == ":reset":
        messages.clear()
        messages.extend(init_messages())
        print("AI> context cleared")
        return True

    if text == ":history":
        show_history_brief(messages)
        return True

    if text == ":root":
        print(f"AI> project_root={PROJECT_ROOT}")
        return True

    if text == ":perm":
        show_permissions()
        return True

    # :perm <tool> <mode>
    if text.startswith(":perm "):
        parts = text.split()
        _, tool_name, mode = parts  # MVP：格式不对就直接抛错
        set_permission(tool_name, mode)
        return True

    return False


def main():
    llm = build_llm().bind_tools(
        [add, mul, list_files, glob_files, read_file, grep_file]
    )

    tools = {
        "add": add,
        "mul": mul,
        "list_files": list_files,
        "glob_files": glob_files,
        "read_file": read_file,
        "grep_file": grep_file,
    }

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

            # 逐个执行模型请求的工具
            for call in tool_calls:
                name = call["name"]
                args = call["args"]
                call_id = call["id"]

                if not check_permission(name, args):
                    messages.append(
                        ToolMessage(
                            content="BLOCKED_BY_PERMISSION",
                            tool_call_id=call_id,
                        )
                    )
                    continue

                result = tools[name].invoke(args)

                # 最小可观测性（固定格式）
                print(f"[tool] {name} args={args}")

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=call_id,
                    )
                )
            # 执行完 tool_calls 继续 invoke，让模型基于工具结果决定下一步


if __name__ == "__main__":
    main()
