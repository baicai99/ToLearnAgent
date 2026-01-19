# -*- coding: utf-8 -*-
"""
标题：L01-06 最小 Coding Agent 闭环：read_file + apply_patch + bash（带写入三态与命令白名单）
执行代码：
  pip install -U langchain langchain-openai
  # 根目录创建 .env：OPENAI_API_KEY=...  可选 OPENAI_BASE_URL=...
  python l01_06_add_bash_tool.py --model gpt-5-nano --policy ask

本课目标：
- 延续：持续对话 ReAct（工具按需调用，不强迫）
- 新增：bash 工具（命令白名单）
- 形成：读（read_file）→ 改（apply_patch）→ 跑（bash） 的最小闭环
- 保持：不堆 try/except（错误暴露便于学习）

注意：
- workdir 仍是 toy_repo；工具参数中的 path/file_path 约定为“相对 workdir”
- 为避免模型传 toy_repo/calc.py 造成重复前缀，本课做确定性归一化（不是 try 兜底）
"""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
    AIMessage,
)
from langchain_core.tools import tool


# =========================
# 0) 从根目录 .env 读取 key/base_url（极简实现）
# =========================


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and (k not in os.environ or os.environ[k] == ""):
            os.environ[k] = v


# =========================
# 1) 玩具仓库
# =========================

TOY_REPO = Path("toy_repo")


def ensure_toy_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "tests").mkdir(parents=True, exist_ok=True)

    calc_py = repo_dir / "calc.py"
    test_py = repo_dir / "tests" / "test_calc.py"

    if not calc_py.exists():
        calc_py.write_text(
            textwrap.dedent(
                """\
                # calc.py
                def add(a: int, b: int) -> int:
                    # BUG: 这里故意写错，让测试失败
                    return a - b
                """
            ),
            encoding="utf-8",
        )

    if not test_py.exists():
        test_py.write_text(
            textwrap.dedent(
                """\
                # tests/test_calc.py
                from calc import add

                def test_add_basic():
                    assert add(2, 3) == 5
                """
            ),
            encoding="utf-8",
        )


def safe_join(workdir: Path, rel_path: str) -> Path:
    p = (workdir / rel_path).resolve()
    wd = workdir.resolve()
    if not str(p).startswith(str(wd)):
        raise ValueError("path escapes workdir")
    return p


def normalize_rel_path(rel_path: str, workdir: Path) -> str:
    """
    确定性归一化（不是 try 兜底）：
    - 工具约定 path 是相对 workdir（toy_repo）的路径，例如 "calc.py" 或 "tests/test_calc.py"
    - 如果模型误传 "toy_repo/calc.py"，则剥掉前缀 "toy_repo/"
    """
    s = rel_path.strip().replace("\\", "/")
    prefix = workdir.as_posix().rstrip("/") + "/"
    if s.startswith(prefix):
        s = s[len(prefix) :]
    return s


def unified_diff(old: str, new: str, filename: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )
    return "".join(diff)


# =========================
# 2) 工具：@tool（LangChain 原生）
# =========================

WORKDIR = TOY_REPO
WRITE_POLICY: str = "ask"

# bash 白名单（你后面会逐步扩展，但新手期先窄）
ALLOWED_CMD_PREFIXES = (
    "python -m pytest",
    "pytest",
    "python ",
    "ls",
    "cat ",
)


@tool("read_file")
def read_file(path: str) -> str:
    """
    读取工作区内文件内容。
    path: 相对 WORKDIR 的路径，例如 "calc.py" 或 "tests/test_calc.py"
    返回：文件全文
    """
    path = normalize_rel_path(path, WORKDIR)
    p = safe_join(WORKDIR, path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return p.read_text(encoding="utf-8", errors="replace")


@tool("apply_patch")
def apply_patch(file_path: str, new_content: str) -> str:
    """
    覆盖写入文件（简化版 patch）。
    file_path: 相对 WORKDIR 的路径，例如 "calc.py"
    new_content: 文件的新全文内容

    写入受 WRITE_POLICY 控制：allow/ask/deny
    """
    file_path = normalize_rel_path(file_path, WORKDIR)
    p = safe_join(WORKDIR, file_path)

    old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    diff = unified_diff(old, new_content, file_path)

    if WRITE_POLICY == "deny":
        return "[DENY] 当前策略禁止写入。以下为建议 diff（未落盘）：\n" + (
            diff if diff.strip() else "(无 diff)"
        )

    if WRITE_POLICY == "ask":
        print("\n========== [WRITE PREVIEW / 需要确认] ==========")
        print(diff if diff.strip() else "(无 diff)")
        print("===============================================")
        ans = input("是否写入该补丁？(y/n) ").strip().lower()
        if ans not in ("y", "yes"):
            return "[ASK->NO] 你拒绝了写入。以下 diff 未落盘：\n" + (
                diff if diff.strip() else "(无 diff)"
            )

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new_content, encoding="utf-8")
    return "[OK] 已写入：{0}\nDiff:\n{1}".format(
        file_path, diff if diff.strip() else "(无 diff)"
    )


@tool("bash")
def bash(command: str) -> str:
    """
    在工作区执行命令（受白名单前缀限制）。
    command: 要执行的命令字符串，例如 "python -m pytest -q"
    返回：returncode + stdout/stderr
    """
    cmd = command.strip()
    if not any(cmd.startswith(pfx) for pfx in ALLOWED_CMD_PREFIXES):
        raise ValueError(f"command not allowed: {cmd!r}")

    proc = subprocess.run(
        cmd,
        cwd=str(WORKDIR),
        shell=True,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    return (
        f"returncode={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
    )


# =========================
# 3) System Prompt：工具按需调用 + 教学型输出 + 鼓励闭环
# =========================

SYSTEM_PROMPT_TMPL = """\
你是一个终端里的 Coding Agent，遵循 ReAct（Agent->Tool->Agent）。

可用工具：
- read_file(path)：读取工作区内文件内容
- apply_patch(file_path, new_content)：覆盖写入文件（可能被权限策略拒绝/需要确认）
- bash(command)：在工作区运行命令（命令受白名单限制）

工作区（workdir）：{workdir}
写入策略（write_policy）：{policy}

关键约束（强制）：
1) 对寒暄/闲聊（例如“你好”）直接回复，不要调用任何工具。
2) 当你需要文件内容才能判断/修改时，必须先调用 read_file；禁止猜文件内容。
3) 修改文件时：先 read_file 获取旧内容，再 apply_patch 写入完整 new_content。
4) 当用户要“修复/验证/能跑起来”时，尽量在修改后调用 bash 跑一次测试：
   - 推荐：python -m pytest -q
5) 工具参数 path/file_path 必须是“相对 workdir”的路径：
   - 正确：calc.py、tests/test_calc.py
   - 不要写：toy_repo/calc.py（无需带 workdir 前缀）

输出风格：
- 教学型：说明你为何需要/不需要工具，你从工具返回里看到了什么，以及下一步计划。
"""


# =========================
# 4) LLM + bind_tools
# =========================


def build_llm(model: str, temperature: float) -> ChatOpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return ChatOpenAI(
        model=model, temperature=temperature, api_key=api_key, base_url=base_url
    )


def get_tool_calls(ai_msg: Any) -> List[Dict[str, Any]]:
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        return ai_msg.tool_calls
    ak = getattr(ai_msg, "additional_kwargs", {}) or {}
    return ak.get("tool_calls", []) or []


def run_one_turn(
    llm_tools: ChatOpenAI, messages: List[BaseMessage], max_tool_hops: int = 10
) -> str:
    tool_map = {"read_file": read_file, "apply_patch": apply_patch, "bash": bash}

    hops = 0
    while True:
        ai = llm_tools.invoke(messages)
        tool_calls = get_tool_calls(ai)

        messages.append(
            ai if isinstance(ai, BaseMessage) else AIMessage(content=str(ai))
        )

        if not tool_calls:
            return ai.content or ""

        hops += 1
        if hops > max_tool_hops:
            return "本轮工具调用次数过多，我已停止以避免循环。你可以把目标拆小或给更明确的指令。"

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            tool_fn = tool_map[name]  # 未知工具会 KeyError（用于学习/修 prompt）

            result = tool_fn.invoke(args)

            tool_call_id = call.get("id")
            messages.append(
                ToolMessage(content=result, tool_name=name, tool_call_id=tool_call_id)
            )


# =========================
# 5) 主程序：持续对话
# =========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L01-06: add bash tool to close the loop"
    )
    parser.add_argument("--model", default="gpt-5-nano", help="默认 gpt-5-nano")
    parser.add_argument("--temperature", type=float, default=0.0, help="建议 0")
    parser.add_argument(
        "--policy", default="ask", choices=["allow", "ask", "deny"], help="写入策略"
    )
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY：请在根目录 .env 或环境变量中设置。")

    global WRITE_POLICY
    WRITE_POLICY = args.policy

    ensure_toy_repo(WORKDIR)

    llm = build_llm(args.model, args.temperature)
    llm_tools = llm.bind_tools([read_file, apply_patch, bash])

    system_prompt = SYSTEM_PROMPT_TMPL.format(
        workdir=WORKDIR.as_posix(), policy=WRITE_POLICY
    )
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    print("已启动 L01-06（持续对话 ReAct + 可控写入 + bash）。输入 exit/quit 退出。")
    print(f"当前写入策略：{WRITE_POLICY}")

    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not user_text:
            continue

        messages.append(HumanMessage(content=user_text))
        reply = run_one_turn(llm_tools, messages)
        print("AI>", reply)


if __name__ == "__main__":
    main()
