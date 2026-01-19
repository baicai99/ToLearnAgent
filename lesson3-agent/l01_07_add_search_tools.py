# -*- coding: utf-8 -*-
"""
标题：L01-07 增强检索：list_files + grep（让 Agent 能探索工作区）——ReAct 闭环更像真实 Coding Agent
执行代码：
  pip install -U langchain langchain-openai
  # 根目录创建 .env：OPENAI_API_KEY=...  可选 OPENAI_BASE_URL=...
  python l01_07_add_search_tools.py --model gpt-5-nano --policy ask

本课目标：
- 在 L01-06（read/patch/bash）基础上新增“检索”：
  1) list_files：列目录（可递归）
  2) grep：全文搜索
- 让 Agent 能自己发现文件，而不是你告诉它文件名
- 仍然：@tool + bind_tools；持续对话；不堆 try/兜底
"""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List

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

TOY_REPO = Path("toy_repo").resolve()


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
    - 工具参数约定是相对 workdir 的路径
    - 如果模型误传 "toy_repo/xxx"，剥掉前缀
    """
    s = rel_path.strip().replace("\\", "/")
    abs_workdir = workdir.resolve().as_posix()
    abs_prefix = abs_workdir.rstrip("/") + "/"
    if s == abs_workdir:
        return "."
    if s.startswith(abs_prefix):
        s = s[len(abs_prefix) :]
    name_prefix = workdir.name.rstrip("/") + "/"
    if s == workdir.name:
        return "."
    if s.startswith(name_prefix):
        s = s[len(name_prefix) :]
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

ALLOWED_CMD_PREFIXES = (
    "python -m pytest",
    "pytest",
    "python ",
    "ls",
    "cat ",
)


@tool("list_files")
def list_files(dir: str = ".", recursive: bool = False, max_results: int = 200) -> str:
    """
    列出目录下的文件。
    dir: 相对 workdir 的目录（默认 "."）
    recursive: 是否递归
    max_results: 最多返回多少条
    返回：每行一个路径（相对 workdir）
    """
    d = normalize_rel_path(dir, WORKDIR)
    root = safe_join(WORKDIR, d)
    if not root.exists():
        raise FileNotFoundError(f"dir not found: {d}")
    if not root.is_dir():
        raise ValueError(f"not a dir: {d}")

    paths: List[str] = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file():
                paths.append(p.relative_to(WORKDIR).as_posix())
                if len(paths) >= max_results:
                    break
    else:
        for p in root.iterdir():
            if p.is_file():
                paths.append(p.relative_to(WORKDIR).as_posix())
                if len(paths) >= max_results:
                    break

    return "\n".join(paths)


@tool("grep")
def grep(pattern: str, path: str = ".", max_results: int = 50) -> str:
    """
    在工作区内搜索文本。
    pattern: 要搜索的字符串/简单正则（这里按 Python in/包含匹配处理，避免引入复杂正则语义差异）
    path: 搜索范围（相对 workdir），可为目录或文件
    max_results: 最多返回多少条命中
    返回格式：<file>:<line_no>:<line_text>
    """
    base = normalize_rel_path(path, WORKDIR)
    target = safe_join(WORKDIR, base)
    if not target.exists():
        raise FileNotFoundError(f"path not found: {base}")

    results: List[str] = []

    def scan_file(fp: Path) -> None:
        nonlocal results
        text = fp.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                rel = fp.relative_to(WORKDIR).as_posix()
                results.append(f"{rel}:{i}:{line}")
                if len(results) >= max_results:
                    return

    if target.is_file():
        scan_file(target)
    else:
        for fp in target.rglob("*"):
            if fp.is_file():
                scan_file(fp)
                if len(results) >= max_results:
                    break

    return "\n".join(results) if results else "(no matches)"


@tool("read_file")
def read_file(path: str) -> str:
    """
    读取工作区内文件内容。
    path: 相对 workdir 的路径，例如 "calc.py"
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
    file_path: 相对 workdir 的路径
    new_content: 文件的新全文内容
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
# 3) Prompt：鼓励先探索再定位
# =========================

SYSTEM_PROMPT_TMPL = """\
你是一个终端里的 Coding Agent，遵循 ReAct（Agent->Tool->Agent）。

可用工具：
- list_files(dir=".", recursive=False)：列文件
- grep(pattern, path=".", max_results=50)：搜索文本
- read_file(path)：读取文件
- apply_patch(file_path, new_content)：写入文件（可能被权限策略拒绝/需要确认）
- bash(command)：运行命令（命令受白名单限制）

工作区（workdir）：{workdir}
写入策略（write_policy）：{policy}

关键约束（强制）：
1) 寒暄/闲聊（如“你好”）直接回复，不要调用工具。
2) 不要假设文件名：当你不知道目标文件在哪时，优先用 list_files / grep 进行探索与定位。
3) 需要文件内容再用 read_file；禁止猜内容。
4) 修改文件时：先 read_file 再 apply_patch，且 apply_patch 的 new_content 必须是“完整文件新内容”。
5) 当目标是“修复/验证能跑”时，修改后尽量 bash 跑一次：python -m pytest -q
6) 工具的路径参数必须是相对 workdir：例如 calc.py、tests/test_calc.py（不要写 toy_repo/xxx）。

输出风格：
- 教学型：说明你为何需要/不需要工具，你从工具结果里看到了什么，下一步计划是什么。
"""


# =========================
# 4) LLM + bind_tools + 多轮 ReAct
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
    llm_tools: ChatOpenAI, messages: List[BaseMessage], max_tool_hops: int = 12
) -> str:
    tool_map = {
        "list_files": list_files,
        "grep": grep,
        "read_file": read_file,
        "apply_patch": apply_patch,
        "bash": bash,
    }

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
            return "本轮工具调用次数过多，我已停止以避免循环。你可以把目标拆小或给更明确的约束。"

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            tool_fn = tool_map[name]  # 未知工具会 KeyError（用于学习与修 prompt）

            print(f"正在使用工具 tool: {name}")
            result = tool_fn.invoke(args)

            tool_call_id = call.get("id")
            messages.append(
                ToolMessage(content=result, tool_name=name, tool_call_id=tool_call_id)
            )


# =========================
# 5) 主程序：持续对话
# =========================


def main() -> None:
    parser = argparse.ArgumentParser(description="L01-07: add search tools")
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
    llm_tools = llm.bind_tools([list_files, grep, read_file, apply_patch, bash])

    system_prompt = SYSTEM_PROMPT_TMPL.format(
        workdir=WORKDIR.as_posix(), policy=WRITE_POLICY
    )
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    print("已启动 L01-07（持续对话 ReAct + 检索工具）。输入 exit/quit 退出。")
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
