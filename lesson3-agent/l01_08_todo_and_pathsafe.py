# -*- coding: utf-8 -*-
"""
标题：L01-08 无崩溃版本：路径绝对化 + Todo 工具（todoread/todowrite）——更像可用 Coding Agent
执行代码：
  pip install -U langchain langchain-openai
  # 根目录创建 .env：OPENAI_API_KEY=...  可选 OPENAI_BASE_URL=...
  python l01_08_todo_and_pathsafe.py --model gpt-5-nano --policy ask

本课目标：
1) 修复上一课的路径崩溃根因：所有路径统一以“绝对 WORKDIR”作为锚点，避免相对/绝对混用。
2) 新增 Todo 工具：
   - todowrite(items): 写入任务清单
   - todoread(): 读取任务清单
   让 Agent 先列清单、再推进，减少反复与跑偏。

工具（@tool）：
- list_files(dir=".", recursive=False, max_results=200)
- grep(pattern, path=".", max_results=50)
- read_file(path)
- apply_patch(file_path, new_content)   # 写入三态 allow/ask/deny
- bash(command)                         # 命令白名单
- todowrite(items)
- todoread()

说明：
- 不做“到处 try/except 兜底”。越界路径、命令不在白名单等会直接报错，利于学习边界。
- 仅保留 max_tool_hops 防止模型陷入无限工具循环（这是产品级必要护栏）。
"""

from __future__ import annotations

import argparse
import difflib
import json
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


# =========================
# 2) 全局运行时配置（由 main 初始化）
# =========================

WORKDIR: Path = Path(".").resolve()      # 绝对路径（关键：全程只用绝对）
WRITE_POLICY: str = "ask"
TODO_FILE: Path = Path(".").resolve() / ".agent_todo.json"

ALLOWED_CMD_PREFIXES = (
    "python -m pytest",
    "pytest",
    "python ",
    "ls",
    "cat ",
)


# =========================
# 3) 路径规范化（关键：不再混用相对/绝对）
# =========================

def normalize_rel_path(user_path: str) -> str:
    """
    将工具入参规范化为“相对 WORKDIR 的 posix 路径字符串”。

    规则：
    - 入参必须是相对路径（禁止绝对路径），防止越权访问。
    - 若模型误传前缀 "toy_repo/xxx"，会剥掉与 WORKDIR 末段同名的前缀。
    - 最终会做 resolve 并校验仍在 WORKDIR 内（防 .. 越界）。
    """
    s = (user_path or "").strip().replace("\\", "/")
    if not s:
        raise ValueError("path is empty")

    # 禁止绝对路径（Windows: C:/..., Unix: /...）
    if Path(s).is_absolute() or (len(s) >= 2 and s[1] == ":" and s[0].isalpha()):
        raise ValueError("absolute path is not allowed")

    # 剥离 "./"
    if s.startswith("./"):
        s = s[2:]

    # 剥离 workdir 末段前缀，例如 WORKDIR=.../toy_repo，模型传 toy_repo/calc.py
    wd_name = WORKDIR.name.replace("\\", "/")
    prefix = wd_name + "/"
    if s.startswith(prefix):
        s = s[len(prefix):]

    # resolve + 越界校验
    candidate = (WORKDIR / s).resolve()
    if not str(candidate).startswith(str(WORKDIR)):
        raise ValueError("path escapes workdir")

    return candidate.relative_to(WORKDIR).as_posix()


def abs_path(rel_norm: str) -> Path:
    """
    将已规范化的相对路径转换为绝对路径对象。
    这里不做额外兜底：normalize_rel_path 已保证安全。
    """
    return (WORKDIR / rel_norm).resolve()


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
# 4) 工具：全部用 @tool（LangChain 原生）
# =========================

@tool("list_files")
def list_files(dir: str = ".", recursive: bool = False, max_results: int = 200) -> str:
    """
    列出目录下文件（返回相对 WORKDIR 的路径，每行一个）。
    """
    rel = normalize_rel_path(dir)
    root = abs_path(rel)
    if not root.exists():
        raise FileNotFoundError(f"dir not found: {rel}")
    if not root.is_dir():
        raise ValueError(f"not a dir: {rel}")

    out: List[str] = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file():
                out.append(p.relative_to(WORKDIR).as_posix())
                if len(out) >= max_results:
                    break
    else:
        for p in root.iterdir():
            if p.is_file():
                out.append(p.relative_to(WORKDIR).as_posix())
                if len(out) >= max_results:
                    break

    return "\n".join(out)


@tool("grep")
def grep(pattern: str, path: str = ".", max_results: int = 50) -> str:
    """
    在工作区内搜索文本（简单包含匹配）。
    返回：<file>:<line_no>:<line_text>
    """
    if not pattern:
        raise ValueError("pattern is empty")

    rel = normalize_rel_path(path)
    target = abs_path(rel)
    if not target.exists():
        raise FileNotFoundError(f"path not found: {rel}")

    results: List[str] = []

    def scan_file(fp: Path) -> None:
        text = fp.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                results.append(f"{fp.relative_to(WORKDIR).as_posix()}:{i}:{line}")
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
    读取文件全文（path 为相对 WORKDIR）。
    """
    rel = normalize_rel_path(path)
    p = abs_path(rel)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {rel}")
    return p.read_text(encoding="utf-8", errors="replace")


@tool("apply_patch")
def apply_patch(file_path: str, new_content: str) -> str:
    """
    覆盖写入文件（简化版 patch）。写入受 WRITE_POLICY 控制：allow/ask/deny
    """
    rel = normalize_rel_path(file_path)
    p = abs_path(rel)

    old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    diff = unified_diff(old, new_content, rel)

    if WRITE_POLICY == "deny":
        return "[DENY] 当前策略禁止写入。以下为建议 diff（未落盘）：\n" + (diff if diff.strip() else "(无 diff)")

    if WRITE_POLICY == "ask":
        print("\n========== [WRITE PREVIEW / 需要确认] ==========")
        print(diff if diff.strip() else "(无 diff)")
        print("===============================================")
        ans = input("是否写入该补丁？(y/n) ").strip().lower()
        if ans not in ("y", "yes"):
            return "[ASK->NO] 你拒绝了写入。以下 diff 未落盘：\n" + (diff if diff.strip() else "(无 diff)")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new_content, encoding="utf-8")
    return "[OK] 已写入：{0}\nDiff:\n{1}".format(rel, diff if diff.strip() else "(无 diff)")


@tool("bash")
def bash(command: str) -> str:
    """
    在工作区执行命令（命令受白名单前缀限制）。
    """
    cmd = (command or "").strip()
    if not cmd:
        raise ValueError("command is empty")
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


@tool("todowrite")
def todowrite(items: List[str]) -> str:
    """
    写入任务清单（覆盖写）。
    """
    if not isinstance(items, list):
        raise ValueError("items must be a list of strings")
    data = {"items": items}
    TODO_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"[OK] todo saved: {len(items)} items"


@tool("todoread")
def todoread() -> str:
    """
    读取任务清单。
    """
    if not TODO_FILE.exists():
        return "(todo is empty)"
    return TODO_FILE.read_text(encoding="utf-8", errors="replace")


# =========================
# 5) Prompt：引导先列 todo，再探索、定位、修改、验证
# =========================

SYSTEM_PROMPT_TMPL = """\
你是一个终端里的 Coding Agent，遵循 ReAct（Agent->Tool->Agent）。

可用工具：
- todowrite(items) / todoread()
- list_files(dir=".", recursive=False)
- grep(pattern, path=".")
- read_file(path)
- apply_patch(file_path, new_content)  # 写入可能被拒绝/需要确认
- bash(command)                        # 命令受白名单限制

工作区（workdir）：{workdir}
写入策略（write_policy）：{policy}

关键规则（强制）：
1) 寒暄/闲聊（如“你好”）直接回复，不要调用工具。
2) 面对“需要多步完成”的需求，先 todowrite 写一份任务清单（3~7条），并在过程中用 todoread 对照推进。
3) 不要假设文件名：不知道文件在哪就用 list_files / grep 探索定位。
4) 需要内容才 read_file；禁止猜文件内容。
5) 修改文件：先 read_file，再 apply_patch，且 new_content 必须是“完整文件新内容”。
6) 需要验证时，优先 bash 跑：python -m pytest -q
7) 工具 path 参数必须是相对 workdir（例如 calc.py），禁止绝对路径。

输出风格：
- 教学型：解释你为何需要/不需要工具，你从工具结果里看到了什么，下一步计划是什么。
"""


# =========================
# 6) LLM + bind_tools + 多轮 ReAct
# =========================

def build_llm(model: str, temperature: float) -> ChatOpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key, base_url=base_url)


def get_tool_calls(ai_msg: Any) -> List[Dict[str, Any]]:
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        return ai_msg.tool_calls
    ak = getattr(ai_msg, "additional_kwargs", {}) or {}
    return ak.get("tool_calls", []) or []


def run_one_turn(llm_tools: ChatOpenAI, messages: List[BaseMessage], max_tool_hops: int = 14) -> str:
    tool_map = {
        "todowrite": todowrite,
        "todoread": todoread,
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

        messages.append(ai if isinstance(ai, BaseMessage) else AIMessage(content=str(ai)))

        if not tool_calls:
            return ai.content or ""

        hops += 1
        if hops > max_tool_hops:
            return "本轮工具调用次数过多，我已停止以避免循环。请你把目标拆小或加更明确约束。"

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            tool_fn = tool_map[name]  # 未知工具会 KeyError（用于学习与修 prompt）

            result = tool_fn.invoke(args)
            tool_call_id = call.get("id")
            messages.append(ToolMessage(content=result, tool_name=name, tool_call_id=tool_call_id))


# =========================
# 7) 主程序：持续对话
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="L01-08: pathsafe + todo tools")
    parser.add_argument("--model", default="gpt-5-nano", help="默认 gpt-5-nano")
    parser.add_argument("--temperature", type=float, default=0.0, help="建议 0")
    parser.add_argument("--policy", default="ask", choices=["allow", "ask", "deny"], help="写入策略")
    parser.add_argument("--workdir", default="toy_repo", help="工作区目录（默认 toy_repo）")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY：请在根目录 .env 或环境变量中设置。")

    global WORKDIR, WRITE_POLICY, TODO_FILE
    WORKDIR = Path(args.workdir).resolve()         # 关键：绝对路径锚点
    WRITE_POLICY = args.policy
    TODO_FILE = (WORKDIR / ".agent_todo.json").resolve()

    ensure_toy_repo(WORKDIR)

    llm = build_llm(args.model, args.temperature)
    llm_tools = llm.bind_tools([todowrite, todoread, list_files, grep, read_file, apply_patch, bash])

    system_prompt = SYSTEM_PROMPT_TMPL.format(workdir=WORKDIR.as_posix(), policy=WRITE_POLICY)
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    print("已启动 L01-08（路径绝对化 + Todo + 检索 + 修改 + 运行）。输入 exit/quit 退出。")
    print(f"workdir={WORKDIR.as_posix()}  write_policy={WRITE_POLICY}")

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
