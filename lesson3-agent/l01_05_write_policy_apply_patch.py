# -*- coding: utf-8 -*-
"""
标题：L01-05 ReAct + 可控写入：@tool apply_patch + 三态权限（allow/ask/deny）
执行代码：
  pip install -U langchain langchain-openai
  # 根目录创建 .env：OPENAI_API_KEY=...  可选 OPENAI_BASE_URL=...
  python l01_05_write_policy_apply_patch.py --model gpt-5-nano --policy ask

本课目标：
- 延续上一课的“持续对话 ReAct”（工具按需调用，不强迫）
- 新增写入工具 apply_patch（仍然是 @tool）
- 写入权限三态：allow / ask / deny
- ask 模式下：打印 unified diff，终端 y/n 决定是否落盘

注意：
- 本课不引入 bash（跑命令）以控制复杂度；下一课再加
- 不做大量兜底：参数错误/越界路径等直接抛异常，便于你学习边界
"""

from __future__ import annotations

import argparse
import difflib
import os
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
# 1) 玩具仓库（本课开始允许写入）
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
                    # BUG: 这里故意写错，让测试失败（后续课会修）
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


def normalize_rel_path(workdir: Path, rel_path: str) -> str:
    rel = rel_path.replace("\\", "/").lstrip("/")
    wd = workdir.as_posix().strip("./")
    if wd and rel.startswith(wd + "/"):
        rel = rel[len(wd) + 1 :]
    return rel


def safe_join(workdir: Path, rel_path: str) -> Path:
    rel_path = normalize_rel_path(workdir, rel_path)
    p = (workdir / rel_path).resolve()
    wd = workdir.resolve()
    if not str(p).startswith(str(wd)):
        raise ValueError("path escapes workdir")
    return p


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
#    - read_file
#    - apply_patch（带三态权限）
# =========================

WORKDIR = TOY_REPO
WRITE_POLICY: str = "ask"  # 由 CLI 设置（allow/ask/deny）


@tool("read_file")
def read_file(path: str) -> str:
    """
    读取工作区内文件内容。
    path: 相对 WORKDIR 的路径，例如 "tests/test_calc.py"
    返回：文件全文
    """
    p = safe_join(WORKDIR, path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return p.read_text(encoding="utf-8", errors="replace")


@tool("apply_patch")
def apply_patch(file_path: str, new_content: str) -> str:
    """
    用“新内容”覆盖写入指定文件（简化版 patch）。
    注意：写入会受 WRITE_POLICY 控制（allow/ask/deny）。

    参数：
      file_path: 相对 WORKDIR 的路径，例如 "calc.py"
      new_content: 文件的新全文内容（字符串）

    返回：
      写入结果 + unified diff（或拒绝信息）
    """
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

    # allow 或 ask->yes：写入
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new_content, encoding="utf-8")
    return "[OK] 已写入：{0}\nDiff:\n{1}".format(
        file_path, diff if diff.strip() else "(无 diff)"
    )


# =========================
# 3) System Prompt：工具按需调用 + 教学型输出 + 写入规则
# =========================

SYSTEM_PROMPT_TMPL = """\
你是一个终端里的 Coding Agent，遵循 ReAct（Agent->Tool->Agent）。

你有两个工具：
- read_file(path): 读取工作区内文件内容
- apply_patch(file_path, new_content): 覆盖写入文件（写入可能被权限策略拒绝或需要确认）

工作区（workdir）：{workdir}
写入策略（write_policy）：{policy}

强制规则：
1) 对于寒暄/闲聊（例如“你好”），直接自然语言回复，不要调用工具。
2) 当你需要文件内容才能回答/修改时，必须先调用 read_file。
3) 当你要修改文件时：
   - 先 read_file 获取旧内容；
   - 再调用 apply_patch 写入 new_content（给出完整的新文件内容，不要只给片段）。
4) 回答要教学型：说明你为什么需要/不需要工具，你从工具返回里看到了什么，因此下一步是什么。

提示：toy_repo 内常见文件：
- tests/test_calc.py
- calc.py
"""


# =========================
# 4) 构建 LLM + 绑定工具
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
    llm_tools: ChatOpenAI, messages: List[BaseMessage], max_tool_hops: int = 8
) -> str:
    """
    单轮对话：允许本轮内多次 Agent->Tool->Agent 跳转。
    - 不做复杂兜底；仅用 max_tool_hops 防止死循环。
    """
    tool_map = {"read_file": read_file, "apply_patch": apply_patch}

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
            tool_fn = tool_map[name]  # 若未知工具会 KeyError（符合“从错误中学习”）

            result = tool_fn.invoke(args)

            tool_call_id = call.get("id")
            messages.append(
                ToolMessage(content=result, tool_name=name, tool_call_id=tool_call_id)
            )


# =========================
# 5) 主程序：持续对话窗口
# =========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L01-05: ReAct chat + apply_patch + write policy"
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
    llm_tools = llm.bind_tools([read_file, apply_patch])

    system_prompt = SYSTEM_PROMPT_TMPL.format(
        workdir=WORKDIR.as_posix(), policy=WRITE_POLICY
    )
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    print("已启动 L01-05（持续对话 ReAct + 可控写入）。输入 exit/quit 退出。")
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
