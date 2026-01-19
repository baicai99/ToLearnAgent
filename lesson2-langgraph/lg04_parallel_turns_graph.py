# -*- coding: utf-8 -*-
"""
标题：L01-04 修正版：持续对话 ReAct（按需工具调用）——@tool + bind_tools + 多轮循环
执行代码：
  pip install -U langchain langchain-openai
  # 根目录创建 .env：OPENAI_API_KEY=...  可选 OPENAI_BASE_URL=...
  python l01_04_react_chat_loop.py --model gpt-5-nano

说明：
- 工具用 @tool 定义，不手写解析器
- 不强迫调用工具：闲聊（如“你好”）应直接回复，不触发 function call
- 支持持续对话：每轮用户输入 -> (可能多次) Agent->Tool->Agent -> 输出 -> 下一轮
- 不做“到处 try/except 兜底”：参数缺失、越界路径等直接抛错，便于学习
"""

from __future__ import annotations

import argparse
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
# 1) 玩具仓库（先只读，后续课再写）
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


def safe_join(workdir: Path, rel_path: str) -> Path:
    p = (workdir / rel_path).resolve()
    wd = workdir.resolve()
    if not str(p).startswith(str(wd)):
        raise ValueError("path escapes workdir")
    return p


# =========================
# 2) 工具：@tool（LangChain 原生工具）
# =========================

WORKDIR = TOY_REPO


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


# =========================
# 3) System Prompt：工具按需调用（关键修复点）
# =========================

SYSTEM_PROMPT = f"""\
你是一个终端里的 Coding Agent，遵循 ReAct（Agent->Tool->Agent）。

你有一个可用工具：
- read_file(path): 读取工作区内文件内容

工作区（workdir）：{WORKDIR.as_posix()}

强制规则（非常重要）：
1) 只有当回答“必须依赖文件内容”时，才调用 read_file。
   - 如果用户只是寒暄、闲聊、泛问答（例如“你好”“你是谁”），直接自然语言回复，不要调用任何工具。
2) 如果你确实需要文件内容，先调用 read_file 获取，再基于工具返回内容回答。
3) 回答风格偏教学型：说明你做了什么决策（为何需要/不需要工具），并尽量清晰。

你可以读取的典型文件包括：
- tests/test_calc.py
- calc.py
但不要无缘无故去读；先判断是否必要。
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
    """
    LangChain 的 AIMessage 上通常有 tool_calls。
    这里不做 try/except 兜底：没有就返回空列表。
    """
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        return ai_msg.tool_calls
    ak = getattr(ai_msg, "additional_kwargs", {}) or {}
    return ak.get("tool_calls", []) or []


# =========================
# 5) 多轮 ReAct：每轮用户输入可触发多次 tool call
# =========================


def run_one_turn(
    llm_tools: ChatOpenAI, messages: List[BaseMessage], max_tool_hops: int = 6
) -> str:
    """
    单轮：用户输入已写入 messages。
    该轮内允许“多次工具跳转”（Agent->Tool->Agent->Tool...），直到没有 tool_calls 或达到 max_tool_hops。
    返回：最终对用户可见的文本回复。
    """
    tool_map = {"read_file": read_file}

    hops = 0
    while True:
        ai = llm_tools.invoke(messages)
        tool_calls = get_tool_calls(ai)

        # 把 AIMessage 记录进上下文（很重要：保留轨迹）
        messages.append(
            ai if isinstance(ai, BaseMessage) else AIMessage(content=str(ai))
        )

        # 1) 不需要工具：直接返回文本（闲聊正常走这里）
        if not tool_calls:
            return ai.content or ""

        # 2) 需要工具：执行工具并回灌 ToolMessage
        hops += 1
        if hops > max_tool_hops:
            # 这里不做复杂兜底策略，只做硬停止，防止死循环
            return (
                "本轮工具调用次数过多，我已停止以避免循环。你可以换一种更明确的指令。"
            )

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            tool_fn = tool_map[
                name
            ]  # 若模型请求未知工具，会 KeyError（便于学习与修 prompt）

            result = tool_fn.invoke(args)

            tool_call_id = call.get("id")  # 用于对齐工具返回
            messages.append(
                ToolMessage(content=result, tool_name=name, tool_call_id=tool_call_id)
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L01-04 fixed: multi-turn ReAct chat loop"
    )
    parser.add_argument("--model", default="gpt-5-nano", help="默认 gpt-5-nano")
    parser.add_argument("--temperature", type=float, default=0.0, help="建议 0")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY：请在根目录 .env 或环境变量中设置。")

    ensure_toy_repo(WORKDIR)

    llm = build_llm(args.model, args.temperature)
    llm_tools = llm.bind_tools([read_file])  # 工具可用，但不强迫调用

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    print("已启动持续对话 ReAct（工具按需调用）。输入 exit/quit 退出。")
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
