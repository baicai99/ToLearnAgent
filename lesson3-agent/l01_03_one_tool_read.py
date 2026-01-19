# -*- coding: utf-8 -*-
"""
标题：L01-03 单工具 ReAct：read_file + ToolMessage 回灌（使用 LangChain Tool Calling｜去兜底版）
执行代码：
  pip install -U langchain langchain-openai pydantic
  # 根目录创建 .env（示例：OPENAI_API_KEY=...  可选 OPENAI_BASE_URL=...）
  python l01_03_one_tool_read.py --model gpt-5-nano --show-messages

本课目标：
- 不再要求模型输出“手写 JSON Action”
- 让模型通过 Tool Calling 机制直接发起 tool_calls
- 只真正执行 1 个工具：read_file
- 将工具返回作为 ToolMessage 写回 messages（这就是 ReAct 的 Observation）
- apply_patch / bash 仍暴露为工具，但返回“本课未启用”

教学约束（重要）：
- 本文件刻意“不做兼容回退、不吞异常”，让你从真实报错中学习
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import uuid
from pathlib import Path
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.tools import tool


# -------------------------
# 0) 读取根目录 .env（不依赖 python-dotenv）
# -------------------------

def load_dotenv(dotenv_path: Path) -> None:
    """
    极简 .env 解析器：
    - 支持 KEY=VALUE
    - 支持引号包裹
    - 忽略空行与 # 注释
    - 仅在环境变量未设置时写入 os.environ（避免覆盖）
    """
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and (k not in os.environ or os.environ[k] == ""):
            os.environ[k] = v


# -------------------------
# 1) 玩具仓库（本课只读）
# -------------------------

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
                    # BUG: 这里故意写错，让测试失败（后续课用）
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


# -------------------------
# 2) 工具：read_file（本课唯一执行工具） + 其它工具（禁用）
# -------------------------

WORKDIR: Path = TOY_REPO


def safe_join(workdir: Path, rel: str) -> Path:
    p = (workdir / rel).resolve()
    wd = workdir.resolve()
    if not str(p).startswith(str(wd)):
        raise ValueError("read_file: path escapes workdir")
    return p


def preview_text(s: str, max_lines: int = 40) -> str:
    lines = s.splitlines()
    return "\n".join(lines[:max_lines])


@tool("read_file")
def read_file(path: str) -> str:
    """
    读取工作区（workdir）内文件内容。path 必须是相对 workdir 的相对路径，例如：
    - tests/test_calc.py
    - calc.py
    """
    p = safe_join(WORKDIR, path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    return p.read_text(encoding="utf-8", errors="replace")


@tool("apply_patch")
def apply_patch(patch: str) -> str:
    """本课未启用：对代码打补丁。"""
    return "[ToolDisabled] apply_patch 本课未启用。请仅使用 read_file 获取信息，或直接给出结论。"


@tool("bash")
def bash(command: str) -> str:
    """本课未启用：执行 shell 命令。"""
    return "[ToolDisabled] bash 本课未启用。请仅使用 read_file 获取信息，或直接给出结论。"


TOOLS = [read_file, apply_patch, bash]
TOOL_MAP = {t.name: t for t in TOOLS}


# -------------------------
# 3) System Prompt：让模型走 Tool Calling（不再要求 JSON Action）
# -------------------------

SYSTEM_PROMPT = f"""\
你是一个终端 Coding Agent 的“决策器”，采用 ReAct（Agent->Tool->Agent）。
你可以调用工具来读取信息；当信息足够时，请直接输出最终结论（中文、教学口吻）。

可用工具：
- read_file(path): 读取工作区内文件内容（你需要文件内容就调用它）
- apply_patch(patch): 本课未启用（会返回 ToolDisabled）
- bash(command): 本课未启用（会返回 ToolDisabled）

你当前工作区（workdir）：{TOY_REPO.as_posix()}
建议你先阅读文件再做判断，例如：tests/test_calc.py、calc.py。

重要约束：
- 不要编造未通过工具读到的内容
- 需要引用证据时，基于 tool 返回的文件内容进行说明
"""


def print_messages(messages: List[BaseMessage], max_chars: int = 220) -> None:
    print("\n--- messages（上下文快照）---")
    for i, m in enumerate(messages):
        role = m.__class__.__name__.replace("Message", "")
        content = (m.content or "").replace("\n", "\\n")
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        print(f"[{i:02d}] {role}: {content}")
    print("----------------------------\n")


# -------------------------
# 4) 初始化 ChatOpenAI（最直接，不做兼容回退）
# -------------------------

def build_llm(model: str, temperature: float) -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()

    if not api_key:
        raise SystemExit("缺少 OPENAI_API_KEY。请在根目录 .env 或环境变量中设置。")

    if base_url:
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key, base_url=base_url)

    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def get_tool_calls(ai: AIMessage) -> List[Dict[str, Any]]:
    """
    只走“标准路径”：ai.tool_calls
    - 如果你的环境里 AIMessage 没有 tool_calls，说明你的 langchain 版本/绑定方式不对
    - 这里不做任何 fallback，直接抛错，利于你定位根因
    """
    tc = getattr(ai, "tool_calls", None)
    if tc is None:
        raise RuntimeError(
            "当前 AIMessage 没有 tool_calls 字段。"
            "请检查：1) 是否成功 llm.bind_tools(TOOLS)；2) langchain/langchain-openai 版本是否支持。"
        )

    out: List[Dict[str, Any]] = []
    for item in tc:
        # 预期 item 是 dict-like：{'id':..., 'name':..., 'args':...}
        out.append(
            {
                "id": item.get("id") or item.get("tool_call_id") or "",
                "name": item.get("name") or "",
                "args": item.get("args") or {},
            }
        )
    return [x for x in out if x["name"]]


# -------------------------
# 5) 主程序：模型 -> tool_calls -> 执行工具 -> ToolMessage 回灌
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="L01-03: one tool read_file + ToolMessage (LangChain Tool Calling) - no fallback"
    )
    parser.add_argument("--model", default="gpt-5-nano", help="模型名（默认 gpt-5-nano）")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度（建议 0）")
    parser.add_argument("--show-messages", action="store_true", help="每轮打印 messages 快照")
    parser.add_argument("--workdir", default=str(TOY_REPO), help="工作区目录（默认 toy_repo）")
    args = parser.parse_args()

    load_dotenv(Path(".env"))

    workdir = Path(args.workdir)
    ensure_toy_repo(workdir)

    global WORKDIR
    WORKDIR = workdir

    llm = build_llm(model=args.model, temperature=args.temperature)

    # 关键：强制绑定工具；失败就让它报错（不吞异常）
    llm = llm.bind_tools(TOOLS)

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    print("已启动 L01-03（去兜底版：不做兼容回退、不吞异常；本课只真正执行 read_file）。")
    print("输入普通文本：让模型决策/调用工具；输入 :next：不新增用户输入继续决策；输入 exit/quit 退出。")

    if args.show_messages:
        print_messages(messages)

    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break

        if user_text == ":next":
            pass
        else:
            if not user_text:
                continue
            messages.append(HumanMessage(content=user_text))

        if args.show_messages:
            print("\n[Debug] 发送给模型之前的上下文：")
            print_messages(messages)

        ai = llm.invoke(messages)  # 失败就直接报错
        messages.append(ai)

        tool_calls = get_tool_calls(ai)  # 不支持就直接报错

        print("\nAI >")
        if (ai.content or "").strip():
            print(ai.content)

        if tool_calls:
            print("\n[ToolCalls] >")
            print(json.dumps(tool_calls, ensure_ascii=False, indent=2))

            for call in tool_calls:
                name = call["name"]
                args_dict = call.get("args") or {}
                call_id = call.get("id") or f"call_{uuid.uuid4().hex[:8]}"

                tool_obj = TOOL_MAP[name]  # 不存在就 KeyError，直接暴露问题
                result = tool_obj.invoke(args_dict)  # 工具内部错误直接抛出

                if name == "read_file" and isinstance(result, str):
                    path = str(args_dict.get("path", "")).strip()
                    obs_preview = (
                        f"read_file OK: path={path}\n"
                        f"--- preview (first 40 lines) ---\n{preview_text(result, 40)}\n"
                        f"--- end preview ---\n"
                        f"(full content is provided to agent in ToolMessage)"
                    )
                    print("\n[Observation] >")
                    print(obs_preview)
                    messages.append(ToolMessage(content=result, tool_call_id=call_id))
                else:
                    print("\n[Observation] >")
                    print(result)
                    messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

            print("\n提示：现在你可以输入 :next，让模型基于这次 observation 再做下一步决策。")

        if args.show_messages:
            print("\n[Debug] 本轮结束后 messages：")
            print_messages(messages)


if __name__ == "__main__":
    main()
