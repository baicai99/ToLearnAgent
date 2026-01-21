# -*- coding: utf-8 -*-
"""
标题：L02-07（稳定版）main.py —— review 不写入；修改需用户明确意图；工具参数缺失不崩溃
执行代码：
  python main.py --model gpt-5-nano --policy ask --workdir toy_repo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from tools import (
    init_tools,
    set_turn_context,
    repo_tree,
    todowrite,
    todoread,
    list_files,
    grep,
    read_file,
    read_file_range,
    apply_hunks,
    bash,
    submit,
)
from agent import run_one_turn


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


SYSTEM_PROMPT = """\
你是一个终端里的 Coding Agent，遵循 ReAct（Agent->Tool->Agent）。

工具：
- repo_tree(dir=".", recursive=True)：用于 review/列目录（只读）
- list_files/grep/read_file/read_file_range：只读工具
- apply_hunks(file_path, hunks)：局部修改工具（仅在用户明确提出“修改/修复/添加/实现”等需求时使用）
- bash：运行 pytest
- submit：完成闸门（tests 任务必须 pytest rc==0）

强制行为：
1) 用户说 review/查看/目录/结构：只用 repo_tree 或 list_files/read_file_range，不要修改任何文件。
2) 用户明确要求修改（修复/添加/实现/重构等）：先定位 -> 局部读取 -> apply_hunks 最小修改 -> pytest -> submit。
3) 工具返回 (error) 或 JSON status=ERROR/WRITE_NOT_REQUESTED/USER_REJECTED 时：不要重复同一动作，应询问用户或换方案。
"""


def build_llm(model: str, temperature: float) -> ChatOpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return ChatOpenAI(
        model=model, temperature=temperature, api_key=api_key, base_url=base_url
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--policy", default="ask", choices=["allow", "ask", "deny"])
    parser.add_argument("--workdir", default="toy_repo")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY：请在 .env 或环境变量中设置。")

    init_tools(Path(args.workdir), args.policy)

    llm = build_llm(args.model, args.temperature)
    llm_tools = llm.bind_tools(
        [
            repo_tree,
            todowrite,
            todoread,
            list_files,
            grep,
            read_file,
            read_file_range,
            apply_hunks,
            bash,
            submit,
        ]
    )

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    print(
        "已启动 L02-07（稳定版）：review 不写入；缺参/越界不崩溃；写入成功不误熔断。exit/quit 退出。"
    )
    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not user_text:
            continue

        # 关键：每轮设置“是否允许写”
        set_turn_context(user_text)

        messages.append(HumanMessage(content=user_text))
        reply = run_one_turn(llm_tools, messages)
        print("AI>", reply)


if __name__ == "__main__":
    main()
