# -*- coding: utf-8 -*-
"""
标题：L02-06 main.py —— 入口：加载 .env、初始化工具、启动对话；完成必须走 submit（证据驱动）
执行代码：
  pip install -U "langchain>=0.2" "langchain-openai>=0.1" "langchain-core>=0.2"
  # 当前目录 .env：OPENAI_API_KEY=...  可选 OPENAI_BASE_URL=...
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
    todowrite,
    todoread,
    list_files,
    grep,
    read_file,
    apply_patch,
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

可用工具：
- todowrite(items) / todoread()
- list_files(dir=".", recursive=False)
- grep(pattern, path=".")
- read_file(path)
- apply_patch(file_path, new_content)  # 写入可能被拒绝/需要确认
- bash(command)                        # 可用于运行 pytest
- submit(final, evidence, task_type)   # 完成闸门：tests 任务必须 pytest 通过才会 ACCEPT

关键规则（强制）：
1) 多步任务先 todowrite（3~7 条），执行过程中用 todoread 对照推进。
2) 不要假设文件名：不知道就先 list_files/grep 定位。
3) 需要内容才 read_file；禁止猜。
4) 修改文件：先 read_file，再 apply_patch，且 new_content 必须是完整文件新内容。
5) 如果用户目标是“修复测试/让 pytest 通过/验证通过”等：
   - task_type 必须设为 "tests"
   - 在 submit 前必须 bash 运行 python -m pytest -q 且 returncode==0
6) 你不能用纯文本宣布“完成”。想结束必须调用 submit。
7) evidence 必须可验证、可复核，至少包含：
   - pytest 结果（tests 任务）
   - 修改的文件与变更摘要（diff/changed_files）
   - 关键定位证据（grep/文件路径等）
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

    # bind_tools：不手写解析器
    llm_tools = llm.bind_tools(
        [
            todowrite,
            todoread,
            list_files,
            grep,
            read_file,
            apply_patch,
            bash,
            submit,
        ]
    )

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    print(
        "已启动 L02-06（证据驱动 Done：submit 闸门 + tests 必须 pytest 通过）。输入 exit/quit 退出。"
    )
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
