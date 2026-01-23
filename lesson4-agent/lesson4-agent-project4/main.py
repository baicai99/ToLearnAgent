# -*- coding: utf-8 -*-
"""
标题：L02-08 main.py —— TaskType 协议 + Evidence 模板 + write_file（稳定创建文件）
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
    evidence_read,
    repo_tree,
    todowrite,
    todoread,
    list_files,
    grep,
    read_file_range,
    write_file,
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

本课新增关键能力：任务类型协议 + 证据模板
- review：只读审阅，不得写文件；用 repo_tree/list_files/grep/read_file_range 获取证据；最后 submit(task_type="review")
- create：创建新文件必须用 write_file；不要用 apply_hunks 传补丁字符串；最后 submit(task_type="create")
- tests：修复测试必须 pytest 通过；最后 submit(task_type="tests")
- implement：实现/修改功能，优先 apply_hunks 做最小改动；最后 submit(task_type="implement")

工具：
- evidence_read：查看当前 evidence（必要时用来写 submit.evidence）
- repo_tree / list_files / grep / read_file_range：只读
- write_file：创建/覆盖文件（新建文件优先）
- apply_hunks：局部修改（hunks 必须是 list[dict] 结构；若创建文件请用 write_file）
- bash：仅允许 python/pytest 前缀
- submit：会按 task_type 模板做硬校验（不满足会 REJECT）

强制规则：
1) “review/查看目录/结构”只读，不写。
2) “创建文件”必须 write_file(file_path, content)。
3) “修复测试”必须 grep/read_range 定位 + apply_hunks 修改 + pytest 通过。
4) 结束必须 submit，并给出 evidence（建议先 evidence_read 查看再写）。
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
            evidence_read,
            repo_tree,
            todowrite,
            todoread,
            list_files,
            grep,
            read_file_range,
            write_file,
            apply_hunks,
            bash,
            submit,
        ]
    )

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    print(
        "已启动 L02-08：TaskType+Evidence+write_file（创建文件稳定）。exit/quit 退出。"
    )
    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not user_text:
            continue

        # 每轮设置“是否允许写”
        set_turn_context(user_text)

        messages.append(HumanMessage(content=user_text))
        reply = run_one_turn(llm_tools, messages)
        print("AI>", reply)


if __name__ == "__main__":
    main()
