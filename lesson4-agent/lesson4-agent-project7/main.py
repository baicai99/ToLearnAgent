# -*- coding: utf-8 -*-
"""
标题：L02-08B main.py —— 用 LangGraph 跑 ReAct（替代 while-loop）
执行代码：
  python main.py --model gpt-5-nano --policy ask --workdir toy_repo

说明：
  [L02-08B FIX] 输出改为打印 state_out["last_reply"]，而不是依赖最后一条 AIMessage.content。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

import core_runtime as cr
from tools import (
    set_turn_context,
    repo_tree, list_files, grep, read_file_range,
    write_file, apply_hunks,
    bash,
    evidence_read,
    todowrite, todoread,
    submit,
)

from graph_agent import build_graph


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

本课关键：用 LangGraph 表达 ReAct 循环：think -> tools -> think ... -> END

任务路由（review/create/tests/implement）：
- review：只读。写工具会返回 WRITE_NOT_ALLOWED
- create：创建新文件必须 write_file(file_path, content)
- tests：定位->修改->pytest通过->submit(task_type="tests")
- implement：定位->最小修改->必要时验证->submit(task_type="implement")

重要：
- 你可以在工具调用阶段不输出自然语言（AIMessage.content 可为空）。
- 用户可见的最终回答请写在 submit(final=...) 的 final 字段中。
"""


def build_llm(model: str, temperature: float) -> ChatOpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key, base_url=base_url)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--policy", default="ask", choices=["allow", "ask", "deny"])
    parser.add_argument("--workdir", default="toy_repo")
    parser.add_argument("--verbose", action="store_true", help="显示图节点执行过程（教学用）")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY：请在根目录 .env 或环境变量中设置。")

    cr.init_runtime(Path(args.workdir), args.policy)

    llm = build_llm(args.model, args.temperature)

    llm_tools = llm.bind_tools([
        repo_tree, list_files, grep, read_file_range,
        write_file, apply_hunks,
        bash,
        evidence_read,
        todowrite, todoread,
        submit,
    ])

    tool_map: Dict[str, Any] = {
        "repo_tree": repo_tree,
        "list_files": list_files,
        "grep": grep,
        "read_file_range": read_file_range,
        "write_file": write_file,
        "apply_hunks": apply_hunks,
        "bash": bash,
        "evidence_read": evidence_read,
        "todowrite": todowrite,
        "todoread": todoread,
        "submit": submit,
    }

    graph = build_graph(
        llm_tools=llm_tools,
        tool_map=tool_map,
        max_tool_hops=32,
        repeat_limit=6,
        verbose=True if args.verbose else False,
    )

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    print("已启动 L02-08B（LangGraph ReAct）。输入 exit/quit 退出。")

    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not user_text:
            continue

        tt = set_turn_context(user_text)
        print(f"[router] task_type={tt}")

        messages.append(HumanMessage(content=f"[task_type={tt}] {user_text}"))

        # [L02-08B FIX] 初始化 state 时提供 last_reply，图内会更新它（尤其是 submit.final）
        state_in = {
            "messages": messages,
            "hops": 0,
            "repeat": {},
            "stop": False,
            "stop_reason": "",
            "last_reply": "",
        }
        state_out = graph.invoke(state_in)

        messages = state_out["messages"]

        # [L02-08B FIX] 以 last_reply 作为用户可见输出（而不是 AIMessage.content）
        reply = (state_out.get("last_reply") or "").strip() or "(no reply)"
        print("AI>", reply)

        if state_out.get("stop_reason"):
            print(f"[graph] stop_reason={state_out['stop_reason']}")


if __name__ == "__main__":
    main()
