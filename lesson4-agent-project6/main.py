# -*- coding: utf-8 -*-
"""
标题：L02-08B main.py —— 用 LangGraph 跑 ReAct（而不是 while-loop）
执行代码：
  python main.py --model gpt-5-nano --policy ask --workdir toy_repo
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
    repo_tree,
    list_files,
    grep,
    read_file_range,
    write_file,
    apply_hunks,
    bash,
    evidence_read,
    todowrite,
    todoread,
    submit,
)

from graph_agent import build_graph  # 【NEW-LANGGRAPH】图执行器


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
本课使用 LangGraph 来显式表达这个循环：think -> tools -> think ... -> END。

任务路由（review/create/tests/implement）仍然存在：
- review：只读（写工具会返回 WRITE_NOT_ALLOWED）
- create：创建文件必须 write_file(file_path, content)
- tests：定位->修改->pytest 通过->submit(task_type="tests")
- implement：定位->最小修改->需要时验证->submit(task_type="implement")

结束必须 submit，并给出 evidence（建议先 evidence_read 再写）。
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
        raise SystemExit("缺少 OPENAI_API_KEY：请在根目录 .env 或环境变量中设置。")

    # 初始化 runtime（与 L02-08A 相同）
    cr.init_runtime(Path(args.workdir), args.policy)

    llm = build_llm(args.model, args.temperature)

    # bind_tools（与 L02-08A 相同）
    llm_tools = llm.bind_tools(
        [
            repo_tree,
            list_files,
            grep,
            read_file_range,
            write_file,
            apply_hunks,
            bash,
            evidence_read,
            todowrite,
            todoread,
            submit,
        ]
    )

    # 【NEW-LANGGRAPH】工具映射（graph 的 tools 节点用它执行工具）
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

    # 【NEW-LANGGRAPH】构建并编译图（一次构建，多轮复用）
    graph = build_graph(
        llm_tools=llm_tools, tool_map=tool_map, max_tool_hops=32, repeat_limit=6
    )

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    print("已启动 L02-08B（LangGraph ReAct）。exit/quit 退出。")
    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not user_text:
            continue

        # 每轮任务路由（与 L02-08A 相同）
        tt = set_turn_context(user_text)
        messages.append(HumanMessage(content=f"[task_type={tt}] {user_text}"))

        # 【NEW-LANGGRAPH】调用图：输入 state，得到新 state（包含累积 messages）
        state_in = {
            "messages": messages,
            "hops": 0,
            "repeat": {},
            "stop": False,
            "stop_reason": "",
            "last_reply": "",
        }
        state_out = graph.invoke(state_in)

        # 更新全局 messages
        messages = state_out["messages"]

        # 输出本轮可见回复
        reply = state_out.get("last_reply") or "(no reply)"
        print("AI>", reply)


if __name__ == "__main__":
    main()
