# -*- coding: utf-8 -*-
"""
标题：L02-08D main.py —— 线程级对话记忆（checkpointer）+ Tool Profile（不再手动维护 messages）
执行代码：
  python main.py --model gpt-5-nano --policy ask --workdir toy_repo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import core_runtime as cr
from tools import (
    init_tools,
    set_turn_context,
    repo_tree,
    list_files,
    grep,
    read_file_range,
    write_file,
    apply_hunks,
    commit_patch,
    reject_patch,
    bash,
    evidence_read,
    todowrite,
    todoread,
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


# [L02-08D NEW] 更清晰的“只在需要时用工具”的提示：配合 tool_profile 生效
SYSTEM_PROMPT = """\
你是一个终端里的 Coding Agent，遵循 ReAct（Agent->Tool->Agent）。

关键规则：
1) 工具不是必用：能直接回答就直接用自然语言回答（尤其是聊天/问候）。
2) review 仅使用只读工具（repo_tree/list_files/grep/read_file_range），不要尝试写入。
3) create/tests/implement 才能写入。ask 策略下写入会先 PREVIEW，等待用户审批。
4) 只有你认为任务已完成时才调用 submit(final=..., evidence=[...], task_type=...)。否则不要 submit。

输出要求：
- 回复要教学型：说明你为何选择（或不选择）工具，以及下一步做什么。
"""


def build_llm(model: str, temperature: float) -> ChatOpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return ChatOpenAI(
        model=model, temperature=temperature, api_key=api_key, base_url=base_url
    )


def pick_tool_profile(user_text: str, task_type: str) -> str:
    """
    [L02-08D NEW] Tool Profile 策略：
    - chat：完全禁用工具
    - review_ro：只读工具
    - full：读写+pytest+submit
    """
    s = (user_text or "").strip().lower()
    greet = ["你好", "hi", "hello", "在吗", "谢谢", "早上好", "晚上好"]
    if len(s) <= 12 and any(w in s for w in greet):
        return "chat"
    if task_type in ("review",):
        return "review_ro"
    if task_type in ("create", "tests", "implement"):
        return "full"
    return "review_ro"


def _get_interrupt_payload(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    intr = result.get("__interrupt__")
    if not intr:
        return None
    first = intr[0]
    val = getattr(first, "value", None)
    if isinstance(val, dict):
        return val
    if isinstance(first, dict) and isinstance(first.get("value"), dict):
        return first["value"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--policy", default="ask", choices=["allow", "ask", "deny"])
    parser.add_argument("--workdir", default="toy_repo")
    parser.add_argument(
        "--thread", default="demo-thread-1", help="线程ID：用于 checkpointer 持久化状态"
    )
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY：请在根目录 .env 或环境变量中设置。")

    init_tools(args.workdir, args.policy)

    llm = build_llm(args.model, args.temperature)

    # 三套 LLM：chat（无工具）、review_ro（只读工具）、full（全工具）
    llm_plain = llm  # 不绑定工具
    llm_review_ro = llm.bind_tools(
        [repo_tree, list_files, grep, read_file_range, evidence_read, todoread]
    )
    llm_full = llm.bind_tools(
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

    tool_map: Dict[str, Any] = {
        "repo_tree": repo_tree,
        "list_files": list_files,
        "grep": grep,
        "read_file_range": read_file_range,
        "write_file": write_file,
        "apply_hunks": apply_hunks,
        "commit_patch": commit_patch,
        "reject_patch": reject_patch,
        "bash": bash,
        "evidence_read": evidence_read,
        "todowrite": todowrite,
        "todoread": todoread,
        "submit": submit,
    }

    memory = InMemorySaver()
    graph = build_graph(
        llm_plain=llm_plain,
        llm_review_ro=llm_review_ro,
        llm_full=llm_full,
        tool_map=tool_map,
        checkpointer=memory,
    )

    config = {
        "configurable": {"thread_id": args.thread}
    }  # 线程级持久化 :contentReference[oaicite:8]{index=8}

    print("已启动 L02-08D（线程级记忆 + Tool Profile）。输入 exit/quit 退出。")

    # [L02-08D NEW] 只在第一次把 SystemMessage 写入线程状态；之后不再手动维护 messages
    bootstrapped = False

    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not user_text:
            continue

        task_type = set_turn_context(user_text)
        profile = pick_tool_profile(user_text, task_type)
        print(f"[router] task_type={task_type} tool_profile={profile}")

        msgs: list[BaseMessage] = []
        if not bootstrapped:
            msgs.append(SystemMessage(content=SYSTEM_PROMPT))
            bootstrapped = True

        msgs.append(HumanMessage(content=f"[task_type={task_type}] {user_text}"))

        # 线程状态由 checkpointer 维护；每次只提交“本轮新增字段 + 新消息”
        result = graph.invoke(
            {
                "messages": msgs,
                "tool_profile": profile,
                "hops": 0,
                "repeat": {},
                "stop": False,
                "stop_reason": "",
                "last_reply": "",
                "pending_patch_id": None,
                "pending_diff": None,
                "pending_tool": None,
            },
            config=config,
        )

        # interrupt：审批写入补丁
        payload = _get_interrupt_payload(result)
        while payload is not None:
            print("\n========== [WRITE PREVIEW / 等待审批] ==========")
            print(payload.get("diff") or "(no diff)")
            print("===============================================")
            ans = (
                input((payload.get("prompt") or "是否允许？(y/n)") + " ")
                .strip()
                .lower()
            )
            approved = ans in ("y", "yes", "true", "1")

            result = graph.invoke(Command(resume=approved), config=config)
            payload = _get_interrupt_payload(result)

        reply = (result.get("last_reply") or "").strip() or "(no reply)"
        print("AI>", reply)

        if result.get("stop_reason"):
            print(f"[graph] stop_reason={result['stop_reason']}")


if __name__ == "__main__":
    main()
