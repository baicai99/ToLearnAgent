# -*- coding: utf-8 -*-
"""
标题：L02-08G main.py —— pytest gate + 每轮输出“本轮过程”（教学）
执行代码：
  python main.py --model gpt-5-nano --policy ask --workdir toy_repo --thread demo-thread-2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import intent as it
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
    run_pytest,  # [L02-08G NEW]
    evidence_read,
    todowrite,
    todoread,
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
        if not k:
            continue

        lk = k.lower()
        if lk == "apikey" and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = v
            continue
        if lk == "baseurl" and not os.environ.get("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = v
            continue

        if k not in os.environ or os.environ[k] == "":
            os.environ[k] = v


SYSTEM_PROMPT = """\
你是一个终端里的 Coding Agent，遵循 ReAct（Agent->Tool->Agent）。

【工具意图门控（L02-08F）】
- 闲聊/概念解释：不要调用任何工具，直接回答。
- review：只读工具取证据（repo_tree/list_files/grep/read_file_range）后再总结。
- create/tests/implement：可以用工具；写入必须走 PREVIEW->人工审批；不允许“猜测 file_path”。

【L02-08G 新增：测试驱动闭环（pytest gate）】
- 当任务涉及“修复/实现/让测试通过”：写入提交后，应优先调用 run_pytest 验证。
- 只有拿到 pytest 证据（rc=0 或失败信息），才允许判断 done/继续修复。
- 输出必须包含“本轮你做了哪些工具调用与依据”（可结合 progress 说明）。

请用教学风格在终端输出：你为什么做这一步、依据是什么、下一步是什么。
"""


def build_llm(model: str, temperature: float) -> ChatOpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return ChatOpenAI(
        model=model, temperature=temperature, api_key=api_key, base_url=base_url
    )


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


def _fallback_reply(result: Dict[str, Any]) -> str:
    reply = (result.get("last_reply") or "").strip()
    if reply:
        return reply
    msgs = result.get("messages") or []
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, AIMessage):
                c = (m.content or "").strip()
                if c:
                    return c
    sr = (result.get("stop_reason") or "").strip()
    if sr:
        return f"(no reply; stop_reason={sr})"
    return "(no reply)"


# [L02-08G NEW] 每轮把 progress 打出来，保证你能回放过程
def _print_progress(result: Dict[str, Any], limit: int = 8) -> None:
    prog = result.get("progress") or []
    if not isinstance(prog, list) or not prog:
        return
    tail = prog[-limit:]
    print("\n【本轮过程（progress）】")
    for x in tail:
        print("-", x)
    print("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--policy", default="ask", choices=["allow", "ask", "deny"])
    parser.add_argument("--workdir", default="toy_repo")
    parser.add_argument("--thread", default="demo-thread-2")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "缺少 OPENAI_API_KEY / apikey：请在根目录 .env 或环境变量中设置。"
        )

    init_tools(args.workdir, args.policy)

    llm = build_llm(args.model, args.temperature)

    llm_plain = llm
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
            run_pytest,  # [L02-08G NEW]
            evidence_read,
            todowrite,
            todoread,
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
        "run_pytest": run_pytest,
        "evidence_read": evidence_read,
        "todowrite": todowrite,
        "todoread": todoread,
    }

    memory = InMemorySaver()
    graph = build_graph(
        llm_plain=llm_plain,
        llm_review_ro=llm_review_ro,
        llm_full=llm_full,
        tool_map=tool_map,
        checkpointer=memory,
    )

    config = {"configurable": {"thread_id": args.thread}}

    print(
        "已启动 L02-08G（pytest gate + 证据链）。输入 exit/quit 退出；help 查看命令。"
    )

    bootstrapped = False

    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not user_text:
            continue

        task_type = set_turn_context(user_text)

        ttype = it.classify_task_type(user_text)
        profile, reason = it.pick_tool_profile(ttype, user_text)
        print(
            f"[router] task_type={task_type} intent_task_type={ttype} tool_profile={profile} reason={reason}"
        )

        msgs: list[BaseMessage] = []
        if not bootstrapped:
            msgs.append(SystemMessage(content=SYSTEM_PROMPT))
            bootstrapped = True
        msgs.append(HumanMessage(content=f"[task_type={task_type}] {user_text}"))

        result = graph.invoke(
            {
                "messages": msgs,
                "hops": 0,
                "repeat": {},
                "stop": False,
                "stop_reason": "",
                "last_reply": "",
                "pending_patch_id": None,
                "pending_diff": None,
                "pending_tool": None,
                "session_active": False,
                "objective": "",
                "constraints": [],
                "done_criteria": [],
                "progress": [],
                "open_questions": [],
                "last_test_rc": None,
                "last_test_cmd": None,
                "last_test_summary": "",
            },
            config=config,
        )

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

        print("AI>", _fallback_reply(result))
        _print_progress(result, limit=10)

        if result.get("stop_reason"):
            print(f"[graph] stop_reason={result['stop_reason']}")


if __name__ == "__main__":
    main()
