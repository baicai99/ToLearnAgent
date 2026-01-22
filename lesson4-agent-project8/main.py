# -*- coding: utf-8 -*-
"""
标题：L02-08C main.py —— CLI 里处理 __interrupt__ 并用 Command(resume=...) 恢复
执行代码：
  python main.py --model gpt-5-nano --policy ask --workdir toy_repo
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from langgraph.checkpoint.memory import (
    InMemorySaver,
)  # :contentReference[oaicite:6]{index=6}
from langgraph.types import Command  # :contentReference[oaicite:7]{index=7}

import core_runtime as cr
from tools import (
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


SYSTEM_PROMPT = """\
你是一个终端里的 Coding Agent，遵循 ReAct（Agent->Tool->Agent）。

本课关键：
- 写入确认不在工具层 input(y/n)，而是在图层用 interrupt 等待人类审批。
- ask 策略下，write_file/apply_hunks 会返回 PREVIEW（patch_id + diff），随后图会进入审批节点。
- 只有用户批准后，系统才会 commit_patch；若用户拒绝，会记录并结束本轮，避免重复尝试。

注意：
- 除非你认为本轮任务已完成，否则不要调用 submit。
- 需要结束时，submit(final=..., evidence=[...], task_type=...) 的 final 必须有可读的中文结论。
"""


def build_llm(model: str, temperature: float) -> ChatOpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return ChatOpenAI(
        model=model, temperature=temperature, api_key=api_key, base_url=base_url
    )


def _get_interrupt_payload(result: Dict[str, Any]) -> Dict[str, Any] | None:
    intr = result.get("__interrupt__")
    if not intr:
        return None
    # docs 中通常是 [Interrupt(...)]；这里做最小兼容读取 value
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY：请在根目录 .env 或环境变量中设置。")

    cr.init_runtime(Path(args.workdir), args.policy)

    llm = build_llm(args.model, args.temperature)

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

    # [L02-08C NEW] interrupt 需要 checkpointer :contentReference[oaicite:8]{index=8}
    memory = InMemorySaver()
    graph = build_graph(
        llm_tools=llm_tools,
        tool_map=tool_map,
        checkpointer=memory,
        verbose=args.verbose,
    )

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    print("已启动 L02-08C（interrupt 写入审批）。输入 exit/quit 退出。")

    turn_id = 0
    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not user_text:
            continue

        turn_id += 1
        thread_id = f"turn-{turn_id}-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}

        tt = set_turn_context(user_text)
        print(f"[router] task_type={tt}")

        messages.append(HumanMessage(content=f"[task_type={tt}] {user_text}"))

        state_in = {
            "messages": messages,
            "hops": 0,
            "repeat": {},
            "stop": False,
            "stop_reason": "",
            "last_reply": "",
            "pending_patch_id": None,
            "pending_diff": None,
            "pending_tool": None,
        }

        result = graph.invoke(state_in, config=config)

        # [L02-08C NEW] 处理 interrupt：读取 __interrupt__，向人类要输入，再 resume
        payload = _get_interrupt_payload(result)
        while payload is not None:
            print("\n========== [WRITE PREVIEW / 等待审批] ==========")
            if payload.get("diff"):
                print(payload["diff"])
            else:
                print("(no diff)")
            print("===============================================")
            ans = (
                input((payload.get("prompt") or "是否允许？(y/n)") + " ")
                .strip()
                .lower()
            )
            approved = ans in ("y", "yes", "true", "1")

            result = graph.invoke(Command(resume=approved), config=config)
            payload = _get_interrupt_payload(result)

        # 更新 messages（本轮累积后的完整历史）
        messages = result.get("messages", messages)

        reply = (result.get("last_reply") or "").strip() or "(no reply)"
        print("AI>", reply)

        if result.get("stop_reason"):
            print(f"[graph] stop_reason={result['stop_reason']}")


if __name__ == "__main__":
    main()
