# -*- coding: utf-8 -*-
"""
执行代码：
  python -m lessonL03_04_commands_modes.main

标题：L03-04（/command + plan/build + doom-loop 护栏）

[L03-04 NEW]
- 显式命令系统：/help /mode /plan /build /review /test /fix
- 仍然是“非命令输入不路由”：由 LLM 决定是否调用工具
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    # Allow running as a script:
    #   python lessonL03_04_commands_modes/main.py
    # while keeping relative imports working.
    import sys
    from pathlib import Path

    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    __package__ = "lessonL03_04_commands_modes"

import json
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import (
    WORKDIR,
    SESSION_FILE,
    CHECKPOINT_DB,
    EVENT_LOG,
    DEFAULT_MODEL,
    DEFAULT_WRITE_POLICY,
    DEFAULT_MODE,
    load_env_from_root,
)
from .event_log import EventLogger
from .core_runtime import Runtime
from .graph_agent import build_graph


def _load_or_create_thread_id(session_file: Path) -> str:
    if session_file.exists():
        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
            tid = str(data.get("thread_id", "")).strip()
            if tid:
                return tid
        except Exception:
            pass
    tid = str(uuid.uuid4())
    session_file.write_text(
        json.dumps({"thread_id": tid}, ensure_ascii=False), encoding="utf-8"
    )
    return tid


def _last_ai_text(messages):
    for m in reversed(messages or []):
        if getattr(m, "type", "") == "ai":
            txt = (getattr(m, "content", "") or "").strip()
            if txt:
                return txt
    return ""


def main():
    api_key, base_url = load_env_from_root()

    if not WORKDIR.exists():
        raise RuntimeError(f"找不到 toy_repo：{WORKDIR}（请确认工作区目录存在）")

    thread_id = _load_or_create_thread_id(SESSION_FILE)
    config = {"configurable": {"thread_id": thread_id}}

    logger = EventLogger(EVENT_LOG)
    rt = Runtime(workdir=WORKDIR)

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        builder = build_graph(
            rt=rt,
            logger=logger,
            api_key=api_key,
            base_url=base_url,
            model=DEFAULT_MODEL,
            default_mode=DEFAULT_MODE,
            default_write_policy=DEFAULT_WRITE_POLICY,
        )
        graph = builder.compile(checkpointer=checkpointer)

        print(
            "已启动 L03-04（/command + plan/build + doom-loop 护栏）。输入 exit/quit 退出。"
        )
        print(f"workdir: {WORKDIR}")
        print(
            "路径约定：所有路径相对 toy_repo，例如 calc.py、tests/test_calc.py（不要写 toy_repo/ 前缀）"
        )
        print(f"model: {DEFAULT_MODEL}")
        print(f"default_mode: {DEFAULT_MODE}")
        print(f"default_write_policy: {DEFAULT_WRITE_POLICY}")
        print(f"thread_id: {thread_id}")
        print(f"checkpoint_db: {CHECKPOINT_DB}")
        print(f"event_log: {EVENT_LOG}")
        print("输入 /help 查看命令列表。")

        while True:
            user = input("You> ").strip()
            if user.lower() in ("exit", "quit"):
                break

            logger.log("user_input", {"text": user})

            interrupted = False
            for event in graph.stream(
                {"messages": [HumanMessage(content=user)]},
                config=config,
                stream_mode="updates",
            ):
                if "__interrupt__" in event:
                    intr = event["__interrupt__"][0]
                    print(intr.value)
                    ans = input("> ").strip().lower()
                    logger.log("approval_input", {"text": ans})
                    interrupted = True

                    for _ in graph.stream(
                        Command(resume=ans), config=config, stream_mode="updates"
                    ):
                        pass
                    break

            st = graph.get_state(config).values
            reply = _last_ai_text(st.get("messages"))
            if reply:
                print(f"AI> {reply}")
            else:
                print("AI> (no reply)")

            mode = st.get("mode", DEFAULT_MODE)
            wp = st.get("write_policy", DEFAULT_WRITE_POLICY)
            print(f"[state] mode={mode} write_policy={wp}")

            traces = st.get("tool_traces", []) or []
            if traces:
                tail = traces[-3:]
                print("[trace] 最近 3 条工具轨迹：")
                for i, tr in enumerate(tail, start=1):
                    name = tr.get("name", "")
                    args = tr.get("args", {})
                    prev = (tr.get("result_preview", "") or "").replace("\n", " ")
                    if len(prev) > 160:
                        prev = prev[:160] + "...(truncated)"
                    print(f"  {i}) {name} args={args} -> {prev}")

            if interrupted:
                logger.log("turn_done_after_resume", {})


if __name__ == "__main__":
    main()
