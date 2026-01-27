# -*- coding: utf-8 -*-
"""
执行代码：
  macOS:    python lessonL03_01_persist/main.py
  Windows:  python .\lessonL03_01_persist\main.py

标题：L03-01（持久化会话 + interrupt 写入审批 + 事件日志）

[L03 NEW]
1) SQLite checkpointer：退出/重启后继续同一 thread
2) session 文件保存 thread_id（.l03_session.json）
3) 事件日志 JSONL（.l03_events.jsonl）
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.types import Command

if __package__ in (None, ""):
    # Allow running as a script: `python .\lessonL03_01_persist\main.py`
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from lessonL03_01_persist.config import (
    WORKDIR,
    SESSION_FILE,
    CHECKPOINT_DB,
    EVENT_LOG,
    load_env_from_root,
)
from lessonL03_01_persist.event_log import EventLogger
from lessonL03_01_persist.tools import init_runtime
from lessonL03_01_persist.graph_agent import build_graph

WritePolicy = Literal["allow", "ask", "deny"]


def _load_or_create_thread_id(session_file: Path, new: bool = False) -> str:
    if new and session_file.exists():
        session_file.unlink()

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


def main():
    api_key, base_url = load_env_from_root()

    if not WORKDIR.exists():
        raise RuntimeError(f"找不到 toy_repo：{WORKDIR}（请确认工作区目录存在）")

    # 你可以改成 allow/deny；本课默认 ask
    write_policy: WritePolicy = "ask"

    logger = EventLogger(EVENT_LOG)
    rt = init_runtime(WORKDIR)

    thread_id = _load_or_create_thread_id(SESSION_FILE, new=False)
    config = {"configurable": {"thread_id": thread_id}}

    graph, memory = build_graph(
        rt=rt,
        logger=logger,
        api_key=api_key,
        base_url=base_url,
        write_policy=write_policy,
        checkpoint_db_path=str(CHECKPOINT_DB),
    )

    print(f"已启动 L03-01（持久化会话 + interrupt 审批）。输入 exit/quit 退出。")
    print(f"workdir: {WORKDIR}")
    print(f"write_policy: {write_policy}")
    print(f"thread_id: {thread_id}")
    print(f"checkpoint_db: {CHECKPOINT_DB}")
    print(f"event_log: {EVENT_LOG}")

    try:
        while True:
            user = input("You> ").strip()
            if user.lower() in ("exit", "quit"):
                break

            logger.log("user_input", {"text": user})

            # 先把用户消息送进图
            out = None
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

                    # resume
                    for event2 in graph.stream(
                        Command(resume=ans), config=config, stream_mode="updates"
                    ):
                        # 这里不做复杂渲染，最终统一从 state 里取最后一条 AIMessage 输出
                        pass
                    break

            # 每轮结束后，从状态里取最后一条 assistant 文本回复
            st = graph.get_state(config).values
            msgs = st.get("messages", [])
            reply = ""
            for m in reversed(msgs):
                if getattr(m, "type", "") == "ai":
                    reply = (getattr(m, "content", "") or "").strip()
                    if reply:
                        break
            if reply:
                print(f"AI> {reply}")
            else:
                print("AI> (no reply)")

    finally:
        # 关闭 sqlite saver
        try:
            memory.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
