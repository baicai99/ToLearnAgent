# -*- coding: utf-8 -*-
"""
执行代码：
  python -m lessonL03_02_safe_persist.main

标题：L03-02（可恢复持久化 + interrupt 审批 + 不持久化 ToolMessage）

[L03-02 NEW]
- 强制模块运行，规避 sys.path 导入问题（不会再 ModuleNotFoundError）
- SqliteSaver.from_conn_string 必须在 with 内部使用（你环境里是上下文管理器）citeturn0search4turn0search8
"""

from __future__ import annotations

# [L03-02 NEW] 若用户误用“按文件路径运行”，直接给出明确提示（而不是隐式导入失败）
if __package__ is None or __package__ == "":
    raise SystemExit("请在项目根目录运行：python -m lessonL03_02_safe_persist.main")

import json
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)  # 来自 langgraph-checkpoint-sqlite citeturn0search4

from .config import (
    WORKDIR,
    SESSION_FILE,
    CHECKPOINT_DB,
    EVENT_LOG,
    DEFAULT_MODEL,
    DEFAULT_WRITE_POLICY,
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

    # [L03-02 NEW] from_conn_string 在你当前版本返回上下文管理器，必须 with 使用 citeturn0search4turn0search8
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        builder = build_graph(
            rt=rt,
            logger=logger,
            api_key=api_key,
            base_url=base_url,
            model=DEFAULT_MODEL,
            write_policy=DEFAULT_WRITE_POLICY,
        )
        graph = builder.compile(checkpointer=checkpointer)

        print(
            "已启动 L03-02（持久化 + interrupt 审批 + 不持久化 ToolMessage）。输入 exit/quit 退出。"
        )
        print(f"workdir: {WORKDIR}")
        print(f"model: {DEFAULT_MODEL}")
        print(f"write_policy: {DEFAULT_WRITE_POLICY}")
        print(f"thread_id: {thread_id}")
        print(f"checkpoint_db: {CHECKPOINT_DB}")
        print(f"event_log: {EVENT_LOG}")

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

                    # resume（会从 approval 节点继续）
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
                # 这里理论上不会出现 noreply；若出现说明图没有产生 AIMessage
                print("AI> (no reply)")

            if interrupted:
                logger.log("turn_done_after_resume", {})


if __name__ == "__main__":
    main()
