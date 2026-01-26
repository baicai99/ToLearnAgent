# -*- coding: utf-8 -*-
"""
执行代码：
  python -m lessonL03_03_routerless_react.main

标题：L03-03（无关键词路由的 ReAct + 持久化 + interrupt 审批）

[L03-03 NEW]
- 不再用关键词 router：LLM 自主决定是否调用工具（看 tool_calls）
- 继续使用 SqliteSaver 的上下文管理器写法（避免把 context manager 当 saver 传入）
"""

from __future__ import annotations

# 若误用“按文件路径运行”，直接给出明确提示
if __package__ is None or __package__ == "":
    raise SystemExit("请在项目根目录运行：python -m lessonL03_03_routerless_react.main")

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
    session_file.write_text(json.dumps({"thread_id": tid}, ensure_ascii=False), encoding="utf-8")
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
            write_policy=DEFAULT_WRITE_POLICY,
        )
        graph = builder.compile(checkpointer=checkpointer)

        print("已启动 L03-03（无关键词路由的 ReAct）。输入 exit/quit 退出。")
        print(f"workdir: {WORKDIR}")
        print("路径约定：所有路径相对 toy_repo，例如 calc.py、tests/test_calc.py（不要写 toy_repo/ 前缀）")
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

            # 触发一次图运行
            interrupted = False
            for event in graph.stream({"messages": [HumanMessage(content=user)]}, config=config, stream_mode="updates"):
                if "__interrupt__" in event:
                    intr = event["__interrupt__"][0]
                    print(intr.value)
                    ans = input("> ").strip().lower()
                    logger.log("approval_input", {"text": ans})
                    interrupted = True

                    for _ in graph.stream(Command(resume=ans), config=config, stream_mode="updates"):
                        pass
                    break

            st = graph.get_state(config).values
            reply = _last_ai_text(st.get("messages"))
            if reply:
                print(f"AI> {reply}")
            else:
                print("AI> (no reply)")

            # [教学型输出] 打印最近一次工具轨迹摘要（不是 tool 消息）
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
