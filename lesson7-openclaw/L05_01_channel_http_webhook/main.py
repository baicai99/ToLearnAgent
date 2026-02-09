# L05_01_channel_http_webhook/main.py
# 运行方式（推荐从项目根目录执行）：
#   uvicorn L05_01_channel_http_webhook.main:app --host 127.0.0.1 --port 18789 --reload

import os
import time
import uuid
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import Body, FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect  # 断开处理写法见 FastAPI WebSocket 文档 :contentReference[oaicite:2]{index=2}

# NEW: LangGraph
from langgraph.graph import StateGraph, MessagesState, START, END

# NEW: InMemory checkpointer（跳过 SQLite，仅进程内 thread/session）
from langgraph.checkpoint.memory import InMemorySaver  # thread_id 用法 :contentReference[oaicite:3]{index=3}

# NEW: ChatOpenAI
from langchain_openai import ChatOpenAI  # 参数见参考 :contentReference[oaicite:4]{index=4}
from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI(title="Mini-OpenClaw Gateway + HTTP Channel", version="0.5.1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2-nano")  # NEW: 默认按你要求
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

# NEW: 开启 streaming（WS 会尝试 token 流；若你的 base_url 不支持则会退化为整段返回）
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    base_url=OPENAI_BASE_URL,
    temperature=0,
    streaming=True,
)

# NEW: 最小 agent node
async def agent_node(state: MessagesState):
    ai_msg = await llm.ainvoke(state["messages"])
    return {"messages": [ai_msg]}

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

checkpointer = InMemorySaver()
app_graph = graph.compile(checkpointer=checkpointer)


@app.get("/health")
async def health():
    return {"ok": True, "model": OPENAI_MODEL}


# 直连入口（保留）
@app.post("/message")
async def direct_message(payload: Dict[str, Any] = Body(...)):
    """
    { "message": "...", "session_id": "可选" }
    """
    user_text = payload.get("message", "")
    session_id = payload.get("session_id") or uuid.uuid4().hex

    t0 = time.perf_counter()
    config = {"configurable": {"thread_id": session_id}}  # thread_id 绑定 checkpointer :contentReference[oaicite:5]{index=5}
    result = await app_graph.ainvoke({"messages": [HumanMessage(content=user_text)]}, config=config)

    reply = result["messages"][-1].content
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {"session_id": session_id, "reply": reply, "latency_ms": latency_ms, "model": OPENAI_MODEL}


# NEW: Channel 入站入口（HTTP Webhook）
@app.post("/channels/http/inbound")
async def http_channel_inbound(payload: Dict[str, Any] = Body(...)):
    """
    一个通用 webhook channel 的最小入站协议：
      {
        "from": "user_123" / "+86..." / "tg:12345" / "slack:Uxxx",
        "text": "你好",
        "session_id": "可选（外部系统自己管理会话时才传）",
        "channel": "可选（默认 http）"
      }

    返回：
      {
        "channel": "http",
        "from": "...",
        "session_id": "...",
        "reply": "...",
        "latency_ms": ...
      }
    """
    sender = payload.get("from", "")
    text = payload.get("text", "")
    channel = payload.get("channel", "http")

    # NEW: “稳定 session_id”策略（不需要额外存储，也不需要一堆 if/else）
    # - 外部显式传 session_id：直接用
    # - 否则：用 uuid5 对 (channel + sender) 做确定性映射
    session_id = payload.get("session_id") or uuid.uuid5(uuid.NAMESPACE_URL, f"{channel}:{sender}").hex

    t0 = time.perf_counter()
    config = {"configurable": {"thread_id": session_id}}  # thread_id 绑定 checkpointer :contentReference[oaicite:6]{index=6}
    result = await app_graph.ainvoke({"messages": [HumanMessage(content=text)]}, config=config)

    reply = result["messages"][-1].content
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "channel": channel,
        "from": sender,
        "session_id": session_id,
        "reply": reply,
        "latency_ms": latency_ms,
        "model": OPENAI_MODEL,
    }


# 本地调试用 WS（保留）：观察 token 是否真流式
@app.websocket("/ws")
async def ws_debug(websocket: WebSocket):
    await websocket.accept()

    conn_session_id = uuid.uuid4().hex
    await websocket.send_json({"type": "session", "session_id": conn_session_id, "model": OPENAI_MODEL})

    try:
        while True:
            payload = await websocket.receive_json()
            user_text = payload.get("message", "")
            session_id = payload.get("session_id") or conn_session_id

            await websocket.send_json({"type": "start", "session_id": session_id})
            config = {"configurable": {"thread_id": session_id}}  # :contentReference[oaicite:7]{index=7}

            t0 = time.perf_counter()

            async for ev in app_graph.astream_events(
                {"messages": [HumanMessage(content=user_text)]},
                config=config,
                version="v2",
            ):
                if ev.get("event") == "on_chat_model_stream":
                    chunk = (ev.get("data") or {}).get("chunk")
                    token = getattr(chunk, "content", None)
                    if token:
                        await websocket.send_json({"type": "token", "token": token})

            latency_ms = int((time.perf_counter() - t0) * 1000)
            await websocket.send_json({"type": "done", "session_id": session_id, "latency_ms": latency_ms})

    except WebSocketDisconnect:
        return