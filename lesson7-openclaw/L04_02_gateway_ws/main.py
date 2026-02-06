# L04_02_gateway_ws/main.py
# 运行方式（推荐从项目根目录执行）：
#   uvicorn L04_02_gateway_ws.main:app --host 127.0.0.1 --port 18789 --reload

import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect

# NEW: LangGraph 最小 Graph API
from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
    END,
)  # :contentReference[oaicite:3]{index=3}

# NEW: OpenAI 兼容 Chat 模型封装
from langchain_openai import ChatOpenAI  # :contentReference[oaicite:4]{index=4}
from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI(title="Mini-OpenClaw Gateway (WS)", version="0.2.0")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # 可选

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    base_url=OPENAI_BASE_URL,
    temperature=0,
)


# NEW: LangGraph 节点（仍然是最小单节点，后续 4.3/4.4 才引入 session/thread 等）
async def agent_node(state: MessagesState):
    ai_msg = await llm.ainvoke(state["messages"])
    return {"messages": [ai_msg]}


graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)
app_graph = graph.compile()


# NEW: 提供一个最小网页客户端，方便你直接验证 WS token/step 流（不引入额外依赖）
@app.get("/")
async def index():
    html = """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>WS Gateway Demo</title></head>
  <body>
    <div style="max-width:900px;margin:24px auto;font-family:ui-monospace, SFMono-Regular, Menlo, monospace;">
      <h3>Mini Gateway WS Demo</h3>
      <div>
        <input id="msg" style="width:80%" value="用一句话解释 LangGraph，并给一个例子" />
        <button onclick="sendMsg()">Send</button>
      </div>
      <pre id="log" style="margin-top:16px;padding:12px;border:1px solid #ddd;white-space:pre-wrap;"></pre>
    </div>
    <script>
      const log = (s) => { document.getElementById("log").textContent += s + "\\n"; };
      const ws = new WebSocket("ws://" + location.host + "/ws");
      ws.onopen = () => log("[ws] connected");
      ws.onmessage = (ev) => {
        try {
          const obj = JSON.parse(ev.data);
          if (obj.type === "token") log(obj.token);
          else log(JSON.stringify(obj));
        } catch (e) {
          log(ev.data);
        }
      };
      ws.onclose = () => log("[ws] closed");
      function sendMsg() {
        document.getElementById("log").textContent = "";
        ws.send(JSON.stringify({message: document.getElementById("msg").value}));
      }
    </script>
  </body>
</html>
"""
    return HTMLResponse(html)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/message")
async def message(message: str = Body(..., embed=True)):
    # 保留一个 HTTP 兜底入口（MVP），便于脚本/CI 测试
    t0 = time.perf_counter()
    result = await app_graph.ainvoke({"messages": [HumanMessage(content=message)]})
    reply = result["messages"][-1].content
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {"reply": reply, "latency_ms": latency_ms, "model": OPENAI_MODEL}


# NEW: WebSocket：流式推送 token + step 事件
# 事件来自 astream_events（v2），事件名格式见 LangChain runnable 文档 :contentReference[oaicite:5]{index=5}
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            payload = await websocket.receive_json()
            user_text = payload.get("message", "")

            t0 = time.perf_counter()
            await websocket.send_json({"type": "start", "model": OPENAI_MODEL})

            # NEW: 关键：用 astream_events 拿到 token/step 等标准事件流 :contentReference[oaicite:6]{index=6}
            async for ev in app_graph.astream_events(
                {"messages": [HumanMessage(content=user_text)]},
                version="v2",
            ):
                event_name: str = ev.get("event", "")
                runnable_name: str = ev.get("name", "")

                # NEW: token 流（on_chat_model_stream）——用 chunk.content 提取增量 token
                if event_name == "on_chat_model_stream":
                    chunk = (ev.get("data") or {}).get("chunk")
                    token = getattr(chunk, "content", None)
                    if token:
                        await websocket.send_json({"type": "token", "token": token})
                    continue

                # NEW: step 边界（node 进入/退出）——MVP 只标注 agent 这个节点
                if runnable_name == "agent" and event_name in (
                    "on_chain_start",
                    "on_chain_end",
                ):
                    await websocket.send_json(
                        {"type": "step", "name": runnable_name, "event": event_name}
                    )
                    continue

                # NEW: 其余事件原样轻量转发（便于你观察生态位：graph/chain/model 的层级关系）
                await websocket.send_json(
                    {
                        "type": "event",
                        "event": event_name,
                        "name": runnable_name,
                        "run_id": str(ev.get("run_id", "")),
                    }
                )

            latency_ms = int((time.perf_counter() - t0) * 1000)
            await websocket.send_json({"type": "done", "latency_ms": latency_ms})

    except WebSocketDisconnect:
        # 断开连接：按 FastAPI/Starlette 约定处理即可 :contentReference[oaicite:7]{index=7}
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=18789)
