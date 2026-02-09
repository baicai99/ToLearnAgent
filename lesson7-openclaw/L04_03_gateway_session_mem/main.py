# L04_03_gateway_session_mem/main.py
# 运行方式（推荐从项目根目录执行）：
#   uvicorn L04_03_gateway_session_mem.main:app --host 127.0.0.1 --port 18789 --reload

import os
import time
import uuid
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import Body, FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver  # 进程内存持久化：threads/thread_id :contentReference[oaicite:3]{index=3}

from langchain_openai import ChatOpenAI  # ChatOpenAI.streaming 参数见参考 :contentReference[oaicite:4]{index=4}
from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI(title="Mini-OpenClaw Gateway (Session + WS)", version="0.3.1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2-nano")  # 默认按你要求
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# NEW: 开启 streaming，否则 astream_events 往往只会在最后给你完整结果 :contentReference[oaicite:5]{index=5}
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    base_url=OPENAI_BASE_URL,
    temperature=0,
    streaming=True,
)

async def agent_node(state: MessagesState):
    # 这里仍然用 ainvoke（返回最终 AIMessage），但因为 llm.streaming=True
    # 过程中的 token 会通过回调事件流（on_chat_model_stream）冒出来 :contentReference[oaicite:6]{index=6}
    ai_msg = await llm.ainvoke(state["messages"])
    return {"messages": [ai_msg]}

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

checkpointer = InMemorySaver()
app_graph = graph.compile(checkpointer=checkpointer)

@app.get("/")
async def index():
    # NEW: token 直接“追加”到同一行，避免你误以为是最后一次性输出
    html = """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Gateway Session WS Demo</title></head>
  <body>
    <div style="max-width:900px;margin:24px auto;font-family:ui-monospace, SFMono-Regular, Menlo, monospace;">
      <h3>Mini Gateway Session + WS Demo</h3>
      <div>session_id: <span id="sid">(none)</span></div>

      <div style="margin-top:8px;">
        <input id="msg" style="width:80%" value="记住我的名字叫白菜。请回复：已记住" />
        <button onclick="sendMsg('msg')">Send 1</button>
      </div>

      <div style="margin-top:8px;">
        <input id="msg2" style="width:80%" value="我叫什么名字？请直接回答名字本身，不要说多余的话。" />
        <button onclick="sendMsg('msg2')">Send 2</button>
      </div>

      <pre id="out" style="margin-top:16px;padding:12px;border:1px solid #ddd;white-space:pre-wrap;"></pre>
    </div>

    <script>
      const out = document.getElementById("out");
      const append = (s) => { out.textContent += s; };
      const appendLine = (s) => { out.textContent += s + "\\n"; };

      let sessionId = null;
      const ws = new WebSocket("ws://" + location.host + "/ws");

      ws.onopen = () => appendLine("[ws] connected");
      ws.onclose = () => appendLine("\\n[ws] closed");

      ws.onmessage = (ev) => {
        const obj = JSON.parse(ev.data);

        if (obj.type === "session") {
          sessionId = obj.session_id;
          document.getElementById("sid").textContent = sessionId;
          appendLine("[session] " + sessionId);
          return;
        }

        if (obj.type === "step") {
          appendLine("\\n[" + obj.event + "] " + obj.name);
          return;
        }

        if (obj.type === "token") {
          append(obj.token);
          return;
        }

        if (obj.type === "done") {
          appendLine("\\n\\n[done] " + obj.latency_ms + " ms");
          return;
        }
      };

      function sendMsg(inputId) {
        out.textContent = "";
        ws.send(JSON.stringify({ message: document.getElementById(inputId).value, session_id: sessionId }));
      }
    </script>
  </body>
</html>
"""
    return HTMLResponse(html)

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/sessions")
async def create_session():
    session_id = uuid.uuid4().hex
    return {"session_id": session_id}

@app.post("/message")
async def message(payload: Dict[str, Any] = Body(...)):
    user_text = payload.get("message", "")
    session_id = payload.get("session_id") or uuid.uuid4().hex

    t0 = time.perf_counter()
    config = {"configurable": {"thread_id": session_id}}  # thread_id 配合 checkpointer :contentReference[oaicite:7]{index=7}
    result = await app_graph.ainvoke({"messages": [HumanMessage(content=user_text)]}, config=config)

    reply = result["messages"][-1].content
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {"session_id": session_id, "reply": reply, "latency_ms": latency_ms, "model": OPENAI_MODEL}

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    conn_default_session = uuid.uuid4().hex
    await websocket.send_json({"type": "session", "session_id": conn_default_session, "model": OPENAI_MODEL})

    try:
        while True:
            payload = await websocket.receive_json()
            user_text = payload.get("message", "")
            session_id = payload.get("session_id") or conn_default_session

            await websocket.send_json({"type": "step", "name": "agent", "event": "start"})
            config = {"configurable": {"thread_id": session_id}}  # :contentReference[oaicite:8]{index=8}

            t0 = time.perf_counter()

            # NEW: astream_events v2：on_chat_model_stream 的 data.chunk 是 AIMessageChunk(content="...") :contentReference[oaicite:9]{index=9}
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
            await websocket.send_json({"type": "step", "name": "agent", "event": "end"})
            await websocket.send_json({"type": "done", "session_id": session_id, "latency_ms": latency_ms})

    except WebSocketDisconnect:
        return

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18789)