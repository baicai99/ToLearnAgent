# L04_04_gateway_integrated/main.py
# 运行方式（推荐从项目根目录执行）：
#   uvicorn L04_04_gateway_integrated.main:app --host 127.0.0.1 --port 18789 --reload

import os
import time
import uuid
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import Body, FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect

# NEW: LangGraph（MessagesState + StateGraph）
from langgraph.graph import StateGraph, MessagesState, START, END

# NEW: InMemory checkpointer（跳过 SQLite，仅进程内）
from langgraph.checkpoint.memory import InMemorySaver  # thread_id 绑定见文档 :contentReference[oaicite:3]{index=3}

# NEW: ChatOpenAI（支持 OpenAI-compatible base_url）
from langchain_openai import ChatOpenAI  # base_url 等见参考 :contentReference[oaicite:4]{index=4}
from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI(title="Mini-OpenClaw Gateway (Integrated)", version="0.4.0")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2-nano")  # NEW: 默认按你要求
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# NEW: 开启 streaming（是否真“token 流”还取决于你对接的网关/模型链路是否支持）
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    base_url=OPENAI_BASE_URL,
    temperature=0,
    streaming=True,
)

# NEW: 最小 agent 节点
async def agent_node(state: MessagesState):
    ai_msg = await llm.ainvoke(state["messages"])
    return {"messages": [ai_msg]}

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

# NEW: 进程内 session/thread 持久化
checkpointer = InMemorySaver()
app_graph = graph.compile(checkpointer=checkpointer)


@app.get("/")
async def index():
    # NEW: 最小 WebChat，便于验证 session + token 流
    html = """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Gateway Integrated Demo</title></head>
  <body>
    <div style="max-width:960px;margin:24px auto;font-family:ui-monospace, SFMono-Regular, Menlo, monospace;">
      <h3>Gateway Integrated Demo (HTTP + WS + Session)</h3>
      <div>session_id: <span id="sid">(none)</span></div>

      <div style="margin-top:10px;">
        <input id="msg1" style="width:82%" value="记住：我的名字叫白菜。请回复：已记住" />
        <button onclick="send('msg1')">Send 1</button>
      </div>

      <div style="margin-top:10px;">
        <input id="msg2" style="width:82%" value="我叫什么名字？只回答名字本身。" />
        <button onclick="send('msg2')">Send 2</button>
      </div>

      <pre id="out" style="margin-top:16px;padding:12px;border:1px solid #ddd;white-space:pre-wrap;"></pre>
    </div>

    <script>
      const out = document.getElementById("out");
      const line = (s) => out.textContent += s + "\\n";
      const append = (s) => out.textContent += s;

      let sessionId = null;
      const ws = new WebSocket("ws://" + location.host + "/ws");

      ws.onopen = () => line("[ws] connected");
      ws.onclose = () => line("\\n[ws] closed");

      ws.onmessage = (ev) => {
        const obj = JSON.parse(ev.data);

        if (obj.type === "session") {
          sessionId = obj.session_id;
          document.getElementById("sid").textContent = sessionId;
          line("[session] " + sessionId + " | model=" + obj.model);
          return;
        }

        if (obj.type === "start") {
          line("[start] message_id=" + obj.message_id);
          return;
        }

        if (obj.type === "step") {
          line("[" + obj.event + "] " + obj.name);
          return;
        }

        if (obj.type === "token") {
          append(obj.token);
          return;
        }

        if (obj.type === "done") {
          line("\\n[done] " + obj.latency_ms + " ms");
          return;
        }

        if (obj.type === "error") {
          line("[error] " + obj.message);
          return;
        }

        line(JSON.stringify(obj));
      };

      function send(inputId) {
        out.textContent = "";
        ws.send(JSON.stringify({
          session_id: sessionId,
          message: document.getElementById(inputId).value
        }));
      }
    </script>
  </body>
</html>
"""
    return HTMLResponse(html)


@app.get("/health")
async def health():
    return {"ok": True, "model": OPENAI_MODEL}


# NEW: 显式创建 session（也可不调：/message 和 /ws 会自动分配）
@app.post("/sessions")
async def create_session():
    return {"session_id": uuid.uuid4().hex}


@app.post("/message")
async def http_message(payload: Dict[str, Any] = Body(...)):
    """
    请求体:
      { "message": "...", "session_id": "可选" }
    """
    user_text = payload.get("message", "")
    session_id = payload.get("session_id") or uuid.uuid4().hex

    t0 = time.perf_counter()

    # NEW: session_id 作为 thread_id，绑定到 checkpointer
    config = {"configurable": {"thread_id": session_id}}  # :contentReference[oaicite:5]{index=5}
    result = await app_graph.ainvoke({"messages": [HumanMessage(content=user_text)]}, config=config)

    reply = result["messages"][-1].content
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return {"session_id": session_id, "reply": reply, "latency_ms": latency_ms, "model": OPENAI_MODEL}


@app.websocket("/ws")
async def ws_message(websocket: WebSocket):
    # WebSocket 基本用法见 FastAPI 文档 :contentReference[oaicite:6]{index=6}
    await websocket.accept()

    # NEW: 连接建立即给一个默认 session，客户端也可覆盖传入 session_id
    conn_session_id = uuid.uuid4().hex
    await websocket.send_json({"type": "session", "session_id": conn_session_id, "model": OPENAI_MODEL})

    try:
        while True:
            payload = await websocket.receive_json()
            user_text = payload.get("message", "")

            # NEW: 客户端可显式指定 session_id（thread_id）
            session_id = payload.get("session_id") or conn_session_id
            message_id = uuid.uuid4().hex

            await websocket.send_json({"type": "start", "session_id": session_id, "message_id": message_id})
            await websocket.send_json({"type": "step", "name": "agent", "event": "start"})

            t0 = time.perf_counter()
            config = {"configurable": {"thread_id": session_id}}  # :contentReference[oaicite:7]{index=7}

            # NEW: 只转发 token（避免大量“无用事件”干扰你判断是否真流式）
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
            await websocket.send_json(
                {"type": "done", "session_id": session_id, "message_id": message_id, "latency_ms": latency_ms}
            )

    except WebSocketDisconnect:
        return
    except Exception as e:
        # NEW: 最小错误事件（不做复杂兜底/分支）
        await websocket.send_json({"type": "error", "message": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18789)