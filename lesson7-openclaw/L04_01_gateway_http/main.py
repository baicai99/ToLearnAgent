# L04_01_gateway_http/main.py
# 运行方式（推荐从项目根目录执行）：
#   uvicorn L04_01_gateway_http.main:app --host 127.0.0.1 --port 18789 --reload

import os
import time

from dotenv import load_dotenv
from fastapi import Body, FastAPI

# NEW: LangGraph 的最小 Graph API（MessagesState / START / END）
from langgraph.graph import StateGraph, MessagesState, START, END  # :contentReference[oaicite:2]{index=2}

# NEW: LangChain 的 OpenAI 兼容 Chat 模型封装
from langchain_openai import ChatOpenAI  # :contentReference[oaicite:3]{index=3}
from langchain_core.messages import HumanMessage


load_dotenv()

app = FastAPI(title="Mini-OpenClaw Gateway (HTTP)", version="0.1.0")

# NEW: 读取模型配置（最小必需：API KEY；model 给一个默认值方便 MVP）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # 可选

# NEW: 初始化 LLM（base_url 可用于 OpenAI 兼容网关/代理） :contentReference[oaicite:4]{index=4}
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    base_url=OPENAI_BASE_URL,
    temperature=0,
)


# NEW: LangGraph 节点：把 messages 丢给 LLM，再把 AI 回复追加回 messages
async def agent_node(state: MessagesState):
    ai_msg = await llm.ainvoke(state["messages"])
    return {"messages": [ai_msg]}


# NEW: 构建最小图：START -> agent -> END（官方 overview 也是这个骨架） :contentReference[oaicite:5]{index=5}
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)
app_graph = graph.compile()


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/message")
async def message(message: str = Body(..., embed=True)):
    """
    请求体:
      { "message": "你好" }

    响应:
      { "reply": "...", "latency_ms": 12, "model": "..." }
    """
    t0 = time.perf_counter()

    result = await app_graph.ainvoke(
        {"messages": [HumanMessage(content=message)]}
    )

    reply = result["messages"][-1].content
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return {"reply": reply, "latency_ms": latency_ms, "model": OPENAI_MODEL}


if __name__ == "__main__":
    # 本地直接 python -m L04_01_gateway_http.main 也能跑（但推荐用 uvicorn 命令）
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=18789)