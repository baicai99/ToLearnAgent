# lg00_min_loop.py
# 目标：最小 StateGraph + 条件边 + 循环终止（不含 LLM）
# 学完本文件你就掌握 LangGraph 的“骨架”：State / Node / Edge / START / END / compile / invoke

import operator
from typing import Annotated, Literal, cast
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # reducer=operator.add：让 aggregate 变成“只追加”的列表（每次 node 返回 ["A"] 会被累加）
    aggregate: Annotated[list[str], operator.add]

def node_a(state: State):
    # 节点：读取 state，返回“更新”（不要原地 mutate）
    return {"aggregate": ["A"]}

def node_b(state: State):
    return {"aggregate": ["B"]}

def route(state: State) -> Literal["b", "__end__"]:
    # 条件边：达到终止条件就返回 END，否则去 b
    if len(state["aggregate"]) >= 6:
        return cast(Literal["__end__"], END)
    return "b"

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)   # a 执行后，根据 route 决定去 b 还是 END
builder.add_edge("b", "a")                  # b 执行后回到 a，形成循环

graph = builder.compile()  # 必须 compile 后才能 invoke/stream :contentReference[oaicite:6]{index=6}

if __name__ == "__main__":
    out = graph.invoke({"aggregate": []}, {"recursion_limit": 50})
    print(out)
