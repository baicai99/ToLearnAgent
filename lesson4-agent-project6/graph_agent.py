# -*- coding: utf-8 -*-
"""
标题：L02-08B graph_agent.py —— 用 LangGraph 表达 ReAct（think->tools->think）
执行代码：
  由 main.py 调用 build_graph(...) 后 graph.invoke(...)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import TypedDict
import operator

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.graph import (
    StateGraph,
    START,
    END,
)  # 【NEW-LANGGRAPH】图/起点/终点 :contentReference[oaicite:1]{index=1}


# =========================
# 【NEW-LANGGRAPH】1) 图的 State 设计
# - messages 用 reducer=operator.add：节点只返回“新增消息列表”，图会自动累加到总 messages
#   这是 LangGraph 常用模式：state["messages"] = state["messages"] + update["messages"] :contentReference[oaicite:2]{index=2}
# =========================
class AgentState(TypedDict):
    messages: List[BaseMessage]
    hops: int
    repeat: Dict[str, int]
    stop: bool
    stop_reason: str
    last_reply: str


def _get_tool_calls(ai_msg: Any) -> List[Dict[str, Any]]:
    """
    兼容从 AIMessage 里取 tool_calls。
    """
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        return ai_msg.tool_calls
    ak = getattr(ai_msg, "additional_kwargs", {}) or {}
    return ak.get("tool_calls", []) or []


def _fingerprint(name: str, args: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))
    h.update(b"\n")
    h.update(json.dumps(args, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def build_graph(
    llm_tools: Any,
    tool_map: Dict[str, Any],
    max_tool_hops: int = 32,
    repeat_limit: int = 6,
):
    """
    【NEW-LANGGRAPH】2) 构建 StateGraph：think -> tools -> think ... -> END
    """

    # 【NEW-LANGGRAPH】节点A：think（调用 LLM，产出 tool_calls 或直接回答）
    def node_think(state: AgentState) -> Dict[str, Any]:
        if state["stop"]:
            return {}

        # hops 超限直接停（防图循环）
        if state["hops"] >= max_tool_hops:
            return {
                "stop": True,
                "stop_reason": "工具调用过多已停止（防循环）。",
                "last_reply": "工具调用过多已停止（防循环）。",
            }

        ai = llm_tools.invoke(state["messages"])
        # 把 AIMessage 追加进 messages（增量返回）
        updates: Dict[str, Any] = {
            "messages": [ai],
            "last_reply": ai.content or "",
        }
        return updates

    # 【NEW-LANGGRAPH】节点B：tools（执行最后一个 AIMessage 请求的工具）
    def node_tools(state: AgentState) -> Dict[str, Any]:
        if state["stop"]:
            return {}

        # 最新一条应为 AIMessage
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return {
                "stop": True,
                "stop_reason": "内部状态异常：最后一条不是 AIMessage。",
                "last_reply": "内部状态异常：最后一条不是 AIMessage。",
            }

        tool_calls = _get_tool_calls(last)
        if not tool_calls:
            # 没工具就不做事
            return {}

        new_msgs: List[BaseMessage] = []
        repeat = state["repeat"]

        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {}) or {}

            if name not in tool_map:
                # 不崩溃：返回工具消息让模型自纠
                new_msgs.append(
                    ToolMessage(
                        content=f"(error) unknown tool: {name}",
                        tool_name=name or "unknown",
                    )
                )
                continue

            fp = _fingerprint(name, args)
            repeat[fp] = repeat.get(fp, 0) + 1
            if repeat[fp] >= repeat_limit:
                return {
                    "stop": True,
                    "stop_reason": "检测到重复尝试相同动作（模型可能卡住）。请你明确下一步：继续、改方案、或停止。",
                    "last_reply": "检测到重复尝试相同动作（模型可能卡住）。请你明确下一步：继续、改方案、或停止。",
                }

            tool_fn = tool_map[name]
            result = tool_fn.invoke(args)

            # tool_call_id 在不同版本中可能存在/不存在；这里不强依赖
            new_msgs.append(ToolMessage(content=result, tool_name=name))

            # 如果 submit=ACCEPT，直接 stop（你也可以选择回到 think 让模型写一个自然语言总结）
            if name == "submit":
                try:
                    obj = json.loads(result)
                    if obj.get("status") == "ACCEPT":
                        return {
                            "messages": new_msgs,
                            "stop": True,
                            "stop_reason": "submit=ACCEPT",
                            "last_reply": "已完成（submit=ACCEPT）。",
                            "hops": state["hops"] + 1,
                            "repeat": repeat,
                        }
                except Exception:
                    # submit 结果非 JSON：交给模型下一步处理，不在这里崩溃
                    pass

        return {
            "messages": new_msgs,
            "hops": state["hops"] + 1,
            "repeat": repeat,
        }

    # 【NEW-LANGGRAPH】3) 条件边：决定 think 后是去 tools 还是结束
    def route_after_think(state: AgentState) -> str:
        if state["stop"]:
            return "end"

        last = state["messages"][-1]
        if isinstance(last, AIMessage) and _get_tool_calls(last):
            return "tools"
        return "end"

    # 【NEW-LANGGRAPH】4) 条件边：tools 后如果 stop 就结束，否则回到 think
    def route_after_tools(state: AgentState) -> str:
        return "end" if state["stop"] else "think"

    # =========================
    # 【NEW-LANGGRAPH】5) 组装图
    # =========================
    sg = StateGraph(AgentState)

    sg.add_node("think", node_think)
    sg.add_node("tools", node_tools)

    sg.add_edge(START, "think")
    sg.add_conditional_edges("think", route_after_think, {"tools": "tools", "end": END})
    sg.add_conditional_edges("tools", route_after_tools, {"think": "think", "end": END})

    graph = sg.compile()
    return graph
