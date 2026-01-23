# -*- coding: utf-8 -*-
"""
标题：L02-08B graph_agent.py —— 用 LangGraph 表达 ReAct（think->tools->think）
执行代码：
  由 main.py 调用 build_graph(...) 后 graph.invoke(...)

说明：
  本文件包含本课全部新增内容。
  另外包含一处关键修复点：tool-calling 场景 AIMessage.content 常为空，用户可见输出应来自 submit.final。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from typing import Annotated

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

# [L02-08B NEW] add_messages reducer：用于安全追加消息状态（图会自动合并增量 messages）
from langgraph.graph.message import add_messages


class AgentState(Dict[str, Any]):
    # [L02-08B NEW] 图状态：messages 采用 add_messages 自动累积
    messages: Annotated[List[BaseMessage], add_messages]
    hops: int
    repeat: Dict[str, int]
    stop: bool
    stop_reason: str

    # [L02-08B FIX] 用户可见输出：不要只依赖 AIMessage.content（tool-calling 常为空）
    last_reply: str


def _tool_calls_from(ai_msg: AIMessage) -> List[Dict[str, Any]]:
    if getattr(ai_msg, "tool_calls", None):
        return ai_msg.tool_calls  # type: ignore[return-value]
    ak = getattr(ai_msg, "additional_kwargs", {}) or {}
    return ak.get("tool_calls", []) or []


def _fp(name: str, args: Dict[str, Any]) -> str:
    return json.dumps({"name": name, "args": args}, ensure_ascii=False, sort_keys=True)


def build_graph(
    llm_tools: Any,
    tool_map: Dict[str, Any],
    *,
    max_tool_hops: int = 32,
    repeat_limit: int = 6,
    verbose: bool = True,
):
    """
    [L02-08B NEW] 构建 ReAct 状态图：
      START -> think -> (tools -> think)* -> END

    [L02-08B FIX] 输出策略：
      - 普通对话：使用 think 产生的 AIMessage.content
      - tool-calling 且最终 submit=ACCEPT：使用 submit(args).final 作为用户可见输出
    """

    def think(state: AgentState) -> Dict[str, Any]:
        if state["stop"]:
            return {}

        if state["hops"] >= max_tool_hops:
            msg = "工具调用过多已停止（防循环）。"
            return {"stop": True, "stop_reason": msg, "last_reply": msg}

        if verbose:
            print("\n==[THINK]==")

        ai: AIMessage = llm_tools.invoke(state["messages"])

        if verbose:
            tc = _tool_calls_from(ai)
            print(f"[THINK] tool_calls={len(tc)}")

        # [L02-08B FIX] tool-calling 的 AIMessage 往往 content 为空；
        # 先把 content 存到 last_reply（若后续 submit=ACCEPT，会用 submit.final 覆盖）
        return {"messages": [ai], "last_reply": (ai.content or "")}

    def tools(state: AgentState) -> Dict[str, Any]:
        if state["stop"]:
            return {}

        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            msg = "状态异常：最后一条消息不是 AIMessage。"
            return {"stop": True, "stop_reason": msg, "last_reply": msg}

        tool_calls = _tool_calls_from(last)
        if not tool_calls:
            return {}

        if verbose:
            print("==[TOOLS]==")

        # [L02-08B NEW] repeat 拷贝更新，避免原地 mutation 难追踪
        repeat = dict(state.get("repeat", {}))

        out_msgs: List[BaseMessage] = []

        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            tool_call_id = call.get("id") or _fp(name, args)

            if name not in tool_map:
                out_msgs.append(
                    ToolMessage(content=f"(error) unknown tool: {name}", tool_name=name, tool_call_id=tool_call_id)
                )
                continue

            fingerprint = _fp(name, args)
            repeat[fingerprint] = repeat.get(fingerprint, 0) + 1
            if repeat[fingerprint] >= repeat_limit:
                msg = "检测到重复尝试相同动作（模型可能卡住）。请明确：继续/改方案/停止。"
                return {"stop": True, "stop_reason": msg, "last_reply": msg, "repeat": repeat}

            if verbose:
                print(f"[TOOLS] {name} args={args}")

            result = tool_map[name].invoke(args)
            out_msgs.append(ToolMessage(content=result, tool_name=name, tool_call_id=tool_call_id))

            # [L02-08B FIX] submit=ACCEPT 时，用户可见答案来自 submit 的 args["final"]
            if name == "submit":
                final_text = (args.get("final") or "").strip()
                obj = json.loads(result)  # core_runtime 保证 submit 返回 JSON
                if obj.get("status") == "ACCEPT":
                    return {
                        "messages": out_msgs,
                        "stop": True,
                        "stop_reason": "submit=ACCEPT",
                        "last_reply": final_text or "已完成（submit=ACCEPT）。",
                        "repeat": repeat,
                        "hops": state["hops"] + 1,
                    }

        return {"messages": out_msgs, "repeat": repeat, "hops": state["hops"] + 1}

    # [L02-08B NEW] 条件边：think 后是否进入 tools
    def after_think(state: AgentState) -> str:
        if state["stop"]:
            return "end"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and _tool_calls_from(last):
            return "tools"
        return "end"

    # [L02-08B NEW] 条件边：tools 后回 think 或结束
    def after_tools(state: AgentState) -> str:
        return "end" if state["stop"] else "think"

    # [L02-08B NEW] 组装图
    sg = StateGraph(AgentState)
    sg.add_node("think", think)
    sg.add_node("tools", tools)

    sg.add_edge(START, "think")
    sg.add_conditional_edges("think", after_think, {"tools": "tools", "end": END})
    sg.add_conditional_edges("tools", after_tools, {"think": "think", "end": END})

    return sg.compile()
