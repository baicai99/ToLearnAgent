# -*- coding: utf-8 -*-
"""
标题：L02-08C graph_agent.py —— interrupt 审批节点（写入确认上移到图层）
执行代码：
  由 main.py 调用 build_graph(...) 后 graph.invoke(...)

要点：
- interrupt() 会让图暂停，并把 payload 通过 __interrupt__ 返回给调用方；
- 用 Command(resume=...) 继续执行，resume 值会成为 interrupt() 的返回值。:contentReference[oaicite:2]{index=2}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict
from typing import Annotated

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# [L02-08C NEW] interrupt + Command（官方推荐的人类介入方式）:contentReference[oaicite:3]{index=3}
from langgraph.types import interrupt, Command


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    hops: int
    repeat: Dict[str, int]
    stop: bool
    stop_reason: str
    last_reply: str

    # [L02-08C NEW] 待审批补丁信息（来自工具 PREVIEW）
    pending_patch_id: Optional[str]
    pending_diff: Optional[str]
    pending_tool: Optional[str]


def _tool_calls_from(ai_msg: AIMessage) -> List[Dict[str, Any]]:
    if getattr(ai_msg, "tool_calls", None):
        return ai_msg.tool_calls  # type: ignore[return-value]
    ak = getattr(ai_msg, "additional_kwargs", {}) or {}
    return ak.get("tool_calls", []) or []


def _fp(name: str, args: Dict[str, Any]) -> str:
    return json.dumps({"name": name, "args": args}, ensure_ascii=False, sort_keys=True)


def _try_json(s: str) -> Optional[Dict[str, Any]]:
    ss = (s or "").strip()
    if not ss.startswith("{"):
        return None
    try:
        obj = json.loads(ss)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def build_graph(
    llm_tools: Any,
    tool_map: Dict[str, Any],
    *,
    checkpointer: Any,
    max_tool_hops: int = 32,
    repeat_limit: int = 6,
    verbose: bool = False,
):
    """
    图结构：
      START -> think -> tools -> (approve -> commit/reject) -> think ... -> END

    关键：
      - write_file/apply_hunks 在 ask 策略下返回 PREVIEW
      - tools 节点检测 PREVIEW，把 patch_id/diff 写入 state 并路由到 approve
      - approve 节点 interrupt 等待人类审批
    """

    def think(state: AgentState) -> Dict[str, Any]:
        if state.get("stop"):
            return {}

        hops = int(state.get("hops", 0))
        if hops >= max_tool_hops:
            msg = "工具调用过多已停止（防循环）。"
            return {"stop": True, "stop_reason": msg, "last_reply": msg}

        if verbose:
            print("\n==[THINK]==")

        ai: AIMessage = llm_tools.invoke(state.get("messages", []))

        # tool-calling 场景 content 可能为空；last_reply 主要由 submit.final 覆盖
        return {"messages": [ai], "last_reply": (ai.content or "")}

    def tools(state: AgentState) -> Dict[str, Any]:
        if state.get("stop"):
            return {}

        messages = state.get("messages", [])
        if not messages:
            return {
                "stop": True,
                "stop_reason": "状态异常：messages 为空。",
                "last_reply": "状态异常：messages 为空。",
            }

        last = messages[-1]
        if not isinstance(last, AIMessage):
            return {
                "stop": True,
                "stop_reason": "状态异常：最后一条不是 AIMessage。",
                "last_reply": "状态异常：最后一条不是 AIMessage。",
            }

        tool_calls = _tool_calls_from(last)
        if not tool_calls:
            # 没有工具调用则结束（本轮由 main 直接输出 last_reply）
            return {"stop": True, "stop_reason": "no_tool_calls"}

        if verbose:
            print("==[TOOLS]==")

        repeat = dict(state.get("repeat", {}))
        out_msgs: List[BaseMessage] = []

        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            tool_call_id = call.get("id") or _fp(name, args)

            if name not in tool_map:
                out_msgs.append(
                    ToolMessage(
                        content=f"(error) unknown tool: {name}",
                        tool_name=name,
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            fingerprint = _fp(name, args)
            repeat[fingerprint] = repeat.get(fingerprint, 0) + 1
            if repeat[fingerprint] >= repeat_limit:
                msg = (
                    "检测到重复尝试相同动作（模型可能卡住）。请明确：继续/改方案/停止。"
                )
                return {
                    "messages": out_msgs,
                    "repeat": repeat,
                    "stop": True,
                    "stop_reason": msg,
                    "last_reply": msg,
                }

            result = tool_map[name].invoke(args)
            tm = ToolMessage(content=result, tool_name=name, tool_call_id=tool_call_id)
            out_msgs.append(tm)

            # [L02-08C NEW] 捕获 PREVIEW，进入审批节点（不再让工具层 input）
            obj = _try_json(result)
            if obj and obj.get("status") == "PREVIEW":
                return {
                    "messages": out_msgs,
                    "repeat": repeat,
                    "hops": int(state.get("hops", 0)) + 1,
                    "pending_patch_id": obj.get("patch_id"),
                    "pending_diff": obj.get("diff"),
                    "pending_tool": name,
                    "stop": False,  # 不停止，路由到 approve
                }

            # submit=ACCEPT：结束
            if name == "submit":
                if obj and obj.get("status") == "ACCEPT":
                    final_text = (args.get("final") or "").strip()
                    return {
                        "messages": out_msgs,
                        "repeat": repeat,
                        "hops": int(state.get("hops", 0)) + 1,
                        "stop": True,
                        "stop_reason": "submit=ACCEPT",
                        "last_reply": final_text or "已完成（submit=ACCEPT）。",
                    }

        # 没有 PREVIEW 且没有 submit=ACCEPT：继续 think
        return {
            "messages": out_msgs,
            "repeat": repeat,
            "hops": int(state.get("hops", 0)) + 1,
        }

    # [L02-08C NEW] 审批节点：interrupt 等人类输入（y/n -> True/False）
    def approve(state: AgentState):
        pid = (state.get("pending_patch_id") or "").strip()
        diff = state.get("pending_diff") or ""
        tool_name = state.get("pending_tool") or "write"

        payload = {
            "type": "approve_patch",
            "patch_id": pid,
            "tool": tool_name,
            "diff": diff,
            "prompt": "是否允许写入该补丁？(y/n)",
        }

        decision = interrupt(
            payload
        )  # resume 值会成为 decision :contentReference[oaicite:4]{index=4}
        approved = bool(decision)

        if approved:
            return Command(
                goto="commit", update={"last_reply": "已批准写入，正在提交补丁..."}
            )
        return Command(
            goto="reject",
            update={
                "last_reply": "已拒绝写入：我不会重复尝试同一补丁。请你指示下一步。"
            },
        )  # 防止循环

    def commit(state: AgentState) -> Dict[str, Any]:
        pid = state.get("pending_patch_id")
        if not pid:
            msg = "状态异常：pending_patch_id 为空，无法提交。"
            return {"stop": True, "stop_reason": msg, "last_reply": msg}

        # 人类批准触发的“系统工具调用”（不依赖 LLM tool_call）
        result = tool_map["commit_patch"].invoke({"patch_id": pid})
        event = AIMessage(content=f"[system] commit_patch => {result}")

        # 清理 pending_*，然后回到 think
        return {
            "messages": [event],
            "pending_patch_id": None,
            "pending_diff": None,
            "pending_tool": None,
        }

    def reject(state: AgentState) -> Dict[str, Any]:
        pid = state.get("pending_patch_id")
        if not pid:
            msg = "状态异常：pending_patch_id 为空，无法拒绝。"
            return {"stop": True, "stop_reason": msg, "last_reply": msg}

        result = tool_map["reject_patch"].invoke({"patch_id": pid})
        event = AIMessage(content=f"[system] reject_patch => {result}")

        # [关键] 拒绝后本轮直接结束，避免模型继续“重试同样补丁”形成循环
        return {
            "messages": [event],
            "pending_patch_id": None,
            "pending_diff": None,
            "pending_tool": None,
            "stop": True,
            "stop_reason": "USER_REJECTED_PATCH",
        }

    def route_after_tools(state: AgentState) -> str:
        if state.get("pending_patch_id"):
            return "approve"
        if state.get("stop"):
            return "end"
        return "think"

    sg = StateGraph(AgentState)
    sg.add_node("think", think)
    sg.add_node("tools", tools)
    sg.add_node("approve", approve)
    sg.add_node("commit", commit)
    sg.add_node("reject", reject)

    sg.add_edge(START, "think")
    sg.add_edge("think", "tools")
    sg.add_conditional_edges(
        "tools", route_after_tools, {"approve": "approve", "think": "think", "end": END}
    )

    sg.add_edge("commit", "think")
    sg.add_edge("reject", END)

    # [L02-08C NEW] 必须带 checkpointer 才能用 interrupt :contentReference[oaicite:5]{index=5}
    return sg.compile(checkpointer=checkpointer)
