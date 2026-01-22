# -*- coding: utf-8 -*-
"""
标题：L02-08D graph_agent.py —— Tool Profile + 线程级状态（checkpointer）+ interrupt 审批
执行代码：
  由 main.py build_graph(...) 后 graph.invoke(...)

学习点（本课新增）：
- [L02-08D NEW] tool_profile：chat / review_ro / full
- [L02-08D NEW] 工具白名单：不同 profile 下，工具调用不允许越权
- [L02-08D CHANGE] 不再把人工审批触发的 commit/reject 写成 ToolMessage(role=tool)
  （避免 OpenAI tool message 序列校验 400）:contentReference[oaicite:2]{index=2}
- interrupt/Command：等待审批并恢复执行 :contentReference[oaicite:3]{index=3}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict
from typing import Annotated

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command  # :contentReference[oaicite:4]{index=4}


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    # [L02-08D NEW] tool_profile：决定本轮 LLM 是否允许工具、允许哪些工具
    tool_profile: str  # "chat" | "review_ro" | "full"

    hops: int
    repeat: Dict[str, int]
    stop: bool
    stop_reason: str
    last_reply: str

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
    llm_plain: Any,
    llm_review_ro: Any,
    llm_full: Any,
    tool_map: Dict[str, Any],
    *,
    checkpointer: Any,
    max_tool_hops: int = 32,
    repeat_limit: int = 6,
):
    # [L02-08D NEW] 不同 profile 的工具白名单
    ALLOW_TOOLS: Dict[str, set[str]] = {
        "chat": set(),  # 不允许任何工具
        "review_ro": {
            "repo_tree",
            "list_files",
            "grep",
            "read_file_range",
            "evidence_read",
            "todoread",
        },
        "full": {
            "repo_tree",
            "list_files",
            "grep",
            "read_file_range",
            "write_file",
            "apply_hunks",
            "commit_patch",
            "reject_patch",
            "bash",
            "evidence_read",
            "todowrite",
            "todoread",
            "submit",
        },
    }

    def _pick_llm(profile: str):
        if profile == "chat":
            return llm_plain
        if profile == "review_ro":
            return llm_review_ro
        return llm_full

    def think(state: AgentState) -> Dict[str, Any]:
        if state.get("stop"):
            return {}

        hops = int(state.get("hops", 0))
        if hops >= max_tool_hops:
            msg = "工具调用过多已停止（防循环）。"
            return {"stop": True, "stop_reason": msg, "last_reply": msg}

        profile = (state.get("tool_profile") or "review_ro").strip()
        llm = _pick_llm(profile)

        ai: AIMessage = llm.invoke(state.get("messages", []))
        # tool-calling 时 content 可能为空；最终用户可见答复通常来自“最后一次 think 的 content”或 submit.final
        return {"messages": [ai], "last_reply": (ai.content or "")}

    def tools(state: AgentState) -> Dict[str, Any]:
        if state.get("stop"):
            return {}

        messages = state.get("messages", [])
        if not messages:
            msg = "状态异常：messages 为空。"
            return {"stop": True, "stop_reason": msg, "last_reply": msg}

        last = messages[-1]
        if not isinstance(last, AIMessage):
            msg = "状态异常：最后一条不是 AIMessage。"
            return {"stop": True, "stop_reason": msg, "last_reply": msg}

        tool_calls = _tool_calls_from(last)
        if not tool_calls:
            # 本轮没有工具调用：直接结束（由 main 打印 last_reply）
            return {"stop": True, "stop_reason": "NO_TOOL_CALLS"}

        profile = (state.get("tool_profile") or "review_ro").strip()
        allowed = ALLOW_TOOLS.get(profile, ALLOW_TOOLS["review_ro"])

        repeat = dict(state.get("repeat", {}))
        out_msgs: List[BaseMessage] = []

        for call in tool_calls:
            name = (call.get("name") or "").strip()
            args = call.get("args", {}) or {}
            tool_call_id = call.get("id") or _fp(name, args)  # 保障 tool_call_id 稳定

            # [L02-08D NEW] 防止 pydantic 在 invoke 前因类型不对崩溃：先做显式校验
            if not isinstance(args, dict):
                out_msgs.append(
                    ToolMessage(
                        content="(error) tool args must be an object/dict",
                        tool_name=name or "unknown",
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            # [L02-08D NEW] 工具越权拦截
            if name not in allowed:
                out_msgs.append(
                    ToolMessage(
                        content=f"(error) tool not allowed in profile={profile}: {name}",
                        tool_name=name or "unknown",
                        tool_call_id=tool_call_id,
                    )
                )
                continue

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
            out_msgs.append(
                ToolMessage(content=result, tool_name=name, tool_call_id=tool_call_id)
            )

            obj = _try_json(result)

            # PREVIEW：进入审批
            if obj and obj.get("status") == "PREVIEW":
                return {
                    "messages": out_msgs,
                    "repeat": repeat,
                    "hops": int(state.get("hops", 0)) + 1,
                    "pending_patch_id": obj.get("patch_id"),
                    "pending_diff": obj.get("diff"),
                    "pending_tool": name,
                    "stop": False,
                }

            # submit=ACCEPT：结束并用 submit.final 作为用户可见输出
            if name == "submit" and obj and obj.get("status") == "ACCEPT":
                final_text = (args.get("final") or "").strip()
                return {
                    "messages": out_msgs,
                    "repeat": repeat,
                    "hops": int(state.get("hops", 0)) + 1,
                    "stop": True,
                    "stop_reason": "submit=ACCEPT",
                    "last_reply": final_text or "已完成（submit=ACCEPT）。",
                }

        return {
            "messages": out_msgs,
            "repeat": repeat,
            "hops": int(state.get("hops", 0)) + 1,
        }

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

        decision = interrupt(payload)  # :contentReference[oaicite:5]{index=5}
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
        )

    def commit(state: AgentState) -> Dict[str, Any]:
        pid = (state.get("pending_patch_id") or "").strip()
        if not pid:
            msg = "状态异常：pending_patch_id 为空，无法提交。"
            return {"stop": True, "stop_reason": msg, "last_reply": msg}

        result = tool_map["commit_patch"].invoke({"patch_id": pid})
        obj = _try_json(result) or {}
        file = obj.get("file")
        status = obj.get("status")

        # [L02-08D CHANGE] 人工审批触发的结果写成 AIMessage（避免 role=tool 的序列约束）:contentReference[oaicite:6]{index=6}
        note = (
            f"[human-approved] commit_patch status={status} patch_id={pid} file={file}"
        )
        return {
            "messages": [AIMessage(content=note)],
            "pending_patch_id": None,
            "pending_diff": None,
            "pending_tool": None,
            "last_reply": "补丁已提交。",
        }

    def reject(state: AgentState) -> Dict[str, Any]:
        pid = (state.get("pending_patch_id") or "").strip()
        if not pid:
            msg = "状态异常：pending_patch_id 为空，无法拒绝。"
            return {"stop": True, "stop_reason": msg, "last_reply": msg}

        result = tool_map["reject_patch"].invoke({"patch_id": pid})
        obj = _try_json(result) or {}
        status = obj.get("status")

        note = f"[human-rejected] reject_patch status={status} patch_id={pid}"
        # 拒绝后结束本轮，避免继续重试形成循环
        return {
            "messages": [AIMessage(content=note)],
            "pending_patch_id": None,
            "pending_diff": None,
            "pending_tool": None,
            "stop": True,
            "stop_reason": "USER_REJECTED_PATCH",
            "last_reply": "已拒绝该补丁。本轮停止，请你指示下一步。",
        }

    def route_after_think(state: AgentState) -> str:
        # chat profile 直接结束（不跑 tools）
        profile = (state.get("tool_profile") or "review_ro").strip()
        if profile == "chat":
            return "end"
        return "tools"

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
    sg.add_conditional_edges("think", route_after_think, {"tools": "tools", "end": END})
    sg.add_conditional_edges(
        "tools", route_after_tools, {"approve": "approve", "think": "think", "end": END}
    )

    sg.add_edge("commit", "think")
    sg.add_edge("reject", END)

    # 线程级持久化：必须提供 thread_id :contentReference[oaicite:7]{index=7}
    return sg.compile(checkpointer=checkpointer)
