# -*- coding: utf-8 -*-
"""
标题：L03-03 LangGraph（无关键词路由的 ReAct）
执行：python -m lessonL03_03_routerless_react.main

[L03-03 NEW]
- 删除“关键词 router”：不再预测任务类型
- 唯一依据：LLM 是否输出 tool_calls
- ToolMessage 只在单回合内存里使用，绝不写入持久化 state（避免恢复后 OpenAI 400）
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Literal, TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from .prompts import SYSTEM_PROMPT
from .event_log import EventLogger
from .core_runtime import Runtime, apply_hunks_to_text, compute_base_sha
from . import tools as tool_mod

WritePolicy = Literal["allow", "ask", "deny"]


class PatchPlan(TypedDict, total=False):
    status: str
    patch_id: str
    file_path: str
    base_sha: str
    fingerprint: str
    diff: str
    mode: str
    hunks: List[Dict[str, Any]]
    content: str


class ToolTrace(TypedDict, total=False):
    name: str
    args: Dict[str, Any]
    result_preview: str


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], ...]
    write_policy: WritePolicy
    pending_patch: Optional[PatchPlan]
    rejected_fingerprints: List[str]
    tool_traces: List[ToolTrace]
    approval: Optional[str]


def _add_messages(existing: List[BaseMessage], new: List[BaseMessage]) -> List[BaseMessage]:
    return existing + new


def _extract_tool_calls(ai: AIMessage) -> List[Dict[str, Any]]:
    if getattr(ai, "tool_calls", None):
        return list(ai.tool_calls)  # type: ignore[arg-type]
    ak = getattr(ai, "additional_kwargs", {}) or {}
    tcs = ak.get("tool_calls") or []
    return list(tcs)


def _latest_user_text(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""


def _trace_summary(traces: List[ToolTrace], max_items: int = 6) -> str:
    if not traces:
        return ""
    tail = traces[-max_items:]
    lines = ["最近工具结果摘要（用于上下文，不是 tool 消息）："]
    for i, tr in enumerate(tail, start=1):
        name = tr.get("name", "")
        prev = (tr.get("result_preview", "") or "").strip()
        if len(prev) > 300:
            prev = prev[:300] + "...(truncated)"
        lines.append(f"{i}. {name}: {prev}")
    return "\n".join(lines)


def build_graph(
    *,
    rt: Runtime,
    logger: EventLogger,
    api_key: str,
    base_url: str,
    model: str,
    write_policy: WritePolicy,
):
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=0,
    )

    tool_fns = [
        tool_mod.list_files,
        tool_mod.read_file_range,
        tool_mod.grep,
        tool_mod.bash,
        tool_mod.propose_hunks,
        tool_mod.propose_write_file,
    ]
    tool_mod.attach_runtime(rt, tool_fns)
    tool_map = {t.name: t for t in tool_fns}

    llm_tools = llm.bind_tools(tool_fns)

    def react_turn(state: AgentState) -> Dict[str, Any]:
        """
        单次用户输入的 ReAct 回合：
        - 让 LLM 自主决定是否调用工具
        - 如果调用工具：执行并回送结果，最多迭代 MAX_STEPS
        - 若生成补丁计划：写入 pending_patch，交给审批/提交节点处理
        """
        # [L03-03 NEW] 硬约束：持久化 state 里不允许 ToolMessage
        for m in state.get("messages", []):
            if isinstance(m, ToolMessage):
                raise RuntimeError("状态中不允许出现 ToolMessage（本课要求：ToolMessage 不持久化）")

        user_text = _latest_user_text(state)
        logger.log("user", {"text": user_text})

        traces: List[ToolTrace] = list(state.get("tool_traces", []))
        rejected = set(state.get("rejected_fingerprints", []))

        sys_msgs: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        summary = _trace_summary(traces)
        if summary:
            sys_msgs.append(SystemMessage(content=summary))

        msgs_for_step: List[BaseMessage] = sys_msgs + state.get("messages", [])

        MAX_STEPS = 8
        pending_patch: Optional[PatchPlan] = None

        for step in range(1, MAX_STEPS + 1):
            ai: AIMessage = llm_tools.invoke(msgs_for_step)  # type: ignore[assignment]
            tcs = _extract_tool_calls(ai)

            logger.log("llm", {"step": step, "tool_calls": [tc.get("name") for tc in tcs], "has_text": bool((ai.content or "").strip())})

            if not tcs:
                # 不需要工具：直接回复（本课核心：由 LLM 自己决定）
                text = (ai.content or "").strip()
                if not text:
                    text = "（模型未返回可见文本）"
                return {"messages": [AIMessage(content=text)], "tool_traces": traces}

            tool_messages: List[ToolMessage] = []
            produced_patch = False

            for tc in tcs:
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                tc_id = tc.get("id", "") or ""

                if name not in tool_map:
                    raise RuntimeError(f"unknown tool: {name}")

                result = tool_map[name].invoke(args)

                # tool 消息只在内存回合里出现，且严格对应 tool_call_id
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc_id))

                prev = str(result)
                if len(prev) > 600:
                    prev = prev[:600] + "...(truncated)"
                traces.append({"name": name, "args": args, "result_preview": prev})
                logger.log("tool", {"name": name})

                # 捕捉写入提案
                if name in ("propose_hunks", "propose_write_file"):
                    plan = json.loads(str(result))
                    fp = plan.get("fingerprint", "")
                    if fp and fp in rejected:
                        logger.log("blocked_rejected", {"fingerprint": fp, "file_path": plan.get("file_path")})
                        return {
                            "messages": [AIMessage(content="你之前拒绝过同一个补丁（相同 fingerprint），我不会重复尝试。请你明确：改方案 / 允许写入 / 放弃。")],
                            "tool_traces": traces,
                        }
                    pending_patch = plan
                    produced_patch = True
                    break

            if produced_patch and pending_patch:
                return {"tool_traces": traces, "pending_patch": pending_patch}

            # 继续下一步：把 ai(含 tool_calls) + tool_messages 加入内存上下文（不持久化）
            msgs_for_step = msgs_for_step + [ai] + tool_messages

        return {"messages": [AIMessage(content="达到本回合最大 ReAct 步数限制，仍未完成。请你缩小任务或明确下一步。")], "tool_traces": traces}

    def approval(state: AgentState) -> Dict[str, Any]:
        plan = state.get("pending_patch")
        if not plan:
            return {"approval": None}

        preview = (
            "\n========== [WRITE PREVIEW / 等待审批] ==========\n"
            f"{plan.get('diff','')}\n"
            "===============================================\n"
            "是否允许写入该补丁？(y/n)"
        )
        logger.log("write_preview", {"file_path": plan.get("file_path"), "fingerprint": plan.get("fingerprint")})

        ans = interrupt(preview)
        return {"approval": str(ans).strip().lower()}

    def commit(state: AgentState) -> Dict[str, Any]:
        plan = state.get("pending_patch")
        if not plan:
            return {"messages": [AIMessage(content="(commit) 没有待提交补丁。")], "pending_patch": None}

        file_path = plan.get("file_path", "")
        base_sha = plan.get("base_sha", "")
        mode = plan.get("mode", "")
        fp = plan.get("fingerprint", "")

        p = rt.abs_path(file_path)
        old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        cur_sha = compute_base_sha(old)

        if base_sha and cur_sha != base_sha:
            logger.log("commit_conflict", {"file_path": file_path, "fingerprint": fp})
            return {
                "pending_patch": None,
                "messages": [AIMessage(content="审批期间文件发生变化（base_sha 不一致），为避免误写，我已中止提交。请重新发起修改。")],
            }

        if mode == "write_file":
            rt.write_text(file_path, plan.get("content", ""))
        else:
            hunks = plan.get("hunks", []) or []
            new = apply_hunks_to_text(old, hunks)
            rt.write_text(file_path, new)

        logger.log("commit_ok", {"file_path": file_path, "fingerprint": fp})
        return {"pending_patch": None, "messages": [AIMessage(content=f"已写入：{file_path}（fingerprint={fp[:10]}...）")]}

    def reject(state: AgentState) -> Dict[str, Any]:
        plan = state.get("pending_patch")
        fp = plan.get("fingerprint", "") if plan else ""
        rej = list(state.get("rejected_fingerprints", []))
        if fp and fp not in rej:
            rej.append(fp)

        logger.log("reject", {"fingerprint": fp})
        return {
            "pending_patch": None,
            "rejected_fingerprints": rej,
            "messages": [AIMessage(content="你已拒绝写入该补丁。我不会重复尝试同一修改。请你选择：改方案 / 允许写入 / 放弃。")],
        }

    builder = StateGraph(AgentState)

    builder.add_node("react_turn", react_turn)
    builder.add_node("approval", approval)
    builder.add_node("commit", commit)
    builder.add_node("reject", reject)

    builder.add_edge(START, "react_turn")

    def route_after_react(state: AgentState) -> str:
        plan = state.get("pending_patch")
        if not plan:
            return END
        policy = state.get("write_policy", write_policy)
        if policy == "allow":
            return "commit"
        if policy == "ask":
            return "approval"
        # deny：这里直接结束（你也可以改为给出“仅输出补丁”的分支）
        return END

    builder.add_conditional_edges("react_turn", route_after_react, {"commit": "commit", "approval": "approval", END: END})

    def route_after_approval(state: AgentState) -> str:
        a = (state.get("approval") or "").strip().lower()
        if a == "y":
            return "commit"
        return "reject"

    builder.add_conditional_edges("approval", route_after_approval, {"commit": "commit", "reject": "reject"})
    builder.add_edge("commit", END)
    builder.add_edge("reject", END)

    return builder
