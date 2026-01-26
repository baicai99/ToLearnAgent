# -*- coding: utf-8 -*-
"""
执行代码：python -m lessonL03_02_safe_persist.main
标题：L03-02 LangGraph ReAct（持久化 + interrupt 审批 + 不持久化 ToolMessage）

[L03-02 NEW]
1) 持久化仅保存：Human/AI 文本消息 + tool_traces（可序列化），不保存 ToolMessage(role=tool)
   - 彻底规避“恢复后 tool 消息序列不合法 => OpenAI 400” citeturn0search5turn0search9
2) approval 节点纯粹 interrupt（无副作用），符合“resume 会从节点开头重跑”的语义 citeturn1search6turn1search0
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

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


def _latest_user_text(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""


def _should_use_tools(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if not t:
        return False

    greetings = ["你好", "hi", "hello", "在吗", "你是谁", "你是什么模型"]
    if any(g in t for g in greetings) and len(t) <= 12:
        return False

    keywords = [
        "toy_repo", "toyrepo", "repo", "仓库", "目录", "文件", ".py", "calc", "test", "pytest",
        "运行", "执行", "报错", "修复", "修改", "创建", "新增", "搜索", "读取", "review",
    ]
    return any(k in t for k in keywords)


def _extract_tool_calls(ai: AIMessage) -> List[Dict[str, Any]]:
    if getattr(ai, "tool_calls", None):
        return list(ai.tool_calls)  # type: ignore[arg-type]
    ak = getattr(ai, "additional_kwargs", {}) or {}
    tcs = ak.get("tool_calls") or []
    return list(tcs)


def _render_tool_trace_summary(traces: List[ToolTrace], max_items: int = 6) -> str:
    if not traces:
        return ""
    tail = traces[-max_items:]
    lines = ["最近工具结果摘要（用于上下文，不代表 tool 消息）："]
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

    def think(state: AgentState) -> Dict[str, Any]:
        user_text = _latest_user_text(state)
        logger.log("gate", {"user_text": user_text, "should_use_tools": _should_use_tools(user_text)})

        # 只持久化 Human/AI；如果有人误塞 ToolMessage，直接报错暴露问题（不兜底）
        for m in state.get("messages", []):
            if isinstance(m, ToolMessage):
                raise RuntimeError("状态中不允许出现 ToolMessage（本课要求：ToolMessage 不持久化）")

        sys_msgs: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        trace_summary = _render_tool_trace_summary(state.get("tool_traces", []))
        if trace_summary:
            sys_msgs.append(SystemMessage(content=trace_summary))

        base_msgs = sys_msgs + state.get("messages", [])

        # 如果不需要工具，直接一次回答（不绑定 tools）
        if not _should_use_tools(user_text):
            ai = llm.invoke(base_msgs)
            if not isinstance(ai, AIMessage):
                ai = AIMessage(content=str(ai))
            logger.log("llm_reply_no_tools", {"has_text": bool((ai.content or "").strip())})
            return {"messages": [ai]}

        # 需要工具：在“单次 turn 的内存里”完成 ReAct 循环
        traces: List[ToolTrace] = list(state.get("tool_traces", []))
        rejected = set(state.get("rejected_fingerprints", []))
        pending_patch: Optional[PatchPlan] = None

        MAX_STEPS = 8
        msgs_for_step = list(base_msgs)

        for step in range(1, MAX_STEPS + 1):
            ai: AIMessage = llm_tools.invoke(msgs_for_step)  # type: ignore[assignment]
            tcs = _extract_tool_calls(ai)
            logger.log("llm_step", {"step": step, "tool_calls": [tc.get("name") for tc in tcs], "has_text": bool((ai.content or "").strip())})

            if not tcs:
                # 无工具调用：完成
                final_text = (ai.content or "").strip()
                if not final_text:
                    final_text = "（模型未返回可见文本）"
                return {"messages": [AIMessage(content=final_text)], "tool_traces": traces}

            # 执行工具（ToolMessage 仅用于本次内存回合，绝不写入 state）
            tool_messages: List[ToolMessage] = []
            wrote_plan_this_step = False

            for tc in tcs:
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                tc_id = tc.get("id", "") or ""

                if name not in tool_map:
                    raise RuntimeError(f"unknown tool: {name}")

                result = tool_map[name].invoke(args)

                # tool 消息必须紧跟 tool_calls（这里在内存中满足 OpenAI 约束）citeturn0search5turn0search9
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc_id))

                prev = str(result)
                if len(prev) > 600:
                    prev = prev[:600] + "...(truncated)"
                traces.append({"name": name, "args": args, "result_preview": prev})
                logger.log("tool_ok", {"name": name})

                if name in ("propose_hunks", "propose_write_file"):
                    plan = json.loads(str(result))
                    fp = plan.get("fingerprint", "")
                    if fp and fp in rejected:
                        # 硬阻断：拒绝过就不再重复同补丁
                        msg = "你之前拒绝过同一个补丁（相同 fingerprint），我不会重复尝试。请你明确：改方案 / 允许写入 / 放弃。"
                        logger.log("write_blocked_rejected", {"fingerprint": fp, "file_path": plan.get("file_path")})
                        return {"messages": [AIMessage(content=msg)], "tool_traces": traces}

                    pending_patch = plan
                    wrote_plan_this_step = True
                    break

            if wrote_plan_this_step and pending_patch:
                policy = state.get("write_policy", write_policy)
                if policy == "deny":
                    fp = pending_patch.get("fingerprint", "")
                    logger.log("write_denied_policy", {"fingerprint": fp})
                    return {
                        "messages": [AIMessage(content="当前写入策略为 deny：我不会写入文件。我可以输出补丁供你手动应用。")],
                        "tool_traces": traces,
                        "pending_patch": None,
                    }
                if policy == "allow":
                    # 直接交给 commit 节点
                    return {"tool_traces": traces, "pending_patch": pending_patch}

                # ask：交给 approval 节点（本节点不调用 interrupt，避免副作用重跑）
                return {"tool_traces": traces, "pending_patch": pending_patch}

            # 继续下一步：把 ai(含 tool_calls) + tool_messages 加入内存消息（不持久化）
            msgs_for_step = msgs_for_step + [ai] + tool_messages

        # 超过步数：明确失败而非兜底
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

        # 注意：不要 try/except 包 interrupt（官方建议）citeturn1search0turn1search11
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

    # ---- build graph ----
    builder = StateGraph(AgentState)

    builder.add_node("think", think)
    builder.add_node("approval", approval)
    builder.add_node("commit", commit)
    builder.add_node("reject", reject)

    builder.add_edge(START, "think")

    def route_after_think(state: AgentState) -> str:
        plan = state.get("pending_patch")
        if not plan:
            return END
        policy = state.get("write_policy", write_policy)
        if policy == "allow":
            return "commit"
        if policy == "ask":
            return "approval"
        return END  # deny 已在 think 中处理并清掉 pending_patch

    builder.add_conditional_edges("think", route_after_think, {"commit": "commit", "approval": "approval", END: END})

    def route_after_approval(state: AgentState) -> str:
        a = (state.get("approval") or "").strip().lower()
        if a == "y":
            return "commit"
        return "reject"

    builder.add_conditional_edges("approval", route_after_approval, {"commit": "commit", "reject": "reject"})
    builder.add_edge("commit", END)
    builder.add_edge("reject", END)

    return builder
