# -*- coding: utf-8 -*-
"""
标题：L02-08F graph_agent.py —— Tool Gating + 参数最小充分 + 永不(no reply)
执行代码：
  由 main.py build_graph(...) 后 graph.invoke(...)

本课新增学习内容：
- [L02-08F NEW] 统一意图：router 内调用 intent.classify/pick_tool_profile，决定 tool_profile
- [L02-08F NEW] 工具参数 sanitize：对空 dir/path 与默认值做“契约级归一化”，避免崩溃
- [L02-08F NEW] 工具调用被禁止时：不执行工具，直接用自然语言要求澄清/改写目标
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict
from typing import Annotated

from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    ToolMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command

import intent as it


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    tool_profile: str  # "chat" | "review_ro" | "full"
    tool_profile_reason: str

    hops: int
    repeat: Dict[str, int]
    stop: bool
    stop_reason: str
    last_reply: str

    pending_patch_id: Optional[str]
    pending_diff: Optional[str]
    pending_tool: Optional[str]

    # Task Session fields（沿用）
    session_active: bool
    objective: str
    constraints: List[str]
    done_criteria: List[str]
    progress: List[str]
    open_questions: List[str]


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


def _extract_user_text_from_last_human(messages: List[BaseMessage]) -> str:
    if not messages or not isinstance(messages[-1], HumanMessage):
        return ""
    s = messages[-1].content or ""
    if "]" in s and s.startswith("[task_type="):
        return s.split("]", 1)[1].strip()
    return s.strip()


def _make_session_context(state: AgentState) -> str:
    active = bool(state.get("session_active", False))
    if not active:
        return "当前没有激活的任务会话。若用户在做明确编码任务，请先明确 objective（可用 new task: ...）。"

    objective = (state.get("objective") or "").strip()
    constraints = state.get("constraints") or []
    done_criteria = state.get("done_criteria") or []
    progress = state.get("progress") or []
    open_q = state.get("open_questions") or []

    def fmt_list(xs: List[str], limit: int = 6) -> str:
        xs2 = [x.strip() for x in xs if x and x.strip()]
        if not xs2:
            return "(空)"
        xs2 = xs2[-limit:]
        return "\n".join([f"- {x}" for x in xs2])

    return (
        "【Task Session】\n"
        f"objective: {objective or '(空)'}\n"
        f"constraints:\n{fmt_list(constraints)}\n"
        f"done_criteria:\n{fmt_list(done_criteria)}\n"
        f"progress(last):\n{fmt_list(progress)}\n"
        f"open_questions:\n{fmt_list(open_q)}\n"
        "规则：每一步必须服务于 objective；若信息不足，先用只读工具取证据或向用户提问。"
    )


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
    ALLOW_TOOLS: Dict[str, set[str]] = {
        "chat": set(),
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
        },
    }

    def _pick_llm(profile: str):
        if profile == "chat":
            return llm_plain
        if profile == "review_ro":
            return llm_review_ro
        return llm_full

    # router：同时处理 Task Session 命令 + 本课新增 tool gating
    def router(state: AgentState) -> Dict[str, Any]:
        msgs = state.get("messages", [])
        user_text = _extract_user_text_from_last_human(msgs)
        sl = user_text.lower().strip()

        # session fields init
        session_active = bool(state.get("session_active", False))
        objective = (state.get("objective") or "").strip()
        constraints = list(state.get("constraints") or [])
        done_criteria = list(state.get("done_criteria") or [])
        progress = list(state.get("progress") or [])
        open_q = list(state.get("open_questions") or [])

        def stop_with(reply: str, **updates: Any) -> Dict[str, Any]:
            upd = {"stop": True, "stop_reason": "ROUTER_STOP", "last_reply": reply}
            upd.update(updates)
            return upd

        # 控制命令：不触发工具
        if sl in ("help", "/help", "?"):
            reply = (
                "可用命令：\n"
                "- new task: <目标>\n"
                "- status\n"
                "- done\n"
                "- abort\n"
            )
            return stop_with(reply, tool_profile="chat", tool_profile_reason="help")

        if (
            sl.startswith("new task")
            or sl.startswith("newtask")
            or sl.startswith("任务:")
            or sl.startswith("任务：")
        ):
            if ":" in user_text:
                obj = user_text.split(":", 1)[1].strip()
            elif "：" in user_text:
                obj = user_text.split("：", 1)[1].strip()
            else:
                obj = user_text.replace("new task", "").replace("newtask", "").strip()

            session_active = True
            objective = obj or "(未提供目标)"
            constraints = []
            done_criteria = [
                "代码变更已通过写入审批（若有写入）",
                "若涉及测试，则 pytest 通过（returncode=0）",
                "最终解释清楚：做了什么、依据是什么、下一步是什么",
            ]
            progress = [f"开启新任务：{objective}"]
            open_q = [] if obj else ["请补充 objective：你要实现/修复/审阅什么？"]

            return stop_with(
                f"已开启任务会话。\nobjective: {objective}\n接下来直接说需求即可。",
                tool_profile="chat",
                tool_profile_reason="new task command",
                session_active=session_active,
                objective=objective,
                constraints=constraints,
                done_criteria=done_criteria,
                progress=progress,
                open_questions=open_q,
            )

        if sl in ("status", "/status"):
            if not session_active:
                return stop_with(
                    "当前没有激活的任务会话。用 new task: <目标> 开启。",
                    tool_profile="chat",
                    tool_profile_reason="status",
                )
            reply = (
                "【Session Status】\n"
                f"objective: {objective}\n"
                f"constraints: {constraints if constraints else '(空)'}\n"
                f"done_criteria: {done_criteria if done_criteria else '(空)'}\n"
                f"progress(last 6): {(progress[-6:] if progress else '(空)')}\n"
                f"open_questions: {open_q if open_q else '(空)'}"
            )
            return stop_with(
                reply,
                tool_profile="chat",
                tool_profile_reason="status",
                session_active=session_active,
                objective=objective,
                constraints=constraints,
                done_criteria=done_criteria,
                progress=progress,
                open_questions=open_q,
            )

        if sl in ("abort", "/abort"):
            return stop_with(
                "已放弃当前任务会话（session 已清空）。",
                tool_profile="chat",
                tool_profile_reason="abort",
                session_active=False,
                objective="",
                constraints=[],
                done_criteria=[],
                progress=[],
                open_questions=[],
            )

        if sl in ("done", "/done"):
            if not session_active:
                return stop_with(
                    "当前没有激活任务会话，无需 done。",
                    tool_profile="chat",
                    tool_profile_reason="done",
                )
            summary = (
                "【任务结束】\n"
                f"objective: {objective}\n"
                "progress(last 10):\n"
                + "\n".join([f"- {x}" for x in (progress[-10:] if progress else [])])
                + ("\n(无记录)" if not progress else "")
            )
            return stop_with(
                summary,
                tool_profile="chat",
                tool_profile_reason="done",
                session_active=False,
                objective="",
                constraints=[],
                done_criteria=[],
                progress=[],
                open_questions=[],
            )

        # 自动开启 session（维持 L02-08E 行为）
        if not session_active and user_text and len(user_text) >= 6:
            session_active = True
            objective = user_text
            constraints = []
            done_criteria = [
                "代码变更已通过写入审批（若有写入）",
                "若涉及测试，则 pytest 通过（returncode=0）",
                "最终解释清楚：做了什么、依据是什么、下一步是什么",
            ]
            progress = [f"自动开启任务：{objective}"]
            open_q = []

        # [L02-08F NEW] tool gating：根据 user_text 决定 tool_profile（统一逻辑）
        task_type = it.classify_task_type(user_text)
        profile, reason = it.pick_tool_profile(task_type, user_text)

        return {
            "session_active": session_active,
            "objective": objective,
            "constraints": constraints,
            "done_criteria": done_criteria,
            "progress": progress,
            "open_questions": open_q,
            "tool_profile": profile,
            "tool_profile_reason": reason,
            "stop": False,
            "stop_reason": "",
        }

    def think(state: AgentState) -> Dict[str, Any]:
        if state.get("stop"):
            return {}

        hops = int(state.get("hops", 0))
        if hops >= max_tool_hops:
            msg = "工具调用过多已停止（防循环）。"
            return {"stop": True, "stop_reason": msg, "last_reply": msg}

        profile = (state.get("tool_profile") or "review_ro").strip()
        llm = _pick_llm(profile)

        # session 上下文注入（不写入历史）
        ctx = _make_session_context(state)
        ctx_msg = SystemMessage(content=ctx)

        msgs = state.get("messages", [])
        ai: AIMessage = llm.invoke([ctx_msg] + msgs)

        # [L02-08F NEW] 绝不允许 last_reply 为空：先用 ai.content，否则用最小提示
        reply = (ai.content or "").strip()
        if not reply:
            reply = (
                "我已收到你的输入。请补充更具体的目标或允许我使用只读工具查看工作区。"
            )

        return {"messages": [ai], "last_reply": reply}

    def tools(state: AgentState) -> Dict[str, Any]:
        if state.get("stop"):
            return {}

        profile = (state.get("tool_profile") or "review_ro").strip()
        allowed = ALLOW_TOOLS.get(profile, ALLOW_TOOLS["review_ro"])

        # 如果 profile=chat，理论上不会到这里；但仍做“硬门控”，避免误跑
        if profile == "chat":
            msg = "当前为 chat 模式：不允许调用工具。若要查看/修改代码，请明确需求（review/create/tests/implement）。"
            return {"stop": True, "stop_reason": "TOOL_BLOCKED_CHAT", "last_reply": msg}

        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            msg = "状态异常：最后一条不是 AIMessage。"
            return {"stop": True, "stop_reason": msg, "last_reply": msg}

        tool_calls = _tool_calls_from(messages[-1])
        if not tool_calls:
            # 没有工具调用：结束本轮（reply 已由 think 给出）
            return {"stop": True, "stop_reason": "NO_TOOL_CALLS"}

        repeat = dict(state.get("repeat", {}))
        out_msgs: List[BaseMessage] = []
        progress = list(state.get("progress") or [])

        for call in tool_calls:
            name = (call.get("name") or "").strip()
            raw_args = call.get("args", {}) or {}
            tool_call_id = call.get("id") or _fp(
                name, raw_args if isinstance(raw_args, dict) else {}
            )

            if name not in allowed:
                out_msgs.append(
                    ToolMessage(
                        content=f"(error) tool not allowed in profile={profile}: {name}",
                        tool_name=name,
                        tool_call_id=tool_call_id,
                    )
                )
                progress.append(f"工具被门控拒绝：{name} profile={profile}")
                continue

            if name not in tool_map:
                out_msgs.append(
                    ToolMessage(
                        content=f"(error) unknown tool: {name}",
                        tool_name=name,
                        tool_call_id=tool_call_id,
                    )
                )
                progress.append(f"未知工具：{name}")
                continue

            # [L02-08F NEW] 参数 sanitize：把“空 dir/path/缺省”归一化，避免缺参/空参崩溃
            ok, args, note = it.sanitize_tool_call(name, raw_args)
            if not ok:
                out_msgs.append(
                    ToolMessage(
                        content=f"(error) {note}",
                        tool_name=name,
                        tool_call_id=tool_call_id,
                    )
                )
                progress.append(f"工具参数非法：{name} note={note}")
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
                    "progress": progress,
                }

            result = tool_map[name].invoke(args)
            out_msgs.append(
                ToolMessage(content=result, tool_name=name, tool_call_id=tool_call_id)
            )
            progress.append(f"已调用工具：{name}（{note}）")

            obj = _try_json(result)
            if obj and isinstance(obj.get("status"), str):
                st = obj["status"]
                if st == "PREVIEW":
                    progress.append(
                        f"生成写入预览：tool={name} file={obj.get('file')} patch_id={obj.get('patch_id')}"
                    )
                    return {
                        "messages": out_msgs,
                        "repeat": repeat,
                        "hops": int(state.get("hops", 0)) + 1,
                        "pending_patch_id": obj.get("patch_id"),
                        "pending_diff": obj.get("diff"),
                        "pending_tool": name,
                        "stop": False,
                        "progress": progress,
                    }

        return {
            "messages": out_msgs,
            "repeat": repeat,
            "hops": int(state.get("hops", 0)) + 1,
            "progress": progress,
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
        decision = interrupt(payload)
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

        note = (
            f"[human-approved] commit_patch status={status} patch_id={pid} file={file}"
        )
        progress = list(state.get("progress") or [])
        progress.append(f"已提交补丁：file={file} patch_id={pid}")

        return {
            "messages": [AIMessage(content=note)],
            "pending_patch_id": None,
            "pending_diff": None,
            "pending_tool": None,
            "last_reply": "补丁已提交。你可以继续下一步（例如运行 pytest）或输入 status 查看进度。",
            "progress": progress,
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
        progress = list(state.get("progress") or [])
        progress.append(f"用户拒绝补丁：patch_id={pid}")

        return {
            "messages": [AIMessage(content=note)],
            "pending_patch_id": None,
            "pending_diff": None,
            "pending_tool": None,
            "stop": True,
            "stop_reason": "USER_REJECTED_PATCH",
            "last_reply": "已拒绝该补丁。本轮停止，请你指示下一步（改方案/放弃/允许写入）。",
            "progress": progress,
        }

    def route_after_router(state: AgentState) -> str:
        if state.get("stop"):
            return "end"
        return "think"

    def route_after_think(state: AgentState) -> str:
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
    sg.add_node("router", router)
    sg.add_node("think", think)
    sg.add_node("tools", tools)
    sg.add_node("approve", approve)
    sg.add_node("commit", commit)
    sg.add_node("reject", reject)

    sg.add_edge(START, "router")
    sg.add_conditional_edges(
        "router", route_after_router, {"think": "think", "end": END}
    )
    sg.add_conditional_edges("think", route_after_think, {"tools": "tools", "end": END})
    sg.add_conditional_edges(
        "tools", route_after_tools, {"approve": "approve", "think": "think", "end": END}
    )
    sg.add_edge("commit", "think")
    sg.add_edge("reject", END)

    return sg.compile(checkpointer=checkpointer)
