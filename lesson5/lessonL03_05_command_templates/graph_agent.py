# -*- coding: utf-8 -*-
"""
标题：L03-05 LangGraph（命令参数化 + 动作模板）
执行：python -m lessonL03_05_command_templates.main

[L03-05 NEW]
1) preprocess 只处理显式命令：
   - 解析参数 -> 生成 TurnTemplate（动作模板）-> 注入 turn_instructions
   - /grep 作为确定性工具命令：preprocess 直接调用工具并回消息（不走 LLM）
2) 用 preprocess_end 显式控制是否进入 react_turn（不再依赖“猜测式条件”）
3) 非命令输入：不路由；LLM 自主决定是否 tool_calls
"""

from __future__ import annotations

import json
import hashlib
from typing import Any, Dict, List, Optional, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from .event_log import EventLogger
from .core_runtime import Runtime, apply_hunks_to_text, compute_base_sha
from . import tools as tool_mod
from .prompts import BASE_SYSTEM_PROMPT, MODE_PLAN_ADDENDUM, MODE_BUILD_ADDENDUM
from .commands import handle_command, render_template, TurnTemplate

WritePolicy = Literal["allow", "ask", "deny"]
AgentMode = Literal["plan", "build"]


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


class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]

    mode: AgentMode
    write_policy: WritePolicy

    # 每回合命令模板注入（字符串），回合结束要清空
    turn_instructions: Optional[str]
    # [L03-05 NEW] 明确控制 preprocess 后是否结束本回合
    preprocess_end: bool

    pending_patch: Optional[PatchPlan]
    rejected_fingerprints: List[str]

    tool_traces: List[ToolTrace]
    recent_action_keys: List[str]

    approval: Optional[str]


def _mode_system_prompt(mode: AgentMode) -> str:
    return MODE_PLAN_ADDENDUM if mode == "plan" else MODE_BUILD_ADDENDUM


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


def _hash_action(name: str, args: Dict[str, Any]) -> str:
    s = json.dumps({"name": name, "args": args}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def build_graph(
    *,
    rt: Runtime,
    logger: EventLogger,
    api_key: str,
    base_url: str,
    model: str,
    default_mode: AgentMode,
    default_write_policy: WritePolicy,
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

    # -----------------------------
    # preprocess：命令解析 + 动作模板注入
    # -----------------------------
    def preprocess(state: AgentState) -> Dict[str, Any]:
        user_text = _latest_user_text(state)
        mode: AgentMode = state.get("mode", default_mode)  # type: ignore[assignment]
        wp: WritePolicy = state.get("write_policy", default_write_policy)  # type: ignore[assignment]

        # 初始化字段
        if "mode" not in state:
            state["mode"] = mode
        if "write_policy" not in state:
            state["write_policy"] = wp
        if "messages" not in state:
            state["messages"] = []

        # 默认：不结束
        out: Dict[str, Any] = {"preprocess_end": False, "turn_instructions": None}

        if not user_text.startswith("/"):
            return out

        # [L03-05 NEW] 解析命令并生成模板或立即回复
        res = handle_command(user_text, current_mode=mode)
        logger.log("command", {"raw": user_text, "end": res.end, "mode_change": res.mode_change, "has_template": bool(res.template)})

        if res.mode_change:
            out["mode"] = res.mode_change

        if res.immediate_reply:
            out["messages"] = [AIMessage(content=res.immediate_reply)]
            out["preprocess_end"] = True
            return out

        if res.template and res.template.name == "grep_direct":
            # [L03-05 NEW] /grep：确定性执行工具（不走 LLM）
            params = res.template.params
            tool_args = {
                "pattern": params["pattern"],
                "path": params["path"],
                "max_results": int(params["max_results"]),
            }
            result = tool_map["grep"].invoke(tool_args)
            # 结果是 JSON 字符串：直接回给用户（教学型）
            out["messages"] = [AIMessage(content=f"已执行 /grep：\n{result}")]
            out["preprocess_end"] = True
            return out

        if res.template:
            out["turn_instructions"] = render_template(res.template)
            out["preprocess_end"] = False
            return out

        # 其他情况：结束
        out["messages"] = [AIMessage(content="命令处理失败。输入 /help 查看用法。")]
        out["preprocess_end"] = True
        return out

    # -----------------------------
    # react_turn：非命令输入不路由 + LLM 自主 tool_calls
    # -----------------------------
    def react_turn(state: AgentState) -> Dict[str, Any]:
        # 硬约束：持久化 state 中不允许 ToolMessage
        for m in state.get("messages", []):
            if isinstance(m, ToolMessage):
                raise RuntimeError("状态中不允许出现 ToolMessage（要求：ToolMessage 不持久化）")

        mode: AgentMode = state.get("mode", default_mode)  # type: ignore[assignment]
        wp: WritePolicy = state.get("write_policy", default_write_policy)  # type: ignore[assignment]
        turn_hint = (state.get("turn_instructions") or "").strip()

        user_text = _latest_user_text(state)
        logger.log("user", {"text": user_text, "mode": mode, "write_policy": wp, "has_turn_hint": bool(turn_hint)})

        traces: List[ToolTrace] = list(state.get("tool_traces", []) or [])
        rejected = set(state.get("rejected_fingerprints", []) or [])
        recent_keys: List[str] = list(state.get("recent_action_keys", []) or [])

        sys_msgs: List[BaseMessage] = [
            SystemMessage(content=BASE_SYSTEM_PROMPT),
            SystemMessage(content=_mode_system_prompt(mode)),
        ]
        if turn_hint:
            sys_msgs.append(SystemMessage(content=f"[命令注入]\n{turn_hint}"))

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
                text = (ai.content or "").strip() or "（模型未返回可见文本）"
                return {
                    "messages": state.get("messages", []) + [AIMessage(content=text)],
                    "tool_traces": traces,
                    "recent_action_keys": recent_keys,
                    "turn_instructions": None,  # 回合结束清空
                }

            tool_messages: List[ToolMessage] = []
            produced_patch = False

            for tc in tcs:
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                tc_id = tc.get("id", "") or ""

                if name not in tool_map:
                    raise RuntimeError(f"unknown tool: {name}")

                # doom-loop：同一动作重复 3 次就停
                ak = _hash_action(name, args)
                recent_keys.append(ak)
                recent_keys = recent_keys[-12:]
                if recent_keys.count(ak) >= 3:
                    logger.log("doom_loop", {"tool": name})
                    msg = "检测到重复尝试相同工具调用（可能陷入循环）。我已停止。请你明确：更换目标/提供更多信息/允许我采用不同方案。"
                    return {
                        "messages": state.get("messages", []) + [AIMessage(content=msg)],
                        "tool_traces": traces,
                        "recent_action_keys": recent_keys,
                        "turn_instructions": None,
                    }

                result = tool_map[name].invoke(args)
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc_id))

                prev = str(result)
                if len(prev) > 600:
                    prev = prev[:600] + "...(truncated)"
                traces.append({"name": name, "args": args, "result_preview": prev})
                logger.log("tool", {"name": name})

                if name in ("propose_hunks", "propose_write_file"):
                    plan = json.loads(str(result))
                    fp = plan.get("fingerprint", "")
                    if fp and fp in rejected:
                        logger.log("blocked_rejected", {"fingerprint": fp, "file_path": plan.get("file_path")})
                        msg = "你之前拒绝过同一个补丁（相同 fingerprint），我不会重复尝试。请你明确：改方案 / 允许写入 / 放弃。"
                        return {
                            "messages": state.get("messages", []) + [AIMessage(content=msg)],
                            "tool_traces": traces,
                            "recent_action_keys": recent_keys,
                            "turn_instructions": None,
                        }

                    pending_patch = plan
                    produced_patch = True
                    break

            if produced_patch and pending_patch:
                if wp == "deny":
                    diff_text = pending_patch.get("diff", "")
                    msg = "当前写入策略为 deny：我不会写入文件。\n\n下面是补丁预览（你可手工应用）：\n" + diff_text
                    return {
                        "messages": state.get("messages", []) + [AIMessage(content=msg)],
                        "tool_traces": traces,
                        "pending_patch": None,
                        "recent_action_keys": recent_keys,
                        "turn_instructions": None,
                    }

                return {
                    "tool_traces": traces,
                    "pending_patch": pending_patch,
                    "recent_action_keys": recent_keys,
                    "turn_instructions": None,
                }

            msgs_for_step = msgs_for_step + [ai] + tool_messages

        return {
            "messages": state.get("messages", []) + [AIMessage(content="达到本回合最大 ReAct 步数限制，仍未完成。请你缩小任务或明确下一步。")],
            "tool_traces": traces,
            "recent_action_keys": recent_keys,
            "turn_instructions": None,
        }

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
            return {"messages": state.get("messages", []) + [AIMessage(content="(commit) 没有待提交补丁。")], "pending_patch": None}

        file_path = plan.get("file_path", "")
        base_sha = plan.get("base_sha", "")
        mode = plan.get("mode", "")
        fp = plan.get("fingerprint", "")

        p = rt.abs_path(file_path)
        old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        cur_sha = compute_base_sha(old)

        if base_sha and cur_sha != base_sha:
            logger.log("commit_conflict", {"file_path": file_path, "fingerprint": fp})
            msg = "审批期间文件发生变化（base_sha 不一致），为避免误写，我已中止提交。请重新发起修改。"
            return {"pending_patch": None, "messages": state.get("messages", []) + [AIMessage(content=msg)]}

        if mode == "write_file":
            rt.write_text(file_path, plan.get("content", ""))
        else:
            hunks = plan.get("hunks", []) or []
            new = apply_hunks_to_text(old, hunks)
            rt.write_text(file_path, new)

        logger.log("commit_ok", {"file_path": file_path, "fingerprint": fp})
        return {"pending_patch": None, "messages": state.get("messages", []) + [AIMessage(content=f"已写入：{file_path}（fingerprint={fp[:10]}...）")]}

    def reject(state: AgentState) -> Dict[str, Any]:
        plan = state.get("pending_patch")
        fp = plan.get("fingerprint", "") if plan else ""
        rej = list(state.get("rejected_fingerprints", []) or [])
        if fp and fp not in rej:
            rej.append(fp)

        logger.log("reject", {"fingerprint": fp})
        msg = "你已拒绝写入该补丁。我不会重复尝试同一修改。请你选择：改方案 / 允许写入 / 放弃。"
        return {"pending_patch": None, "rejected_fingerprints": rej, "messages": state.get("messages", []) + [AIMessage(content=msg)]}

    builder = StateGraph(AgentState)
    builder.add_node("preprocess", preprocess)
    builder.add_node("react_turn", react_turn)
    builder.add_node("approval", approval)
    builder.add_node("commit", commit)
    builder.add_node("reject", reject)

    builder.add_edge(START, "preprocess")

    # [L03-05 NEW] preprocess_end 显式控制流程
    def route_after_preprocess(state: AgentState) -> str:
        if state.get("preprocess_end", False):
            return END
        return "react_turn"

    builder.add_conditional_edges("preprocess", route_after_preprocess, {"react_turn": "react_turn", END: END})

    def route_after_react(state: AgentState) -> str:
        plan = state.get("pending_patch")
        if not plan:
            return END
        wp: WritePolicy = state.get("write_policy", default_write_policy)  # type: ignore[assignment]
        if wp == "allow":
            return "commit"
        if wp == "ask":
            return "approval"
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
