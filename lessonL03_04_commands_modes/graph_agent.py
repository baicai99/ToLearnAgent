# -*- coding: utf-8 -*-
"""
标题：L03-04 LangGraph（/command + plan/build + 反重复动作护栏）
执行：python -m lessonL03_04_commands_modes.main

[L03-04 NEW]
1) preprocess 节点：只处理“显式命令”
   - /mode /plan /build：确定性切换模式（不让 LLM 猜）
   - /review /test /fix：注入 turn_instructions（让 LLM 更像 OpenCode 那样执行）
2) 反重复动作（doom-loop）：
   - 同一工具调用（name+args）重复达到阈值 => 强制停下并询问用户
3) 仍旧不持久化 ToolMessage：ToolMessage 仅在单回合内存里使用
"""

from __future__ import annotations

import json
import hashlib
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
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from .event_log import EventLogger
from .core_runtime import Runtime, apply_hunks_to_text, compute_base_sha
from . import tools as tool_mod
from .prompts import (
    BASE_SYSTEM_PROMPT,
    MODE_PLAN_ADDENDUM,
    MODE_BUILD_ADDENDUM,
    COMMAND_HELP,
)

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


def _append_list(a: Optional[List[Any]], b: Optional[List[Any]]) -> List[Any]:
    return list(a or []) + list(b or [])


class AgentState(TypedDict, total=False):
    # messages 必须累积，才能“持续对话 + 可恢复”
    messages: Annotated[List[BaseMessage], add_messages]

    mode: AgentMode
    write_policy: WritePolicy

    # 本回合临时提示（命令注入）。[L03-04 NEW]：每回合结束要清空
    turn_instructions: Optional[str]

    pending_patch: Optional[PatchPlan]
    rejected_fingerprints: Annotated[List[str], _append_list]

    tool_traces: Annotated[List[ToolTrace], _append_list]

    # [L03-04 NEW] 反重复动作：记录最近工具调用签名
    recent_action_keys: Annotated[List[str], _append_list]

    approval: Optional[str]


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


def _mode_system_prompt(mode: AgentMode) -> str:
    if mode == "plan":
        return MODE_PLAN_ADDENDUM
    return MODE_BUILD_ADDENDUM


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
    # [L03-04 NEW] preprocess：只处理显式命令
    # -----------------------------
    def preprocess(state: AgentState) -> Dict[str, Any]:
        user_text = _latest_user_text(state)
        mode: AgentMode = state.get("mode", default_mode)  # type: ignore[assignment]
        wp: WritePolicy = state.get("write_policy", default_write_policy)  # type: ignore[assignment]

        # 初始化缺失字段（首次运行）
        out: Dict[str, Any] = {}
        if "mode" not in state:
            out["mode"] = mode
        if "write_policy" not in state:
            out["write_policy"] = wp

        if not user_text.startswith("/"):
            # 非命令：不分流，不注入
            out["turn_instructions"] = None
            return out

        parts = user_text.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        logger.log("command", {"raw": user_text, "cmd": cmd, "args": args})

        if cmd in ("/help", "/?"):
            out["turn_instructions"] = None
            out["pending_patch"] = None
            out["messages"] = [AIMessage(content=COMMAND_HELP)]
            return out

        if cmd in ("/plan",):
            out["mode"] = "plan"
            out["turn_instructions"] = None
            out["messages"] = [AIMessage(content="已切换到 plan 模式。")]
            return out

        if cmd in ("/build",):
            out["mode"] = "build"
            out["turn_instructions"] = None
            out["messages"] = [AIMessage(content="已切换到 build 模式。")]
            return out

        if cmd == "/mode":
            if not args or args[0].lower() not in ("plan", "build"):
                out["turn_instructions"] = None
                out["messages"] = [AIMessage(content="用法：/mode plan 或 /mode build")]
                return out
            out["mode"] = args[0].lower()
            out["turn_instructions"] = None
            out["messages"] = [AIMessage(content=f"已切换到 {out['mode']} 模式。")]
            return out

        # 下列命令不“替你做 workflow”，只注入明确意图（仍由 LLM 决定如何用工具）
        if cmd == "/review":
            out["turn_instructions"] = (
                "用户执行 /review：请审阅 toy_repo。建议步骤："
                "1) 先 list_files 看结构；2) 识别关键文件（如 calc.py、tests/）；"
                "3) 若需要再 read_file_range 查看；4) 给出下一步建议。"
            )
            return out

        if cmd == "/test":
            out["turn_instructions"] = (
                "用户执行 /test：请在 toy_repo 下运行测试并汇报。"
                "默认使用 bash 执行：pytest -q。若项目不用 pytest，请先 list_files 判断。"
            )
            return out

        if cmd == "/fix":
            out["turn_instructions"] = (
                "用户执行 /fix：请尝试修复失败测试。建议闭环："
                "1) 先 /test 的思路运行测试；2) 定位失败文件/函数；"
                "3) 通过 propose_* 提出最小补丁；4) 再跑测试验证；5) 总结。"
            )
            return out

        out["turn_instructions"] = None
        out["messages"] = [AIMessage(content=f"未知命令：{cmd}。输入 /help 查看支持的命令。")]
        return out

    # -----------------------------
    # ReAct 回合（无关键词路由）
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

        traces: List[ToolTrace] = list(state.get("tool_traces", []))
        rejected = set(state.get("rejected_fingerprints", []))
        recent_keys: List[str] = list(state.get("recent_action_keys", []))

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
                # [L03-04 NEW] 每回合清空 turn_instructions，避免“命令注入”污染后续回合
                return {"messages": [AIMessage(content=text)], "tool_traces": traces, "turn_instructions": None}

            tool_messages: List[ToolMessage] = []
            produced_patch = False

            for tc in tcs:
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                tc_id = tc.get("id", "") or ""

                if name not in tool_map:
                    raise RuntimeError(f"unknown tool: {name}")

                # [L03-04 NEW] 反重复动作护栏（doom-loop）：同一动作重复 3 次就停
                ak = _hash_action(name, args)
                recent_keys.append(ak)
                # 只保留最近 12 个
                recent_keys = recent_keys[-12:]
                if recent_keys.count(ak) >= 3:
                    logger.log("doom_loop", {"tool": name})
                    return {
                        "messages": [AIMessage(content="检测到重复尝试相同工具调用（可能陷入循环）。我已停止。请你明确：更换目标/提供更多信息/允许我采用不同方案。")],
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
                        return {
                            "messages": [AIMessage(content="你之前拒绝过同一个补丁（相同 fingerprint），我不会重复尝试。请你明确：改方案 / 允许写入 / 放弃。")],
                            "tool_traces": traces,
                            "recent_action_keys": recent_keys,
                            "turn_instructions": None,
                        }

                    pending_patch = plan
                    produced_patch = True
                    break

            if produced_patch and pending_patch:
                # [L03-04 NEW] deny 策略不允许挂着 pending_patch 直接结束，否则会造成“无回复/状态脏”
                if wp == "deny":
                    diff_text = pending_patch.get("diff", "")
                    msg = (
                        "当前写入策略为 deny：我不会写入文件。\n\n"
                        "下面是补丁预览（你可手工应用）：\n"
                        f"{diff_text}"
                    )
                    return {
                        "messages": [AIMessage(content=msg)],
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
            "messages": [AIMessage(content="达到本回合最大 ReAct 步数限制，仍未完成。请你缩小任务或明确下一步。")],
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

    # -------- graph wiring --------
    builder = StateGraph(AgentState)

    builder.add_node("preprocess", preprocess)
    builder.add_node("react_turn", react_turn)
    builder.add_node("approval", approval)
    builder.add_node("commit", commit)
    builder.add_node("reject", reject)

    builder.add_edge(START, "preprocess")

    # preprocess 后：如果 preprocess 已经直接生成 AIMessage（/help 或 /mode），就结束；否则进入 react
    def route_after_preprocess(state: AgentState) -> str:
        # 若 preprocess 产生了最后一条 AIMessage（例如 /help /mode），就 END
        # 判断方式：本回合 preprocess 只会“追加”一条 AIMessage，且 turn_instructions 为 None
        # 非命令输入 preprocess 不会追加 AIMessage
        # 为避免复杂标记，这里用一个简洁规则：若最新一条消息是 AI 且用户输入是命令且 turn_instructions 为空 => END
        user_text = _latest_user_text(state)
        if user_text.startswith("/") and not (state.get("turn_instructions") or "").strip():
            # 对于 /review /test /fix turn_instructions 非空，会进入 react_turn
            # 对于 /help /mode /plan /build turn_instructions 为空，直接结束
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
