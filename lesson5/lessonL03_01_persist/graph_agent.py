# -*- coding: utf-8 -*-
"""
标题：L03-01 LangGraph ReAct（SQLite 持久化 + interrupt 审批）
执行：由 main.py 调用

[L03 NEW]
- 使用 SqliteSaver 持久化线程状态（跨进程/重启可恢复）:contentReference[oaicite:3]{index=3}
- interrupt 审批：暂停等待 y/n，再 Command(resume=...) 恢复 :contentReference[oaicite:4]{index=4}
- 工程化“硬约束”避免无限循环：对被拒绝 fingerprint 的补丁，运行时直接阻断，不让模型反复提交
"""

from __future__ import annotations

import atexit
import json
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from .event_log import EventLogger
from .prompts import SYSTEM_PROMPT
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
    hunks: List[Dict[str, Any]]
    mode: str
    content: str


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], ...]  # 运行时由 builder 指定 reducer
    write_policy: WritePolicy
    pending_patch: Optional[PatchPlan]
    rejected_fingerprints: List[str]


def _add_messages_reducer(
    existing: List[BaseMessage], new: List[BaseMessage]
) -> List[BaseMessage]:
    return existing + new


def _normalize_messages_for_model(msgs: List[BaseMessage]) -> List[BaseMessage]:
    """
    OpenAI chat format requires role=tool messages to be direct responses to an
    immediately preceding assistant message with tool_calls.

    Because we persist messages across runs (and some integrations may drop the
    tool_calls linkage), normalize tool-related messages into plain text
    Human/AI messages before sending them back to the model.
    """
    out: List[BaseMessage] = []
    for m in msgs:
        if isinstance(m, ToolMessage):
            out.append(AIMessage(content=f"[tool_result] {m.content}"))
            continue

        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            names = [tc.get("name") for tc in (m.tool_calls or []) if tc.get("name")]
            marker = (
                f"(assistant requested tools: {', '.join(names)})" if names else ""
            )
            text = (m.content or "").strip()
            if text and marker:
                text = f"{text}\n\n{marker}"
            elif not text:
                text = marker or "(assistant requested tools)"
            out.append(AIMessage(content=text))
            continue

        if isinstance(m, (HumanMessage, AIMessage)):
            out.append(m)
        else:
            out.append(AIMessage(content=str(getattr(m, "content", ""))))

    return out


def _latest_user_text(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""


def _should_use_tools(user_text: str) -> bool:
    """
    [L03 NEW] 工程化 tool-guard：避免“你好也执行工具”
    这里先用可解释的启发式（后续 L03-xx 会升级为可插拔策略/LLM classifier）。
    """
    t = (user_text or "").strip().lower()
    if not t:
        return False

    # 典型问候/闲聊
    greetings = ["你好", "hi", "hello", "在吗", "你是谁", "你是什么模型"]
    if any(g in t for g in greetings) and len(t) <= 12:
        return False

    keywords = [
        "toy_repo",
        "toyrepo",
        "repo",
        "仓库",
        "目录",
        "文件",
        ".py",
        "calc",
        "test",
        "pytest",
        "运行",
        "执行",
        "报错",
        "修复",
        "修改",
        "创建",
        "新增",
        "grep",
        "搜索",
        "读取",
        "review",
    ]
    return any(k in t for k in keywords)


def build_graph(
    *,
    rt: Runtime,
    logger: EventLogger,
    api_key: str,
    base_url: str,
    write_policy: WritePolicy,
    checkpoint_db_path: str,
):
    # LLM
    llm = ChatOpenAI(
        model="gpt-5-nano",
        api_key=api_key,
        base_url=base_url or None,
        temperature=0,
    )

    # Tools
    tool_fns = [
        tool_mod.list_files,
        tool_mod.read_file_range,
        tool_mod.grep,
        tool_mod.bash,
        tool_mod.propose_hunks,
        tool_mod.propose_write_file,
    ]
    tool_mod.attach_runtime_to_tools(rt, tool_fns)
    tool_map = {t.name: t for t in tool_fns}  # StructuredTool

    llm_tools = llm.bind_tools(tool_fns)

    def gate(state: AgentState) -> Dict[str, Any]:
        user_text = _latest_user_text(state)
        use_tools = _should_use_tools(user_text)
        logger.log("gate", {"user_text": user_text, "use_tools": use_tools})

        if not use_tools:
            # 直接纯聊天：不让模型看到 tools，避免它“顺手 tool call”
            prompt = [("system", SYSTEM_PROMPT)] + _normalize_messages_for_model(
                state.get("messages", [])
            )
            ai = llm.invoke(prompt)
            if not isinstance(ai, AIMessage):
                ai = AIMessage(content=str(ai))
            return {"messages": [ai]}

        return {}  # 继续到 think

    def think(state: AgentState) -> Dict[str, Any]:
        msgs = _normalize_messages_for_model(state.get("messages", []))
        # 强制带 system prompt
        full = [("system", SYSTEM_PROMPT)] + msgs  # type: ignore[list-item]
        ai: AIMessage = llm_tools.invoke(full)  # type: ignore[assignment]

        logger.log(
            "think",
            {
                "assistant_has_text": bool((ai.content or "").strip()),
                "tool_calls": [tc.get("name") for tc in (ai.tool_calls or [])],
            },
        )
        return {"messages": [ai]}

    def _extract_tool_calls(ai: AIMessage) -> List[Dict[str, Any]]:
        # 兼容：优先 ai.tool_calls
        if getattr(ai, "tool_calls", None):
            return list(ai.tool_calls)  # type: ignore[arg-type]
        # 兜底：additional_kwargs
        ak = getattr(ai, "additional_kwargs", {}) or {}
        tcs = ak.get("tool_calls") or []
        return list(tcs)

    def run_tools(state: AgentState) -> Dict[str, Any]:
        msgs = state.get("messages", [])
        last = msgs[-1] if msgs else None
        if not isinstance(last, AIMessage):
            return {}

        tool_calls = _extract_tool_calls(last)
        if not tool_calls:
            return {}

        rejected = set(state.get("rejected_fingerprints", []))
        updates: Dict[str, Any] = {"messages": []}
        pending_patch: Optional[PatchPlan] = None

        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("args", {}) or {}
            tc_id = tc.get("id", "") or ""

            if name not in tool_map:
                err = f"(error) unknown tool: {name}"
                updates["messages"].append(ToolMessage(content=err, tool_call_id=tc_id))
                logger.log("tool_error", {"name": name, "args": args, "error": err})
                continue

            # [L03 NEW] 边界层异常控制：不吞错误、不崩溃
            try:
                result = tool_map[name].invoke(args)
            except Exception as e:
                err = f"(tool exception) {type(e).__name__}: {e}"
                updates["messages"].append(ToolMessage(content=err, tool_call_id=tc_id))
                logger.log("tool_exception", {"name": name, "args": args, "error": err})
                continue

            # ToolMessage 必须紧跟对应 tool_call_id（避免你之前遇到的 OpenAI 400）
            updates["messages"].append(
                ToolMessage(content=str(result), tool_call_id=tc_id)
            )
            logger.log("tool_ok", {"name": name, "args": args})

            # 捕获“写入提案”
            if name in ("propose_hunks", "propose_write_file"):
                plan = json.loads(str(result))
                fp = plan.get("fingerprint", "")
                if fp and fp in rejected:
                    # [L03 NEW] 硬阻断：被拒绝的 fingerprint 不能再次进入审批，避免无限循环
                    logger.log(
                        "write_blocked_rejected",
                        {"fingerprint": fp, "file_path": plan.get("file_path")},
                    )
                    updates["messages"].append(
                        AIMessage(
                            content="我检测到你之前拒绝过同一个补丁（相同 fingerprint）。为避免无限重复，我不会再次尝试写入。请你明确：要改方案、还是放弃、还是允许写入？"
                        )
                    )
                    return updates

                pending_patch = (
                    plan  # 本轮只处理最后一个写入提案（简化；后续可扩展队列）
                )

        if pending_patch:
            updates["pending_patch"] = pending_patch

        return updates

    def maybe_submit(state: AgentState) -> Dict[str, Any]:
        plan = state.get("pending_patch")
        if not plan:
            return {}

        policy = state.get("write_policy", "ask")
        if policy == "deny":
            # 直接拒绝，并记入 rejected
            fp = plan.get("fingerprint", "")
            rejected = list(state.get("rejected_fingerprints", []))
            if fp and fp not in rejected:
                rejected.append(fp)
            logger.log(
                "write_denied", {"fingerprint": fp, "file_path": plan.get("file_path")}
            )
            return {
                "pending_patch": None,
                "rejected_fingerprints": rejected,
                "messages": [
                    AIMessage(
                        content="当前写入策略为 deny：我不会写入文件。我可以给你建议或输出补丁供你手动应用。"
                    )
                ],
            }

        if policy == "allow":
            # 直接 commit
            return commit(state)

        # ask：进入 interrupt
        diff_text = plan.get("diff", "")
        preview = (
            "\n========== [WRITE PREVIEW / 等待审批] ==========\n"
            f"{diff_text}\n"
            "===============================================\n"
            "是否允许写入该补丁？(y/n)"
        )
        logger.log(
            "write_preview",
            {
                "file_path": plan.get("file_path"),
                "fingerprint": plan.get("fingerprint"),
            },
        )
        ans = interrupt(preview)
        return {"_approval": str(ans).strip().lower()}

    def commit(state: AgentState) -> Dict[str, Any]:
        plan = state.get("pending_patch")
        if not plan:
            return {}

        file_path = plan["file_path"]
        base_sha = plan.get("base_sha", "")
        mode = plan.get("mode", "")
        fp = plan.get("fingerprint", "")

        # 读取当前文件，做漂移检测
        p = rt.abs_path(file_path)
        old = ""
        if p.exists():
            old = p.read_text(encoding="utf-8", errors="replace")
        cur_sha = compute_base_sha(old)

        if base_sha and cur_sha != base_sha:
            # 不自动合并：工程化要可解释
            logger.log("commit_conflict", {"file_path": file_path, "fingerprint": fp})
            return {
                "pending_patch": None,
                "messages": [
                    AIMessage(
                        content="审批期间文件内容发生变化（base_sha 不一致），为避免误写，我已中止提交。请你重新发起一次修改（我会重新读取文件并生成新补丁）。"
                    )
                ],
            }

        # 落盘
        if mode == "write_file":
            rt.write_text(file_path, plan.get("content", ""))
        else:
            # hunks 模式
            hunks = plan.get("hunks", []) or []
            new = apply_hunks_to_text(old, hunks)
            rt.write_text(file_path, new)

        logger.log("commit_ok", {"file_path": file_path, "fingerprint": fp})
        return {
            "pending_patch": None,
            "messages": [
                AIMessage(
                    content=f"已写入补丁：{file_path}（fingerprint={fp[:10]}...）"
                )
            ],
        }

    def reject(state: AgentState) -> Dict[str, Any]:
        plan = state.get("pending_patch")
        fp = plan.get("fingerprint", "") if plan else ""
        rejected = list(state.get("rejected_fingerprints", []))
        if fp and fp not in rejected:
            rejected.append(fp)
        logger.log(
            "reject",
            {"fingerprint": fp, "file_path": plan.get("file_path") if plan else ""},
        )

        # [L03 NEW] 关键：拒绝后“硬停”，并明确要求用户指令（避免模型自嗨重试）
        return {
            "pending_patch": None,
            "rejected_fingerprints": rejected,
            "messages": [
                AIMessage(
                    content="你已拒绝写入该补丁。我将停止继续尝试同一修改。请你选择：1）改方案 2）允许写入 3）放弃本次任务。"
                )
            ],
        }

    # Graph
    builder = StateGraph(AgentState)

    builder.add_node("gate", gate)
    builder.add_node("think", think)
    builder.add_node("tools", run_tools)
    builder.add_node("submit", maybe_submit)
    builder.add_node("commit", commit)
    builder.add_node("reject", reject)

    builder.add_edge(START, "gate")

    def route_after_gate(state: AgentState) -> str:
        # gate 若已经直接回复，会追加 AIMessage，此时直接 END
        # 否则继续 think
        user_text = _latest_user_text(state)
        return "think" if _should_use_tools(user_text) else END

    builder.add_conditional_edges(
        "gate", route_after_gate, {"think": "think", END: END}
    )

    def route_after_think(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and (_extract_tool_calls(last)):
            return "tools"
        return END

    builder.add_conditional_edges(
        "think", route_after_think, {"tools": "tools", END: END}
    )

    builder.add_edge("tools", "submit")

    def route_after_submit(state: AgentState) -> str:
        # allow/deny 可能直接 commit/reply；ask 会写入 _approval
        approval = state.get("_approval")
        if approval == "y":
            return "commit"
        if approval == "n":
            return "reject"
        # allow/deny 走完会清 pending_patch 或追加消息，回到 think 继续
        if state.get("pending_patch") is None:
            return "think"
        return "reject"

    builder.add_conditional_edges(
        "submit",
        route_after_submit,
        {"commit": "commit", "reject": "reject", "think": "think"},
    )
    builder.add_edge("commit", "think")
    builder.add_edge("reject", END)

    # [L03 NEW] SQLite 持久化 checkpointer（跨进程可恢复）
    class _CheckpointerWrapper:
        def __init__(self, maybe_cm):
            self._cm = None
            if hasattr(maybe_cm, "__enter__") and hasattr(maybe_cm, "__exit__"):
                self._cm = maybe_cm
                self._cp = maybe_cm.__enter__()
            else:
                self._cp = maybe_cm
            atexit.register(self.close)

        def close(self) -> None:
            if self._cm is None:
                return
            cm, self._cm = self._cm, None
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass

        def __getattr__(self, name: str):
            return getattr(self._cp, name)

    memory = _CheckpointerWrapper(SqliteSaver.from_conn_string(checkpoint_db_path))
    graph = builder.compile(checkpointer=memory)

    return graph, memory
