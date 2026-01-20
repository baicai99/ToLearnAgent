# -*- coding: utf-8 -*-
"""
标题：L02-08 agent.py —— ReAct 循环 + 重复调用熔断（写入成功重置计数）+ 任务类型 submit
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage

from tools import (
    evidence_read,
    repo_tree,
    todowrite,
    todoread,
    list_files,
    grep,
    read_file_range,
    write_file,
    apply_hunks,
    bash,
    submit,
)


def get_tool_calls(ai_msg: Any) -> List[Dict[str, Any]]:
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        return ai_msg.tool_calls
    ak = getattr(ai_msg, "additional_kwargs", {}) or {}
    return ak.get("tool_calls", []) or []


def _submit_accept(tool_result: str) -> Tuple[bool, str]:
    obj = json.loads(tool_result)
    return obj.get("status") == "ACCEPT", obj.get("reason", "")


def _call_fingerprint(name: str, args: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))
    h.update(b"\n")
    h.update(json.dumps(args, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def run_one_turn(
    llm_tools: Any, messages: List[BaseMessage], max_tool_hops: int = 32
) -> str:
    tool_map = {
        "evidence_read": evidence_read,
        "repo_tree": repo_tree,
        "todowrite": todowrite,
        "todoread": todoread,
        "list_files": list_files,
        "grep": grep,
        "read_file_range": read_file_range,
        "write_file": write_file,
        "apply_hunks": apply_hunks,
        "bash": bash,
        "submit": submit,
    }

    call_counts: Dict[str, int] = {}
    hops = 0

    while True:
        ai = llm_tools.invoke(messages)
        tool_calls = get_tool_calls(ai)
        messages.append(
            ai if isinstance(ai, BaseMessage) else AIMessage(content=str(ai))
        )

        if not tool_calls:
            return ai.content or ""

        hops += 1
        if hops > max_tool_hops:
            return "本轮工具调用过多已停止（防循环）。请明确：仅审阅/允许修改/放弃。"

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}

            fp = _call_fingerprint(name, args)
            call_counts[fp] = call_counts.get(fp, 0) + 1

            if call_counts[fp] >= 5:
                return "检测到重复尝试相同动作（模型可能卡住）。请你明确下一步：继续、改方案、或停止。"

            tool_fn = tool_map[name]
            result = tool_fn.invoke(args)

            tool_call_id = call.get("id")
            messages.append(
                ToolMessage(content=result, tool_name=name, tool_call_id=tool_call_id)
            )

            # 写入类：成功则重置计数，避免误熔断
            if name in ("write_file", "apply_hunks"):
                obj = json.loads(result)
                if obj.get("status") in ("APPLIED", "ALREADY_APPLIED"):
                    call_counts[fp] = 0

            if name == "submit":
                ok, _reason = _submit_accept(result)
                if ok:
                    return ai.content or "已完成（submit=ACCEPT）。"
