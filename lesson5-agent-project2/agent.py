# -*- coding: utf-8 -*-
"""
标题：L02-06 agent.py —— ReAct 循环 + submit 闸门解析（证据驱动 Done）
执行代码：
  被 main.py 导入后使用（不要单独运行）
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage

from tools import (
    todowrite, todoread,
    list_files, grep, read_file, apply_patch, bash,
    submit,
)


def get_tool_calls(ai_msg: Any) -> List[Dict[str, Any]]:
    """从 AIMessage 提取 tool_calls（无 try/except 兜底：没有就返回空列表）。"""
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        return ai_msg.tool_calls
    ak = getattr(ai_msg, "additional_kwargs", {}) or {}
    return ak.get("tool_calls", []) or []


def _is_submit_accept(tool_result: str) -> Tuple[bool, str]:
    """
    submit 工具返回 JSON 字符串：{"status":"ACCEPT"|"REJECT","reason":"..."}
    这里解析用于在引擎层决定是否结束该轮。
    """
    obj = json.loads(tool_result)
    status = obj.get("status")
    reason = obj.get("reason", "")
    return status == "ACCEPT", reason


def run_one_turn(llm_tools: Any, messages: List[BaseMessage], max_tool_hops: int = 18) -> str:
    """
    单轮：允许多次 Agent->Tool->Agent 跳转。
    新增：当模型调用 submit 且被 ACCEPT，直接返回模型的 final（由模型在上一条 AIMessage 中给出文本总结）。
    """
    tool_map = {
        "todowrite": todowrite,
        "todoread": todoread,
        "list_files": list_files,
        "grep": grep,
        "read_file": read_file,
        "apply_patch": apply_patch,
        "bash": bash,
        "submit": submit,
    }

    hops = 0
    while True:
        ai = llm_tools.invoke(messages)
        tool_calls = get_tool_calls(ai)

        messages.append(ai if isinstance(ai, BaseMessage) else AIMessage(content=str(ai)))

        # 不再让“纯文本结束”成为完成：如果需要结束，必须 submit
        if not tool_calls:
            return ai.content or ""

        hops += 1
        if hops > max_tool_hops:
            return "本轮工具调用次数过多，我已停止以避免循环。请把目标拆小或加更明确约束。"

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            tool_fn = tool_map[name]  # 未知工具会 KeyError（用于学习/修 prompt）

            result = tool_fn.invoke(args)

            tool_call_id = call.get("id")
            messages.append(ToolMessage(content=result, tool_name=name, tool_call_id=tool_call_id))

            # submit 被 ACCEPT：结束该轮并把“上一条 ai.content”作为用户可见回复
            if name == "submit":
                ok, reason = _is_submit_accept(result)
                if ok:
                    # 最后一次 AIMessage（模型在发 submit 之前的那条）通常包含总结文本
                    # 这里直接返回 ai.content（已经记录在 messages）
                    return ai.content or "已完成（submit=ACCEPT）。"
                else:
                    # REJECT：继续让模型根据 ToolMessage 纠正
                    continue