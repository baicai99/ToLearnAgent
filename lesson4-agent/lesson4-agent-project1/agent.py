# -*- coding: utf-8 -*-
"""
标题：L02-01 agent.py —— ReAct 循环引擎（多轮 tool hop），不变成 workflow
执行代码：
  被 main.py 导入后使用（不要单独运行）
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage

from tools import (
    todowrite, todoread,
    list_files, grep, read_file, apply_patch, bash,
)


def get_tool_calls(ai_msg: Any) -> List[Dict[str, Any]]:
    # 不做 try/except 兜底：没有就返回空
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        return ai_msg.tool_calls
    ak = getattr(ai_msg, "additional_kwargs", {}) or {}
    return ak.get("tool_calls", []) or []


def run_one_turn(llm_tools: Any, messages: List[BaseMessage], max_tool_hops: int = 14) -> str:
    """
    单轮：允许多次 Agent->Tool->Agent 跳转，直到模型不再发 tool_calls。
    这就是防止“写成 workflow”的关键：下一步做什么由模型决定，而不是你排死顺序。
    """
    tool_map = {
        "todowrite": todowrite,
        "todoread": todoread,
        "list_files": list_files,
        "grep": grep,
        "read_file": read_file,
        "apply_patch": apply_patch,
        "bash": bash,
    }

    hops = 0
    while True:
        ai = llm_tools.invoke(messages)
        tool_calls = get_tool_calls(ai)

        messages.append(ai if isinstance(ai, BaseMessage) else AIMessage(content=str(ai)))

        if not tool_calls:
            return ai.content or ""

        hops += 1
        if hops > max_tool_hops:
            return "本轮工具调用次数过多，我已停止以避免循环。请你把目标拆小或加更明确约束。"

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            tool_fn = tool_map[name]  # 未知工具会 KeyError（暴露错误，利于学习）

            result = tool_fn.invoke(args)
            tool_call_id = call.get("id")
            messages.append(ToolMessage(content=result, tool_name=name, tool_call_id=tool_call_id))