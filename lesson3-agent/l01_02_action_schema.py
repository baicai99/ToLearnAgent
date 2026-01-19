# -*- coding: utf-8 -*-
"""
标题：L01-02 Action 协议（JSON 决策）——从 Chat 到 ReAct 的第一步
执行代码：
  pip install -U langchain langchain-openai pydantic
  # 在项目根目录创建 .env（示例见下方说明）
  python l01_02_action_schema.py --model gpt-5-nano --show-messages

本课目标：
- 模型不再“自由聊天”，而是输出可解析的 JSON Action：
  1) {"type":"tool", "tool": {"name":"read_file|apply_patch|bash", "args": {...}}}
  2) {"type":"done", "final":"..."}
- 本课不执行工具，只做：提示词约束 -> 解析 JSON -> Pydantic 校验 -> 打印结构化结果
- 这是从“聊天”过渡到“Agent->Tool->Agent”的关键：先把“决策”变成机器可读。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage


# -------------------------
# 1) 读取根目录 .env（不依赖 python-dotenv）
# -------------------------

def load_dotenv(dotenv_path: Path) -> None:
    """
    极简 .env 解析器：
    - 支持 KEY=VALUE
    - 支持引号包裹
    - 忽略空行与 # 注释
    - 仅在环境变量未设置时写入 os.environ（避免覆盖）
    """
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and (k not in os.environ or os.environ[k] == ""):
            os.environ[k] = v


# -------------------------
# 2) Action 协议（结构化）
# -------------------------

class ToolCall(BaseModel):
    # 这里先把 3 个核心工具名字定死（对齐你后续目标），但本课不执行
    name: Literal["read_file", "apply_patch", "bash"] = Field(..., description="工具名")
    args: Dict[str, Any] = Field(default_factory=dict, description="工具参数")


class AgentAction(BaseModel):
    type: Literal["tool", "done"] = Field(..., description="tool 或 done")
    tool: Optional[ToolCall] = Field(default=None, description="type=tool 时必须提供")
    final: Optional[str] = Field(default=None, description="type=done 时必须提供")


def extract_json(text: str) -> Dict[str, Any]:
    """
    强约束：要求模型输出 JSON 对象（不允许夹杂额外文本）。
    但考虑到模型偶发加前后缀，这里做一个兜底：抓取第一个 {...}。
    """
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)

    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        raise ValueError(f"模型输出里找不到 JSON：\n{text}")
    return json.loads(m.group(0))


# -------------------------
# 3) System Prompt：让模型只输出 JSON Action
# -------------------------

SYSTEM_PROMPT = """\
你是一个终端 Coding Agent 的“决策器”。你必须输出一个 JSON 对象（且只输出 JSON，不要解释）。

你只能输出以下两种 JSON 之一：

A) 请求调用工具（注意：本系统稍后会执行工具；你现在只负责决策）
{
  "type": "tool",
  "tool": {
    "name": "read_file" | "apply_patch" | "bash",
    "args": { ... }
  }
}

B) 结束任务
{
  "type": "done",
  "final": "用教学口吻说明：你为什么可以结束。要求引用你已知的信息；不要凭空编造。"
}

约束：
- 如果你需要文件内容或命令输出才能判断，优先选择 type=tool，并用 read_file 或 bash 获取信息。
- 你不能声称你看过文件，除非你之前请求过 read_file 并从工具返回中获得了内容（在本课里你不会真的拿到工具返回，因此要更谨慎）。
- 不要输出 Markdown，不要输出多余字符。
"""


def print_messages(messages: List[BaseMessage], max_chars: int = 220) -> None:
    print("\n--- messages（上下文快照）---")
    for i, m in enumerate(messages):
        role = m.__class__.__name__.replace("Message", "")
        content = (m.content or "").replace("\n", "\\n")
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        print(f"[{i:02d}] {role}: {content}")
    print("----------------------------\n")


# -------------------------
# 4) 主程序：终端循环（只做决策，不执行工具）
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="L01-02: Action schema only (no tools executed)")
    parser.add_argument("--model", default="gpt-5-nano", help="模型名（默认 gpt-5-nano）")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度（建议 0）")
    parser.add_argument("--show-messages", action="store_true", help="每轮打印 messages 快照")
    args = parser.parse_args()

    # 读取根目录 .env
    load_dotenv(Path(".env"))

    # 基础校验：至少要有 key；base_url 可选（只在你用代理/路由时需要）
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY。请在根目录 .env 或环境变量中设置。")

    # 初始化 LLM（langchain-openai 会读取 OPENAI_API_KEY / OPENAI_BASE_URL）
    llm = ChatOpenAI(model=args.model, temperature=args.temperature)

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    print("已启动 Action 决策窗口（本课不执行工具）。输入 exit/quit 退出。")
    if args.show_messages:
        print_messages(messages)

    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not user_text:
            continue

        messages.append(HumanMessage(content=user_text))

        if args.show_messages:
            print("\n[Debug] 发送给模型之前的上下文：")
            print_messages(messages)

        ai = llm.invoke(messages)
        raw = ai.content

        # 解析与校验
        try:
            obj = extract_json(raw)
            action = AgentAction.model_validate(obj)
        except (ValueError, json.JSONDecodeError, ValidationError) as e:
            print("\n[ParseError] 模型输出未满足 JSON Action 协议：")
            print(raw)
            print("\n[Error Detail]", repr(e))
            # 为了继续对话，这里不把坏输出写入 messages（避免污染上下文）
            continue

        # 把“合格的 action”写回 messages（作为可追溯轨迹）
        messages.append(AIMessage(content=json.dumps(action.model_dump(), ensure_ascii=False)))

        # 教学型输出：结构化展示
        print("\nAI(Action JSON) >")
        print(json.dumps(action.model_dump(), ensure_ascii=False, indent=2))

        if args.show_messages:
            print("\n[Debug] 写回 action 后，上下文变成：")
            print_messages(messages)


if __name__ == "__main__":
    main()
