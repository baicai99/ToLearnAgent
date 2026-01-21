# -*- coding: utf-8 -*-
"""
标题：L02-08A main.py —— 任务路由 + 工具契约固化 + 3 文件组合
执行代码：
  python main.py --model gpt-5-nano --policy ask --workdir toy_repo
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
    AIMessage,
)

import core_runtime as cr
from tools import (
    set_turn_context,
    repo_tree,
    list_files,
    grep,
    read_file_range,
    write_file,
    apply_hunks,
    bash,
    evidence_read,
    todowrite,
    todoread,
    submit,
)


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and (k not in os.environ or os.environ[k] == ""):
            os.environ[k] = v


SYSTEM_PROMPT = """\
你是一个终端里的 Coding Agent，遵循 ReAct（Agent->Tool->Agent）。

本课核心：任务路由（Task Router）+ 工具契约（Tool Contract）
- 每轮会被路由为：review / create / tests / implement
- review：只读。只能用 repo_tree/list_files/grep/read_file_range/evidence_read/todoread
- create：创建新文件必须用 write_file(file_path, content)，禁止用 apply_hunks 拼补丁字符串
- tests：修复测试必须：定位（grep/read_file_range）-> 修改（apply_hunks 或 write_file）-> python -m pytest -q 通过 -> submit(task_type="tests")
- implement：实现/修改功能：定位 -> 最小修改 -> 需要时运行验证 -> submit(task_type="implement")

硬规则：
1) review 任务禁止写入：若调用 write_file/apply_hunks，会返回 WRITE_NOT_ALLOWED，你必须改为询问用户是否要修改/创建。
2) apply_hunks 的 hunks 必须是 list[dict] 结构；如果用户要“创建文件”，必须用 write_file。
3) 结束必须 submit，并提供可核验 evidence（建议先 evidence_read 再写）。
"""


def build_llm(model: str, temperature: float) -> ChatOpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return ChatOpenAI(
        model=model, temperature=temperature, api_key=api_key, base_url=base_url
    )


def get_tool_calls(ai_msg: Any) -> List[Dict[str, Any]]:
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        return ai_msg.tool_calls
    ak = getattr(ai_msg, "additional_kwargs", {}) or {}
    return ak.get("tool_calls", []) or []


def run_one_turn(
    llm_tools: Any, messages: List[BaseMessage], max_tool_hops: int = 32
) -> str:
    tool_map = {
        "repo_tree": repo_tree,
        "list_files": list_files,
        "grep": grep,
        "read_file_range": read_file_range,
        "write_file": write_file,
        "apply_hunks": apply_hunks,
        "bash": bash,
        "evidence_read": evidence_read,
        "todowrite": todowrite,
        "todoread": todoread,
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
            return "本轮工具调用过多已停止（防循环）。请明确：仅审阅 / 明确修改需求 / 停止。"

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}

            # 重复动作熔断（宽松阈值，避免误杀）
            fp = json.dumps(
                {"name": name, "args": args}, ensure_ascii=False, sort_keys=True
            )
            call_counts[fp] = call_counts.get(fp, 0) + 1
            if call_counts[fp] >= 6:
                return "检测到重复尝试相同动作（模型可能卡住）。请你明确下一步：继续、改方案、或停止。"

            tool_fn = tool_map[name]
            result = tool_fn.invoke(args)
            messages.append(
                ToolMessage(content=result, tool_name=name, tool_call_id=call.get("id"))
            )

            # 写入成功则重置计数，避免“接受写入仍误熔断”
            if name in ("write_file", "apply_hunks"):
                obj = json.loads(result)
                if obj.get("status") in ("APPLIED", "ALREADY_APPLIED"):
                    call_counts[fp] = 0

            if name == "submit":
                obj = json.loads(result)
                if obj.get("status") == "ACCEPT":
                    return ai.content or "已完成（submit=ACCEPT）。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--policy", default="ask", choices=["allow", "ask", "deny"])
    parser.add_argument("--workdir", default="toy_repo")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY：请在根目录 .env 或环境变量中设置。")

    # 初始化 runtime（最稳：直接调用 core）
    cr.init_runtime(Path(args.workdir), args.policy)

    llm = build_llm(args.model, args.temperature)
    llm_tools = llm.bind_tools(
        [
            repo_tree,
            list_files,
            grep,
            read_file_range,
            write_file,
            apply_hunks,
            bash,
            evidence_read,
            todowrite,
            todoread,
            submit,
        ]
    )

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    print("已启动 L02-08A（任务路由+契约固化+3文件组合）。输入 exit/quit 退出。")

    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not user_text:
            continue

        # 每轮任务路由
        tt = set_turn_context(user_text)
        # 同步到 core（set_turn_context 已经写进 core 的 STATE）
        messages.append(HumanMessage(content=f"[task_type={tt}] {user_text}"))

        reply = run_one_turn(llm_tools, messages)
        print("AI>", reply)


if __name__ == "__main__":
    main()
