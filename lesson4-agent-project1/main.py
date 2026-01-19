# -*- coding: utf-8 -*-
"""
标题：L02-01 main.py —— 入口：加载 .env、初始化工具配置、启动持续对话窗口
执行代码：
  pip install -U langchain langchain-openai
  # 根目录/当前目录 .env：OPENAI_API_KEY=...  可选 OPENAI_BASE_URL=...
  python main.py --model gpt-5-nano --policy ask --workdir toy_repo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from tools import init_tools, todowrite, todoread, list_files, grep, read_file, apply_patch, bash
from agent import run_one_turn


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


SYSTEM_PROMPT_TMPL = """\
你是一个终端里的 Coding Agent，遵循 ReAct（Agent->Tool->Agent）。

可用工具：
- todowrite(items) / todoread()
- list_files(dir=".", recursive=False)
- grep(pattern, path=".")
- read_file(path)
- apply_patch(file_path, new_content)  # 写入可能被拒绝/需要确认
- bash(command)                        # 命令受白名单限制

工作区（workdir）：{workdir}
写入策略（write_policy）：{policy}

关键规则（强制）：
1) 寒暄/闲聊（如“你好”）直接回复，不要调用工具。
2) 多步任务先 todowrite 列 3~7 条 todo，并在执行中 todoread 对照推进。
3) 不要假设文件名：不知道就先 list_files/grep 定位。
4) 需要内容才 read_file；禁止猜文件内容。
5) 修改文件：先 read_file，再 apply_patch，且 new_content 必须是完整文件新内容。
6) 需要验证时，优先 bash 跑：python -m pytest -q
7) 工具 path 参数必须是相对 workdir 的路径，禁止绝对路径。
"""


def build_llm(model: str, temperature: float) -> ChatOpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key, base_url=base_url)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--policy", default="ask", choices=["allow", "ask", "deny"])
    parser.add_argument("--workdir", default="toy_repo")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY：请在 .env 或环境变量中设置。")

    # 初始化工具配置（关键：workdir 绝对化 + todo 文件位置 + toy_repo 准备）
    init_tools(Path(args.workdir), args.policy)

    llm = build_llm(args.model, args.temperature)

    # 绑定工具（不手写解析器）：模型原生产出 tool_calls
    llm_tools = llm.bind_tools([todowrite, todoread, list_files, grep, read_file, apply_patch, bash])

    system_prompt = SYSTEM_PROMPT_TMPL.format(workdir=Path(args.workdir).resolve().as_posix(), policy=args.policy)
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    print("已启动 L02-01（2~3 文件结构的 ReAct Coding Agent）。输入 exit/quit 退出。")
    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not user_text:
            continue

        messages.append(HumanMessage(content=user_text))
        reply = run_one_turn(llm_tools, messages)
        print("AI>", reply)


if __name__ == "__main__":
    main()