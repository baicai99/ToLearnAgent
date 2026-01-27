# -*- coding: utf-8 -*-
"""
标题：L03-05 命令解析与动作模板（Command Templates）
执行：python -m lessonL03_05_command_templates.main

[L03-05 NEW]
- 只解析以 / 开头的命令；自然语言不解析
- 参数解析使用 shlex.split：支持引号
- 将命令转换为 turn_template（动作模板）：供 graph_agent 注入给 LLM
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple, Any

AgentMode = Literal["plan", "build"]


@dataclass(frozen=True)
class TurnTemplate:
    """动作模板：给 LLM 的“本回合明确目标 + 建议步骤 + 关键参数”"""
    name: str
    goal: str
    steps: List[str]
    params: Dict[str, Any]


@dataclass(frozen=True)
class CommandResult:
    # 若 end=True：preprocess 直接回复并结束本回合（例如 /help /mode /grep）
    end: bool
    # immediate_reply：若 end=True，输出给用户的文本
    immediate_reply: Optional[str]
    # mode_change：若非 None，切换 mode
    mode_change: Optional[AgentMode]
    # template：若非 None，注入给 LLM 让其执行（/test /review /fix）
    template: Optional[TurnTemplate]


def _split_cmd(text: str) -> List[str]:
    # 统一 posix=True：Windows 也能用引号把带空格参数包起来
    return shlex.split(text, posix=True)


def _parse_k_and_rest(tokens: List[str]) -> Tuple[Optional[str], List[str]]:
    """
    解析：-k EXPR；以及 `--` 之后的剩余参数（原样保留）
    返回：(k_expr, extra_args)
    """
    k_expr: Optional[str] = None
    extra: List[str] = []

    i = 0
    after_ddash = False
    while i < len(tokens):
        t = tokens[i]
        if after_ddash:
            extra.append(t)
            i += 1
            continue
        if t == "--":
            after_ddash = True
            i += 1
            continue
        if t in ("-k",):
            if i + 1 >= len(tokens):
                raise ValueError("缺少 -k 的参数，例如：/test -k add")
            k_expr = tokens[i + 1]
            i += 2
            continue
        extra.append(t)
        i += 1

    return k_expr, extra


def _parse_n(tokens: List[str], default_n: int) -> Tuple[int, List[str]]:
    """
    解析 -n N（或 --max N），返回：(n, remaining_positional)
    """
    n = default_n
    rest: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("-n", "--max"):
            if i + 1 >= len(tokens):
                raise ValueError("缺少 -n 的参数，例如：/review -n 80")
            n = int(tokens[i + 1])
            i += 2
            continue
        rest.append(t)
        i += 1
    return n, rest


def render_template(tt: TurnTemplate) -> str:
    """
    把 TurnTemplate 渲染成给 LLM 的“命令注入文本”
    """
    lines = [
        f"[动作模板] name={tt.name}",
        f"目标：{tt.goal}",
        "建议步骤：",
    ]
    for idx, s in enumerate(tt.steps, start=1):
        lines.append(f"{idx}. {s}")
    if tt.params:
        lines.append("参数：")
        for k, v in tt.params.items():
            lines.append(f"- {k} = {v}")
    return "\n".join(lines)


def handle_command(text: str, current_mode: AgentMode) -> CommandResult:
    """
    主入口：给 preprocess 调用。
    """
    tokens = _split_cmd(text)
    if not tokens:
        return CommandResult(end=True, immediate_reply="空命令。输入 /help 查看用法。", mode_change=None, template=None)

    cmd = tokens[0].lower()
    args = tokens[1:]

    if cmd in ("/help", "/?"):
        from .prompts import COMMAND_HELP
        return CommandResult(end=True, immediate_reply=COMMAND_HELP, mode_change=None, template=None)

    if cmd in ("/plan",):
        return CommandResult(end=True, immediate_reply="已切换到 plan 模式。", mode_change="plan", template=None)

    if cmd in ("/build",):
        return CommandResult(end=True, immediate_reply="已切换到 build 模式。", mode_change="build", template=None)

    if cmd == "/mode":
        if not args or args[0].lower() not in ("plan", "build"):
            return CommandResult(end=True, immediate_reply="用法：/mode plan 或 /mode build", mode_change=None, template=None)
        m = args[0].lower()  # type: ignore[assignment]
        return CommandResult(end=True, immediate_reply=f"已切换到 {m} 模式。", mode_change=m, template=None)

    # --- [L03-05 NEW] 参数化模板：/review /test /fix ---
    if cmd == "/review":
        n, rest = _parse_n(args, default_n=120)
        if rest:
            return CommandResult(end=True, immediate_reply="用法：/review [-n N]", mode_change=None, template=None)
        tt = TurnTemplate(
            name="review",
            goal="审阅 toy_repo 的结构与关键文件，给出下一步建议（教学型输出）。",
            steps=[
                f"先调用 list_files 列出文件结构（必要时只展示前 {n} 个）。",
                "识别关键文件（例如：入口、核心逻辑、tests、配置）。",
                "如需证据，再用 read_file_range 打开关键文件的头部/关键函数。",
                "最后输出：关键文件清单 + 你建议的下一步（例如运行 /test 或开始修复）。",
            ],
            params={"max_files_to_show": n},
        )
        return CommandResult(end=False, immediate_reply=None, mode_change=None, template=tt)

    if cmd == "/test":
        k_expr, extra = _parse_k_and_rest(args)
        # 默认：pytest -q；允许用户通过 `--` 追加 pytest 参数（如 -q -x）
        base = ["pytest", "-q"]
        if k_expr:
            base += ["-k", k_expr]
        base += extra
        # 这里不做 shell 级别的复杂转义：把最终命令作为“建议命令字符串”交给 LLM 决定如何调用 bash
        suggested_cmd = " ".join([f'"{x}"' if (" " in x or "\t" in x) else x for x in base])

        tt = TurnTemplate(
            name="test",
            goal="运行测试并汇报失败原因（必须报告 returncode、失败摘要、下一步建议）。",
            steps=[
                "若不确定项目是否使用 pytest：先 list_files 判断；否则直接 bash 运行测试。",
                "bash 执行建议命令。",
                "解析输出：若失败，指出失败测试/文件/断言信息（基于日志证据，不要猜）。",
                "给出下一步：是否 /fix（以及你将如何定位与验证）。",
            ],
            params={"suggested_test_cmd": suggested_cmd, "k_expr": k_expr, "extra_args": extra},
        )
        return CommandResult(end=False, immediate_reply=None, mode_change=None, template=tt)

    if cmd == "/fix":
        k_expr, extra = _parse_k_and_rest(args)
        base = ["pytest", "-q"]
        if k_expr:
            base += ["-k", k_expr]
        base += extra
        suggested_cmd = " ".join([f'"{x}"' if (" " in x or "\t" in x) else x for x in base])

        tt = TurnTemplate(
            name="fix",
            goal="修复失败测试：读→改→测→总结（最小改动，必须可验证）。",
            steps=[
                "先 bash 运行测试（或只跑相关测试），收集失败证据。",
                "定位：grep / read_file_range 找到失败相关代码与测试。",
                "提出最小补丁：用 propose_hunks 或 propose_write_file 生成补丁计划。",
                "写入策略为 ask 时等待审批；写入后再次 bash 跑测试验证。",
                "总结：修复点、验证命令、风险与后续建议。",
            ],
            params={"suggested_test_cmd": suggested_cmd, "k_expr": k_expr, "extra_args": extra},
        )
        return CommandResult(end=False, immediate_reply=None, mode_change=None, template=tt)

    # --- [L03-05 NEW] 确定性工具命令：/grep ---
    if cmd == "/grep":
        # 语法：/grep PATTERN [PATH] [-n N]
        n, rest = _parse_n(args, default_n=50)
        if not rest:
            return CommandResult(end=True, immediate_reply="用法：/grep PATTERN [PATH] [-n N]", mode_change=None, template=None)
        pattern = rest[0]
        path = rest[1] if len(rest) >= 2 else "."
        tt = TurnTemplate(
            name="grep_direct",
            goal="确定性执行 grep（不走 LLM 决策）。",
            steps=[],
            params={"pattern": pattern, "path": path, "max_results": n},
        )
        # 标记为 end=True：由 preprocess 直接执行工具并回复
        return CommandResult(end=True, immediate_reply=None, mode_change=None, template=tt)

    return CommandResult(end=True, immediate_reply=f"未知命令：{cmd}。输入 /help 查看支持的命令。", mode_change=None, template=None)
