# -*- coding: utf-8 -*-
"""
标题：L02-08F intent.py —— 工具意图门控 + 参数最小充分策略（核心新增）
执行代码：
  由 main.py / graph_agent.py 导入使用，不直接运行
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple


TaskType = Literal["chat", "review", "create", "tests", "implement"]
ToolProfile = Literal["chat", "review_ro", "full"]


# [L02-08F NEW] 统一的“意图分类”结果：同一套逻辑在 main 与 graph 中复用，避免口径不一致
@dataclass(frozen=True)
class Intent:
    task_type: TaskType
    tool_profile: ToolProfile
    reason: str


def classify_task_type(user_text: str) -> TaskType:
    s = (user_text or "").strip().lower()
    if not s:
        return "chat"

    greet = ["你好", "hi", "hello", "在吗", "谢谢", "早上好", "晚上好"]
    if len(s) <= 12 and any(w in s for w in greet):
        return "chat"

    # 概念解释/元问题：强制 chat（不需要碰工作区）
    concept = [
        "为什么",
        "作用",
        "原理",
        "区别",
        "怎么理解",
        "是什么",
        "解释一下",
        "langgraph",
        "langchain",
        "react",
    ]
    if any(w in s for w in concept) and not any(
        x in s
        for x in [
            "toy_repo",
            "toyrepo",
            "calc.py",
            "pytest",
            "test",
            "修复",
            "改代码",
            "创建文件",
        ]
    ):
        return "chat"

    create_markers = [
        "创建",
        "新建",
        "生成",
        "命名",
        "create",
        "write file",
        "new file",
    ]
    tests_markers = ["pytest", "测试", "test", "让测试通过", "跑测试"]
    implement_markers = [
        "修复",
        "修改",
        "添加",
        "新增",
        "实现",
        "重构",
        "更新",
        "fix",
        "bug",
        "patch",
    ]
    review_markers = [
        "review",
        "查看",
        "浏览",
        "列出",
        "结构",
        "目录",
        "tree",
        "overview",
        "审阅",
        "有什么",
        "里面有什么",
    ]

    if any(m in s for m in create_markers):
        return "create"
    if any(m in s for m in tests_markers):
        return "tests"
    if any(m in s for m in implement_markers):
        return "implement"
    if any(m in s for m in review_markers):
        return "review"
    return "review"


def pick_tool_profile(task_type: TaskType, user_text: str) -> Tuple[ToolProfile, str]:
    # [L02-08F NEW] “工具门控”的第一层：能不碰工作区就不碰
    s = (user_text or "").strip().lower()

    # 显式命令（status/done/abort/help/new task）属于对话控制，不应触发工具
    ctrl = [
        "status",
        "/status",
        "done",
        "/done",
        "abort",
        "/abort",
        "help",
        "/help",
        "new task",
        "newtask",
        "任务:",
    ]
    if any(c in s for c in ctrl):
        return "chat", "control command -> no tools"

    if task_type == "chat":
        return "chat", "chat/concept -> no tools"

    if task_type == "review":
        return "review_ro", "review -> read-only tools"

    # create/tests/implement：允许 full，但写入仍由 policy+interrupt 控制
    return "full", f"{task_type} -> full tools"


# [L02-08F NEW] 工具参数“最小充分”规约：不靠 try/except，而是显式规范与归一化
def sanitize_tool_call(name: str, args: Any) -> Tuple[bool, Dict[str, Any], str]:
    """
    返回 (ok, new_args, note)
    - ok=False 表示 args 无法成为 dict；调用方应生成 ToolMessage 错误，而不是崩溃
    - 对 dir/path 为空字符串的场景，统一转为 "."（只读工具）或保持为空并由运行时返回可读错误（写工具）
    """
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return False, {}, "tool args must be an object/dict"

    a = dict(args)

    # 只读工具：dir/path 空字符串 -> "."
    if name in ("repo_tree", "list_files"):
        d = a.get("dir", ".")
        if d is None or (isinstance(d, str) and d.strip() == ""):
            a["dir"] = "."
        if "recursive" not in a:
            a["recursive"] = name == "repo_tree"
        if "max_results" not in a:
            a["max_results"] = 200

    if name == "grep":
        # pattern 为空时不修复，让运行时返回清晰错误，促使模型/你补充
        if "path" not in a or (
            isinstance(a.get("path"), str) and a.get("path", "").strip() == ""
        ):
            a["path"] = "."
        if "max_results" not in a:
            a["max_results"] = 50

    if name == "read_file_range":
        # path 不做“猜测修复”，避免错误读到别的文件；缺参时让运行时返回清晰错误
        if "start_line" not in a:
            a["start_line"] = 1
        if "end_line" not in a:
            a["end_line"] = 200

    if name == "bash":
        # command 缺参不修复
        pass

    # 写工具：不对 file_path 做“自动猜测”；避免误写
    # write_file/apply_hunks 的缺参由运行时返回 JSON ERROR，不会崩溃

    return True, a, "sanitized"
