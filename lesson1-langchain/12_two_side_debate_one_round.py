# 12_two_side_debate_one_round.py
# 目标：完成“2 辩手 + 1 裁判”的单轮辩论闭环（无投票、无搜索）
# 一个知识点：把多个 create_agent 的结构化输出拼成一个可裁决的最小系统
#
# 运行：
#   python 12_two_side_debate_one_round.py --topic "..."
#
# 依赖：
#   pip install -U "langchain[openai]" 或 "langchain[anthropic]"
#   pip install pydantic
#
# 说明：
# - structured_response 的返回位置见 structured output 文档。
# - ToolStrategy(ResponseFormat) 用于更稳地拿到结构化结果。

import os
import argparse
from typing import List, Literal, Dict, Any

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class DebaterTurn(BaseModel):
    stance: Literal["pro", "con"]
    claims: List[str]
    rebuttals: List[str]
    concessions: List[str]
    questions: List[str]


class JudgeDecision(BaseModel):
    winner: Literal["pro", "con", "tie", "insufficient"] = Field(description="获胜方或无法裁决")
    rationale: List[str] = Field(description="裁决理由要点（3-8条）")
    pro_strong_points: List[str] = Field(description="正方最强点（1-4条）")
    con_strong_points: List[str] = Field(description="反方最强点（1-4条）")
    key_uncertainties: List[str] = Field(description="关键不确定点（0-5条）")
    confidence: float = Field(description="置信度 0~1")


def debater_prompt(stance: str) -> str:
    role = "正方" if stance == "pro" else "反方"
    return f"""
你是辩手（{role}）。
你必须输出结构化结果（DebaterTurn），不得输出多余文字。

要求：
1) 全中文
2) claims 每条一句话（3-6条）
3) rebuttals 反驳对方可能最强论点（2-4条）
4) concessions 克制真实（0-3条）
5) questions 1-3条短问句
""".strip()


def judge_prompt() -> str:
    return """
你是裁判，不站队。
你必须输出结构化结果（JudgeDecision），不得输出多余文字。

评分维度（写入 rationale 时体现）：
1) 论证一致性（是否自洽、定义是否清楚）
2) 关键反驳质量（是否击中对方核心）
3) 覆盖面与边界条件（是否处理了关键限制）
4) 不确定性处理（是否承认信息不足并合理降低置信度）

注意：
- 不允许引入外部资料，只基于双方发言内容裁决。
- 如果双方都缺少关键前提，winner=insufficient。
""".strip()


def call_structured(agent, user_content: str) -> Dict[str, Any]:
    r = agent.invoke({"messages": [{"role": "user", "content": user_content}]})
    return r.get("structured_response")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="辩题")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-4.1"))
    args = parser.parse_args()

    # 两个辩手
    pro_agent = create_agent(
        model=args.model,
        system_prompt=debater_prompt("pro"),
        response_format=ToolStrategy(DebaterTurn),
    )
    con_agent = create_agent(
        model=args.model,
        system_prompt=debater_prompt("con"),
        response_format=ToolStrategy(DebaterTurn),
    )

    # 裁判
    judge_agent = create_agent(
        model=args.model,
        system_prompt=judge_prompt(),
        response_format=ToolStrategy(JudgeDecision),
    )

    # 1) 开篇立论（单轮：这里就让双方各自给出一份 turn）
    pro_turn = call_structured(
        pro_agent,
        f"辩题：{args.topic}\n请输出 DebaterTurn（正方）。"
    )
    con_turn = call_structured(
        con_agent,
        f"辩题：{args.topic}\n请输出 DebaterTurn（反方）。"
    )

    # 2) 裁判裁决（仅基于双方结构化内容）
    judge_input = f"""
辩题：{args.topic}

正方发言（结构化）：
{pro_turn}

反方发言（结构化）：
{con_turn}

请输出 JudgeDecision。
""".strip()

    decision = call_structured(judge_agent, judge_input)

    print("=== PRO ===")
    print(pro_turn)
    print("\n=== CON ===")
    print(con_turn)
    print("\n=== JUDGE ===")
    print(decision)


if __name__ == "__main__":
    main()
