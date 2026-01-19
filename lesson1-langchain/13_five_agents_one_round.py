# 13_five_agents_one_round.py
# 目标：5 Agent 单轮辩论（2正2反1裁判），无投票、无搜索
# 一个知识点：多角色结构化发言 + 裁判结构化裁决 + 产出可复盘 JSON
#
# 依赖：
#   pip install -U "langchain[openai]" 或 "langchain[anthropic]"
#   pip install pydantic
#
# 说明：
# - create_agent 可配置 response_format，让结构化结果返回在 structured_response。:contentReference[oaicite:1]{index=1}

import os
import json
import argparse
from typing import List, Literal, Dict, Any

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class DebaterTurn(BaseModel):
    agent_id: str = Field(description="辩手ID，如 pro_a / con_b")
    stance: Literal["pro", "con"] = Field(description="立场：pro 正方 / con 反方")
    claims: List[str] = Field(description="本方核心论点（3-6条，每条一句话）")
    rebuttals: List[str] = Field(description="对对方可能最强论点的反驳（2-4条）")
    concessions: List[str] = Field(description="本方承认的限制/不确定点（0-3条）")
    questions: List[str] = Field(description="向对方提问（1-3条，短问句）")


class JudgeDecision(BaseModel):
    winner: Literal["pro", "con", "tie", "insufficient"] = Field(description="获胜方或无法裁决")
    rationale: List[str] = Field(description="裁决理由要点（3-8条）")
    pro_strong_points: List[str] = Field(description="正方最强点（1-4条）")
    con_strong_points: List[str] = Field(description="反方最强点（1-4条）")
    key_uncertainties: List[str] = Field(description="关键不确定点（0-5条）")
    confidence: float = Field(description="置信度 0~1")


def debater_system_prompt(agent_id: str, stance: str, style: str) -> str:
    role = "正方" if stance == "pro" else "反方"
    return f"""
你是辩手（{role}），你的辩手ID是 {agent_id}。
你的论证风格：{style}

输出必须为结构化 DebaterTurn，不得输出多余文字。

硬性要求：
1) 全中文
2) claims 每条一句话，3-6条
3) rebuttals 反驳对方“可能最强论点”，2-4条
4) concessions 克制真实，0-3条
5) questions 1-3条短问句
6) 不允许引用外部资料，只用一般常识与逻辑推演
""".strip()


def judge_system_prompt() -> str:
    return """
你是裁判，不站队。
输出必须为结构化 JudgeDecision，不得输出多余文字。

评分维度（写进 rationale）：
1) 论证一致性：定义清楚、前提合理、自洽
2) 关键反驳质量：是否击中对方核心，而非稻草人
3) 覆盖面与边界：是否识别关键限制与适用条件
4) 不确定性处理：是否承认信息不足并合理降低置信度

注意：
- 不允许引入外部资料，只基于双方发言内容裁决
- 若双方关键前提都缺失：winner=insufficient
""".strip()


def call_structured(agent, user_content: str) -> Any:
    res = agent.invoke({"messages": [{"role": "user", "content": user_content}]})
    return res.get("structured_response")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="辩题")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-4.1"))
    parser.add_argument("--out", default="", help="可选：输出 JSON 文件路径")
    args = parser.parse_args()

    # 两个正方、两个反方：先用“风格差异”做轻量区分（不增加系统复杂度）
    debaters = [
        {"agent_id": "pro_a", "stance": "pro", "style": "偏逻辑链路：定义-论点-推导-结论，尽量可检验"},
        {"agent_id": "pro_b", "stance": "pro", "style": "偏执行落地：关注成本/流程/激励与反直觉风险的应对"},
        {"agent_id": "con_a", "stance": "con", "style": "偏风险控制：识别失败模式、外部性、激励扭曲与边界条件"},
        {"agent_id": "con_b", "stance": "con", "style": "偏公平与长期：关注组织文化、人才结构、协作摩擦与长期副作用"},
    ]

    agents = {}
    for d in debaters:
        agents[d["agent_id"]] = create_agent(
            model=args.model,
            system_prompt=debater_system_prompt(d["agent_id"], d["stance"], d["style"]),
            response_format=ToolStrategy(DebaterTurn),
        )

    judge = create_agent(
        model=args.model,
        system_prompt=judge_system_prompt(),
        response_format=ToolStrategy(JudgeDecision),
    )

    # 1) 4 位辩手依次发言（先不并行，降低复杂度）
    turns: List[Dict[str, Any]] = []
    for d in debaters:
        uid = d["agent_id"]
        user_msg = f"辩题：{args.topic}\n请以你的立场输出 DebaterTurn。"
        turn = call_structured(agents[uid], user_msg)
        turns.append(turn)

    # 2) 裁判裁决（只基于双方发言）
    judge_input = {
        "topic": args.topic,
        "turns": turns,
        "instruction": "请基于 turns 输出 JudgeDecision。",
    }
    decision = call_structured(judge, json.dumps(judge_input, ensure_ascii=False))

    output = {
        "topic": args.topic,
        "model": args.model,
        "turns": turns,
        "judge_decision": decision,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[已写入] {args.out}")


if __name__ == "__main__":
    main()
