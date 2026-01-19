# 14_voting_mechanism.py
# 目标：实现“4 票 + 裁判平票裁决”的投票机制（不引入搜索、不生成辩论）
# 一个知识点：投票输出结构化 + 统计 + 平票处理（tie-break）
#
# 输入：L13 产生的 debate.json
# 运行：
#   python 14_voting_mechanism.py --in debate.json
#
# 依赖：
#   pip install -U "langchain[openai]" 或 "langchain[anthropic]"
#   pip install pydantic

import os
import json
import argparse
from typing import List, Literal, Dict, Any

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class Vote(BaseModel):
    voter_id: str = Field(description="投票人ID，如 pro_a / con_b")
    vote: Literal["pro", "con", "tie", "insufficient"] = Field(description="投票结果")
    rationale: List[str] = Field(description="投票理由要点（2-6条）")
    confidence: float = Field(description="置信度 0~1")


class TieBreak(BaseModel):
    final_winner: Literal["pro", "con", "tie", "insufficient"] = Field(description="最终裁决")
    rationale: List[str] = Field(description="平票裁决理由（2-6条）")
    confidence: float = Field(description="置信度 0~1")


def voter_system_prompt(voter_id: str) -> str:
    return f"""
你是投票人（voter_id={voter_id}）。
注意：此阶段你不是辩手，而是陪审员（juror），要求尽量客观。

规则：
1) 只能基于输入 JSON 中提供的 turns 与 judge_decision 投票
2) 不允许引入外部资料
3) 输出必须为结构化 Vote，不得输出多余文字
4) vote 可以投 pro / con / tie / insufficient；不要求投自己阵营
5) rationale 要点式，指向“哪一方在关键点更强/更弱”
""".strip()


def tiebreak_system_prompt() -> str:
    return """
你是平票裁决裁判（tie-break judge），只处理“统计平票或争议”的情况。
输出必须为结构化 TieBreak，不得输出多余文字。

规则：
1) 只能基于 turns、judge_decision 与 votes 做最终裁决
2) 不允许引入外部资料
3) 如果双方都无法支持关键前提：final_winner=insufficient
4) rationale 解释你为什么在平票下选择该结果（或为何仍为 tie）
""".strip()


def call_structured(agent, user_content: str) -> Any:
    res = agent.invoke({"messages": [{"role": "user", "content": user_content}]})
    return res.get("structured_response")


def tally(votes: List[Dict[str, Any]]) -> Dict[str, int]:
    cnt = {"pro": 0, "con": 0, "tie": 0, "insufficient": 0}
    for v in votes:
        k = v.get("vote")
        if k in cnt:
            cnt[k] += 1
    return cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="L13 输出的 JSON 文件路径")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-4.1"))
    args = parser.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        debate = json.load(f)

    topic = debate["topic"]
    turns = debate["turns"]
    judge_decision = debate["judge_decision"]

    voter_ids = [t.get("agent_id", f"v{i}") for i, t in enumerate(turns, start=1)]

    voters = {}
    for vid in voter_ids:
        voters[vid] = create_agent(
            model=args.model,
            system_prompt=voter_system_prompt(vid),
            response_format=ToolStrategy(Vote),
        )

    tiebreak_judge = create_agent(
        model=args.model,
        system_prompt=tiebreak_system_prompt(),
        response_format=ToolStrategy(TieBreak),
    )

    payload = {
        "topic": topic,
        "turns": turns,
        "judge_decision": judge_decision,
    }

    votes: List[Dict[str, Any]] = []
    for vid in voter_ids:
        v = call_structured(voters[vid], json.dumps(payload, ensure_ascii=False))
        votes.append(v)

    counts = tally(votes)

    # 初步结果：只在 pro 与 con 之间看多数；tie/insufficient 先作为信号
    pro_n = counts["pro"]
    con_n = counts["con"]

    if pro_n > con_n:
        final = {"final_winner": "pro", "rationale": ["多数票支持正方。"], "confidence": 0.7}
    elif con_n > pro_n:
        final = {"final_winner": "con", "rationale": ["多数票支持反方。"], "confidence": 0.7}
    else:
        # 平票：交给 tie-break judge
        tb_input = {
            "topic": topic,
            "turns": turns,
            "judge_decision": judge_decision,
            "votes": votes,
            "counts": counts,
            "instruction": "出现平票/争议，请输出 TieBreak。",
        }
        final = call_structured(tiebreak_judge, json.dumps(tb_input, ensure_ascii=False))

    out = {
        "topic": topic,
        "model": args.model,
        "votes": votes,
        "counts": counts,
        "final_decision": final,
        "judge_decision_ref": judge_decision,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
