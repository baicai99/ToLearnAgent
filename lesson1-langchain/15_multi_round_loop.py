# 15_multi_round_loop.py
# 目标：多轮辩论循环（5 Agent：2正2反1裁判）+ 投票 + 终止条件
# 约束：不使用 web_search，不使用 LangGraph；单文件独立可跑
# 默认模型：gpt-5-nano（可用环境变量 LC_MODEL 覆盖）

import os
import json
import argparse
from typing import List, Literal, Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


# -------------------------
# 1) 结构化数据模型
# -------------------------

class DebateTurn(BaseModel):
    agent_id: str
    stance: Literal["pro", "con"]
    round: int
    phase: Literal["opening", "rebuttal"]

    claims: List[str] = Field(description="核心论点（3-6条，每条一句话）")
    rebuttals: List[str] = Field(description="针对对方最强点的反驳（2-5条）")
    answers: List[str] = Field(description="对上一轮对方问题的回应（0-4条，可为空）")
    concessions: List[str] = Field(description="承认的限制/不确定点（0-3条）")
    questions: List[str] = Field(description="向对方提问（1-3条，短问句）")


class JudgeDecision(BaseModel):
    winner: Literal["pro", "con", "tie", "insufficient"] = Field(description="本轮裁决")
    confidence: float = Field(description="置信度 0~1")

    rationale: List[str] = Field(description="裁决理由要点（3-8条）")
    pro_strong_points: List[str] = Field(description="正方最强点（1-4条）")
    con_strong_points: List[str] = Field(description="反方最强点（1-4条）")
    key_uncertainties: List[str] = Field(description="关键不确定点（0-5条）")

    focus_next_round: List[str] = Field(description="下一轮建议聚焦的争议点（0-4条）")


class Vote(BaseModel):
    voter_id: str
    vote: Literal["pro", "con", "tie", "insufficient"]
    confidence: float = Field(description="置信度 0~1")
    rationale: List[str] = Field(description="投票理由要点（2-6条）")


class TieBreak(BaseModel):
    final_winner: Literal["pro", "con", "tie", "insufficient"]
    confidence: float = Field(description="置信度 0~1")
    rationale: List[str] = Field(description="平票裁决理由（2-6条）")


# -------------------------
# 2) Prompt 工厂（保持简单可控）
# -------------------------

def debater_system_prompt(agent_id: str, stance: str, style: str) -> str:
    role = "正方" if stance == "pro" else "反方"
    return f"""
你是辩手（{role}），你的辩手ID是 {agent_id}。
你的论证风格：{style}

硬性规则：
1) 只输出结构化 DebateTurn，不得输出多余文字
2) 全中文
3) 不允许引用外部资料，只能基于题目、常识与双方已给出的内容
4) 每条 claims/rebuttals/questions 必须短、明确、可比较
""".strip()


def judge_system_prompt() -> str:
    return """
你是裁判，不站队。
只输出结构化 JudgeDecision，不得输出多余文字。

裁决只基于双方发言内容（不引入外部资料）。

评分维度（写入 rationale）：
1) 论证一致性：定义/前提是否清楚自洽
2) 反驳质量：是否击中对方核心，而非稻草人
3) 覆盖面与边界：是否处理关键限制条件
4) 不确定性处理：是否承认信息不足并合理降低置信度

当双方关键前提都缺失：winner=insufficient。
""".strip()


def voter_system_prompt(voter_id: str) -> str:
    return f"""
你是投票人（voter_id={voter_id}），处于陪审员模式（juror），尽量客观。
只输出结构化 Vote，不得输出多余文字。

规则：
1) 只能基于 turns 与 judge_decision 投票
2) 不允许引入外部资料
3) vote 可为 pro/con/tie/insufficient，不要求投自己阵营
""".strip()


def tiebreak_system_prompt() -> str:
    return """
你是平票裁决裁判（tie-break judge）。
只输出结构化 TieBreak，不得输出多余文字。

规则：
1) 只能基于 turns/judge_decision/votes/counts 做最终裁决
2) 不允许引入外部资料
3) 若关键前提不足：final_winner=insufficient
""".strip()


# -------------------------
# 3) 小工具函数
# -------------------------

def call_structured(agent, user_content: str) -> Any:
    """统一的结构化调用入口。"""
    res = agent.invoke({"messages": [{"role": "user", "content": user_content}]})
    return res.get("structured_response")


def tally_votes(votes: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"pro": 0, "con": 0, "tie": 0, "insufficient": 0}
    for v in votes:
        k = v.get("vote")
        if k in counts:
            counts[k] += 1
    return counts


def should_stop(
    round_idx: int,
    judge: Dict[str, Any],
    prev_winner: Optional[str],
    conf_threshold: float,
    insufficient_threshold: float,
) -> bool:
    """
    终止策略（示例，可按你的产品策略改）：
    - 如果 winner 是 pro/con 且置信度>=阈值，并且连续两轮相同 winner => stop
    - 如果 winner=insufficient 且置信度>=阈值 => stop
    - tie 默认不断（除非到 max_rounds）
    """
    winner = judge.get("winner")
    conf = float(judge.get("confidence", 0))

    if winner == "insufficient" and conf >= insufficient_threshold:
        return True

    if winner in ("pro", "con") and conf >= conf_threshold and prev_winner == winner:
        return True

    return False


# -------------------------
# 4) 主流程：多轮辩论
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="辩题")
    parser.add_argument("--max-rounds", type=int, default=3, help="最大轮数")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-5-nano"), help="默认 gpt-5-nano，可覆盖")
    parser.add_argument("--out", default="", help="可选：输出 JSON 文件")
    parser.add_argument("--conf-threshold", type=float, default=0.78, help="连续两轮同胜方的置信度阈值")
    parser.add_argument("--insufficient-threshold", type=float, default=0.70, help="insufficient 终止置信度阈值")
    args = parser.parse_args()

    # 2正2反：用“风格差异”做区分（低成本、可控）
    debaters = [
        {"agent_id": "pro_a", "stance": "pro", "style": "偏逻辑链路：定义-论点-推导-结论"},
        {"agent_id": "pro_b", "stance": "pro", "style": "偏执行落地：流程、成本、激励、可操作性"},
        {"agent_id": "con_a", "stance": "con", "style": "偏风险控制：失败模式、外部性、激励扭曲"},
        {"agent_id": "con_b", "stance": "con", "style": "偏长期影响：公平、文化、协作摩擦、人才结构"},
    ]

    # 创建辩手 agents
    debater_agents = {}
    for d in debaters:
        debater_agents[d["agent_id"]] = create_agent(
            model=args.model,
            system_prompt=debater_system_prompt(d["agent_id"], d["stance"], d["style"]),
            response_format=ToolStrategy(DebateTurn),
        )

    # 裁判 agent
    judge_agent = create_agent(
        model=args.model,
        system_prompt=judge_system_prompt(),
        response_format=ToolStrategy(JudgeDecision),
    )

    # 投票 agents（4 个陪审员）
    voter_agents = {}
    for d in debaters:
        vid = d["agent_id"]
        voter_agents[vid] = create_agent(
            model=args.model,
            system_prompt=voter_system_prompt(vid),
            response_format=ToolStrategy(Vote),
        )

    # 平票裁决
    tiebreak_agent = create_agent(
        model=args.model,
        system_prompt=tiebreak_system_prompt(),
        response_format=ToolStrategy(TieBreak),
    )

    rounds: List[Dict[str, Any]] = []
    prev_winner: Optional[str] = None

    # 只携带“上一轮”作为上下文（L16 再做摘要压缩与更长记忆）
    last_round_turns: List[Dict[str, Any]] = []
    last_round_judge: Optional[Dict[str, Any]] = None

    for r in range(1, args.max_rounds + 1):
        phase = "opening" if r == 1 else "rebuttal"

        # 1) 生成本轮 4 位辩手发言
        turns: List[Dict[str, Any]] = []
        for d in debaters:
            aid = d["agent_id"]
            stance = d["stance"]

            # 给辩手的输入：题目 +（可选）上一轮对手发言 + 裁判聚焦点
            payload = {
                "topic": args.topic,
                "round": r,
                "phase": phase,
                "your_agent_id": aid,
                "your_stance": stance,
                "previous_round_turns": last_round_turns,
                "previous_round_judge": last_round_judge,
                "instruction": (
                    "若 phase=opening：给出你的初始立论。"
                    "若 phase=rebuttal：请基于上一轮对手内容与裁判 focus_next_round，进行反驳与补强，并回答对方问题。"
                ),
                "constraints": {
                    "no_external_sources": True,
                    "claims_range": "3-6",
                    "rebuttals_range": "2-5",
                    "questions_range": "1-3",
                },
            }

            turn = call_structured(
                debater_agents[aid],
                json.dumps(payload, ensure_ascii=False),
            )
            turns.append(turn)

        # 2) 裁判裁决
        judge_payload = {
            "topic": args.topic,
            "round": r,
            "phase": phase,
            "turns": turns,
            "instruction": "基于 turns 输出 JudgeDecision；不要引入外部资料。",
        }
        judge = call_structured(judge_agent, json.dumps(judge_payload, ensure_ascii=False))

        # 3) 投票
        votes: List[Dict[str, Any]] = []
        vote_payload = {
            "topic": args.topic,
            "round": r,
            "turns": turns,
            "judge_decision": judge,
            "instruction": "请以陪审员模式投票。",
        }
        for d in debaters:
            vid = d["agent_id"]
            v = call_structured(voter_agents[vid], json.dumps(vote_payload, ensure_ascii=False))
            votes.append(v)

        counts = tally_votes(votes)

        # 4) 汇总本轮最终裁决（多数票优先；平票走 tie-break）
        pro_n = counts["pro"]
        con_n = counts["con"]

        if pro_n > con_n:
            final = {"final_winner": "pro", "confidence": 0.70, "rationale": ["多数票支持正方。"]}
        elif con_n > pro_n:
            final = {"final_winner": "con", "confidence": 0.70, "rationale": ["多数票支持反方。"]}
        else:
            tb_payload = {
                "topic": args.topic,
                "round": r,
                "turns": turns,
                "judge_decision": judge,
                "votes": votes,
                "counts": counts,
                "instruction": "出现平票/争议，请输出 TieBreak。",
            }
            final = call_structured(tiebreak_agent, json.dumps(tb_payload, ensure_ascii=False))

        round_out = {
            "round": r,
            "phase": phase,
            "turns": turns,
            "judge_decision": judge,
            "votes": votes,
            "vote_counts": counts,
            "final_decision": final,
        }
        rounds.append(round_out)

        # 5) 终止判断（用“裁判 winner+confidence”而不是投票，以便下一课接摘要更一致）
        stop = should_stop(
            round_idx=r,
            judge=judge,
            prev_winner=prev_winner,
            conf_threshold=args.conf_threshold,
            insufficient_threshold=args.insufficient_threshold,
        )

        # 更新“上一轮上下文”
        last_round_turns = turns
        last_round_judge = judge
        prev_winner = judge.get("winner")

        if stop:
            break

    output = {
        "topic": args.topic,
        "model": args.model,
        "max_rounds": args.max_rounds,
        "rounds_ran": len(rounds),
        "rounds": rounds,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[已写入] {args.out}")


if __name__ == "__main__":
    main()
