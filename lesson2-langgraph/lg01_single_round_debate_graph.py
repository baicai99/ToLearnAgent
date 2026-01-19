# lg01_single_round_debate_graph.py
# 目标：用 LangGraph 的 StateGraph 编排“单轮 5 Agent（2正2反1裁判）+ 投票”
# 约束：
# - 单文件、可直接运行
# - 不引入循环、不引入摘要、不引入搜索
# - 提示词不提前模板化：尽量写在调用点附近，降低初学者理解成本
# 默认模型：gpt-5-nano（可通过 --model 或环境变量 LC_MODEL 覆盖）

import os
import json
import argparse
from typing import List, Literal, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict

import operator
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from langgraph.graph import StateGraph, START, END


# -------------------------
# 1) 结构化输出（你已经熟悉的部分）
# -------------------------

class DebateTurn(BaseModel):
    agent_id: str
    stance: Literal["pro", "con"]
    round: int
    phase: Literal["opening", "rebuttal"]

    claims: List[str] = Field(description="核心论点（3-6条，每条一句话）")
    rebuttals: List[str] = Field(description="针对对方最强点的反驳（2-5条）")
    answers: List[str] = Field(description="对对方问题的回应（0-4条，可为空）")
    concessions: List[str] = Field(description="承认的限制/不确定点（0-3条）")
    questions: List[str] = Field(description="向对方提问（1-3条，短问句）")


class JudgeDecision(BaseModel):
    winner: Literal["pro", "con", "tie", "insufficient"]
    confidence: float = Field(description="置信度 0~1")
    rationale: List[str] = Field(description="裁决理由要点（3-8条）")
    pro_strong_points: List[str] = Field(description="正方最强点（1-4条）")
    con_strong_points: List[str] = Field(description="反方最强点（1-4条）")
    key_uncertainties: List[str] = Field(description="关键不确定点（0-5条）")
    focus_next_round: List[str] = Field(description="下一轮建议聚焦点（0-4条，LG02 会用到）")


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
# 2) LangGraph State（关键：用 state 串联各节点）
# -------------------------

class State(TypedDict):
    topic: str
    model: str

    # 这里用 reducer=operator.add：允许节点返回 {"turns":[...]} 时自动追加
    turns: Annotated[List[Dict[str, Any]], operator.add]

    judge: Dict[str, Any]

    votes: Annotated[List[Dict[str, Any]], operator.add]
    vote_counts: Dict[str, int]
    final_decision: Dict[str, Any]


# -------------------------
# 3) 小工具
# -------------------------

def tally_votes(votes: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"pro": 0, "con": 0, "tie": 0, "insufficient": 0}
    for v in votes:
        k = v.get("vote")
        if k in counts:
            counts[k] += 1
    return counts


def call_structured(agent, payload: Dict[str, Any]) -> Any:
    """统一：把 payload 作为 user content 传入，并读取 structured_response。"""
    res = agent.invoke({"messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]})
    return res.get("structured_response")


# -------------------------
# 4) 节点实现（提示词尽量贴近调用点，不做工程化抽象）
# -------------------------

def node_generate_turns(state: State) -> Dict[str, Any]:
    topic = state["topic"]
    model_id = state["model"]

    debaters = [
        {"agent_id": "pro_a", "stance": "pro", "style": "偏逻辑链路：定义-论点-推导-结论"},
        {"agent_id": "pro_b", "stance": "pro", "style": "偏执行落地：流程、成本、激励、可操作性"},
        {"agent_id": "con_a", "stance": "con", "style": "偏风险控制：失败模式、外部性、激励扭曲"},
        {"agent_id": "con_b", "stance": "con", "style": "偏长期影响：公平、文化、协作摩擦、人才结构"},
    ]

    turns_out: List[Dict[str, Any]] = []

    for d in debaters:
        role = "正方" if d["stance"] == "pro" else "反方"

        agent = create_agent(
            model=model_id,
            system_prompt=(
                f"你是辩手（{role}），ID={d['agent_id']}。\n"
                f"你的论证风格：{d['style']}。\n"
                "硬性规则：\n"
                "1) 只输出结构化 DebateTurn，不得输出多余文字。\n"
                "2) 全中文。\n"
                "3) 不允许引用外部资料，只能基于题目、常识与对话内信息。\n"
                "4) 每条 claims/rebuttals/questions 必须短、明确、可比较。\n"
            ),
            response_format=ToolStrategy(DebateTurn),
        )

        payload = {
            "topic": topic,
            "round": 1,
            "phase": "opening",
            "your_agent_id": d["agent_id"],
            "your_stance": d["stance"],
            "instruction": (
                "请输出 DebateTurn（开篇立论）。\n"
                "不要引用外部资料；只用常识与逻辑推演。"
            ),
            "format_constraints": {
                "claims_range": "3-6",
                "rebuttals_range": "2-5",
                "questions_range": "1-3",
            },
        }

        turn = call_structured(agent, payload)
        turns_out.append(turn)

    return {"turns": turns_out}


def node_judge(state: State) -> Dict[str, Any]:
    model_id = state["model"]

    judge_agent = create_agent(
        model=model_id,
        system_prompt=(
            "你是裁判，不站队。\n"
            "只输出结构化 JudgeDecision，不得输出多余文字。\n"
            "裁决只基于双方发言内容（不引入外部资料）。\n"
            "评分维度（写入 rationale）：\n"
            "1) 论证一致性（定义/前提是否清楚自洽）\n"
            "2) 反驳质量（是否击中对方核心）\n"
            "3) 覆盖面与边界（是否处理关键限制条件）\n"
            "4) 不确定性处理（是否承认信息不足并降低置信度）\n"
            "当双方关键前提都缺失：winner=insufficient。\n"
        ),
        response_format=ToolStrategy(JudgeDecision),
    )

    payload = {
        "topic": state["topic"],
        "round": 1,
        "phase": "opening",
        "turns": state["turns"],
        "instruction": "请输出 JudgeDecision。不得引入外部资料。",
    }

    judge = call_structured(judge_agent, payload)
    return {"judge": judge}


def node_vote(state: State) -> Dict[str, Any]:
    model_id = state["model"]

    votes_out: List[Dict[str, Any]] = []
    for t in state["turns"]:
        voter_id = t.get("agent_id", "voter")

        voter_agent = create_agent(
            model=model_id,
            system_prompt=(
                f"你是投票人（voter_id={voter_id}），处于陪审员模式（juror），尽量客观。\n"
                "只输出结构化 Vote，不得输出多余文字。\n"
                "规则：\n"
                "1) 只能基于 turns 与 judge_decision 投票\n"
                "2) 不允许引入外部资料\n"
                "3) vote 可为 pro/con/tie/insufficient，不要求投自己阵营\n"
            ),
            response_format=ToolStrategy(Vote),
        )

        payload = {
            "topic": state["topic"],
            "round": 1,
            "turns": state["turns"],
            "judge_decision": state["judge"],
            "instruction": "请以陪审员模式投票。",
        }

        v = call_structured(voter_agent, payload)
        votes_out.append(v)

    counts = tally_votes(votes_out)
    return {"votes": votes_out, "vote_counts": counts}


def node_finalize(state: State) -> Dict[str, Any]:
    model_id = state["model"]
    counts = state["vote_counts"]
    pro_n = counts["pro"]
    con_n = counts["con"]

    # 多数票直接出结果；平票交给 tie-break
    if pro_n > con_n:
        return {"final_decision": {"final_winner": "pro", "confidence": 0.70, "rationale": ["多数票支持正方。"]}}
    if con_n > pro_n:
        return {"final_decision": {"final_winner": "con", "confidence": 0.70, "rationale": ["多数票支持反方。"]}}

    tiebreak_agent = create_agent(
        model=model_id,
        system_prompt=(
            "你是平票裁决裁判（tie-break judge）。\n"
            "只输出结构化 TieBreak，不得输出多余文字。\n"
            "规则：\n"
            "1) 只能基于 turns/judge_decision/votes/counts 做最终裁决\n"
            "2) 不允许引入外部资料\n"
            "3) 若关键前提不足：final_winner=insufficient\n"
        ),
        response_format=ToolStrategy(TieBreak),
    )

    payload = {
        "topic": state["topic"],
        "round": 1,
        "turns": state["turns"],
        "judge_decision": state["judge"],
        "votes": state["votes"],
        "counts": state["vote_counts"],
        "instruction": "出现平票，请输出 TieBreak。",
    }

    final = call_structured(tiebreak_agent, payload)
    return {"final_decision": final}


# -------------------------
# 5) 组装图：START -> turns -> judge -> vote -> finalize -> END
# -------------------------

def build_graph():
    builder = StateGraph(State)
    builder.add_node("turns", node_generate_turns)
    builder.add_node("judge", node_judge)
    builder.add_node("vote", node_vote)
    builder.add_node("finalize", node_finalize)

    builder.add_edge(START, "turns")
    builder.add_edge("turns", "judge")
    builder.add_edge("judge", "vote")
    builder.add_edge("vote", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="辩题")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-5-nano"), help="默认 gpt-5-nano，可覆盖")
    parser.add_argument("--out", default="", help="可选：输出 JSON 文件路径")
    args = parser.parse_args()

    graph = build_graph()

    # 初始 state：注意 turns/votes 用空列表初始化，配合 reducer=operator.add
    init_state: State = {
        "topic": args.topic,
        "model": args.model,
        "turns": [],
        "judge": {},
        "votes": [],
        "vote_counts": {"pro": 0, "con": 0, "tie": 0, "insufficient": 0},
        "final_decision": {},
    }

    out = graph.invoke(init_state)

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[已写入] {args.out}")


if __name__ == "__main__":
    main()
