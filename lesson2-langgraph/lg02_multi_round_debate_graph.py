# lg02_multi_round_debate_graph.py
# 目标：LangGraph 多轮辩论（2正2反1裁判）+ 投票 + 终止条件 + 滚动摘要
# 约束：
# - 单文件、可直接运行
# - 不使用 web_search
# - 提示词尽量写在节点调用点附近（不提前抽象模板）
# 默认模型：gpt-5-nano（可通过 --model 或环境变量 LC_MODEL 覆盖）

import os
import json
import argparse
import operator
from typing import List, Dict, Any, Optional, Literal, Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END


# -------------------------
# 1) 结构化输出
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
    winner: Literal["pro", "con", "tie", "insufficient"]
    confidence: float = Field(description="置信度 0~1")
    rationale: List[str] = Field(description="裁决理由要点（3-8条）")
    pro_strong_points: List[str] = Field(description="正方最强点（1-4条）")
    con_strong_points: List[str] = Field(description="反方最强点（1-4条）")
    key_uncertainties: List[str] = Field(description="关键不确定点（0-5条）")
    focus_next_round: List[str] = Field(description="下一轮建议聚焦点（0-4条）")


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
# 2) LangGraph State
# -------------------------
class State(TypedDict):
    topic: str
    model: str

    # 控制回合
    round: int
    max_rounds: int

    # 终止条件参数
    conf_threshold: float
    insufficient_threshold: float

    # 上一轮信息（用于“连续两轮同胜方且置信度足够”）
    last_winner: Optional[str]
    last_conf_ok: bool

    # 本轮信息（由 finalize 产出，route 使用）
    current_winner: Optional[str]
    current_conf_ok: bool

    # 上下文管理（L16 思路迁移到图中）
    rolling_summary: str
    recent_window: List[Dict[str, Any]]

    # 本轮产物（每轮会覆盖）
    turns: List[Dict[str, Any]]
    judge: Dict[str, Any]
    votes: List[Dict[str, Any]]
    vote_counts: Dict[str, int]
    final_decision: Dict[str, Any]

    # 轮次日志（用 reducer 做“追加式合并”）:contentReference[oaicite:2]{index=2}
    rounds: Annotated[List[Dict[str, Any]], operator.add]

    # summarize 节点要用的“本轮压缩包”
    round_pack: Dict[str, Any]


# -------------------------
# 3) 小工具
# -------------------------


def call_structured(agent, payload: Dict[str, Any]) -> Any:
    res = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ]
        }
    )
    return res.get("structured_response")


def tally_votes(votes: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"pro": 0, "con": 0, "tie": 0, "insufficient": 0}
    for v in votes:
        k = v.get("vote")
        if k in counts:
            counts[k] += 1
    return counts


# -------------------------
# 4) 节点：生成本轮辩手发言
# -------------------------


def node_turns(state: State) -> Dict[str, Any]:
    topic = state["topic"]
    model_id = state["model"]
    r = state["round"]
    phase: Literal["opening", "rebuttal"] = "opening" if r == 1 else "rebuttal"

    debaters = [
        {
            "agent_id": "pro_a",
            "stance": "pro",
            "style": "偏逻辑链路：定义-论点-推导-结论",
        },
        {
            "agent_id": "pro_b",
            "stance": "pro",
            "style": "偏执行落地：流程、成本、激励、可操作性",
        },
        {
            "agent_id": "con_a",
            "stance": "con",
            "style": "偏风险控制：失败模式、外部性、激励扭曲",
        },
        {
            "agent_id": "con_b",
            "stance": "con",
            "style": "偏长期影响：公平、文化、协作摩擦、人才结构",
        },
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
                "3) 不允许引用外部资料；只能基于题目、常识与输入上下文。\n"
                "4) 每条 claims/rebuttals/questions 必须短、明确、可比较。\n"
            ),
            response_format=ToolStrategy(DebateTurn),
        )

        payload = {
            "topic": topic,
            "round": r,
            "phase": phase,
            "your_agent_id": d["agent_id"],
            "your_stance": d["stance"],
            # 关键：每轮只喂 summary + recent_window（而不是全量 transcript）
            "rolling_summary": state["rolling_summary"],
            "recent_window": state["recent_window"],
            "instruction": (
                "请输出 DebateTurn。\n"
                "若 phase=opening：给出初始立论。\n"
                "若 phase=rebuttal：基于 rolling_summary + recent_window 反驳补强，并回答对方问题。\n"
                "不要引用外部资料。"
            ),
            "format_constraints": {
                "claims": "3-6",
                "rebuttals": "2-5",
                "questions": "1-3",
            },
        }

        turn = call_structured(agent, payload)
        turns_out.append(turn)

    return {"turns": turns_out}


# -------------------------
# 5) 节点：裁判
# -------------------------


def node_judge(state: State) -> Dict[str, Any]:
    model_id = state["model"]
    r = state["round"]
    phase = "opening" if r == 1 else "rebuttal"

    judge_agent = create_agent(
        model=model_id,
        system_prompt=(
            "你是裁判，不站队。\n"
            "只输出结构化 JudgeDecision，不得输出多余文字。\n"
            "裁决只基于 turns（结合 rolling_summary/recent_window 作为历史背景），不得引入外部资料。\n"
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
        "round": r,
        "phase": phase,
        "rolling_summary": state["rolling_summary"],
        "recent_window": state["recent_window"],
        "turns": state["turns"],
        "instruction": "请输出 JudgeDecision。",
    }

    judge = call_structured(judge_agent, payload)
    return {"judge": judge}


# -------------------------
# 6) 节点：投票
# -------------------------


def node_vote(state: State) -> Dict[str, Any]:
    model_id = state["model"]
    r = state["round"]

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
            "round": r,
            "turns": state["turns"],
            "judge_decision": state["judge"],
            "instruction": "请投票。",
        }

        v = call_structured(voter_agent, payload)
        votes_out.append(v)

    counts = tally_votes(votes_out)
    return {"votes": votes_out, "vote_counts": counts}


# -------------------------
# 7) 节点：本轮定案 + 产出 round_pack + 设置 current_winner/current_conf_ok
# -------------------------


def node_finalize(state: State) -> Dict[str, Any]:
    model_id = state["model"]
    r = state["round"]
    phase = "opening" if r == 1 else "rebuttal"

    counts = state["vote_counts"]
    pro_n, con_n = counts["pro"], counts["con"]

    # 多数票先出结果；平票交给 tie-break
    if pro_n > con_n:
        final = {
            "final_winner": "pro",
            "confidence": 0.70,
            "rationale": ["多数票支持正方。"],
        }
    elif con_n > pro_n:
        final = {
            "final_winner": "con",
            "confidence": 0.70,
            "rationale": ["多数票支持反方。"],
        }
    else:
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
            "round": r,
            "turns": state["turns"],
            "judge_decision": state["judge"],
            "votes": state["votes"],
            "counts": counts,
            "instruction": "出现平票，请输出 TieBreak。",
        }
        final = call_structured(tiebreak_agent, payload)

    # 终止条件我们以“裁判 winner/confidence”为准（与你 L15/L16 一致）
    judge_winner = state["judge"].get("winner")
    judge_conf = float(state["judge"].get("confidence", 0))
    current_conf_ok = judge_conf >= float(state["conf_threshold"])

    # 给 summarize 用的“压缩包”：只保留必要字段，避免 recent_window 冗余膨胀
    round_pack = {
        "round": r,
        "phase": phase,
        "turns_brief": [
            {
                "agent_id": t.get("agent_id"),
                "stance": t.get("stance"),
                "claims": (t.get("claims") or [])[:4],
                "rebuttals": (t.get("rebuttals") or [])[:4],
                "questions": (t.get("questions") or [])[:3],
                "concessions": (t.get("concessions") or [])[:2],
            }
            for t in state["turns"]
        ],
        "judge": {
            "winner": judge_winner,
            "confidence": judge_conf,
            "rationale": (state["judge"].get("rationale") or [])[:6],
            "focus_next_round": (state["judge"].get("focus_next_round") or [])[:4],
            "key_uncertainties": (state["judge"].get("key_uncertainties") or [])[:4],
        },
        "vote_counts": counts,
        "final_decision": final,
    }

    # rounds 用 reducer 追加（Annotated + operator.add）:contentReference[oaicite:3]{index=3}
    return {
        "final_decision": final,
        "current_winner": judge_winner,
        "current_conf_ok": current_conf_ok,
        "round_pack": round_pack,
        "rounds": [round_pack],
    }


# -------------------------
# 8) 节点：更新 rolling_summary + recent_window（滚动摘要）
# -------------------------


def node_summarize(state: State) -> Dict[str, Any]:
    model_id = state["model"]
    summarizer = init_chat_model(model_id, temperature=0)

    rolling_summary = state["rolling_summary"]
    recent_window = list(state["recent_window"])  # copy
    round_pack = state["round_pack"]

    # 你可按实际成本调参；这里保守一些，便于观察压缩是否触发
    MAX_RECENT_CHARS = 6000
    KEEP_LAST_ITEMS = 6
    MAX_SUMMARY_CHARS = 2500

    def recent_chars(x: List[Dict[str, Any]]) -> int:
        return len(json.dumps(x, ensure_ascii=False))

    recent_window.append(round_pack)

    # recent_window 过长：把旧片段压进 rolling_summary
    if (
        recent_chars(recent_window) > MAX_RECENT_CHARS
        and len(recent_window) > KEEP_LAST_ITEMS
    ):
        to_compress = recent_window[:-KEEP_LAST_ITEMS]
        remain = recent_window[-KEEP_LAST_ITEMS:]

        prompt = (
            "你在做多轮辩论的滚动摘要压缩。\n"
            "请把“新增历史片段”合并进“既有摘要”，输出新的摘要。\n"
            "要求：\n"
            "1) 全中文\n"
            "2) 要点式（最多 10 条）\n"
            "3) 必须保留：双方关键主张、关键反驳、裁判结论变化、尚未解决的争议点\n"
            "4) 不要扩写，不要引入外部资料\n\n"
            f"既有摘要：\n{rolling_summary if rolling_summary else '(空)'}\n\n"
            f"新增历史片段：\n{json.dumps(to_compress, ensure_ascii=False)}\n\n"
            "输出新的摘要："
        )
        rolling_summary = summarizer.invoke(prompt).content.strip()
        recent_window = remain

    # summary 自身太长：再压一次
    if len(rolling_summary) > MAX_SUMMARY_CHARS:
        prompt2 = (
            "请将以下摘要再次压缩，要求：\n"
            "1) 全中文\n"
            "2) 要点式（最多 10 条）\n"
            "3) 不丢失关键争议与结论变化\n\n"
            f"摘要：\n{rolling_summary}\n\n"
            "输出更短摘要："
        )
        rolling_summary = summarizer.invoke(prompt2).content.strip()

    return {"rolling_summary": rolling_summary, "recent_window": recent_window}


# -------------------------
# 9) 路由：决定 END 还是进入下一轮（conditional edges）
# -------------------------
def route_after_summarize(state: State) -> Literal["advance", "__end__"]:
    r = state["round"]
    max_rounds = state["max_rounds"]

    # 规则 1：达到最大轮数 => END
    if r >= max_rounds:
        return END

    judge = state["judge"]
    winner = judge.get("winner")
    conf = float(judge.get("confidence", 0))

    # 规则 2：insufficient 且置信度足够 => END
    if winner == "insufficient" and conf >= float(state["insufficient_threshold"]):
        return END

    # 规则 3：连续两轮同胜方，且两轮都“conf_ok” => END
    last_winner = state["last_winner"]
    last_conf_ok = bool(state["last_conf_ok"])
    current_winner = state["current_winner"]
    current_conf_ok = bool(state["current_conf_ok"])

    if (
        current_winner in ("pro", "con")
        and last_winner == current_winner
        and last_conf_ok
        and current_conf_ok
    ):
        return END

    return "advance"


# -------------------------
# 10) 节点：推进到下一轮（更新 last_*，round+1，清理本轮字段）
# -------------------------
def node_advance(state: State) -> Dict[str, Any]:
    next_round = int(state["round"]) + 1
    return {
        "round": next_round,
        "last_winner": state["current_winner"],
        "last_conf_ok": state["current_conf_ok"],
        # 清理本轮数据（不影响 rounds 日志、summary/window）
        "turns": [],
        "judge": {},
        "votes": [],
        "vote_counts": {"pro": 0, "con": 0, "tie": 0, "insufficient": 0},
        "final_decision": {},
        "round_pack": {},
        "current_winner": None,
        "current_conf_ok": False,
    }


# -------------------------
# 11) 组装图：START -> turns -> judge -> vote -> finalize -> summarize -> (route) -> advance -> turns ... -> END
# -------------------------
def build_graph():
    builder = StateGraph(State)

    builder.add_node("turns", node_turns)
    builder.add_node("judge", node_judge)
    builder.add_node("vote", node_vote)
    builder.add_node("finalize", node_finalize)
    builder.add_node("summarize", node_summarize)
    builder.add_node("advance", node_advance)

    # START/END 与 add_edge/add_conditional_edges 用法见官方 Graph API 概述 :contentReference[oaicite:4]{index=4}
    builder.add_edge(START, "turns")
    builder.add_edge("turns", "judge")
    builder.add_edge("judge", "vote")
    builder.add_edge("vote", "finalize")
    builder.add_edge("finalize", "summarize")

    # conditional edge：决定走 advance 还是 END（loop 的关键机制）:contentReference[oaicite:5]{index=5}
    builder.add_conditional_edges("summarize", route_after_summarize)
    builder.add_edge("advance", "turns")

    return builder.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-5-nano"))
    parser.add_argument("--out", default="")
    parser.add_argument("--conf-threshold", type=float, default=0.78)
    parser.add_argument("--insufficient-threshold", type=float, default=0.70)
    args = parser.parse_args()

    graph = build_graph()

    init_state: State = {
        "topic": args.topic,
        "model": args.model,
        "round": 1,
        "max_rounds": args.max_rounds,
        "conf_threshold": args.conf_threshold,
        "insufficient_threshold": args.insufficient_threshold,
        "last_winner": None,
        "last_conf_ok": False,
        "current_winner": None,
        "current_conf_ok": False,
        "rolling_summary": "",
        "recent_window": [],
        "turns": [],
        "judge": {},
        "votes": [],
        "vote_counts": {"pro": 0, "con": 0, "tie": 0, "insufficient": 0},
        "final_decision": {},
        "rounds": [],
        "round_pack": {},
    }

    # 有 loop 时建议设置 recursion_limit，官方文档说明可避免无限 supersteps :contentReference[oaicite:6]{index=6}
    out = graph.invoke(init_state, {"recursion_limit": 200})
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[已写入] {args.out}")


if __name__ == "__main__":
    main()
