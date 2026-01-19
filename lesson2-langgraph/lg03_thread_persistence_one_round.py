# lg03_thread_persistence_one_round.py
# 目标：LangGraph 线程级持久化（checkpointer + thread_id）
# 设计：每次 invoke 只跑“1 轮辩论”，然后 END；下一次用同一 thread_id 再 invoke，就从已保存的 state 接着跑下一轮。
# 默认模型：gpt-5-nano（可用 --model 或环境变量 LC_MODEL 覆盖）
#
# 关键点（对应官方文档用法）：
# 1) compile(checkpointer=...) 开启 checkpointing :contentReference[oaicite:3]{index=3}
# 2) invoke 时传 config={"configurable":{"thread_id":...}} 让 runtime 知道加载/写入哪个线程的 state :contentReference[oaicite:4]{index=4}

import os
import json
import sqlite3
import argparse
import operator
from typing import List, Dict, Any, Optional, Literal, Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


# -------------------------
# 1) 结构化输出（沿用你已掌握的格式）
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
# 2) State（本课关键：round/rolling_summary/recent_window 会被持久化）
# -------------------------

class State(TypedDict):
    topic: str
    model: str

    # “下一次要跑第几轮”：本课每次 invoke 跑 1 轮，然后把 round +1
    round: int
    last_completed_round: int

    rolling_summary: str
    recent_window: List[Dict[str, Any]]

    # 本次调用产出（保存在 state 里，下一次仍可读取）
    last_round_pack: Dict[str, Any]

    # 日志（追加式合并）
    rounds: Annotated[List[Dict[str, Any]], operator.add]


def call_structured(agent, payload: Dict[str, Any]) -> Any:
    res = agent.invoke({"messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]})
    return res.get("structured_response")


def tally_votes(votes: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"pro": 0, "con": 0, "tie": 0, "insufficient": 0}
    for v in votes:
        k = v.get("vote")
        if k in counts:
            counts[k] += 1
    return counts


# -------------------------
# 3) Node：跑“1 轮”（turns -> judge -> vote -> finalize），并生成 round_pack
# -------------------------

def node_run_one_round(state: State) -> Dict[str, Any]:
    topic = state["topic"]
    model_id = state["model"]
    r = state["round"]
    phase: Literal["opening", "rebuttal"] = "opening" if r == 1 else "rebuttal"

    debaters = [
        {"agent_id": "pro_a", "stance": "pro", "style": "偏逻辑链路：定义-论点-推导-结论"},
        {"agent_id": "pro_b", "stance": "pro", "style": "偏执行落地：流程、成本、激励、可操作性"},
        {"agent_id": "con_a", "stance": "con", "style": "偏风险控制：失败模式、外部性、激励扭曲"},
        {"agent_id": "con_b", "stance": "con", "style": "偏长期影响：公平、文化、协作摩擦、人才结构"},
    ]

    # 1) 4 位辩手发言（每轮只喂 rolling_summary + recent_window）
    turns: List[Dict[str, Any]] = []
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
            "rolling_summary": state["rolling_summary"],
            "recent_window": state["recent_window"],
            "instruction": (
                "请输出 DebateTurn。\n"
                "若 phase=opening：给出初始立论。\n"
                "若 phase=rebuttal：基于 rolling_summary + recent_window 反驳补强，并回答对方问题。\n"
                "不要引用外部资料。"
            ),
        }
        turns.append(call_structured(agent, payload))

    # 2) 裁判
    judge_agent = create_agent(
        model=model_id,
        system_prompt=(
            "你是裁判，不站队。\n"
            "只输出结构化 JudgeDecision，不得输出多余文字。\n"
            "裁决只基于 turns（结合 rolling_summary/recent_window 作为历史背景），不得引入外部资料。\n"
            "评分维度（写入 rationale）：\n"
            "1) 论证一致性\n"
            "2) 反驳质量\n"
            "3) 覆盖面与边界\n"
            "4) 不确定性处理\n"
            "当双方关键前提都缺失：winner=insufficient。\n"
        ),
        response_format=ToolStrategy(JudgeDecision),
    )
    judge = call_structured(
        judge_agent,
        {
            "topic": topic,
            "round": r,
            "phase": phase,
            "rolling_summary": state["rolling_summary"],
            "recent_window": state["recent_window"],
            "turns": turns,
            "instruction": "请输出 JudgeDecision。",
        },
    )

    # 3) 投票
    votes: List[Dict[str, Any]] = []
    for t in turns:
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
        votes.append(
            call_structured(
                voter_agent,
                {"topic": topic, "round": r, "turns": turns, "judge_decision": judge, "instruction": "请投票。"},
            )
        )

    counts = tally_votes(votes)

    # 4) 多数票定案；平票 tie-break
    pro_n, con_n = counts["pro"], counts["con"]
    if pro_n > con_n:
        final = {"final_winner": "pro", "confidence": 0.70, "rationale": ["多数票支持正方。"]}
    elif con_n > pro_n:
        final = {"final_winner": "con", "confidence": 0.70, "rationale": ["多数票支持反方。"]}
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
        final = call_structured(
            tiebreak_agent,
            {"topic": topic, "round": r, "turns": turns, "judge_decision": judge, "votes": votes, "counts": counts},
        )

    # 5) round_pack（用于 recent_window 与 rounds 日志）
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
            for t in turns
        ],
        "judge": {
            "winner": judge.get("winner"),
            "confidence": judge.get("confidence"),
            "rationale": (judge.get("rationale") or [])[:6],
            "focus_next_round": (judge.get("focus_next_round") or [])[:4],
            "key_uncertainties": (judge.get("key_uncertainties") or [])[:4],
        },
        "vote_counts": counts,
        "final_decision": final,
    }

    return {
        "last_completed_round": r,
        "last_round_pack": round_pack,
        "rounds": [round_pack],
        # 这里不自增 round；交给 node_advance 做，保证“每次 invoke 固定跑 1 轮”
    }


# -------------------------
# 4) Node：滚动摘要（把 last_round_pack 合并进 rolling_summary/recent_window）
# -------------------------

def node_update_memory(state: State) -> Dict[str, Any]:
    summarizer = init_chat_model(state["model"], temperature=0)

    rolling_summary = state["rolling_summary"]
    recent_window = list(state["recent_window"])
    pack = state["last_round_pack"]

    MAX_RECENT_CHARS = 6000
    KEEP_LAST_ITEMS = 6
    MAX_SUMMARY_CHARS = 2500

    def recent_chars(x: List[Dict[str, Any]]) -> int:
        return len(json.dumps(x, ensure_ascii=False))

    recent_window.append(pack)

    if recent_chars(recent_window) > MAX_RECENT_CHARS and len(recent_window) > KEEP_LAST_ITEMS:
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

    if len(rolling_summary) > MAX_SUMMARY_CHARS:
        prompt2 = (
            "请将以下摘要再次压缩：\n"
            "1) 全中文\n"
            "2) 要点式（最多 10 条）\n"
            "3) 不丢失关键争议与结论变化\n\n"
            f"摘要：\n{rolling_summary}\n\n"
            "输出更短摘要："
        )
        rolling_summary = summarizer.invoke(prompt2).content.strip()

    return {"rolling_summary": rolling_summary, "recent_window": recent_window}


# -------------------------
# 5) Node：推进 round（让下一次 invoke 进入下一轮）
# -------------------------

def node_advance(state: State) -> Dict[str, Any]:
    return {"round": int(state["round"]) + 1}


def build_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("run_one_round", node_run_one_round)
    builder.add_node("update_memory", node_update_memory)
    builder.add_node("advance", node_advance)

    builder.add_edge(START, "run_one_round")
    builder.add_edge("run_one_round", "update_memory")
    builder.add_edge("update_memory", "advance")
    builder.add_edge("advance", END)

    # 开启 checkpointer：每个 superstep 会保存 state，用于后续按 thread_id 续跑 :contentReference[oaicite:5]{index=5}
    return builder.compile(checkpointer=checkpointer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="首次初始化线程（必须提供 --topic）")
    parser.add_argument("--topic", default="", help="仅 init 时需要")
    parser.add_argument("--thread-id", required=True, help="线程ID：相同ID会续跑，不同ID是新线程")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-5-nano"))
    parser.add_argument("--db", default="lg03_checkpoints.db", help="SQLite checkpoint DB 路径")
    args = parser.parse_args()

    checkpointer = SqliteSaver(sqlite3.connect(args.db))
    graph = build_graph(checkpointer)

    # thread_id 通过 config.configurable 传入 :contentReference[oaicite:6]{index=6}
    config = {"configurable": {"thread_id": args.thread_id}}

    if args.init:
        if not args.topic:
            raise SystemExit("init 模式必须提供 --topic")
        init_state: State = {
            "topic": args.topic,
            "model": args.model,
            "round": 1,
            "last_completed_round": 0,
            "rolling_summary": "",
            "recent_window": [],
            "last_round_pack": {},
            "rounds": [],
        }
        out = graph.invoke(init_state, config=config)
    else:
        # 继续执行：不再传完整 state，直接用 {} 触发从 checkpointer 加载上一次 state 并推进一轮
        out = graph.invoke({}, config=config)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[线程] {args.thread_id}  已完成轮次: {out.get('last_completed_round')}  下一轮将运行: {out.get('round')}")


if __name__ == "__main__":
    main()
