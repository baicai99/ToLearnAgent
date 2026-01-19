# 16_multi_round_with_summary.py
# 目标：在 L15 的多轮辩论基础上，只新增“上下文摘要管理”：
#   - rolling_summary：压缩旧回合内容
#   - recent_window：保留最近若干回合细节
# 每轮只把 summary + recent_window 喂给辩手/裁判/投票人，避免上下文爆炸。
#
# 约束：不使用 web_search，不使用 LangGraph；单文件独立可跑
# 默认模型：gpt-5-nano（可用环境变量 LC_MODEL 覆盖）

import os
import json
import argparse
from typing import List, Literal, Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model


# -------------------------
# 结构化输出：辩手/裁判/投票
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


def tally_votes(votes: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"pro": 0, "con": 0, "tie": 0, "insufficient": 0}
    for v in votes:
        k = v.get("vote")
        if k in counts:
            counts[k] += 1
    return counts


def should_stop(
    judge: Dict[str, Any],
    prev_winner: Optional[str],
    conf_threshold: float,
    insufficient_threshold: float,
) -> bool:
    winner = judge.get("winner")
    conf = float(judge.get("confidence", 0))

    if winner == "insufficient" and conf >= insufficient_threshold:
        return True

    if winner in ("pro", "con") and conf >= conf_threshold and prev_winner == winner:
        return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="辩题")
    parser.add_argument("--max-rounds", type=int, default=5, help="最大轮数")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-5-nano"), help="默认 gpt-5-nano，可覆盖")
    parser.add_argument("--out", default="", help="可选：输出 JSON 文件")
    parser.add_argument("--conf-threshold", type=float, default=0.78, help="连续两轮同胜方的置信度阈值")
    parser.add_argument("--insufficient-threshold", type=float, default=0.70, help="insufficient 终止置信度阈值")
    args = parser.parse_args()

    # 2正2反：用“风格差异”做区分（不额外引入工程化模板）
    debaters = [
        {"agent_id": "pro_a", "stance": "pro", "style": "偏逻辑链路：定义-论点-推导-结论"},
        {"agent_id": "pro_b", "stance": "pro", "style": "偏执行落地：流程、成本、激励、可操作性"},
        {"agent_id": "con_a", "stance": "con", "style": "偏风险控制：失败模式、外部性、激励扭曲"},
        {"agent_id": "con_b", "stance": "con", "style": "偏长期影响：公平、文化、协作摩擦、人才结构"},
    ]

    # -------------------------
    # 创建 agents（提示词直接写在此处，便于初学者观察）
    # -------------------------

    debater_agents = {}
    for d in debaters:
        role = "正方" if d["stance"] == "pro" else "反方"
        debater_agents[d["agent_id"]] = create_agent(
            model=args.model,
            system_prompt=(
                f"你是辩手（{role}），ID={d['agent_id']}。\n"
                f"你的论证风格：{d['style']}。\n"
                "硬性规则：\n"
                "1) 只输出结构化 DebateTurn，不得输出多余文字。\n"
                "2) 全中文。\n"
                "3) 不允许引用外部资料，只能基于题目、常识与对话中已给出的内容。\n"
                "4) 每条 claims/rebuttals/questions 必须短、明确、可比较。\n"
            ),
            response_format=ToolStrategy(DebateTurn),
        )

    judge_agent = create_agent(
        model=args.model,
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

    voter_agents = {}
    for d in debaters:
        voter_agents[d["agent_id"]] = create_agent(
            model=args.model,
            system_prompt=(
                f"你是投票人（voter_id={d['agent_id']}），处于陪审员模式（juror），尽量客观。\n"
                "只输出结构化 Vote，不得输出多余文字。\n"
                "规则：\n"
                "1) 只能基于 turns 与 judge_decision 投票\n"
                "2) 不允许引入外部资料\n"
                "3) vote 可为 pro/con/tie/insufficient，不要求投自己阵营\n"
            ),
            response_format=ToolStrategy(Vote),
        )

    tiebreak_agent = create_agent(
        model=args.model,
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

    # 用于“摘要压缩”的模型（同样默认 gpt-5-nano）
    summarizer = init_chat_model(args.model, temperature=0)

    # -------------------------
    # Rolling Summary + Recent Window（本课新增内容）
    # -------------------------
    rolling_summary = ""            # 压缩层
    recent_window: List[Dict[str, Any]] = []  # 细节层：存最近若干轮的结构化片段

    # 简化预算：用字符数近似（足够教学）
    MAX_RECENT_CHARS = 6000         # recent_window 超过就触发压缩
    KEEP_LAST_ITEMS = 6             # recent_window 压缩后保留的“片段”数量
    MAX_SUMMARY_CHARS = 2500        # summary 过长也会再压缩（简单截断+再总结）

    def recent_chars() -> int:
        return len(json.dumps(recent_window, ensure_ascii=False))

    def update_summary_and_window(new_round_pack: Dict[str, Any]) -> None:
        """把新一轮内容加入 recent_window；超过预算则把旧内容压入 rolling_summary。"""
        nonlocal rolling_summary, recent_window

        recent_window.append(new_round_pack)

        # 若 recent_window 太长，压缩最旧的一段
        if recent_chars() > MAX_RECENT_CHARS and len(recent_window) > KEEP_LAST_ITEMS:
            to_compress = recent_window[:-KEEP_LAST_ITEMS]
            remain = recent_window[-KEEP_LAST_ITEMS:]

            compress_text = json.dumps(to_compress, ensure_ascii=False)
            prompt = (
                "你在做多轮辩论的滚动摘要压缩。\n"
                "请把“新增历史片段”合并进“既有摘要”，输出新的摘要。\n"
                "要求：\n"
                "1) 全中文\n"
                "2) 要点式（最多 10 条）\n"
                "3) 必须保留：双方关键主张、关键反驳、裁判结论变化、尚未解决的争议点\n"
                "4) 不要寒暄，不要扩写，不要引入外部资料\n\n"
                f"既有摘要：\n{rolling_summary if rolling_summary else '(空)'}\n\n"
                f"新增历史片段：\n{compress_text}\n\n"
                "输出新的摘要："
            )
            rolling_summary = summarizer.invoke(prompt).content.strip()
            recent_window = remain

        # 若 rolling_summary 自身太长，再压一次（避免 summary 也膨胀）
        if len(rolling_summary) > MAX_SUMMARY_CHARS:
            prompt2 = (
                "请将以下摘要再次压缩，要求：\n"
                "1) 全中文\n"
                "2) 要点式（最多 10 条）\n"
                "3) 不丢失关键争议与结论\n\n"
                f"摘要：\n{rolling_summary}\n\n"
                "输出更短摘要："
            )
            rolling_summary = summarizer.invoke(prompt2).content.strip()

    # -------------------------
    # 多轮主循环
    # -------------------------
    rounds: List[Dict[str, Any]] = []
    prev_winner: Optional[str] = None

    # 为了让初学者更直观看到“喂了什么”，我们把给 agent 的 payload 直接写在循环里
    for r in range(1, args.max_rounds + 1):
        phase = "opening" if r == 1 else "rebuttal"

        # 1) 生成本轮 4 位辩手发言
        turns: List[Dict[str, Any]] = []
        for d in debaters:
            aid = d["agent_id"]
            stance = d["stance"]

            user_payload = {
                "topic": args.topic,
                "round": r,
                "phase": phase,
                "your_agent_id": aid,
                "your_stance": stance,

                # 这就是“上下文管理”的核心：只喂 summary + recent_window
                "rolling_summary": rolling_summary,
                "recent_window": recent_window,

                "instruction": (
                    "请输出 DebateTurn。\n"
                    "若 phase=opening：给出你的初始立论。\n"
                    "若 phase=rebuttal：基于 rolling_summary 与 recent_window 的信息，进行反驳与补强，并回答对方问题。\n"
                    "注意：不要引用外部资料，只用常识与对话内信息。"
                ),
                "format_constraints": {
                    "claims_range": "3-6",
                    "rebuttals_range": "2-5",
                    "questions_range": "1-3",
                },
            }

            res = debater_agents[aid].invoke({"messages": [{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}]})
            turn = res.get("structured_response")
            turns.append(turn)

        # 2) 裁判裁决（同样只基于 summary+recent_window+本轮 turns）
        judge_payload = {
            "topic": args.topic,
            "round": r,
            "phase": phase,
            "rolling_summary": rolling_summary,
            "recent_window": recent_window,
            "turns": turns,
            "instruction": "请基于 turns（结合 rolling_summary/recent_window 作为历史背景）输出 JudgeDecision。不得引入外部资料。",
        }
        judge_res = judge_agent.invoke({"messages": [{"role": "user", "content": json.dumps(judge_payload, ensure_ascii=False)}]})
        judge = judge_res.get("structured_response")

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
            v_res = voter_agents[vid].invoke({"messages": [{"role": "user", "content": json.dumps(vote_payload, ensure_ascii=False)}]})
            votes.append(v_res.get("structured_response"))

        counts = tally_votes(votes)

        # 4) 最终裁决（多数票；平票走 tie-break）
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
            tb_res = tiebreak_agent.invoke({"messages": [{"role": "user", "content": json.dumps(tb_payload, ensure_ascii=False)}]})
            final = tb_res.get("structured_response")

        round_out = {
            "round": r,
            "phase": phase,
            "rolling_summary_before": rolling_summary,
            "recent_window_before": recent_window,
            "turns": turns,
            "judge_decision": judge,
            "votes": votes,
            "vote_counts": counts,
            "final_decision": final,
        }
        rounds.append(round_out)

        # 5) 更新摘要与窗口（把本轮关键内容打包进去）
        # 只把“必要信息”进入 recent_window，避免把所有字段都塞进去
        new_round_pack = {
            "round": r,
            "phase": phase,
            "turns_brief": [
                {
                    "agent_id": t.get("agent_id"),
                    "stance": t.get("stance"),
                    "claims": t.get("claims", [])[:4],
                    "rebuttals": t.get("rebuttals", [])[:4],
                    "questions": t.get("questions", [])[:3],
                    "concessions": t.get("concessions", [])[:2],
                } for t in turns
            ],
            "judge": {
                "winner": judge.get("winner"),
                "confidence": judge.get("confidence"),
                "rationale": judge.get("rationale", [])[:6],
                "focus_next_round": judge.get("focus_next_round", [])[:4],
                "key_uncertainties": judge.get("key_uncertainties", [])[:4],
            },
        }
        update_summary_and_window(new_round_pack)

        # 6) 终止判断（基于裁判胜负+置信度）
        stop = should_stop(
            judge=judge,
            prev_winner=prev_winner,
            conf_threshold=args.conf_threshold,
            insufficient_threshold=args.insufficient_threshold,
        )
        prev_winner = judge.get("winner")

        if stop:
            break

    output = {
        "topic": args.topic,
        "model": args.model,
        "max_rounds": args.max_rounds,
        "rounds_ran": len(rounds),
        "final_rolling_summary": rolling_summary,
        "final_recent_window": recent_window,
        "rounds": rounds,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[已写入] {args.out}")


if __name__ == "__main__":
    main()
