# 11_search_gate.py
# 目标：实现“受控检索门控”（Search Gate）——纯规则、确定性、可复盘
# 你将学到：
# 1) 去重：同 query 合并
# 2) 预算：每轮最多允许多少个 query，每个 query 最多返回多少条
# 3) 时效：freshness 默认策略（可覆盖）
# 4) 输出：approved / rejected（带理由）
#
# 这是后续“辩论系统由裁判统一调度搜索”的核心组件。

import argparse
import json
import re
from typing import Any, Dict, List, Tuple


ALLOWED_FRESHNESS = {"noLimit", "oneYear", "oneMonth", "oneWeek", "oneDay"}  # 常见枚举


def normalize_query(q: str) -> str:
    """对 query 做归一化，用于去重。"""
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def default_freshness(purpose: str) -> str:
    """
    默认时效策略（可按你业务再调）：
    - api_ref / docs：偏长期（oneYear / noLimit）
    - news / current：偏近期（oneWeek / oneDay）
    - 其他：oneMonth
    """
    p = (purpose or "").strip().lower()
    if p in {"api_ref", "docs", "spec"}:
        return "oneYear"
    if p in {"news", "current", "today", "recent"}:
        return "oneWeek"
    if p in {"fact_check"}:
        return "oneMonth"
    return "oneMonth"


def gate_requests(
    requests_: List[Dict[str, Any]],
    max_queries: int = 2,
    max_count_per_query: int = 8,
) -> Dict[str, Any]:
    """
    输入：requests_（用户/子模块提出的搜索请求）
    输出：
      - approved: 允许执行的 queries（已去重、已应用预算与默认 freshness）
      - rejected: 被拒绝的请求（含原因）
      - merged: 去重合并信息（便于复盘）
    """
    rejected: List[Dict[str, Any]] = []
    merged_map: Dict[str, Dict[str, Any]] = {}

    # 1) 基础校验 + 去重合并
    for idx, r in enumerate(requests_):
        raw_q = (r.get("query") or "").strip()
        if not raw_q:
            rejected.append({"request": r, "reason": "empty_query"})
            continue

        nq = normalize_query(raw_q)
        purpose = (r.get("purpose") or "").strip() or "general"

        freshness = (r.get("freshness") or "").strip() or default_freshness(purpose)
        if freshness not in ALLOWED_FRESHNESS:
            rejected.append({"request": r, "reason": f"invalid_freshness:{freshness}"})
            continue

        count = r.get("count", None)
        try:
            count = int(count) if count is not None else max_count_per_query
        except Exception:
            rejected.append({"request": r, "reason": f"invalid_count:{r.get('count')}"})
            continue
        count = max(1, min(count, max_count_per_query))

        item = {
            "query": raw_q,
            "norm_query": nq,
            "purpose": purpose,
            "freshness": freshness,
            "count": count,
            "sources": [{"index": idx, "raw": r}],
        }

        if nq in merged_map:
            # 合并策略：保留更“紧”的 freshness（oneDay < oneWeek < oneMonth < oneYear < noLimit）
            # 简化实现：按优先级表比较
            priority = {"oneDay": 1, "oneWeek": 2, "oneMonth": 3, "oneYear": 4, "noLimit": 5}
            old = merged_map[nq]
            if priority[item["freshness"]] < priority[old["freshness"]]:
                old["freshness"] = item["freshness"]
            old["count"] = max(old["count"], item["count"])
            old["sources"].extend(item["sources"])
        else:
            merged_map[nq] = item

    merged = list(merged_map.values())

    # 2) 应用预算：只允许 max_queries 个 query
    # 排序策略：purpose 越“严肃”优先；你可按业务扩展
    purpose_rank = {"api_ref": 1, "docs": 1, "fact_check": 2, "general": 3, "news": 4, "current": 4}
    merged.sort(key=lambda x: purpose_rank.get(x["purpose"].lower(), 99))

    approved = merged[:max_queries]
    overflow = merged[max_queries:]

    for it in overflow:
        rejected.append({"request": it, "reason": f"budget_exceeded:max_queries={max_queries}"})

    return {
        "approved": [
            {"query": a["query"], "freshness": a["freshness"], "count": a["count"], "purpose": a["purpose"]}
            for a in approved
        ],
        "rejected": rejected,
        "merged_debug": [
            {"norm_query": m["norm_query"], "query": m["query"], "freshness": m["freshness"], "count": m["count"], "sources": len(m["sources"])}
            for m in merged
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-file", default="", help="JSON 文件路径（数组）")
    parser.add_argument("--max-queries", type=int, default=2)
    parser.add_argument("--max-count-per-query", type=int, default=8)
    args = parser.parse_args()

    if args.requests_file:
        with open(args.requests_file, "r", encoding="utf-8") as f:
            requests_ = json.load(f)
    else:
        # 内置示例：包含重复、缺省 freshness、缺省 count
        requests_ = [
            {"query": "Bocha web-search API query freshness summary count", "purpose": "api_ref"},
            {"query": "LangChain create_agent structured_response", "purpose": "docs", "count": 10},
            {"query": "LangChain create_agent structured_response", "purpose": "duplicate_test"},
            {"query": "  ", "purpose": "bad"},
        ]

    out = gate_requests(
        requests_=requests_,
        max_queries=args.max_queries,
        max_count_per_query=args.max_count_per_query,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
