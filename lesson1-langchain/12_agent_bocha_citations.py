# 12_agent_bocha_citations.py
# 目标：create_agent + Bocha 搜索工具 + 结构化输出 + 引用 evidence_id
#
# 本课只教一个核心点：
#   Agent 必须先调用 bocha_web_search() 获取证据，再输出结构化答案：
#   { answer, citations:[e1,e2], confidence, missing_info }
#
# 依赖：
#   pip install -U "langchain[openai]" 或 "langchain[anthropic]"
#   pip install requests pydantic
#
# Bocha 官方开放平台给出的 Web Search API 示例包括：
#   endpoint: https://api.bochaai.com/v1/web-search
#   header: Authorization: Bearer <API-KEY>
#   body: query/freshness/summary/count   :contentReference[oaicite:6]{index=6}
#
# LangChain structured output 文档说明 structured_response 返回位置。:contentReference[oaicite:7]{index=7}

import os
import json
import argparse
from typing import List, Dict, Any

import requests
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy  # Quickstart 示范用法 :contentReference[oaicite:8]{index=8}


BOCHA_ENDPOINT = "https://api.bochaai.com/v1/web-search"

# 用于教学演示：记录最近一次工具调用返回的 evidence，便于后置校验 citations 合法性。
# 生产环境应放入 state / 数据库，而不是全局变量。
LAST_EVIDENCE: List[Dict[str, Any]] = []


def bocha_web_search(query: str, freshness: str = "oneYear", count: int = 6, summary: bool = True) -> List[Dict[str, Any]]:
    """
    Bocha Web Search Tool
    入参：query/freshness/count/summary
    出参：标准化 evidence 列表，每条包含 evidence_id/title/url/summary/snippet/datePublished
    """
    api_key = os.getenv("BOCHA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少环境变量 BOCHA_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "freshness": freshness,
        "summary": bool(summary),
        "count": int(count),
    }

    resp = requests.post(BOCHA_ENDPOINT, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # 兼容不同包装层：有的返回直接在根，有的在 data 字段
    root = data.get("data") if isinstance(data, dict) and "data" in data else data
    web_pages = root.get("webPages", {}) if isinstance(root, dict) else {}
    items = web_pages.get("value", []) if isinstance(web_pages, dict) else []

    evidence: List[Dict[str, Any]] = []
    for i, it in enumerate(items, start=1):
        evidence.append({
            "evidence_id": f"e{i}",
            "title": it.get("name", ""),
            "url": it.get("url", ""),
            "siteName": it.get("siteName", ""),
            "snippet": it.get("snippet", ""),
            "summary": it.get("summary", ""),
            "datePublished": it.get("datePublished", ""),
        })

    global LAST_EVIDENCE
    LAST_EVIDENCE = evidence
    return evidence


class AnswerWithCitations(BaseModel):
    answer: str = Field(description="中文回答，尽量简洁、条理清楚")
    citations: List[str] = Field(description="引用的 evidence_id 列表，例如 ['e1','e3']，必须来自本次搜索返回")
    confidence: float = Field(description="置信度 0~1")
    missing_info: List[str] = Field(description="为了更准确还缺少的信息（若无则空数组）")


SYSTEM_PROMPT = """
你是一个严谨的中文助手。你必须遵守以下规则：

1) 在回答任何需要外部事实/资料的问题前，必须先调用工具 bocha_web_search 获取证据。
2) 最终输出必须是结构化输出（AnswerWithCitations），不要输出多余文本。
3) citations 只能填写本次 bocha_web_search 返回的 evidence_id（如 e1/e2/...）。
4) answer 中不要粘贴大段网页内容，按要点归纳即可。
5) 如果搜索结果不足以支持结论，降低 confidence，并把缺口写入 missing_info。
""".strip()


def validate_citations(structured: AnswerWithCitations, evidence: List[Dict[str, Any]]) -> List[str]:
    """返回 citations 中不合法的那些（不在 evidence_id 集合内）。"""
    valid_ids = {e["evidence_id"] for e in evidence}
    bad = [c for c in structured.citations if c not in valid_ids]
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="用户问题")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-4.1"))
    parser.add_argument("--freshness", default="oneYear")
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()

    # 把 bocha_web_search 暴露给 agent 作为工具（函数工具）
    # 同时用 ToolStrategy 强制结构化输出走工具调用策略，提高稳定性。:contentReference[oaicite:9]{index=9}
    agent = create_agent(
        model=args.model,
        tools=[bocha_web_search],
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(AnswerWithCitations),
    )

    # 提示：为了让 Agent “知道如何传 freshness/count”，我们把参数写入用户消息中（
