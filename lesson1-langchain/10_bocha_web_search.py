# 10_bocha_web_search.py
# 目标：把 Bocha Web Search API 封装成一个“稳定函数”，返回标准化 evidence 列表
# 运行：
#   export BOCHA_API_KEY="你的key"
#   python 10_bocha_web_search.py --q "阿里巴巴2024年ESG报告重点" --count 5 --freshness oneYear
#
# 依赖：
#   pip install requests
#
# 参考：
# - Bocha 开放平台给出 web-search endpoint 与请求体字段。:contentReference[oaicite:8]{index=8}
# - freshness 可选值示例见官方 MCP repo 描述。:contentReference[oaicite:9]{index=9}

import os
import argparse
import requests


BOCHA_ENDPOINT = "https://api.bochaai.com/v1/web-search"


def bocha_web_search(query: str, freshness: str = "noLimit", count: int = 8, summary: bool = True) -> list[dict]:
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
        "summary": summary,
        "count": count,
    }

    resp = requests.post(BOCHA_ENDPOINT, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # 按开放平台示例，主体结果在 data.webPages.value（字段名以实际返回为准）
    web_pages = (data.get("data") or data).get("webPages", {})
    items = web_pages.get("value", []) if isinstance(web_pages, dict) else []

    evidence = []
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
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="搜索 query")
    parser.add_argument("--freshness", default="noLimit")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    ev = bocha_web_search(args.q, freshness=args.freshness, count=args.count, summary=True)

    print(f"共返回 {len(ev)} 条 evidence：\n")
    for e in ev:
        print(f"[{e['evidence_id']}] {e['title']}")
        print(f"  - url: {e['url']}")
        if e["summary"]:
            print(f"  - summary: {e['summary'][:160]}{'...' if len(e['summary'])>160 else ''}")
        else:
            print(f"  - snippet: {e['snippet'][:160]}{'...' if len(e['snippet'])>160 else ''}")
        print()


if __name__ == "__main__":
    main()
