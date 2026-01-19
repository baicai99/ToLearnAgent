# 11_debater_structured.py
# 目标：让“辩手 Agent”输出强结构化内容，避免散文。
# 一个知识点：结构化辩论发言（claims/rebuttals/concessions/questions）
#
# 运行：
#   python 11_debater_structured.py --topic "..." --stance pro
#
# 依赖：
#   pip install -U "langchain[openai]" 或 "langchain[anthropic]"
#   pip install pydantic
#
# 说明：
# - create_agent 的结构化输出会出现在 structured_response。
# - ToolStrategy(ResponseFormat) 用于提高结构化输出稳定性。

import os
import argparse
from typing import List, Literal

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class DebaterTurn(BaseModel):
    stance: Literal["pro", "con"] = Field(description="立场：pro 正方 / con 反方")
    claims: List[str] = Field(description="本方核心论点（3-6条，每条一句话）")
    rebuttals: List[str] = Field(description="对对方可能论点的预判与反驳（2-4条）")
    concessions: List[str] = Field(description="本方承认的限制/不确定点（0-3条）")
    questions: List[str] = Field(description="向对方提问（1-3条，尖锐但不长篇）")


def build_system_prompt(stance: str) -> str:
    role = "正方" if stance == "pro" else "反方"
    return f"""
你是辩手（{role}）。你必须输出结构化结果（DebaterTurn），不得输出多余文字。

要求：
1) 全中文
2) claims 每条一句话，尽量可检验、可比较
3) rebuttals 针对对方“可能的最强论点”进行反驳
4) concessions 真实、克制（不要为了显得客观而乱写）
5) questions 用于下一步交叉质询（短、聚焦）
""".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="辩题")
    parser.add_argument("--stance", choices=["pro", "con"], required=True, help="pro 正方 / con 反方")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-4.1"))
    args = parser.parse_args()

    agent = create_agent(
        model=args.model,
        system_prompt=build_system_prompt(args.stance),
        response_format=ToolStrategy(DebaterTurn),
    )

    user_msg = f"""
辩题：{args.topic}

请以你的立场输出 DebaterTurn。
注意：不要引用外部资料；只用一般常识与逻辑推演。
""".strip()

    result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
    structured = result.get("structured_response")

    print(structured)


if __name__ == "__main__":
    main()
