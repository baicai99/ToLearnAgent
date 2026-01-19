# 07_structured_output.py
# 目标：让 create_agent 直接返回“可解析”的结构化输出（Pydantic）
# 运行：
#   python 07_structured_output.py --text "John Doe, [email protected], (555) 123-4567"
#
# 依赖：
#   pip install -U "langchain[openai]"  或  pip install -U "langchain[anthropic]"
#   pydantic 通常会随依赖安装；若缺失：pip install pydantic
#
# 说明：
# - structured output 文档：create_agent 会把结构化结果放在 structured_response 键里。:contentReference[oaicite:1]{index=1}

import os
import argparse
from pydantic import BaseModel, Field
from langchain.agents import create_agent


class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address")
    phone: str = Field(description="The phone number")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="待抽取文本")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-4.1"))
    args = parser.parse_args()

    agent = create_agent(
        model=args.model,
        response_format=ContactInfo,   # 关键：指定 schema
        system_prompt="Extract contact info. Return only structured output.",
    )

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": f"Extract contact info from: {args.text}"}
        ]
    })

    # 结构化结果在 structured_response 中。:contentReference[oaicite:2]{index=2}
    print("structured_response =>")
    print(result.get("structured_response"))

    # 同时打印 keys，方便你观察 agent state 里还有什么
    print("\nstate keys =>", list(result.keys()))


if __name__ == "__main__":
    main()
