# 02_messages.py
# 目标：学会两种“对话消息”传参方式（dict 列表 / Message objects）
# 运行：
#   python 02_messages.py --mode dict
#   python 02_messages.py --mode objects
#
# 参考：LangChain Models 文档展示了两种消息输入方式。:contentReference[oaicite:13]{index=13}

import os
import argparse
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage  # 文档示例同路径 :contentReference[oaicite:14]{index=14}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dict", "objects"], default="dict")
    parser.add_argument("--model", default=os.getenv("LC_MODEL", "gpt-4.1"))
    args = parser.parse_args()

    model = init_chat_model(args.model, temperature=0)

    if args.mode == "dict":
        conversation = [
            {"role": "system", "content": "你是一个严谨的中文助手。"},
            {"role": "user", "content": "把英文 'I love programming.' 翻译成中文。"},
        ]
        resp = model.invoke(conversation)
        print(resp.content)
    else:
        conversation = [
            SystemMessage("你是一个严谨的中文助手。"),
            HumanMessage("把英文 'I love programming.' 翻译成中文。"),
            AIMessage("我喜欢编程。"),
            HumanMessage("把英文 'I love building applications.' 翻译成中文。"),
        ]
        resp = model.invoke(conversation)
        print(resp.content)


if __name__ == "__main__":
    main()
