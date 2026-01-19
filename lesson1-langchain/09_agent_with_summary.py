# 09_agent_with_summary.py
# 目标：把 Rolling Summary + Recent Window 接入 create_agent 的 messages 输入
# 运行：
#   python 09_agent_with_summary.py
#
# 说明：
# - 本文件只教一个点：构造“喂给 agent 的上下文”：
#   system + summary(压缩层) + recent window(细节层) + user(本轮)
# - 长对话需要遗忘/压缩，否则上下文窗口与效果会恶化。:contentReference[oaicite:5]{index=5}

import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


def summarize(model, prev_summary: str, old_dialogue: str) -> str:
    prompt = f"""
你在做对话摘要压缩。请把“新增对话内容”合并进“既有摘要”，输出新的摘要。

要求：
1) 用中文
2) 要点式（最多 8 条）
3) 保留：事实、偏好、决策、约束、待办
4) 不要保留寒暄

既有摘要：
{prev_summary if prev_summary else "(空)"}

新增对话内容：
{old_dialogue}

输出新的摘要：
"""
    return model.invoke(prompt).content.strip()


def build_messages(summary: str, window: list[dict], user_text: str) -> list[dict]:
    messages = [{"role": "system", "content": "你是一个严谨的中文助手，回答要简洁、可执行。"}]
    if summary:
        messages.append({"role": "system", "content": f"【对话摘要】\n{summary}"})
    messages.extend(window)  # recent window：保留细节
    messages.append({"role": "user", "content": user_text})
    return messages


def main():
    model_id = os.getenv("LC_MODEL", "gpt-4.1")

    # agent 用 create_agent（你前面已经跑通）
    agent = create_agent(
        model=model_id,
        system_prompt="你是一个严谨的中文助手。",
    )

    # 摘要用同一个 provider 的 chat model（也可单独用更便宜的模型）
    summarizer = init_chat_model(model_id, temperature=0)

    summary = ""
    window = []

    MAX_WINDOW_CHARS = 1600
    KEEP_LAST_N_MSG = 8

    print("进入对话（输入 exit 退出）。我会自动用 summary 控制上下文长度。\n")

    while True:
        user_text = input("你：").strip()
        if user_text.lower() in ("exit", "quit"):
            break

        # 先构造喂给 agent 的 messages（不拼全量历史）
        messages = build_messages(summary, window, user_text)

        result = agent.invoke({"messages": messages})
        # agent 的返回一般包含 messages；取最后一条 assistant
        assistant_msg = result["messages"][-1]
        print("助理：", assistant_msg.get("content", ""))

        # 更新 window：只存最近的真实对话（user + assistant）
        window.append({"role": "user", "content": user_text})
        window.append({"role": "assistant", "content": assistant_msg.get("content", "")})

        # 触发压缩
        total_chars = sum(len(m["content"]) for m in window)
        if total_chars > MAX_WINDOW_CHARS and len(window) > KEEP_LAST_N_MSG:
            compress_part = window[:-KEEP_LAST_N_MSG]
            old_dialogue = "\n".join([f'{m["role"]}: {m["content"]}' for m in compress_part])

            summary = summarize(summarizer, summary, old_dialogue)
            window = window[-KEEP_LAST_N_MSG:]

            print("\n[系统] 已摘要压缩。当前 summary（节选）：")
            print(summary[:300] + ("..." if len(summary) > 300 else ""))
            print()

    print("\n=== 最终 Summary ===")
    print(summary if summary else "(空)")


if __name__ == "__main__":
    main()
