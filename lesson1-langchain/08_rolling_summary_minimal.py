# 08_rolling_summary_minimal.py
# 目标：实现“Recent Window + Rolling Summary”的最小可运行版本（不接 agent）
# 运行：
#   python 08_rolling_summary_minimal.py
#
# 你将得到：
# - summary：滚动摘要（压缩旧内容）
# - window：最近若干轮对话原文（保留细节）
#
# 说明：
# - 长对话容易超过上下文窗口，且成本/效果都变差，因此需要压缩/遗忘策略。:contentReference[oaicite:4]{index=4}

import os
from langchain.chat_models import init_chat_model


def summarize(model, prev_summary: str, old_dialogue: str) -> str:
    """
    用模型把 old_dialogue 压缩进 summary。
    这里刻意写得“很硬”，避免生成散文跑偏：输出要求是要点列表。
    """
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
    resp = model.invoke(prompt)
    return resp.content.strip()


def main():
    model_id = os.getenv("LC_MODEL", "gpt-4.1")
    model = init_chat_model(model_id, temperature=0)

    summary = ""         # 滚动摘要（压缩层）
    window = []          # recent window（细节层），元素是 {"role":..., "content":...}

    # 用字符数近似做预算（足够入门；后面再换 token 计数）
    MAX_WINDOW_CHARS = 1200
    KEEP_LAST_N_MSG = 6   # 压缩后仍保留的最近消息数（含 user/assistant）

    print("进入对话（输入 exit 退出）。我会在 window 过长时自动摘要压缩。\n")

    while True:
        user_text = input("你：").strip()
        if user_text.lower() in ("exit", "quit"):
            break

        window.append({"role": "user", "content": user_text})

        # 这里为了演示，不生成 assistant，只做“窗口管理 + 摘要压缩”
        # 你也可以把 assistant 假设为固定回执
        window.append({"role": "assistant", "content": f"(回执) 已收到：{user_text[:20]}..."})

        # 触发压缩：window 总长度超阈值
        total_chars = sum(len(m["content"]) for m in window)
        if total_chars > MAX_WINDOW_CHARS and len(window) > KEEP_LAST_N_MSG:
            # 需要被压缩的“旧内容”
            compress_part = window[:-KEEP_LAST_N_MSG]
            old_dialogue = "\n".join([f'{m["role"]}: {m["content"]}' for m in compress_part])

            summary = summarize(model, summary, old_dialogue)
            window = window[-KEEP_LAST_N_MSG:]

            print("\n[系统] 已触发摘要压缩。当前 summary：")
            print(summary)
            print("[系统] 已保留 recent window 条数：", len(window), "\n")

    print("\n=== 最终 Summary ===")
    print(summary if summary else "(空)")
    print("\n=== 最终 Recent Window ===")
    for m in window:
        print(m["role"] + ":", m["content"])


if __name__ == "__main__":
    main()
