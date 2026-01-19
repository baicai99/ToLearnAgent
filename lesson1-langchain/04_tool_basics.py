# 04_tool_basics.py
# 目标：学会把一个函数变成“工具”，并理解：工具的 docstring/参数名会进入模型提示词
# 运行：
#   python 04_tool_basics.py
#
# 参考：Quickstart 展示了 @tool 以及 ToolRuntime（进阶），这里先做最小版。:contentReference[oaicite:16]{index=16}

from langchain.tools import tool  # Quickstart 示例里使用了 @tool :contentReference[oaicite:17]{index=17}


@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


def main():
    # 这里只演示“工具是可执行函数”，还不让模型调度
    print("tool name:", add.name)
    print("tool description:", add.description)
    print("call result:", add.invoke({"a": 2, "b": 3}))


if __name__ == "__main__":
    main()
