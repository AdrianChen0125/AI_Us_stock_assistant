from langgraph.graph import StateGraph, END
from typing import TypedDict
from .nodes import (
    should_translate,
    translate_service,
    generate_report,
    AgentState
)

def build_report_graph():
    builder = StateGraph(AgentState)

    # 加入節點
    builder.add_node("generate_report", generate_report)
    builder.add_node("translate", translate_service)

    # 起點
    builder.set_entry_point("generate_report")

    # 條件判斷：是否翻譯
    builder.add_conditional_edges(
        "generate_report",
        should_translate,
        {
            "translate": "translate",
            "skip": END  # 直接結束
        }
    )

    # 翻譯後結束
    builder.add_edge("translate", END)

    return builder.compile()