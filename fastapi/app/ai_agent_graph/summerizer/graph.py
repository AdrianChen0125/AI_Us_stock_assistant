from langgraph.graph import StateGraph, END
from typing import TypedDict
from .nodes import (
    should_translate,
    translate_inputs,
    skip_translation,
    generate_report,
    AgentState
)



def build_report_graph():
    builder = StateGraph(AgentState)

    # 加入實際節點
    builder.add_node("translate", translate_inputs)
    builder.add_node("skip_translation", skip_translation)
    builder.add_node("generate_report", generate_report)

    # 加入條件節點（這不是一個 function，而是「虛擬選擇器」）
    builder.set_conditional_entry_point(
        should_translate,
        {
            "translate": "translate",
            "skip": "skip_translation"
        }
    )

    # 定義流程
    builder.add_edge("translate", "generate_report")
    builder.add_edge("skip_translation", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile()