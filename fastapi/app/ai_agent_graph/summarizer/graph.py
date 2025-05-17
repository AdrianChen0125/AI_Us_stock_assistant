from langgraph.graph import StateGraph, END
from typing import TypedDict
from .nodes import get_inputs, generate_overall_report, translate_to_zh

class OverallReportState(TypedDict, total=False):
    economic_summary: str
    market_sentiment_summary: str
    stock_recommender_summary: str
    language: str
    final_report_en: str
    final_report: str

def create_overall_market_graph():
    builder = StateGraph(OverallReportState)

    builder.add_node("collect_input", get_inputs)
    builder.add_node("generate_report", generate_overall_report)
    builder.add_node("translate_report", translate_to_zh)

    builder.set_entry_point("collect_input")
    builder.add_edge("collect_input", "generate_report")
    builder.add_edge("generate_report", "translate_report")
    builder.add_edge("translate_report", END)

    return builder.compile()