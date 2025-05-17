# graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict

from .nodes import (
    get_overall_sentiment,
    get_sentiment_summary,
    get_sector_summary,
    get_symbol_summary,
    generate_sentiment_report
)

class MarketState(TypedDict, total=False):
    reddit_df: str
    sentiment_summary: str
    sector_summary: str
    symbol_summary: str
    report: str

def create_market_report_graph():
    builder = StateGraph(dict)

    builder.add_node("get_reddit_df", get_overall_sentiment)
    builder.add_node("get_sentiment", get_sentiment_summary)
    builder.add_node("get_sectors", get_sector_summary)
    builder.add_node("get_symbols", get_symbol_summary)
    builder.add_node("generate_report", generate_sentiment_report)

    builder.set_entry_point("get_reddit_df")
    builder.add_edge("get_reddit_df", "get_sentiment")
    builder.add_edge("get_sentiment", "get_sectors")
    builder.add_edge("get_sectors", "get_symbols")
    builder.add_edge("get_symbols", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile()