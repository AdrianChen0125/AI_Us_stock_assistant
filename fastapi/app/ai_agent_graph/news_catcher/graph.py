# graph.py

from typing import List, Dict, Optional, TypedDict
from langgraph.graph import StateGraph, END

from .nodes import (
    get_sentiment_summary,
    filter_similar_topic,
    multiple_search_tool,
    summarize_results
)

class NewsAgentState(TypedDict, total=False):
    sentiment_summary: List[str]
    filtered_topics: List[str]
    search_results: List[Dict]
    summarized_results: List[Dict]


def build_langgraph():
    builder = StateGraph(state_schema=NewsAgentState)

    builder.add_node("get_sentiment_summary", get_sentiment_summary)
    builder.add_node("filter_similar_topic", filter_similar_topic)
    builder.add_node("multiple_search_tool", multiple_search_tool)
    builder.add_node("summarize_results", summarize_results)

    builder.set_entry_point("get_sentiment_summary")
    builder.add_edge("get_sentiment_summary", "filter_similar_topic")
    builder.add_edge("filter_similar_topic", "multiple_search_tool")
    builder.add_edge("multiple_search_tool", "summarize_results")
    builder.add_edge("summarize_results", END)

    return builder.compile()