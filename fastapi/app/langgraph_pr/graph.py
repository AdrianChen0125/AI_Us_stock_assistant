from langgraph.graph import StateGraph, END
from langgraph.channels import LastValue
from typing import TypedDict,Annotated
import pandas as pd
from .nodes import *

class State(TypedDict):
    econ_df: Annotated[pd.DataFrame, LastValue(pd.DataFrame)]
    market_df: Annotated[pd.DataFrame, LastValue(pd.DataFrame)]
    econ_summary: Annotated[str, LastValue(str)]
    market_summary: Annotated[str, LastValue(str)]
    summary: Annotated[str, LastValue(str)]
    report: Annotated[str, LastValue(str)]

def get_graph():
    graph = StateGraph(State)

    graph.add_node("fetch_econ", fetch_econ)
    graph.add_node("fetch_market", fetch_market)
    graph.add_node("summarize_econ", summarize_econ)
    graph.add_node("summarize_market", summarize_market)
    graph.add_node("combine_summary", combine_summary)
    graph.add_node("llm", call_llm)

    graph.add_edge("__start__", "fetch_econ")
    graph.add_edge("fetch_econ", "summarize_econ")
    graph.add_edge("summarize_econ", "fetch_market")
    graph.add_edge("fetch_market", "summarize_market")
    graph.add_edge("summarize_market", "combine_summary")
    graph.add_edge("combine_summary", "llm")
    graph.add_edge("llm", END)

    return graph.compile()