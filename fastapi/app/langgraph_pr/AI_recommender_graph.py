from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Annotated
from .nodes.AI_recommender import *

class RecommenderState(TypedDict):
    holdings: list[str]
    recommended: list[str]
    style_preference: list[str]
    risk_tolerance: str
    stock_df: Annotated[pd.DataFrame, ...]
    prompt: str
    analysis: str

def build_graph():
    graph = StateGraph(RecommenderState)
    graph.add_node("fetch", RunnableLambda(fetch_stock_data))
    graph.add_node("build_prompt", RunnableLambda(build_prompt))
    graph.add_node("llm", RunnableLambda(run_llm))
    
    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "build_prompt")
    graph.add_edge("build_prompt", "llm")
    graph.add_edge("llm", END)
    return graph.compile()