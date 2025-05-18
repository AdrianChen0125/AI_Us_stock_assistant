from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import pandas as pd
import mlflow

from .nodes import fetch_stock_data, build_prompt_node, run_llm

# 初始化 MLflow autolog
mlflow.set_tracking_uri("http://mlflow:5001")
mlflow.set_experiment("stock_recommendation_v1")
mlflow.langchain.autolog()

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
    graph.add_node("fetch", fetch_stock_data)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node("llm", run_llm)

    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "build_prompt")
    graph.add_edge("build_prompt", "llm")
    graph.add_edge("llm", END)

    return graph.compile()