from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from .tools import wiki_tool, news_tool, rag_tool, stock_tool

import mlflow
import os
os.environ["MLFLOW_LANGCHAIN_ENABLE_DEBUG"] = "true"

mlflow.set_tracking_uri("http://mlflow:5001")
mlflow.set_experiment("chat_bot")
mlflow.langchain.autolog()

# Initialize language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Directly use the defined Tool objects
tools = [wiki_tool, news_tool, rag_tool, stock_tool]

# Create LangGraph ReAct agent
chat_graph = create_react_agent(
    model=llm,
    tools=tools,
    prompt="""
        You are a financial assistant that uses tools to answer user questions.

        Always consider using `RAGQuery` first to search internal domain-specific documents.
        If `RAGQuery` is not enough, consider using SearchNews or Wikipedia.
    """
    )