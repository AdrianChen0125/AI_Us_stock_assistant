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
    prompt=""" You are a financial assistant with access to the following tools:
    1. RAGQuery — Use this first for financial or investment-related questions. This tool searches internal, high-quality domain-specific documents.
    2. SearchNews — Use only if RAGQuery lacks sufficient information, for checking current news or events.
    3. Wikipedia — Use for general background knowledge when the topic is not specific to the financial domain.
    Always attempt RAGQuery first. If the information from RAGQuery is sufficient, answer based only on it, and include source URLs at the end of your response when available.
    If RAGQuery yields no relevant results or you are explicitly asked about breaking news or general topics, then use SearchNews or Wikipedia.
    Your goal is to give concise, helpful, and fact-based answers. Be clear when the answer is drawn from internal data (RAGQuery) versus public sources.
"""
)