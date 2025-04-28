
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from openai import OpenAI
import pandas as pd
import os

from utils import (
    fetch_sentiment_topic_summary,
    fetch_economic_index_summary,
    fetch_market_price_summary,
    fetch_market_price_last_7_days,
    fetch_overall_sentiment_summary,
    fetch_top10_symbols_this_week,
    fetch_top5_sectors_this_week,
    search_news_for_keywords,
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ReportState(TypedDict, total=False):
    age: str
    experience: str
    interest: List[str]
    sources: str
    risk: str
    language: str
    email: str
    sentiment_summary: str
    sentiment_df: pd.DataFrame
    eco_index_summary: str
    user_interest_news: str
    market_price_df: pd.DataFrame
    market_price_summary: str
    final_prompt: str
    final_report: str
    
# --- Define all your Nodes here ---
def fetch_sentiment_summary(state):
    df, summary = fetch_sentiment_topic_summary()

    if "keywords" in df.columns:
        df['news'] = df["keywords"].apply(search_news_for_keywords)
    else:
        df["news"] = "No Keywords Available"

    state["sentiment_summary"] = summary
    state["sentiment_df"] = df
    return state

def fetch_economic_index(state):
    index_df, eco_index = fetch_economic_index_summary()
    state["eco_index_summary"] = eco_index
    return state

def fetch_user_interest_news(state):
    interests = state["sources"]
    news = search_news_for_keywords(interests)
    state["user_interest_news"] = news
    return state

def fetch_market_price_summary(state):
    df, summary = fetch_market_price_last_7_days()
    state["market_price_df"] = df
    state["market_price_summary"] = summary
    return state

def assemble_prompt(state):
    profile_info = f"""
    User Profile:
    - Age: {state['age']}
    - Experience: {state['experience']}
    - Preferences: {state['interest']}
    - Sources: {state['sources']}
    - Risk Tolerance: {state['risk']}
    """

    prompt = f"""
    use {state['language']} for all report  
    You are a professional investment advisor. Based on the following information, provide a clear and structured investment analysis and suggestion based on 
    【User Profile】{profile_info}
    ---
    1. 【Economic Analysis】
    Analyze the economic indicators over the past 6 months, identify key turning points, and briefly describe the macroeconomic trends:
    {state['eco_index_summary']},
    {state["market_price_summary"]}
    ---
    2. 【Market Sentiment & News】
    Based on the sentiment summary below and related news, determine market sentiment (bullish/bearish), and highlight impacted sectors or stocks also mention the news you found:
    {state['sentiment_summary']}
    News:
    {state['sentiment_df']['news'].tolist()}
    ---
    3. 【User's Focused Stocks or Topics】
    News related to user's interest:
    {state['user_interest_news']}
    ---
    Provide insights, risk/opportunity assessment, and give a recommendation based on the above context.
    ---
    Present your response in this structure:
    - Macroeconomic Overview
    - Market Sentiment
    - Stock/Topic Recommendation
    """
    state["final_prompt"] = prompt
    return state

def generate_report_with_openai(state):
    prompt = state["final_prompt"]
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        state["final_report"] = response.choices[0].message.content.strip()
    except Exception as e:
        state["final_report"] = f"❌ Error generating report: {e}"
    return state
# --- Setup Graph ---

def setup_langgraph():
    graph = StateGraph(ReportState)

    graph.add_node("fetch_sentiment_summary", fetch_sentiment_summary)
    graph.add_node("fetch_economic_index", fetch_economic_index)
    graph.add_node("fetch_user_interest_news", fetch_user_interest_news)
    graph.add_node("fetch_market_price_summary", fetch_market_price_summary)
    graph.add_node("assemble_prompt", assemble_prompt)
    graph.add_node("generate_report_with_openai", generate_report_with_openai)

    graph.set_entry_point("fetch_sentiment_summary")

    graph.add_edge("fetch_sentiment_summary", "fetch_economic_index")
    graph.add_edge("fetch_economic_index", "fetch_market_price_summary")
    graph.add_edge("fetch_market_price_summary", "fetch_user_interest_news")
    graph.add_edge("fetch_user_interest_news", "assemble_prompt")
    graph.add_edge("assemble_prompt", "generate_report_with_openai")
    graph.add_edge("generate_report_with_openai", END)

    return graph.compile()

def generate_personal_report_via_langgraph(age, experience, interest, sources, risk, language, email):

    if isinstance(interest, set):
        interest = list(interest)
        
    inputs = {
        "age": age,
        "experience": experience,
        "interest": interest,
        "sources": ",".join(sources) if isinstance(sources, list) else sources,
        "risk": risk,
        "language": language,
        "email": email,
    }
    print("👉 Inputs Received:", inputs)
    
    graph = setup_langgraph()
    result = graph.invoke(inputs)
    return result["final_report"]