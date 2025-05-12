import os
import pandas as pd
from typing import Dict, Any
from openai import OpenAI
from async_db import get_db
from services.fetch_sp500_stock import fetch_sp500_service

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🔍 Fetch stock data from the latest snapshot
async def fetch_stock_data(state: Dict[str, Any]) -> Dict[str, Any]:
    symbols = state.get("recommended", [])
    if not symbols:
        return {**state, "stock_df": pd.DataFrame()}

    db_gen = get_db()
    db = await db_gen.__anext__()

    try:
        df = await fetch_sp500_service(symbols, db)
    finally:
        await db.close() 

    return {**state, "stock_df": df}

# 🧠 Build natural language prompt for LLM
def build_prompt(state: Dict[str, Any]) -> Dict[str, Any]:
    holdings = state.get("holdings", [])
    recommended = state.get("recommended", [])
    style = ", ".join(state.get("style_preference", []))
    risk = state.get("risk_tolerance", "")
    df = state.get("stock_df")

    table = df.to_string(index=False) if not df.empty else "(No data available)"

    prompt = (
        "You are a senior investment advisor. Based on the following information, provide a clear and professional investment analysis:\n\n"
        f"[Current Holdings] {', '.join(holdings)}\n"
        f"[Recommended Stocks] {', '.join(recommended)}\n"
        f"[Recommended Stock Data]\n{table}\n\n"
        f"[Investment Preferences] {style}\n[Risk Tolerance] {risk}\n\n"
        "Please provide:\n"
        "correlation between your  Holdings and Recommended Stocks\n"
        "- A brief assessment of the current holdings and their strengths/weaknesses\n"
        "- A detailed explanation for each recommended stock, including why it fits the investor's profile\n"
        "- When applicable, suggest similar stocks (e.g., same sector or function) to consider as alternatives or complements\n"
        "- In addition, introduce a few stocks from different sectors within the S&P 500 to provide diversification and broaden the opportunity set\n"
        "- Based on the stated risk tolerance, offer portfolio allocation guidance and highlight any risks the investor should be especially aware of\n"
        "- Conclude with a concise recommendation strategy suitable for the investor’s preferences and risk profile"
    )

    return {**state, "prompt": prompt}

# 🤖 Call OpenAI LLM using new SDK format
def run_llm(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = state["prompt"]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a professional investment analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    answer = response.choices[0].message.content
    return {**state, "analysis": answer}