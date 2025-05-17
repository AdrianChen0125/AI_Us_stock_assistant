# nodes.py
import os
import pandas as pd
from typing import Dict, Any
from openai import OpenAI
from async_db import get_db
from services.fetch_sp500_stock import fetch_sp500_service
from .prompt import build_prompt 

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🔍 Node to fetch S&P 500 stock data based on recommended symbols
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

# 🧠 Node to build prompt using user's state and stock data
def build_prompt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return build_prompt(state)

# 🤖 Node to call OpenAI LLM and generate the report
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