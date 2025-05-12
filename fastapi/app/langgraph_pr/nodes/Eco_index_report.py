import pandas as pd
from ..prompts.Eco_index_prompt import call_llm_report
from typing import Dict, Any
from database import SessionLocal
from services.market_price import get_market_price_df
from services.economic_index import get_economic_index_df

async def fetch_econ(state):
    
    db = SessionLocal()
    try:
        df = get_economic_index_df(db)
        return {**state, "econ_df": df}
    finally:
        db.close()

async def fetch_market(state):
    db = SessionLocal()
    try:
        df = get_market_price_df(db)
        return {**state, "market_df": df}
    finally:
        db.close()

def summarize_econ(state):
    df = state["econ_df"]
    lines = []
    for name in df["index_name"].unique():
        sub = df[df["index_name"] == name].sort_values("date").tail(6)
        values = ", ".join([f"{d}={round(v,2)}" for d, v in zip(sub["date"], sub["value"])])
        lines.append(f"{name}：{values}")
    return {**state, "econ_summary": "\n".join(lines)}

def summarize_market(state):
    df = state["market_df"]
    lines = []
    for mkt in df["market"].unique():
        sub = df[df["market"] == mkt].sort_values("date").tail(6)
        values = ", ".join([f"{d}={round(v,2)}" for d, v in zip(sub["date"], sub["price"])])
        lines.append(f"{mkt}：{values}")
    return {**state, "market_summary": "\n".join(lines)}

def combine_summary(state):
    econ = state.get("econ_summary", "[Missing econ_summary]")
    market = state.get("market_summary", "[Missing market_summary]")
    summary = f"## Economic Summary\n{econ}\n\n## Market Summary\n{market}"
    print("✅ Combined summary keys:", state.keys())
    print("📄 Summary content preview:", summary[:200])
    return {**state, "summary": summary}

async def call_llm(state):
    print("🧾 LLM node received state keys:", state.keys())
    language = state.get("language", "English")
    report = await call_llm_report(state["summary"],language)
    return {**state, "report": report}

