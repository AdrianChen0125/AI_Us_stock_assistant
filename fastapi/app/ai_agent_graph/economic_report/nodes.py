import os
import logging
from typing import Dict, Any, List
import openai
from datetime import datetime
from collections import defaultdict

from database import SessionLocal
from services.market_price import get_market_price_data
from services.economic_index import get_economic_index_data
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from .prompt import build_economic_report_prompt

logger = logging.getLogger(__name__)
client = wrap_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))


async def fetch_econ(state: Dict[str, Any]) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        data = get_economic_index_data(db)
        logger.info("Fetched economic data")
        return {**state, "econ_df": data}
    finally:
        db.close()


async def fetch_market(state: Dict[str, Any]) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        data = get_market_price_data(db)
        logger.info("Fetched market data")
        return {**state, "market_df": data}
    finally:
        db.close()


def summarize_econ(state: Dict[str, Any]) -> Dict[str, Any]:
    data: List[Dict[str, Any]] = state.get("econ_df", [])
    if not data:
        return {**state, "econ_summary": "No economic data available."}

    grouped = defaultdict(list)
    for row in data:
        date = row["date"]
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        grouped[row["index_name"]].append((date, row["value"]))

    lines = []
    for name, entries in grouped.items():
        sorted_entries = sorted(entries, key=lambda x: x[0])
        tail = sorted_entries[-6:]
        values = ", ".join([f"{d.strftime('%Y-%m-%d')}={round(v, 2)}" for d, v in tail])
        lines.append(f"{name}: {values}")

    return {**state, "econ_summary": "\n".join(lines)}


def summarize_market(state: Dict[str, Any]) -> Dict[str, Any]:
    data: List[Dict[str, Any]] = state.get("market_df", [])
    if not data:
        return {**state, "market_summary": "No market data available."}

    grouped = defaultdict(list)
    for row in data:
        date = row["date"]
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        grouped[row["market"]].append((date, row["price"]))

    lines = []
    for name, entries in grouped.items():
        sorted_entries = sorted(entries, key=lambda x: x[0])
        tail = sorted_entries[-6:]
        values = ", ".join([f"{d.strftime('%Y-%m-%d')}={round(v, 2)}" for d, v in tail])
        lines.append(f"{name}: {values}")

    return {**state, "market_summary": "\n".join(lines)}


def combine_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    econ = state.get("econ_summary", "[Missing econ summary]")
    market = state.get("market_summary", "[Missing market summary]")
    summary = f"## Economic Summary\n{econ}\n\n## Market Summary\n{market}"
    print(summary)
    return {**state, "summary": summary}


@traceable(name="generate_economic_report")
async def generate_economic_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    summary = state.get("summary", "")
    language = state.get("language", "English")
    prompt = build_economic_report_prompt(summary, language)

    logger.info("Sending economic prompt to OpenAI")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    report = response.choices[0].message.content
    logger.info("Received economic report from LLM")
    return {**state, "economic_report": report.strip()}