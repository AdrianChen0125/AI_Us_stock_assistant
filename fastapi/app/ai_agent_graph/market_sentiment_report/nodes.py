import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from .prompt import build_market_sentiment_prompt
from async_db import get_db
from services.sentiment_reddit_service import fetch_reddit_summary
from services.sentiment_topic_service import fetch_latest_sentiment_topics
from services.sentiment_sp500_sector_service import fetch_top_sectors_this_week
from services.sentiment_sp500_top_service import fetch_top_symbols

logger = logging.getLogger(__name__)




llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7).with_config({
    "run_name": "generate_sentiment_report"
})


async def get_overall_sentiment(state: Dict[str, Any]) -> Dict[str, Any]:
    days = state.get("days", 30)
    db_gen = get_db()
    db = await db_gen.__anext__()
    try:
        df = await fetch_reddit_summary(days=days, db=db)
        logger.info("Fetched Reddit sentiment for %d days", days)
        return {**state, "reddit_df": df}
    finally:
        await db.aclose()


async def get_sentiment_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    db_gen = get_db()
    db = await db_gen.__anext__()
    try:
        summary = await fetch_latest_sentiment_topics(db)
        logger.info("Fetched topic sentiment summary")
        return {**state, "sentiment_summary": summary}
    finally:
        await db.aclose()


async def get_sector_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    db_gen = get_db()
    db = await db_gen.__anext__()
    try:
        summary = await fetch_top_sectors_this_week(db)
        logger.info("Fetched top sectors this week")
        return {**state, "sector_summary": summary}
    finally:
        await db.aclose()


async def get_symbol_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    limit = state.get("limit", 10)
    db_gen = get_db()
    db = await db_gen.__anext__()
    try:
        summary = await fetch_top_symbols(limit, db)
        logger.info("Fetched top %d symbols", limit)
        return {**state, "symbol_summary": summary}
    finally:
        await db.aclose()


def generate_sentiment_report(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_market_sentiment_prompt(state)
    logger.info("Sending prompt to OpenAI (MLflow autolog enabled)")

    chain = llm | StrOutputParser()
    result = chain.invoke(prompt)

    return {**state, "report": result.strip()}