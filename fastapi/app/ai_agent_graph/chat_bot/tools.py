import os
import yfinance as yf
import requests
import asyncio
from datetime import datetime, timedelta

from langchain_community.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from services.rag_service import process_rag_question
from async_db import get_pgvector_conn

# ----------------------------
# Wikipedia Tool
# ----------------------------

wiki_tool = Tool(
    name="Wikipedia",
    func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="en")),
    description="Search Wikipedia for general knowledge."
)

# ----------------------------
# Stock Price Tool
# ----------------------------

def get_stock_price(ticker: str) -> str:
    """Check latest stock price using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        price = data["Close"].iloc[-1]
        return f"The latest closing price of {ticker.upper()} is ${price:.2f}."
    except Exception as e:
        return f"Error fetching stock price for {ticker}: {str(e)}"

stock_tool = Tool(
    name="StockPriceChecker",
    func=get_stock_price,
    description="Check current stock price by ticker symbol (e.g., AAPL, TSLA)."
)

# ----------------------------
# News Tool
# ----------------------------

def search_news_for_keywords(query: str) -> str:
    """Search recent English news articles using NewsAPI."""
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    from_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    url = (
        f"https://newsapi.org/v2/everything?q={query}"
        f"&sortBy=publishedAt&from={from_date}&language=en&pageSize=3&apiKey={NEWS_API_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        if not articles:
            return "No English news found."

        return "\n\n".join([
            f"【{a['title']}】\n{a.get('description', 'No description')}\nLink: {a['url']}"
            for a in articles
        ])
    except Exception as e:
        return f"News search error: {e}"

news_tool = Tool(
    name="SearchNews",
    func=search_news_for_keywords,
    description="Search recent news articles by keyword."
)

# ----------------------------
# RAG Tool (sync wrapper)
# ----------------------------

def rag_tool_sync(query: str) -> str:
    """Query internal vector DB for relevant domain knowledge."""
    async def wrapper():
        db = await get_pgvector_conn()
        try:
            result = await process_rag_question(query, top_k=5, db=db)
            return result["answer"]
        finally:
            await db.close()
    
    return asyncio.run(wrapper())

rag_tool = Tool(
    name="RAGQuery",
    func=rag_tool_sync,
    description="Search internal vector DB for relevant domain-specific context."
)