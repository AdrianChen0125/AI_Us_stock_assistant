import os
import pandas as pd
from typing import Dict, Any
import yfinance as yf

from .prompt import build_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# === 建立 LLM Chain ===
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7).with_config({
    "run_name": "stock_analysis_llm"
})
chain = llm | StrOutputParser()

# === 擷取股票資料 (現價 + 52W低點 + 買點分析) ===
async def fetch_stock_data(state: Dict[str, Any]) -> Dict[str, Any]:
    symbols = state.get("recommended", [])
    if not symbols:
        return {**state, "stock_df": pd.DataFrame()}

    try:
        tickers = yf.Tickers(" ".join(symbols))
        records = []

        for symbol in symbols:
            info = tickers.tickers[symbol].info

            price = info.get("regularMarketPrice")
            low_52w = info.get("fiftyTwoWeekLow")

            if price is not None and low_52w is not None and low_52w > 0:
                distance_pct = (price - low_52w) / low_52w * 100
                buy_signal = distance_pct < 15  # 離低點小於15%，認定為潛在買點
            else:
                distance_pct = None
                buy_signal = False

            records.append({
                "symbol": symbol,
                "price": price,
                "52w_low": low_52w,
                "from_low_%": round(distance_pct, 2) if distance_pct else None,
                "is_near_buy_point": buy_signal
            })

        df = pd.DataFrame(records)

    except Exception as e:
        print(f"Error fetching stock data from yfinance: {e}")
        df = pd.DataFrame()

    return {**state, "stock_df": df}

# === 建立提示語 ===
def build_prompt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return build_prompt(state)

# === 呼叫 LLM 分析 ===
def run_llm(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = state["prompt"]
    result = chain.invoke(prompt)
    return {**state, "analysis": result}