import os
import pandas as pd
from typing import Dict, Any
from async_db import get_db
from services.fetch_sp500_stock import fetch_sp500_service
from .prompt import build_prompt

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 使用 ChatOpenAI 並設定 run_name，支援 MLflow autolog
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7).with_config({
    "run_name": "stock_analysis_llm"
})
chain = llm | StrOutputParser()

# 擷取股票資料
async def fetch_stock_data(state: Dict[str, Any]) -> Dict[str, Any]:
    symbols = state.get("recommended", [])
    if not symbols:
        return {**state, "stock_df": pd.DataFrame()}

    db_gen = get_db()
    db = await db_gen.__anext__()
    try:
        data = await fetch_sp500_service(symbols, db)
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        df = pd.DataFrame()
    finally:
        await db.aclose()

    return {**state, "stock_df": df}

# 建立提示語
def build_prompt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return build_prompt(state)

# 呼叫 LLM 生成分析
def run_llm(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = state["prompt"]
    result = chain.invoke(prompt)
    return {**state, "analysis": result}