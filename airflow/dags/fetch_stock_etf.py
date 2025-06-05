from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import pandas as pd
import boto3
import yfinance as yf
import time
from io import StringIO

# --- DAG config ---
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
}

dag = DAG(
    'fetch_stock_etf_with_group',
    default_args=default_args,
    description='Fetch ETF and stock info using yfinance',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

BUCKET = "etoro-data-prj"
ETF_KEY = "stocks/nasdaq_etf.csv"
STOCK_KEY = "stocks/nasdaq_tickers.csv"
OUTPUT_KEY = "stocks/item_features.csv"
REGION = "us-east-1"
BATCH_SIZE = 4000


def read_symbols():
    s3 = boto3.client("s3", region_name=REGION)

    def read_s3_csv(key):
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    etf_df = read_s3_csv(ETF_KEY)
    stock_df = read_s3_csv(STOCK_KEY)

    etf_df["asset_type"] = "etf"
    stock_df["asset_type"] = "stock"

    etf_df.rename(columns={"SYMBOL": "symbol", "NAME": "Name"}, inplace=True)
    stock_df.rename(columns={"Symbol": "symbol", "Name": "Name"}, inplace=True)

    all_df = pd.concat([
        etf_df[["symbol", "Name", "asset_type"]],
        stock_df[["symbol", "Name", "asset_type"]]
    ])

    all_df.to_csv("/tmp/all_symbols.csv", index=False)


def fetch_stock_batch(start_idx, end_idx):
    df = pd.read_csv("/tmp/all_symbols.csv")
    df = df[df["asset_type"] == "stock"].iloc[start_idx:end_idx]
    results = []
    for _, row in df.iterrows():
        symbol = row["symbol"]
        try:
            time.sleep(1)
            info = yf.Ticker(symbol).info
            results.append({
                "symbol": symbol,
                "name": row["Name"],
                "asset_type": "stock",
                "marketCap": info.get("marketCap"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "dividendYield": info.get("dividendYield"),
                "priceToBook": info.get("priceToBook"),
                "averageVolume": info.get("averageVolume"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                "fiftyDayAverage": info.get("fiftyDayAverage"),
            })
        except Exception as e:
            print(f"[stock] failed: {symbol}, error: {e}")
    pd.DataFrame(results).to_csv(f"/tmp/item_features_stock_{start_idx}_{end_idx}.csv", index=False)


def fetch_etf():
    df = pd.read_csv("/tmp/all_symbols.csv")
    df = df[df["asset_type"] == "etf"]
    results = []
    for _, row in df.iterrows():
        symbol = row["symbol"]
        try:
            time.sleep(1)
            info = yf.Ticker(symbol).info
            results.append({
                "symbol": symbol,
                "name": row["Name"],
                "asset_type": "etf",
                "marketCap": info.get("marketCap"),
                "sector": None,
                "industry": None,
                "dividendYield": info.get("yield"),
                "priceToBook": info.get("priceToBook"),
                "averageVolume": info.get("averageVolume"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                "fiftyDayAverage": info.get("fiftyDayAverage"),
            })
        except Exception as e:
            print(f"[etf] failed: {symbol}, error: {e}")
    pd.DataFrame(results).to_csv("/tmp/item_features_etf.csv", index=False)


def merge_and_upload():
    import glob
    stock_parts = glob.glob("/tmp/item_features_stock_*.csv")
    df_stock = pd.concat([pd.read_csv(f) for f in stock_parts]) if stock_parts else pd.DataFrame()
    df_etf = pd.read_csv("/tmp/item_features_etf.csv")
    merged = pd.concat([df_stock, df_etf])
    buf = StringIO()
    merged.to_csv(buf, index=False)

    s3 = boto3.client("s3", region_name=REGION)
    s3.put_object(Bucket=BUCKET, Key=OUTPUT_KEY, Body=buf.getvalue())
    print(f"Uploaded to s3://{BUCKET}/{OUTPUT_KEY}")


with dag:
    t1 = PythonOperator(
        task_id="read_symbols",
        python_callable=read_symbols,
    )

    with TaskGroup("fetch_stock_group") as stock_group:
        df = pd.read_csv("/tmp/all_symbols.csv")
        stock_df = df[df["asset_type"] == "stock"]
        for i in range(0, len(stock_df), BATCH_SIZE):
            PythonOperator(
                task_id=f"fetch_stock_{i}_{i+BATCH_SIZE}",
                python_callable=lambda start=i, end=i + BATCH_SIZE: fetch_stock_batch(start, end)
            )

    t2_etf = PythonOperator(
        task_id="fetch_etf",
        python_callable=fetch_etf,
    )

    t3 = PythonOperator(
        task_id="merge_and_upload",
        python_callable=merge_and_upload,
    )

    t1 >> [stock_group, t2_etf] >> t3