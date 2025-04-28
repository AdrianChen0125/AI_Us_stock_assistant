from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import yfinance as yf
import requests
from psycopg2.extras import execute_values


# Market tickers to fetch
TICKERS = {
    "NASDAQ": "^IXIC",          # Nasdaq Composite Index
    "DOW_JONES": "^DJI",         # Dow Jones Industrial Average
    "SP500": "^GSPC",            # S&P 500 Index
    "US10Y_BOND": "^TNX",        # US 10-Year Treasury Yield (in %)
    "BITCOIN": "BTC-USD",        # Bitcoin Price in USD
    "GOLD": "GC=F",              # COMEX Gold Futures
    "SILVER": "SI=F",            # COMEX Silver Futures
    "CRUDE_OIL": "CL=F",         # NYMEX Crude Oil Futures
    "NATURAL_GAS": "NG=F"        # NYMEX Natural Gas Futures
}

def fetch_market_data(**kwargs):
    # execution_date is a datetime object
    exec_date = kwargs["execution_date"].date()

    data = {}

    # Fetch indexes & bitcoin using yfinance
    for name, ticker in TICKERS.items():
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(start=exec_date, end=exec_date + timedelta(days=1))
        
        if hist.empty:
            # Possibly a non-trading day (weekend or holiday)
            close_price = None
        else:
            close_price = hist['Close'].iloc[0]
        
        data[name] = float(close_price) if close_price else None


    # Save to XCom
    kwargs['ti'].xcom_push(key='market_data', value=data)

def load_market_data_to_postgres(**kwargs):

    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='fetch_market_data', key='market_data')

    exec_date = kwargs["execution_date"].date()

    # Setup DB connection
    hook = PostgresHook(postgres_conn_id = "aws_pg")
    conn = hook.get_conn()
    cursor = conn.cursor()


    # (準備好 rows)
    rows = [
        (market, 
         exec_date, 
         price) 
        for market, price in data.items()
    ]

    insert_sql = """
    INSERT INTO raw_data.market_snapshots (
        market, 
        snapshot_time, 
        price
    )
    VALUES %s
    ON CONFLICT (market, snapshot_time)
    DO UPDATE SET
        price = EXCLUDED.price
    """
    execute_values(cursor, insert_sql, rows)

    conn.commit()
    cursor.close()
    conn.close()

default_args = {
    "owner": "DE_Adrian",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id = "fetch_market_data",
    default_args = default_args,
    start_date = datetime(2025, 4, 1),
    schedule_interval = "0 5 * * *",
    catchup = False, 
    tags = ["raw", "crypto", "bonds", "indexes"]

) as dag:

    fetch_market = PythonOperator(
        task_id = "fetch_market_data",
        python_callable=fetch_market_data
    )

    load_to_postgres = PythonOperator(
        task_id = "load_market_data_to_postgres",
        python_callable = load_market_data_to_postgres,
    )

    fetch_market >> load_to_postgres