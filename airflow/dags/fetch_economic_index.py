import requests, os 
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from psycopg2.extras import execute_values

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "CPI": "CPIAUCSL",
    "Fed Funds Rate": "FEDFUNDS",
    "10Y Treasury": "GS10",
    "Unemployment Rate": "UNRATE",
    "Retail Sales": "RSAFS",
    "Consumer Sentiment": "UMCSENT"
}

def fetch_fred_indicators(**kwargs):
    
    execution_date = kwargs['execution_date'].date() 
    start_date = execution_date - timedelta(days=30)
    end_date = execution_date
    results = []

    for name, sid in FRED_SERIES.items():
        res = requests.get(FRED_API_URL, params={
            "series_id": sid,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": start_date.isoformat(),
            "observation_end": end_date.isoformat(),
        })

        
        
        for obs in res.json().get("observations", []):
            if obs["value"] == ".":
                continue
            results.append({
                "indicator": name,
                "series_id": sid,
                "value": float(obs["value"]),
                "date": obs["date"]
            })

    kwargs["ti"].xcom_push(key="fred_data", value=results)

def store_fred_to_postgres(**kwargs):
    rows = kwargs["ti"].xcom_pull(key="fred_data")
    if not rows:
        print("[INFO] No data to insert.")
        return

    hook = PostgresHook(postgres_conn_id="aws_pg")  
    conn = hook.get_conn()
    cursor = conn.cursor()

    insert_sql = """
    INSERT INTO raw_data.economic_indicators (
        indicator, 
        series_id, 
        value, 
        date, 
        fetched_at)
    VALUES %s
    ON CONFLICT (series_id, date) DO NOTHING;
    """

    values = [
        (
            row["indicator"],
            row["series_id"],
            row["value"],
            row["date"],
            datetime.utcnow()
        )
        for row in rows
    ]

    execute_values(cursor, insert_sql, values)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[INFO] Inserted {len(values)} rows.")


default_args = {
    "owner": "DE_Adrian",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id = "fetch_economic_index",
    default_args = default_args,
    start_date = datetime (2025, 4, 25),
    schedule_interval = "@weekly",  
    catchup = False,
    tags = ["raw", "economic"]
    
) as dag: 
    
    fetch_data = PythonOperator(
        task_id = "fetch_fred_data",
        python_callable = fetch_fred_indicators
        )
    
    store_data = PythonOperator(
        task_id = "store_fred_data",
        python_callable = store_fred_to_postgres
        )

    fetch_data >> store_data