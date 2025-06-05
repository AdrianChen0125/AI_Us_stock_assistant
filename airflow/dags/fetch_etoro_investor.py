from airflow import DAG
from airflow.operators.python import PythonOperator, get_current_context
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import boto3
from bs4 import BeautifulSoup
import pandas as pd
from psycopg2.extras import execute_values

# 共用參數
S3_BUCKET = "etoro-data-prj"
S3_HTML_KEY = "investor/html/etoro_investor.html"
TABLE_NAME = "raw_data.etoro_investors"
POSTGRES_CONN_ID = "aws_pg"

def map_risk_level(risk):
    if pd.isna(risk):
        return "unknown"
    elif 1 <= risk <= 3:
        return "low"
    elif 3 < risk <= 6:
        return "moderate"
    elif risk > 6:
        return "high"
    else:
        return "unknown"

def fetch_html_and_parse():
    context = get_current_context()

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_HTML_KEY)
    html = obj["Body"].read().decode("utf-8")
    soup = BeautifulSoup(html, "lxml")

    usernames = soup.find_all("div", attrs={"automation-id": "discover-people-results-list-item-nickname"})
    risks = soup.find_all("span", attrs={"automation-id": "discover-people-results-list-item-risk-score"})
    copiers = soup.find_all("span", attrs={"automation-id": "discover-people-results-list-item-copiers-num"})
    revenues = soup.find_all("span", attrs={"automation-id": "discover-people-results-list-item-gain"})
    countries = soup.find_all("span", attrs={"automation-id": "discover-people-results-list-item-country"})

    records = []
    for i in range(len(usernames)):
        username = usernames[i].text.strip() if i < len(usernames) else "N/A"
        copiers_val = copiers[i].text.strip().replace(",", "") if i < len(copiers) else "0"
        revenue_val = revenues[i].text.strip().replace("%", "") if i < len(revenues) else "0.0"
        country_val = countries[i].get_text(separator=" ").strip().replace(" •", "") if i < len(countries) else "N/A"
        risk_val = 0
        if i < len(risks) and risks[i].get("class"):
            for cls in risks[i]["class"]:
                if cls.startswith("risk-") and cls != "risk-label":
                    risk_val = int(cls.replace("risk-", ""))
                    break

        risk_level = map_risk_level(risk_val)

        records.append((username, risk_val, int(copiers_val), float(revenue_val), country_val, risk_level))

    df = pd.DataFrame(records, columns=["username", "risk", "copiers", "revenue", "country", "risk_level"])
    context['ti'].xcom_push(key="investor_data", value=df.to_json(orient="records"))

def load_to_postgres():
    context = get_current_context()
    data_json = context['ti'].xcom_pull(task_ids="fetch_html_and_parse", key="investor_data")
    df = pd.read_json(data_json, orient="records")
    df = df.drop_duplicates(subset=["username", "country"], keep="last")

    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = hook.get_conn()
    cursor = conn.cursor()

    cursor.execute(f"TRUNCATE TABLE {TABLE_NAME}")

    values = df.to_records(index=False).tolist()
    insert_sql = f"""
        INSERT INTO {TABLE_NAME} (username, risk, copiers, revenue, country, risk_level)
        VALUES %s
        ON CONFLICT (username, country) DO UPDATE SET
            risk = EXCLUDED.risk,
            copiers = EXCLUDED.copiers,
            revenue = EXCLUDED.revenue,
            country = EXCLUDED.country,
            risk_level = EXCLUDED.risk_level
    """
    execute_values(cursor, insert_sql, values)
    conn.commit()

    print(f" {len(df)} rows inserted to {TABLE_NAME} with risk_level mapping.")

# DAG 定義
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fetch_etoro_investors",
    default_args=default_args,
    start_date=datetime(2025, 5, 27),
    schedule_interval= None,
    catchup=False,
    tags=["raw", "etoro","investor"]
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_html_and_parse",
        python_callable=fetch_html_and_parse,
    )

    load_task = PythonOperator(
        task_id="load_to_postgres",
        python_callable=load_to_postgres,
    )

    fetch_task >> load_task