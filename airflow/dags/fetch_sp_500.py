from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.operators.python import get_current_context
from datetime import datetime
import pandas as pd
import yfinance as yf
import time
import psycopg2
from psycopg2.extras import execute_values
import json

default_args = {
    'owner': 'airflow',
    'start_date': '2025-04-01',
}

dag = DAG(
    dag_id = 'fetch_sp500',
    default_args = default_args,
    schedule_interval = '@daily',
    catchup = False,
    tags = ['stock','sp500'],
)


def fetch_sp500_data(**kwargs):
    context = get_current_context()
    snapshot_date = context['execution_date']

    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url)[0][['Symbol', 'Security']]
    df.columns = ['symbol', 'company']

    data = []
    for symbol in df['symbol']:
        print(symbol)
        try:
            info = yf.Ticker(symbol).info
            print(info)
            data.append({
                'symbol': symbol,
                'company': info.get('longName') or symbol,
                'sector': info.get('sector'),
                'sub_industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'volume': info.get('volume'),
                'previous_close': info.get('previousClose'),
                'open': info.get('open'),
                'day_high': info.get('dayHigh'),
                'day_low': info.get('dayLow'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'high_52w': info.get('fiftyTwoWeekHigh'),
                'low_52w': info.get('fiftyTwoWeekLow'),
                'snapshot_date': str(snapshot_date)
            })
        except:
            continue
        time.sleep(0.5)

    ti = kwargs['ti']
    ti.xcom_push(key='sp500_data', value = json.dumps(data))


def insert_to_postgres(**kwargs):
    ti = kwargs['ti']
    json_data = ti.xcom_pull(key='sp500_data', task_ids='fetch_sp500_data')
    records = json.loads(json_data)

    # 轉成 tuple list
    values = [
        (
            r['symbol'], r['company'], r['sector'], r['sub_industry'],
            r['market_cap'], r['volume'], r['previous_close'], r['open'],
            r['day_high'], r['day_low'], r['pe_ratio'], r['forward_pe'],
            r['dividend_yield'], r['beta'], r['high_52w'], r['low_52w'],
            r['snapshot_date']
        )
        for r in records
    ]
    
    # Create connection 

    hook = PostgresHook(postgres_conn_id="aws_pg")  
    conn = hook.get_conn()
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO sp500_snapshots (
            symbol, company, sector, sub_industry, market_cap, volume,
            previous_close, open, day_high, day_low, pe_ratio, forward_pe,
            dividend_yield, beta, high_52w, low_52w, snapshot_date
        ) VALUES %s
    """
    execute_values(cur, insert_query, values)
    conn.commit()
    cur.close()
    conn.close()

def clean_old_snapshots():
    hook = PostgresHook(postgres_conn_id='aws_pg')  
    conn = hook.get_conn()
    cursor = conn.cursor()

    delete_sql = """
        DELETE FROM sp500_snapshots
        WHERE snapshot_date < CURRENT_DATE - INTERVAL '60 days';
    """

    cursor.execute(delete_sql)
    deleted = cursor.rowcount  
    conn.commit()
    cursor.close()
    conn.close()

    print(f"[INFO] Deleted {deleted} old rows from sp500_snapshots.")

task_fetch = PythonOperator(
    task_id = 'fetch_sp500_data',
    python_callable = fetch_sp500_data,
    provide_context = True,
    dag = dag,
)

task_insert = PythonOperator(
    task_id ='insert_to_postgres',
    python_callable = insert_to_postgres,
    provide_context = True,
    dag = dag,
)

task_clean_old_data = PythonOperator(
    task_id='clean_old_snapshots',
    python_callable=clean_old_snapshots,
    dag=dag,
)

task_fetch >> task_insert >> task_clean_old_data 