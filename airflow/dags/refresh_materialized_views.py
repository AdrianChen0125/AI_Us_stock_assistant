from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="refresh_materialized_views",
    default_args=default_args,
    start_date=datetime(2025, 4, 6),
    schedule_interval="0 6 * * 6",  # 每小時執行一次
    catchup=False,
    tags=["materialized", "postgres"]
) as dag:

    refresh_mv_sentiment_by_date = PostgresOperator(
        task_id="refresh_mv_sentiment_by_date",
        postgres_conn_id="postgres_default",
        sql="REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sentiment_by_date;"
    )

    refresh_mv_sentiment_by_title = PostgresOperator(
        task_id="refresh_mv_sentiment_by_title",
        postgres_conn_id="postgres_default",
        sql="REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sentiment_by_title;"
    )

    refresh_mv_sentiment_by_topic = PostgresOperator(
        task_id="refresh_mv_sentiment_by_topic",
        postgres_conn_id="postgres_default",
        sql="REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sentiment_by_topic;"
    )

    # 可以依序執行，或並行
    refresh_mv_sentiment_by_date >> [refresh_mv_sentiment_by_title, refresh_mv_sentiment_by_topic]
