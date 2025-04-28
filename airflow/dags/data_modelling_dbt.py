from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="dbt_us_stock",
    default_args=default_args,
    start_date=datetime(2025, 4, 24),
    schedule_interval="0 5 * * *",
    catchup=False,
    tags=["dbt", "docker"]
) as dag:

    run_dbt = BashOperator(
        task_id="run_dbt_project",
        bash_command="""
            cd /opt/airflow/dbt/dbt_us_stock_data && \
            dbt run 
        """
    )