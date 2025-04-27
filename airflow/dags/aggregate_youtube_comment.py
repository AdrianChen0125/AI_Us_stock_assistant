from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import os
from datetime import datetime, timedelta
from openai import OpenAI

def extract_and_aggregate(**kwargs):
    pg_hook = PostgresHook(postgres_conn_id = 'aws_pg')
    execution_date = kwargs['ds']
    query = f"""
        SELECT 
            DATE(processed_at) AS topic_date,
            topic_tags AS topic,
            keywords,
            COUNT(*) AS comments_count,
            SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS neg_count,
            SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) AS pos_count
        FROM processed_data.youtube_comments
        WHERE processed_at::date = DATE '{execution_date}'
        GROUP BY topic_date, topic, keywords
        Order by comments_count DESC
        LIMIT 5;
    """
    records = pg_hook.get_records(query)
    kwargs['ti'].xcom_push(key='aggregated_data', value=records)

def generate_topic_summaries(**kwargs):
    records = kwargs['ti'].xcom_pull(key='aggregated_data')
    summaries = []

    for row in records:
        topic_date, topic_tag, keywords, comments_count, neg_count, pos_count = row
        prompt = (
        f"""
        These keywords are collected from Youtube comments related to U.S. stock market discussions, " 
        Generate a simple title in one or two sentences and suitable for social media based on the following keywords: {', '.join(keywords)}
        """
)
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
        )
        topic_summary = response.choices[0].message.content.strip()

        summaries.append({
            'topic_date': topic_date,
            'topic_tags': topic_tag,
            'keywords': keywords,
            'topic_summary': topic_summary,
            'comments_count': comments_count,
            'neg_count': neg_count,
            'pos_count': pos_count
        })

    kwargs['ti'].xcom_push(key='summaries', value=summaries)

def insert_into_table(**kwargs):
    pg_hook = PostgresHook(postgres_conn_id='aws_pg')
    summaries = kwargs['ti'].xcom_pull(key='summaries')

    execution_date = kwargs['ds']  

    insert_query = """
        INSERT INTO processed_data.youtube_topic (
            topic_date, 
            topic_tags, 
            keywords, 
            topic_summary, 
            comments_count,  
            neg_count, 
            pos_count,
            created_at
        ) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    for record in summaries:
        pg_hook.run(insert_query, parameters=(
            record['topic_date'],
            record['topic_tags'],
            record['keywords'],
            record['topic_summary'],
            record['comments_count'],
            record['neg_count'],
            record['pos_count'],
            execution_date  
        ))



default_args = {
    'owner': 'DE_Adrian',
    'start_date': datetime(2025, 4, 24),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id = 'youtube_topic_summary',
    default_args = default_args,
    schedule_interval = '0 5 * * 6',
    catchup = True,
    tags = ['processed','youtube']
)as dag:

    extract_task = PythonOperator(
        task_id='extract_and_aggregate',
        python_callable=extract_and_aggregate
    )

    summary_task = PythonOperator(
        task_id='generate_topic_summaries',
        python_callable=generate_topic_summaries
    )

    insert_task = PythonOperator(
        task_id='insert_into_reddit_topic',
        python_callable=insert_into_table
    )

    extract_task >> summary_task >> insert_task
