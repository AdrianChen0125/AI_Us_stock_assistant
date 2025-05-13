from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from psycopg2.extras import execute_values

from datetime import datetime, timedelta, timezone 
from dateutil.parser import isoparse

import praw
import os
import json

# Reddit API credentials (from environment variables)
def get_reddit():
    return praw.Reddit(
        client_id = os.getenv("REDDIT_CLIENT_ID"),
        client_secret = os.getenv("REDDIT_SECRET"),
        user_agent = os.getenv("REDDIT_AGENT", "airflow_reddit_agent")
    )

def get_top10_per_sector(**kwargs):

    hook = PostgresHook(postgres_conn_id='aws_pg')
    conn = hook.get_conn()
    cursor = conn.cursor()

    query = """

    WITH latest AS (
        SELECT MAX(snapshot_date) FROM raw_data.sp500_snapshots
    ),
    ranked AS (
        SELECT symbol, company, sector, market_cap,
               RANK() OVER (PARTITION BY sector ORDER BY market_cap DESC) AS rk
        FROM raw_data.sp500_snapshots
        WHERE snapshot_date = (SELECT * FROM latest)
    )
    SELECT symbol, company, sector
    FROM ranked
    WHERE rk <= 20;
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    top_stocks = [{'symbol': r[0], 'company': r[1], 'sector': r[2]} for r in rows]
    
    ti = kwargs['ti']
    ti.xcom_push(key='top10_stocks', value=json.dumps(top_stocks))


def get_reddit_comments(**kwargs):
    
    execution_date = kwargs["execution_date"]
    cutoff_start = execution_date - timedelta(days=6)
    cutoff_end = execution_date + timedelta(days=1)

    symbols = kwargs["ti"].xcom_pull(task_ids = 'get_top10_per_sector', key = "top10_stocks")
    top_stocks = json.loads(symbols)

    reddit = get_reddit()
    subreddit = reddit.subreddit("stocks+wallstreetbets+StockMarket")

    comments = []
    max_comments = 10000
    count = 0

    for stock in top_stocks:
        symbol = stock["symbol"]
        company_name = stock["company"]
        query = f'"{symbol}" OR "{company_name}"'

        submissions = reddit.subreddit("stocks+wallstreetbets+StockMarket").search(
            query = symbol,
            sort = "top",
            time_filter = "month",
            limit = 60
        )

        for post in submissions:
            post_created = datetime.utcfromtimestamp(post.created_utc).replace(tzinfo=timezone.utc)
            if not (cutoff_start <= post_created < cutoff_end):
                continue

            submission = reddit.submission(id=post.id)
            submission.comments.replace_more(limit=0)

            for comment in submission.comments:
                if isinstance(comment, praw.models.MoreComments):
                    continue

                comment_created = datetime.utcfromtimestamp(comment.created_utc).replace(tzinfo=timezone.utc)

                if not (execution_date <= comment_created < cutoff_end):
                    continue

                comments.append({
                    "symbol": symbol,
                    "company": company_name,
                    "post_id": submission.id,
                    "post_title":submission.title,
                    "comment_id": comment.id,
                    "subreddit": str(submission.subreddit),
                    "author": str(comment.author),
                    "text": comment.body,
                    "score": comment.score,
                    "created_utc": comment_created.isoformat()
                })
                count += 1
                if count >= max_comments:
                    break
            if count >= max_comments:
                break
        if count >= max_comments:
            break

    print(f"[INFO] Collected {len(comments)} comments across {len(symbols)} symbols.")
    
    ti = kwargs['ti']
    ti.xcom_push(key="reddit_comments", value=comments)


def store_comments_to_postgres(**kwargs):
    comments = kwargs["ti"].xcom_pull(task_ids="get_reddit_comments", key="reddit_comments")
    execution_date = kwargs["execution_date"].date()

    # Setup DB connection
    hook = PostgresHook(postgres_conn_id="aws_pg")
    conn = hook.get_conn()
    cursor = conn.cursor()

    insert_sql = """
    INSERT INTO raw_data.reddit_comments_sp500 (
        symbol, 
        company, 
        post_id,
        post_title, 
        comment_id, 
        subreddit, 
        author, 
        comment_text,
        score, 
        created_utc, 
        fetched_at
    )
    VALUES %s
    ON CONFLICT (comment_id) DO NOTHING
"""

    values = []
    for row in comments:
        try:
            created_utc = row.get("created_utc")
            # Safely parse created_utc, fallback if needed
            if isinstance(created_utc, str):
                created_utc = isoparse(created_utc)
            elif not created_utc:
                created_utc = execution_date  

            values.append((
                str(row.get("symbol", "")),
                str(row.get("company", "")),
                str(row.get("post_id", "")),
                str(row.get("post_title","")),
                str(row.get("comment_id", "")),
                str(row.get("subreddit", "")),
                str(row.get("author") or "unknown"),  
                str(row.get("text", "")),              
                int(row.get("score", 0)),
                created_utc,
                execution_date,
            ))
        except Exception as e:
            print(f"[ERROR] Skipping comment due to error: {e}, data: {row}")

    if not values:
        print("[INFO] No valid comments to insert after processing.")
        cursor.close()
        conn.close()
        return

            
    execute_values(cursor, insert_sql, values)
    conn.commit()

    cursor.close()
    conn.close()
    print(f"[INFO] Successfully inserted {len(values)} comments.")

default_args = {
    "owner": "DE_Adrian",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id = "fetch_reddit_comment_sp500_top10_by_sector",
    default_args = default_args,
    start_date = datetime(2025, 4, 14),
    schedule_interval = "@daily",
    catchup =False,
    tags = ["raw", "reddit","sp500"]
) as dag:
    
    t1 = PythonOperator(
        task_id="get_top10_per_sector",
        python_callable=get_top10_per_sector
        )
    
    t2 = PythonOperator(
    task_id="get_reddit_comments",
    python_callable=get_reddit_comments
    )
    
    t3 = PythonOperator(
    task_id = "store_comments_to_postgres",
    python_callable = store_comments_to_postgres
    )

    t1 >> t2 >> t3