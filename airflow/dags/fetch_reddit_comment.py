from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from psycopg2.extras import execute_values

from datetime import datetime, timedelta, timezone
import praw
import os

# Reddit API credentials (from environment variables)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_AGENT = os.getenv("REDDIT_AGENT")

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="fetch_reddit_comment",
    default_args=default_args,
    start_date=datetime(2025, 4, 24),
    schedule_interval="@daily",
    catchup=True,
)



def get_reddit_posts(**context):
    """Fetch Reddit posts related to US stock market created on execution date"""
    from datetime import timezone

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=REDDIT_AGENT
    )

    # Get execution date from Airflow context
    execution_date = context["execution_date"]
    start_time = execution_date - timedelta(days=2)
    end_time = execution_date + timedelta(days=1)

    subreddit = reddit.subreddit("stocks+wallstreetbets+StockMarket")
    posts = []

    for submission in subreddit.search(
        query="us stock market",
        sort="top",
        time_filter="month",   # still search over past month, then filter precisely
        limit=200
    ):
        created_dt = datetime.utcfromtimestamp(submission.created_utc).replace(tzinfo=timezone.utc)

        # Only keep posts created on the execution date
        if not (start_time <= created_dt < end_time):
            continue

        posts.append({
            "id": submission.id,
            "title": submission.title,
            "url": submission.url,
            "created_utc": created_dt.isoformat(),
            "score": submission.score,
            "num_comments": submission.num_comments,
            "permalink": submission.permalink
        })

    print(f"[INFO] Collected {len(posts)} posts created on {execution_date.date()}")
    context['ti'].xcom_push(key='reddit_posts', value=posts)


def extract_reddit_comments(**context):
    from datetime import datetime, timezone, timedelta
    import os
    import praw

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_SECRET"),
        user_agent=os.getenv("REDDIT_AGENT")
    )

    posts = context['ti'].xcom_pull(task_ids='get_reddit_posts', key='reddit_posts')
    if not posts:
        print("[INFO] No posts received from previous task.")
        return

    # Define time window for filtering comments
    execution_date = context['execution_date']
    start_time = execution_date
    end_time = execution_date + timedelta(days=1)

    comments = []
    total_count = 0
    MAX_COMMENTS = 10000

    for post in posts:
        submission = reddit.submission(id=post["id"])

        # Only iterate over top-level comments (no nested replies)
        for comment in submission.comments:
            if total_count >= MAX_COMMENTS:
                break

            if isinstance(comment, praw.models.MoreComments):
                continue  # skip expandable placeholder

            created_dt = datetime.utcfromtimestamp(comment.created_utc).replace(tzinfo=timezone.utc)

            if not (start_time <= created_dt < end_time):
                continue

            comments.append({
                "post_id": comment.fullname,
                "subreddit": str(submission.subreddit),
                "text": comment.body,
                "created_utc": created_dt.isoformat(),
                "score": comment.score,
                "author": str(comment.author)
            })

            total_count += 1

        if total_count >= MAX_COMMENTS:
            print(f"[INFO] Reached maximum comment limit of {MAX_COMMENTS}.")
            break

    print(f"[INFO] Collected {len(comments)} top-level comments in total.")
    context['ti'].xcom_push(key='reddit_comments', value=comments)




def store_to_postgres(**context):
    """Insert Reddit comments into PostgreSQL"""
    rows = context['ti'].xcom_pull(task_ids='extract_reddit_comments', key='reddit_comments')

    if not rows:
        print("[INFO] No comments to insert.")
        return

    hook = PostgresHook(postgres_conn_id='aws_pg')
    conn = hook.get_conn()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO raw_data.reddit_comments (
            post_id, subreddit, author, comment_text, created_utc, fetched_at
        )
        VALUES %s
        ON CONFLICT (post_id) DO NOTHING;
    """

    values = [
        (
            row["post_id"],
            row["subreddit"],
            row["author"],
            row["text"],
            row["created_utc"],
            datetime.utcnow()
        )
        for row in rows
    ]

    execute_values(cursor, insert_sql, values)
    conn.commit()
    cursor.close()
    conn.close()

    print(f"[INFO] Successfully batch-inserted {len(values)} comments.")


# DAG task definitions
get_posts = PythonOperator(
    task_id="get_reddit_posts",
    python_callable=get_reddit_posts,
    provide_context=True,
    dag=dag,
)

extract_comments = PythonOperator(
    task_id="extract_reddit_comments",
    python_callable=extract_reddit_comments,
    provide_context=True,
    dag=dag,
)

save_to_pg = PythonOperator(
    task_id="store_to_postgres",
    python_callable=store_to_postgres,
    provide_context=True,
    dag=dag,
)

# Task dependencies
get_posts >> extract_comments >> save_to_pg
