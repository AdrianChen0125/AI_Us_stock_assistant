from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
from airflow.utils.dates import days_ago
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import os, requests, random
from openai import OpenAI

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def search_youtube(**kwargs):    
    query = "us stock market news"
    execution_date = kwargs["execution_date"]

    published_after = execution_date - timedelta(days=90)
    published_before = execution_date

    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "key": YOUTUBE_API_KEY,
        "q": query,
        "part": "snippet",
        "type": "video",
        "maxResults": 100,
    }

    search_res = requests.get(search_url, params=search_params).json()

    filtered_video_ids = []
    for item in search_res.get("items", []):
        video_id = item["id"].get("videoId")
        published_str = item["snippet"]["publishedAt"] 
        published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))

        if published_after <= published_at < published_before:
            filtered_video_ids.append(video_id)

    if not filtered_video_ids:
        kwargs['ti'].xcom_push(key='raw_videos', value=[])
        return

    #  Fetch details
    video_url = "https://www.googleapis.com/youtube/v3/videos"
    video_params = {
        "key": YOUTUBE_API_KEY,
        "part": "snippet,statistics",
        "id": ",".join(filtered_video_ids)
    }

    video_res = requests.get(video_url, params=video_params).json()

    videos = []
    for item in video_res.get("items", []):
        stats = item.get("statistics", {})
        view_count = int(stats.get("viewCount", 0))
        videos.append({
            "video_id": item["id"],
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "published_at": item["snippet"]["publishedAt"],
            "views": view_count
        })

    # Sort by views
    top_videos = sorted(videos, key=lambda x: x["views"], reverse=True)

    # Push to XCom
    kwargs['ti'].xcom_push(key='top_videos', value=top_videos)

def filter_with_openai(**kwargs):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Pull XCom value using correct key
    videos = kwargs['ti'].xcom_pull(key='top_videos', task_ids='search_youtube')
    filtered = []

    if not videos:
        print(" No videos found to filter.")
        kwargs['ti'].xcom_push(key='filtered_videos', value=[])
        return

    for vid in videos:
        prompt = f"""
        The following is a YouTube video title: \"{vid['title']}\"
        Is this video related to the U.S. stock market, financial markets, or investing topics?
        If the connection is even somewhat relevant, respond with \"yes\", otherwise \"no\".
        """

        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = res.choices[0].message.content.strip().lower()
        if "yes" in answer:
            filtered.append(vid)

    kwargs['ti'].xcom_push(key='filtered_videos', value=filtered)

def collect_comments(**kwargs):
    top_videos = kwargs['ti'].xcom_pull(key='filtered_videos')
    all_comments = []

    execution_date = kwargs['execution_date']  
    start_time = execution_date - timedelta(days=7)
    end_time = execution_date 

    for video in top_videos:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            "part": "snippet",
            "videoId": video["video_id"],
            "key": YOUTUBE_API_KEY,
            "maxResults": 150,
            "textFormat": "plainText"
        }

        count = 0
        next_page_token = None

        while count < 2000:
            if next_page_token:
                params["pageToken"] = next_page_token

            res = requests.get(url, params=params).json()

            for item in res.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                published_at = snippet.get("publishedAt")

                if not published_at:
                    continue

                # Parse to UTC-aware datetime
                published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))

                # Only collect comments from yesterday up to execution time
                if start_time <= published_dt <= end_time:
                    all_comments.append({
                        "video_id": video["video_id"],
                        "comment_id": item["snippet"]["topLevelComment"]["id"],
                        "title": video["title"],
                        "channel": video["channel"],
                        "author": snippet.get("authorDisplayName"),
                        "text": snippet.get("textDisplay"),
                        "likes": snippet.get("likeCount", 0),
                        "published_at": published_at
                    })

                    count += 1
                    if count >= 2000:
                        break

            next_page_token = res.get("nextPageToken")
            if not next_page_token:
                break

    # Example: store in XCom, or return it
    kwargs['ti'].xcom_push(key="collected_comments", value=all_comments)

def bulk_insert_comments(**kwargs):
    rows = kwargs['ti'].xcom_pull(key='collected_comments')
    hook = PostgresHook(postgres_conn_id='aws_pg')
    conn = hook.get_conn()
    cursor = conn.cursor()
    execution_date = kwargs['execution_date'].date()

    insert_sql = """
        INSERT INTO  raw_data.youtube_comments(
        video_id,
        comment_id,
        title,
        channel,
        text, 
        author, 
        likes, 
        published_at, 
        collected_at
        )
        VALUES %s
        ON CONFLICT (comment_id) DO NOTHING
    """

    values = [
        (
            row["video_id"],
            row['comment_id'],
            row.get("title"),
            row.get("channel"),
            row["text"],
            row.get("author"),
            row.get("likes", 0),
            row["published_at"],
            execution_date
        )
        for row in rows
    ]

    execute_values(cursor, insert_sql, values)
    conn.commit()
    cursor.close()
    conn.close()

    print(f"Successfully batch-inserted {len(values)} comments.")

default_args = {
    "owner": "DE_Adrian",
    "retries": 1,
    "retry_delay": timedelta(minutes=3)
}

with DAG(
    dag_id = "fetch_youtube_comment",
    default_args = default_args,
    start_date = datetime(2025, 4, 24),
    schedule_interval = "0 0 * * 6",
    catchup = False,
    tags=["raw", "youtube", "us stocks"],
) as dag:

    t1 = PythonOperator(
        task_id = "search_youtube", 
        python_callable = search_youtube
        )
    
    t2 = PythonOperator(
        task_id = "filter_with_openai", 
        python_callable = filter_with_openai
        )
    
    t3 = PythonOperator(
        task_id = "collect_comments", 
        python_callable = collect_comments
        )
    
    t4 = PythonOperator(
        task_id="load_to_postgres", 
        python_callable=bulk_insert_comments
        )

    t1 >> t2 >> t3 >> t4 
