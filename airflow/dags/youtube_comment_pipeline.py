from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
from airflow.utils.dates import days_ago
from psycopg2.extras import execute_values
from datetime import datetime,  timedelta
import os, requests,random
from openai import OpenAI

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Flexible query (fallback to default)
def get_search_query(**context):
    # 股市關鍵字清單
    stock_keywords = [
        "us stocks",
        "stock market",
        "nasdaq",
        "dow jones",
        "s&p 500",
    ]
    
    dt = datetime.now()       # 取得目前時間（本地時區）
    year = dt.year  
    
    # 隨機選擇 N 個關鍵字組合在一起（可依需要調整 N）
    selected_keywords = random.sample(stock_keywords, k=3)
    # 使用 OR 串接關鍵字（YouTube API 支援布林搜尋）
    query = " OR ".join(f'"{kw}"' for kw in selected_keywords)
    query += f"and {year}"
    # 如果 Variable 有設，就用 Variable（保留彈性）
    return Variable.get("youtube_search_query", default_var=query)

def search_youtube(**context):
    query = get_search_query()
    print(query)
    # 設定時間範圍：過去兩週
    end_time = context['execution_date']
    start_time = end_time - timedelta(days=30)
    print(end_time,start_time)

    # 第一階段：用搜尋 API 抓影片 ID
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "key": YOUTUBE_API_KEY,
        "q": query,
        "part": "snippet",
        "type": "video",
        "maxResults": 50,
    }

    search_res = requests.get(search_url, params=search_params).json()

    # 提取 videoIds
    video_ids = [
        item["id"]["videoId"]
        for item in search_res.get("items", [])
        if item["id"].get("videoId")
    ]

    if not video_ids:
        context['ti'].xcom_push(key='raw_videos', value=[])
        return

    # 第二階段：用 videos API 抓詳細資訊（含 viewCount）
    video_url = "https://www.googleapis.com/youtube/v3/videos"
    video_params = {
        "key": YOUTUBE_API_KEY,
        "part": "snippet,statistics",
        "id": ",".join(video_ids)
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

    # 根據觀看數排序，取前 10 名
    top_videos = sorted(videos, key=lambda x: x["views"], reverse=True)[:20]

    # 推送到 XCom
    context['ti'].xcom_push(key='top_videos', value=top_videos)

def filter_with_openai(**context):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # 抓取正確的 XCom key
    videos = context['ti'].xcom_pull(key='top_videos', task_ids='search_youtube')
    filtered = []

    if not videos:
        print("⚠️ No videos found to filter.")
        context['ti'].xcom_push(key='filtered_videos', value=[])
        return

    for vid in videos:
        prompt = f"""
        Is this YouTube video title related to the U.S. stock market?
        Title: "{vid['title']}" Answer yes or no:
        """

        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = res.choices[0].message.content.strip().lower()
        if "yes" in answer:
            filtered.append(vid)

    context['ti'].xcom_push(key='filtered_videos', value=filtered)


def select_top_videos(**context):
    videos = context['ti'].xcom_pull(key='filtered_videos')
    ids = ",".join([v["video_id"] for v in videos])
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "key": YOUTUBE_API_KEY,
        "id": ids,
        "part": "statistics,snippet"
    }
    res = requests.get(url, params=params).json()
    sorted_videos = sorted(res["items"], key=lambda v: int(v["statistics"].get("viewCount", 0)), reverse=True)
    top_15 = [{
        "video_id": v["id"],
        "title": v["snippet"]["title"],
        "channel": v["snippet"]["channelTitle"],
        "views": v["statistics"]["viewCount"]
    } for v in sorted_videos[:15]]
    context['ti'].xcom_push(key='top_videos_', value=top_15)

def collect_comments(**context):
    top_videos = context['ti'].xcom_pull(key='top_videos_')
    all_comments = []

    execution_date = context['execution_date']  # Airflow 提供的執行時間 (UTC)
    start_time = execution_date - timedelta(days=1)  
    end_time = execution_date  # 今天 00:00 UTC

    for video in top_videos:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            "part": "snippet",
            "videoId": video["video_id"],
            "key": YOUTUBE_API_KEY,
            "maxResults": 100,
            "textFormat": "plainText"
        }

        count = 0
        next_page_token = None

        while count < 1000:
            if next_page_token:
                params["pageToken"] = next_page_token

            res = requests.get(url, params=params).json()

            for item in res.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                published_at = snippet.get("publishedAt")

                if not published_at:
                    continue

                # 轉換成 datetime 物件（保留時間）
                published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                print(published_dt,start_time,end_time)
                # 僅收集昨天 (UTC) 發佈的留言
                if not (start_time <= published_dt < end_time):
                    continue

                all_comments.append({
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "channel": video["channel"],
                    "author": snippet.get("authorDisplayName"),
                    "text": snippet.get("textDisplay"),
                    "likes": snippet.get("likeCount", 0),
                    "published_at": published_at
                })

                count += 1
                if count >= 1000:
                    break

            next_page_token = res.get("nextPageToken")
            if not next_page_token:
                break

    context['ti'].xcom_push(key='youtube_comments', value=all_comments)
    print(f" Collected {len(all_comments)} comments.")

def bulk_insert_comments(**context):
    rows = context['ti'].xcom_pull(key='youtube_comments')

    hook = PostgresHook(postgres_conn_id='aws_pg')
    conn = hook.get_conn()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO raw_comments (video_id, title, channel, text, author, likes, published_at, collected_at)
        VALUES %s
    """

    values = [
        (
            row["video_id"],
            row.get("title"),
            row.get("channel"),
            row["text"],
            row.get("author"),
            row.get("likes", 0),
            row["published_at"],
            datetime.utcnow()
        )
        for row in rows
    ]

    execute_values(cursor, insert_sql, values)
    conn.commit()
    cursor.close()
    conn.close()

    print(f"批次插入 {len(values)} 筆留言成功")

# DAG definition
default_args = {
    "start_date": datetime(2025, 4, 1),
}

with DAG(
    dag_id="youtube_comment_pipeline_us_stocks_daily",
    default_args=default_args,
    schedule_interval="0 8 * * *",
    catchup=True,
    tags=["youtube", "daily", "us stocks"],
) as dag:

    t1 = PythonOperator(task_id="search_youtube", python_callable=search_youtube)
    t2 = PythonOperator(task_id="filter_with_openai", python_callable=filter_with_openai)
    t3 = PythonOperator(task_id="select_top_videos", python_callable=select_top_videos)
    t4 = PythonOperator(task_id="collect_comments", python_callable=collect_comments)
    t5 = PythonOperator(task_id="load_to_postgres", python_callable=bulk_insert_comments)

    t1 >> t2 >> t3 >> t4 >> t5
