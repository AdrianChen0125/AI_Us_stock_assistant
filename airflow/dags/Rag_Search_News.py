from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from newspaper import Article
from airflow.utils.timezone import make_naive  # <--- 加入這個
import os, ast, requests
from openai import OpenAI

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/opt/airflow/models")

default_args = {
    "owner": "DE_Adrian",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    dag_id="Rag_Search_News",
    default_args=default_args,
    start_date=datetime(2025, 5, 16),
    schedule_interval="0 6 * * 6",  # 每週六早上 6:00
    catchup=False,
)

# --- Task 1: Fetch keywords ---
def fetch_key_words(**kwargs):
    hook = PostgresHook(postgres_conn_id='aws_pg')
    conn = hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT topic_tags
        FROM dbt_us_stock_data_production."top_5_Topic_with_sentiment"
        WHERE topic_date::date = (
            SELECT MAX(topic_date::date)
            FROM dbt_us_stock_data_production."top_5_Topic_with_sentiment"
        )
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    all_tags = set()
    for row in rows:
        tags = row[0]
        if isinstance(tags, list):
            all_tags.update(tags)
    kwargs["ti"].xcom_push(key="topic_tags", value=list(all_tags))

# --- Task 2: Refine with GPT ---
def refine_keywords_with_gpt(**kwargs):
    topic_tags = kwargs["ti"].xcom_pull(key="topic_tags", task_ids="fetch_key_words")
    if not topic_tags:
        return
    prompt = f"""
    Given the topic tags: {', '.join(topic_tags)}, extract high-value search keywords such as:
    - human names
    - country names
    - economic or political terms
    Return them as a Python list of strings.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional financial journalist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )
    try:
        raw = response.choices[0].message.content
        keywords = ast.literal_eval(raw.strip())
    except Exception:
        keywords = topic_tags
    kwargs["ti"].xcom_push(key="search_keywords", value=keywords)

# --- Task 3: Fetch News ---
def fetch_news(**kwargs):
    execution_date = kwargs["execution_date"]
    today = make_naive(execution_date).date()
    from_date = today - timedelta(days=14)

    all_keywords = kwargs["ti"].xcom_pull(key="search_keywords", task_ids="refine_keywords_with_gpt")
    all_articles = []
    seen_urls = set()

    for keyword in all_keywords:
        for page in range(1, 3):
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": keyword,
                "pageSize": 20,
                "page": page,
                "from": from_date.isoformat(),
                "to": today.isoformat(),
                "sortBy": "relevancy",
                "language": "en",
                "apiKey": NEWS_API_KEY,
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                for article in articles:
                    article_url = article.get("url")
                    if not article_url or article_url in seen_urls:
                        continue
                    article["query_keyword"] = keyword
                    all_articles.append(article)
                    seen_urls.add(article_url)
            else:
                print(f"Failed to fetch for '{keyword}', page {page}: {response.status_code} {response.text}")

    kwargs["ti"].xcom_push(key="all_articles", value=all_articles)

# --- Task 4: Vectorize and Store ---
def vectorize_and_store(**kwargs):
    articles = kwargs["ti"].xcom_pull(key="all_articles", task_ids="fetch_news")
    if not articles:
        return

    hook = PostgresHook(postgres_conn_id='aws_pg')
    conn = hook.get_conn()
    cursor = conn.cursor()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for article in articles:
        title = article.get("title", "") or ""
        content = article.get("description", "") or ""
        url = article.get("url", "")
        published_at = article.get("publishedAt")
        keyword = article.get("query_keyword", "")

        try:
            art = Article(url)
            art.download()
            art.parse()
            full_text = art.text.strip()
        except:
            full_text = ""

        text_for_embedding = full_text if full_text and len(full_text) > 800 else f"{title}. {content}"

        cursor.execute("""
            INSERT INTO rag_docs.news_articles (title, content, published_at, url, query_keyword)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING
            RETURNING id
        """, (title, content, published_at, url, keyword))

        result = cursor.fetchone()
        if result is None:
            continue

        article_id = result[0]
        chunks = splitter.split_text(text_for_embedding)
        for idx, chunk in enumerate(chunks):
            vector = model.encode([chunk])[0].tolist()
            cursor.execute("""
                INSERT INTO rag_docs.news_chunks (article_id, chunk_index, chunk_text, embedding, model_name)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (article_id, chunk_index) DO NOTHING
            """, (article_id, idx, chunk, vector, "all-MiniLM-L6-v2"))

    conn.commit()
    cursor.close()
    conn.close()

# --- DAG Tasks ---
t1 = PythonOperator(task_id="fetch_key_words", python_callable=fetch_key_words, provide_context=True, dag=dag)
t2 = PythonOperator(task_id="refine_keywords_with_gpt", python_callable=refine_keywords_with_gpt, provide_context=True, dag=dag)
t3 = PythonOperator(task_id="fetch_news", python_callable=fetch_news, provide_context=True, dag=dag)
t4 = PythonOperator(task_id="vectorize_and_store", python_callable=vectorize_and_store, provide_context=True, dag=dag)

t1 >> t2 >> t3 >> t4