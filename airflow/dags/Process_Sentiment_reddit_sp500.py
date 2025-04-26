from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import re, os
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model_base = os.getenv("MODEL_PATH", "/opt/airflow/models")
sentiment_path = os.path.join(model_base, "nlptown/bert-base-multilingual-uncased-sentiment")

tokenizer = None
model = None



def fetch_recent_comments(**context):
    exec_date = context['execution_date'].date()
    start_date = exec_date - timedelta(days=7)
    end_date = exec_date

    hook = PostgresHook(postgres_conn_id='aws_pg')
    conn = hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            symbol,
            post_id,
            post_title, 
            comment_id, 
            subreddit, 
            author, 
            comment_text, 
            score, 
            created_utc
        FROM raw_data.reddit_comments
        WHERE DATE(created_utc) BETWEEN %s AND %s
    """, (start_date, end_date))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    comments = [{
        "symbol": r[0],
        "post_id": r[1],
        "post_title": r[2],
        "comment_id": r[3],
        "subreddit": r[4],
        "author": r[5],
        "text": r[6],
        "score": r[7],
        "created_utc": r[8],
    } for r in rows]

    context['ti'].xcom_push(key='raw_comments', value=comments)


def clean_and_analyze(**context):
    global tokenizer, model
    
    nltk.download('punkt')

    if tokenizer is None or model is None:

        tokenizer = AutoTokenizer.from_pretrained(sentiment_path)
        model = AutoModelForSequenceClassification.from_pretrained(sentiment_path)

    stop_words = set(stopwords.words('english'))
    raw = context['ti'].xcom_pull(key='raw_comments')

    def clean(text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = text.lower().strip()
        tokens = word_tokenize(text,language='english')
        filtered_tokens = [w for w in tokens if w not in stop_words]
        return " ".join(filtered_tokens) if len(filtered_tokens) > 3 else None

    def analyze_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted = torch.argmax(scores, dim=1).item()
            if predicted <= 1:
                return 'negative'
            elif predicted == 2:
                return 'neutral'
            else:
                return 'positive'

    results = []
    for r in raw:
        cleaned_text = clean(r["text"])
        if not cleaned_text:
            continue
        try:
            if detect(cleaned_text) == 'en':
                sentiment = analyze_sentiment(cleaned_text)
                r["sentiment"] = sentiment
                results.append(r)
        except LangDetectException:
            continue

    context['ti'].xcom_push(key='analyzed_comments', value=results)


def save_to_postgres(**context):
    comments = context['ti'].xcom_pull(key='analyzed_comments')
    processed_at = context['execution_date']

    hook = PostgresHook(postgres_conn_id='aws_pg')
    conn = hook.get_conn()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO processed_data.reddit_comments_sp500 (
            symbol, 
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
    """
    values = [
        (
            r["symbol"],
            r["post_id"],
            r["post_title"],
            r["comment_id"],
            r["subreddit"],
            r["author"],
            r["text"],
            r["score"],
            r["created_utc"],
            fetched_at,
        ) for r in comments
    ]

    execute_values(cursor, insert_sql, values)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[INFO] Saved {len(comments)} comments with sentiment.")


with DAG(
    dag_id="reddit_sentiment_analysis_sp500",
    start_date=datetime(2025, 4, 24),
    schedule_interval="0 4 * * 6",  
    catchup=True,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=60),
    tags=["nlp", "reddit", "sentiment"]
) as dag:

    t1 = PythonOperator(task_id="fetch_recent_comments", python_callable=fetch_recent_comments)
    t2 = PythonOperator(task_id="clean_and_analyze", python_callable=clean_and_analyze)
    t3 = PythonOperator(task_id="save_to_postgres", python_callable=save_to_postgres)

    t1 >> t2 >> t3