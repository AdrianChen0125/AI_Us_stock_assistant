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
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
import random


model_base = os.getenv("MODEL_PATH", "/opt/airflow/models")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=os.getenv("MODEL_PATH", "/opt/airflow/models"))

tokenizer = None
model = None


def fetch_recent_comments(**kwargs):
    exec_date = kwargs['execution_date'].date()
    start_date = exec_date - timedelta(days=7)
    end_date = exec_date 

    hook = PostgresHook(postgres_conn_id='aws_pg')
    conn = hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT comment_id, text FROM raw_data.youtube_comments
        WHERE DATE(published_at) BETWEEN %s AND %s
    """, (start_date, end_date))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    kwargs['ti'].xcom_push(key='raw_comment_texts', value=[{"comment_id": r[0], "text": r[1]} for r in rows])



def clean_texts(**kwargs):
    raw = kwargs['ti'].xcom_pull(key='raw_comment_texts')
    stop_words = set(stopwords.words('english'))

    def clean(text):
        text = re.sub(r"http\S+", "", text)  # remove URLs
        text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
        text = text.lower().strip()
        tokens = word_tokenize(text,language='english')
        filtered_tokens = [w for w in tokens if w not in stop_words]
        return " ".join(filtered_tokens) if len(filtered_tokens) > 3 else None

    cleaned = []
    for r in raw:
        raw_text = r["text"]
        cleaned_text = clean(raw_text)
        if not cleaned_text:
            continue
        try:
            if detect(cleaned_text) == 'en':
                cleaned.append({
                    "comment_id": r["comment_id"],
                    "text": cleaned_text
                })
        except LangDetectException:
            continue  # skip if language detection fails

    kwargs['ti'].xcom_push(key='cleaned_comments', value=cleaned)
    print(f"Filtered and cleaned {len(cleaned)} English comments.")

def analyze_sentiment(text):
    global tokenizer, model

    if tokenizer is None or model is None:
        base_path = os.getenv("MODEL_PATH", "/opt/airflow/models")
        sentiment_path = os.path.join(base_path, "models--nlptown--bert-base-multilingual-uncased-sentiment")
        
        snapshots_path = os.path.join(sentiment_path, "snapshots")
        snapshot_folders = os.listdir(snapshots_path)

        if len(snapshot_folders) != 1:
            raise ValueError(f"Found multiple snapshots: {snapshot_folders}")
        
        sentiment_path = os.path.join(snapshots_path, snapshot_folders[0])


        tokenizer = AutoTokenizer.from_pretrained(sentiment_path)
        model = AutoModelForSequenceClassification.from_pretrained(sentiment_path)

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

def run_bertopic(**kwargs):
    comments = kwargs['ti'].xcom_pull(key='cleaned_comments')

    # If number of comments exceeds 5000, randomly sample 5000
    if len(comments) > 5000:
        print(f'comments {len(comments)} over 5000, sampling to 5000')
        comments = random.sample(comments, 5000)

    texts = [c['text'] for c in comments]
    ids = [c['comment_id'] for c in comments]

    # Decide whether to specify number of topics based on number of comments

    if len(comments) < 1000:
        
        topic_model = BERTopic(
            embedding_model = embedding_model,
            calculate_probabilities=False,
            language="english"
        )
    else:
        # Manually set number of topics to 20
        topic_model = BERTopic(
            nr_topics=20,
            embedding_model = embedding_model,
            calculate_probabilities=False,
            language="english"
        )

    topics, _ = topic_model.fit_transform(texts)

    enriched = []
    for i, topic_id in enumerate(topics):
        keywords = [kw for kw, _ in topic_model.get_topic(topic_id)]
        sentiment = analyze_sentiment(texts[i])
        enriched.append({
            "comment_id": ids[i],
            "keywords": keywords,
            "topic_tags": keywords[:3] if keywords else [],
            "sentiment": sentiment
        })

    kwargs['ti'].xcom_push(key='bertopic_results', value=enriched)



def save_processed_comments(**kwargs):
    rows = kwargs['ti'].xcom_pull(key='bertopic_results')
    hook = PostgresHook(postgres_conn_id='aws_pg')
    conn = hook.get_conn()
    cursor = conn.cursor()
    insert_sql = """
        INSERT INTO processed_data.youtube_comments (
            comment_id,
            sentiment,
            topic_tags,
            keywords,
            processed_at
            )
        VALUES %s
    """
    values = [
        (
            row["comment_id"],
            row.get("sentiment", "neutral"),
            row["topic_tags"],
            row["keywords"],
            kwargs['execution_date'].date()
        ) for row in rows
    ]
    execute_values(cursor, insert_sql, values)
    conn.commit()
    cursor.close()
    conn.close()

default_args = {
    "owner": "DE_Adrian",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id = "Process_Sentiment_topicmodelling_youtube",
    default_args = default_args,
    start_date = datetime(2025, 4, 24),
    schedule_interval = "0 3 * * 6",
    catchup = False,
    max_active_runs = 1,
    dagrun_timeout = timedelta(minutes=60),
    tags=["processed", "youtube", "sentiment", "topic"]
    
) as dag:

    t1 = PythonOperator(task_id="fetch_recent_comments", python_callable=fetch_recent_comments)
    t2 = PythonOperator(task_id="clean_texts", python_callable=clean_texts)
    t3 = PythonOperator(task_id="run_bertopic", python_callable=run_bertopic)
    t4 = PythonOperator(task_id="save_processed_comments", python_callable=save_processed_comments)

    t1 >> t2 >> t3 >> t4
