from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import boto3
import pickle
import mlflow
import os
from io import StringIO
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k

# === DAG 設定 ===
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}
dag = DAG(
    "lightfm_training_pipeline",
    default_args=default_args,
    description="Train LightFM in stages",
    schedule_interval= None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

BASE_PATH = "/tmp/lightfm_pipeline"

# --- Task 1: Extract ---
def extract_data():
    os.makedirs(BASE_PATH, exist_ok=True)
    s3 = boto3.client("s3", region_name="us-east-1")
    
    def fetch_csv(bucket, key):
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
    
    fetch_csv("etoro-data-prj", "interaction/investor_holdings.csv").to_csv(f"{BASE_PATH}/interactions.csv", index=False)
    fetch_csv("etoro-data-prj", "investor/csv/etoro_investors.csv").to_csv(f"{BASE_PATH}/users.csv", index=False)
    fetch_csv("etoro-data-prj", "stocks/nasdaq_tickers.csv").to_csv(f"{BASE_PATH}/stocks.csv", index=False)
    fetch_csv("etoro-data-prj", "stocks/nasdaq_etf.csv").to_csv(f"{BASE_PATH}/etf.csv", index=False)

# --- Task 2: Clean ---
def clean_data():
    interactions = pd.read_csv(f"{BASE_PATH}/interactions.csv")
    users = pd.read_csv(f"{BASE_PATH}/users.csv")
    stocks = pd.read_csv(f"{BASE_PATH}/stocks.csv")
    etf
    interactions["invested"] = (
        interactions["invested"]
        .astype(str)
        .str.replace(",", "")
        .str.replace("%", "")
        .str.replace("-", "")
        .str.strip()
    )
    interactions["invested"] = pd.to_numeric(interactions["invested"], errors="coerce").fillna(0)
    
    popular = interactions["symbol"].value_counts()
    popular = popular[popular >= 100].index
    interactions = interactions[interactions["symbol"].isin(popular)]

    stocks.rename(columns={"Symbol": "symbol"}, inplace=True)
    merged = pd.merge(interactions, stocks, on="symbol", how="left")

    merged.to_csv(f"{BASE_PATH}/cleaned.csv", index=False)
    users.to_csv(f"{BASE_PATH}/users_cleaned.csv", index=False)

# --- Task 3: Train ---
def train_model():
    mlflow.set_tracking_uri("http://mlflow:5001")
    mlflow.set_experiment("lightfm-recommender")

    merged = pd.read_csv(f"{BASE_PATH}/cleaned.csv")
    users = pd.read_csv(f"{BASE_PATH}/users_cleaned.csv")

    triples = list(zip(merged["username"], merged["symbol"], merged["invested"] / 100.0))

    item_features_dict = {
        row["symbol"]: [f"sector:{row['Sector']}", f"industry:{row['Industry']}"]
        for _, row in merged.dropna(subset=["Sector", "Industry"]).iterrows()
    }

    def normalize_risk(r):
        r = int(r)
        return "low" if r <= 3 else "moderate" if r <= 6 else "high"

    user_features_dict = {
        row["username"]: [f"risk:{normalize_risk(row['risk'])}"]
        for _, row in users.iterrows()
    }

    dataset = Dataset()
    all_users = set(merged["username"]).union(user_features_dict.keys())

    dataset.fit(
        users=all_users.union({"__cold_user__"}),
        items=merged["symbol"],
        user_features=[f for v in user_features_dict.values() for f in v],
        item_features=[f for v in item_features_dict.values() for f in v],
    )

    interactions, _ = dataset.build_interactions(triples)
    user_features = dataset.build_user_features(user_features_dict.items())
    item_features = dataset.build_item_features(item_features_dict.items())

    best_score = 0
    best_model = None
    best_config = {}

    mlflow.end_run()
    for k in [10, 20, 50]:
        for lr in [0.005, 0.01, 0.05]:
            with mlflow.start_run(run_name=f"k={k}_lr={lr}"):
                model = LightFM(no_components=k, learning_rate=lr, loss="warp")
                model.fit(interactions, user_features=user_features, item_features=item_features, epochs=20, num_threads=4)
                score = precision_at_k(model, interactions, user_features=user_features, item_features=item_features, k=5).mean()

                mlflow.log_param("no_components", k)
                mlflow.log_param("learning_rate", lr)
                mlflow.log_metric("precision_at_5", score)

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_config = {"no_components": k, "learning_rate": lr}

    # 儲存模型
    with open(f"{BASE_PATH}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(f"{BASE_PATH}/config.json", "w") as f:
        import json
        json.dump(best_config, f)

# --- Task 4: Register ---
def register_model():
    import json
    from pathlib import Path
    from mlflow.pyfunc import log_model

    mlflow.set_tracking_uri("http://mlflow:5001")
    mlflow.set_experiment("lightfm-recommender")

    # 讀取模型與 config
    with open(f"{BASE_PATH}/config.json") as f:
        best_config = json.load(f)

    model_path = f"{BASE_PATH}/best_model.pkl"
    with open(model_path, "rb") as f:
        best_model = pickle.load(f)

    # 註冊到 MLflow，模型會上傳到 artifact S3
    with mlflow.start_run(run_name="register_summary"):
        mlflow.log_params(best_config)
        mlflow.log_artifact(model_path)

        log_model(
            artifact_path="model",
            python_model=LightFMWrapper(),
            artifacts={"model": model_path},
            registered_model_name="LightFM-Recommender"
        )

# === DAG Tasks ===
t1 = PythonOperator(task_id="extract_data", python_callable=extract_data, dag=dag)
t2 = PythonOperator(task_id="clean_data", python_callable=clean_data, dag=dag)
t3 = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)
t4 = PythonOperator(task_id="register_model", python_callable=register_model, dag=dag)

t1 >> t2 >> t3 >> t4  # Task flow