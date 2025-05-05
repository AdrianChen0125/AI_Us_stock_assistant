from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import pandas as pd
import os,sys
import tempfile
import joblib
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
import mlflow
from dotenv import load_dotenv

include_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "include"))
if include_path not in sys.path:
    sys.path.insert(0, include_path)

from stock_model import StockRecommenderModel


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
EXPERIMENT_NAME = "stock_rc"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def load_latest_sp500_data():
    engine = create_engine(DATABASE_URL)
    query = """
    SELECT * FROM dbt_us_stock_data_production.sp500_price
    WHERE snapshot_date = (
        SELECT MAX(snapshot_date) FROM dbt_us_stock_data_production.sp500_price
    );
    """
    return pd.read_sql(query, engine)

def train_model(ti):
    df = load_latest_sp500_data()
    if df.empty:
        raise ValueError(" No data loaded from database.")

    df = df.dropna()
    features = ["sector", "sub_industry", "market_cap", "volume", "previous_close", "pe_ratio", "dividend_yield"]
    df = df[["symbol"] + features]
    df_encoded = pd.get_dummies(df[["sector", "sub_industry"]])
    df_numeric = df[["market_cap", "volume", "previous_close", "pe_ratio", "dividend_yield"]]
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_numeric), columns=df_numeric.columns)
    X = pd.concat([df_encoded.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("nn", NearestNeighbors(n_neighbors=5, metric="cosine"))
    ])
    pipeline.fit(X)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.pkl")
        joblib.dump(pipeline, model_path)

        artifacts = {"model_path": model_path}
        input_example = X.iloc[:1]

        with mlflow.start_run() as run:
            mlflow.log_param("model", "Pipeline: Scaler + NearestNeighbors")
            mlflow.log_param("num_features", X.shape[1])
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=StockRecommenderModel(),
                artifacts=artifacts,
                input_example=input_example,
            )

            # ✅ 傳出 run_id
            ti.xcom_push(key="run_id", value=run.info.run_id)

def push_model(ti):
    run_id = ti.xcom_pull(key="run_id", task_ids="train_model_task")
    if not run_id:
        raise ValueError("❌ No run_id found in XCom.")

    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=EXPERIMENT_NAME)

    client = mlflow.tracking.MlflowClient()
    latest_ver = client.get_latest_versions(EXPERIMENT_NAME, stages=["None"])[0].version
    client.transition_model_version_stage(
        name=EXPERIMENT_NAME,
        version=latest_ver,
        stage="Production",
        archive_existing_versions=True
    )

# === DAG ===
with DAG(
    dag_id="stock_recommender_model_training",
    default_args={"retries": 1},
    schedule_interval="0 2 * * 6",  
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "mlflow", "stocks"]
) as dag:

    train_task = PythonOperator(
        task_id="train_model_task",
        python_callable=train_model,
    )

    push_task = PythonOperator(
        task_id="push_model_task",
        python_callable=push_model,
    )

    train_task >> push_task