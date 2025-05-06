import os
import joblib
import pandas as pd
import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)



def load_latest_model(experiment_name: str = "stock_rc") -> mlflow.pyfunc.PyFuncModel:
    """
    load latest model by experiment 
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        filter_string="attributes.status = 'FINISHED'",
        max_results=1
    )

    if runs.empty:
        raise ValueError(" No successful MLflow run found.")

    latest_run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{latest_run_id}/model"
    print(f" Loading model from {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)