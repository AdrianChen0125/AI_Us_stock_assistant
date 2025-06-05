from fastapi import APIRouter
from fastapi import Request
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd
import mlflow
import os

# === MLflow 設定 ===
mlflow.set_tracking_uri("http://mlflow:5001")  # 改成你的 MLflow URI
experiment = mlflow.get_experiment_by_name("lightfm-recommender")
runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
run_id = runs.iloc[0]["run_id"]
artifact_path = "/tmp/mlflow_artifacts"

# === Artifact 下載 ===
def load_pickle_from_mlflow(filename: str):
    local_path = os.path.join(artifact_path, filename)
    if not os.path.exists(local_path):
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=filename, dst_path=artifact_path)
    with open(local_path, "rb") as f:
        return pickle.load(f)

model = load_pickle_from_mlflow("best_lightfm_model.pkl")
dataset = load_pickle_from_mlflow("lightfm_dataset.pkl")
item_features_dict = load_pickle_from_mlflow("item_features_dict.pkl")
stock_df = load_pickle_from_mlflow("stock_df.pkl")

item_features = dataset.build_item_features(item_features_dict.items())

# === Router ===
router = APIRouter(prefix="/recommend", tags=["recommendation"])

class ColdUserInput(BaseModel):
    risk: str
    interest_sectors: list[str]
    interest_asset_types: list[str]
    top_n: int = 10

def recommend_as_cold_user(model, dataset, item_features, stock_df, risk, interest_sectors, interest_asset_types, top_n=10, raw_top_k=100):
    user_traits = [f"risk:{risk.lower()}"]
    user_traits += [f"interest_sector:{s.strip().lower()}" for s in interest_sectors]
    user_traits += [f"interest_asset_type:{t.strip().lower()}" for t in interest_asset_types]

    user_features_temp = dataset.build_user_features(
        [("__cold_user__", user_traits)],
        normalize=False
    )

    num_items = dataset.interactions_shape()[1]
    scores = model.predict(
        user_ids=dataset.mapping()[0]["__cold_user__"],
        item_ids=np.arange(num_items),
        user_features=user_features_temp,
        item_features=item_features
    )

    reverse_item_mapping = {v: k for k, v in dataset.mapping()[2].items()}
    symbol_to_sector = stock_df.set_index("symbol")["sector"].str.lower().to_dict()
    symbol_to_asset = stock_df.set_index("symbol")["asset_type"].str.lower().to_dict()

    top_indices = np.argsort(-scores)[:raw_top_k]
    etf_list, sector_list, fallback_list = [], [], []

    for idx in top_indices:
        symbol = reverse_item_mapping[idx]
        score = scores[idx]
        sector = symbol_to_sector.get(symbol, "")
        asset_type = symbol_to_asset.get(symbol, "")
        if "etf" in interest_asset_types and asset_type == "etf" and len(etf_list) < 5:
            etf_list.append((symbol, score))
        elif sector in interest_sectors:
            sector_list.append((symbol, score))
        else:
            fallback_list.append((symbol, score))

    seen, final = set(), []
    for group in [etf_list, sector_list, fallback_list]:
        for symbol, score in group:
            if symbol not in seen:
                final.append((symbol, score))
                seen.add(symbol)
            if len(final) >= top_n:
                break
        if len(final) >= top_n:
            break

    return [sym for sym, _ in final]

@router.post("/")
async def get_recommendations(input: ColdUserInput,request: Request):
    
    recs = recommend_as_cold_user(
        model=model,
        dataset=dataset,
        item_features=item_features,
        stock_df=stock_df,
        risk=input.risk,
        interest_sectors=input.interest_sectors,
        interest_asset_types=input.interest_asset_types,
        top_n=input.top_n
    )
    return {"recommendations": recs}