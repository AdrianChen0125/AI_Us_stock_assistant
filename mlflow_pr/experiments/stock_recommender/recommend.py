import argparse
import os
import pandas as pd
import mlflow.pyfunc
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "stock_rc"

# Load latest model
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    filter_string="attributes.status = 'FINISHED'",
    max_results=1
)

if runs.empty:
    raise Exception("❌ No successful MLflow run found.")

latest_run_id = runs.iloc[0].run_id
model_uri = f"runs:/{latest_run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)


def load_latest_data():
    engine = create_engine(DATABASE_URL)
    query = """
    SELECT * FROM dbt_us_stock_data_production.sp500_price
    WHERE snapshot_date = (
        SELECT MAX(snapshot_date) FROM dbt_us_stock_data_production.sp500_price
    );
    """
    return pd.read_sql(query, engine)


def preprocess(df):
    features = ["sector", "sub_industry", "market_cap", "volume", "previous_close", "pe_ratio", "dividend_yield"]
    df = df[["symbol"] + features].dropna()

    df_encoded = pd.get_dummies(df[["sector", "sub_industry"]]).astype(bool)
    df_numeric = df[["market_cap", "volume", "previous_close", "pe_ratio", "dividend_yield"]]
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_numeric), columns=df_numeric.columns)

    X = pd.concat([df_encoded.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)
    X[df_encoded.columns] = X[df_encoded.columns].astype(bool)
    return df.reset_index(drop=True), X


def recommend(symbols):
    df_raw = load_latest_data()
    df, X = preprocess(df_raw)

    target_mask = df["symbol"].isin(symbols)
    if target_mask.sum() == 0:
        print("❌ No valid symbols found.")
        return

    target_df = X[target_mask].mean(axis=0).to_frame().T
    target_df = target_df.reindex(columns=X.columns, fill_value=0)

    # 強制類別欄轉 boolean（避免 float64 對 bool schema 衝突）
    cat_cols = target_df.columns.difference(["market_cap", "volume", "previous_close", "pe_ratio", "dividend_yield"])
    target_df[cat_cols] = target_df[cat_cols].astype(bool)

    result = model.predict(target_df)
    neighbor_indices = result.values[0]
    recommendations = df.iloc[neighbor_indices]["symbol"].tolist()

    print("\n✅ Top Recommendations:")
    for i, symbol in enumerate(recommendations, 1):
        print(f"{i}. {symbol}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True, help="List of stock symbols to base recommendations on")
    args = parser.parse_args()
    recommend(args.symbols)