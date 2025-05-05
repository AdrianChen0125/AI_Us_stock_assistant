import pandas as pd
import os
import tempfile
import joblib
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
import mlflow
from dotenv import load_dotenv
load_dotenv()

# CONFIG 
DATABASE_URL = os.getenv("DATABASE_URL")

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment('stock_rc')

# Custom PyFunc wrapper for NearestNeighbors pipeline
class StockRecommenderModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        distances, indices = self.model.named_steps["nn"].kneighbors(model_input)
        return pd.DataFrame(indices, columns=[f"neighbor_{i+1}" for i in range(indices.shape[1])])

def load_latest_sp500_data():
    engine = create_engine(DATABASE_URL)
    query = """
    SELECT * FROM dbt_us_stock_data_production.sp500_price
    WHERE snapshot_date = (
        SELECT MAX(snapshot_date) FROM dbt_us_stock_data_production.sp500_price
    );
    """
    return pd.read_sql(query, engine)

def train_and_log_model(df):
    df = df.dropna()
    features = ["sector", "sub_industry", "market_cap", "volume", "previous_close", "pe_ratio", "dividend_yield"]
    df = df[["symbol"] + features]

    # Encode & scale
    df_encoded = pd.get_dummies(df[["sector", "sub_industry"]])
    df_numeric = df[["market_cap", "volume", "previous_close", "pe_ratio", "dividend_yield"]]
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_numeric), columns=df_numeric.columns)
    X = pd.concat([df_encoded.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

    # Build pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("nn", NearestNeighbors(n_neighbors=5, metric="cosine"))
    ])
    pipeline.fit(X)

    # Save model & log to MLflow
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


if __name__ == "__main__":
    print("Loading data...")
    df = load_latest_sp500_data()
    if df.empty:
        print("❌ No data loaded from database.")
    else:
        print(f"✅ Loaded {len(df)} rows.")
        train_and_log_model(df)