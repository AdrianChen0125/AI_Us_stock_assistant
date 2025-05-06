from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from recommender.model import load_latest_model
import os

DATABASE_URL = os.getenv("DATABASE_URL")

model = load_latest_model()  

async def load_latest_data(db: AsyncSession) -> pd.DataFrame:
    query = text("""
        SELECT * FROM dbt_us_stock_data_production.sp500_price
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date) FROM dbt_us_stock_data_production.sp500_price
        );
    """)
    result = await db.execute(query)
    rows = result.fetchall()
    columns = result.keys()
    return pd.DataFrame(rows, columns=columns)

def preprocess(df):
    features = ["sector", "sub_industry", "market_cap", "volume", "previous_close", "pe_ratio", "dividend_yield"]
    df = df[["symbol"] + features].dropna()

    df_encoded = pd.get_dummies(df[["sector", "sub_industry"]]).astype(bool)
    df_numeric = df[["market_cap", "volume", "previous_close", "pe_ratio", "dividend_yield"]]
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_numeric), columns=df_numeric.columns)

    X = pd.concat([df_encoded.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)
    X[df_encoded.columns] = X[df_encoded.columns].astype(bool)
    return df.reset_index(drop=True), X

async def get_recommendations(symbols: list[str], db: AsyncSession):
    df_raw = await load_latest_data(db)
    df, X = preprocess(df_raw)

    target_mask = df["symbol"].isin(symbols)
    if target_mask.sum() == 0:
        return {"error": "No valid symbols found."}

    target_df = X[target_mask].mean(axis=0).to_frame().T
    target_df = target_df.reindex(columns=X.columns, fill_value=0)

    cat_cols = target_df.columns.difference(["market_cap", "volume", "previous_close", "pe_ratio", "dividend_yield"])
    target_df[cat_cols] = target_df[cat_cols].astype(bool)

    result = model.predict(target_df)
    neighbor_indices = result.values[0]
    recommendations = df.iloc[neighbor_indices]["symbol"].tolist()

    return recommendations