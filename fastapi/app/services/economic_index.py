# services/economic_index.py
from sqlalchemy.orm import Session
from database import get_db
from models import EconomicIndex as EconomicIndexModel
import pandas as pd
from datetime import datetime, timedelta

def get_economic_index_df(db: Session) -> pd.DataFrame:
    results = (
        db.query(EconomicIndexModel)
        .filter(EconomicIndexModel.month_date >= datetime.utcnow().date() - timedelta(days=180))
        .order_by(EconomicIndexModel.series_id, EconomicIndexModel.month_date)
        .all()
    )
    return pd.DataFrame([
        {
            "date": r.month_date,
            "index_name": r.series_id,
            "value": r.current_month_value,
        } for r in results
    ])