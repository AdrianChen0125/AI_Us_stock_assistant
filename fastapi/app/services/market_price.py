from sqlalchemy.orm import Session
from models import MarketPrice as MarketPriceModel
from datetime import datetime, timedelta
import pandas as pd

def get_market_price_df(db: Session) -> pd.DataFrame:
    one_month_ago = datetime.utcnow().date() - timedelta(days=30)
    
    results = (
        db.query(MarketPriceModel)
        .filter(MarketPriceModel.snapshot_time >= one_month_ago)
        .order_by(MarketPriceModel.market, MarketPriceModel.snapshot_time)
        .all()
    )

    return pd.DataFrame([
        {
            "date": r.snapshot_time,
            "market": r.market,
            "price": r.price
        } for r in results
    ])