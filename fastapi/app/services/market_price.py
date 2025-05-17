from sqlalchemy.orm import Session
from models import MarketPrice as MarketPriceModel
from datetime import datetime, timedelta
from typing import List, Dict

def get_market_price_data(db: Session) -> List[Dict]:
    one_month_ago = datetime.utcnow().date() - timedelta(days=30)
    
    results = (
        db.query(MarketPriceModel)
        .filter(MarketPriceModel.snapshot_time >= one_month_ago)
        .order_by(MarketPriceModel.market, MarketPriceModel.snapshot_time)
        .all()
    )

    return [
        {
            "date": r.snapshot_time.isoformat(),  # Convert datetime to string
            "market": r.market,
            "price": float(r.price)  # Ensure it's JSON-serializable
        } for r in results
    ]