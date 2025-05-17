from typing import List, Dict
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from models.economic_index import EconomicIndex

def get_economic_index_data(db: Session) -> List[Dict]:
    results = (
        db.query(EconomicIndex)
        .filter(EconomicIndex.month_date >= datetime.utcnow().date() - timedelta(days=180))
        .order_by(EconomicIndex.series_id, EconomicIndex.month_date)
        .all()
    )

    return [
        {
            "date": r.month_date.isoformat(),
            "index_name": r.series_id,
            "value": r.current_month_value,
        }
        for r in results
    ]