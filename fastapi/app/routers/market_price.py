from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import timedelta
from typing import List, Optional

from database import get_db
from models.market_price import MarketPrice as MarketPriceModel
from schemas.market_price import MarketPrice

router = APIRouter(prefix="/market_price", tags=["Market Price"])

@router.get("/", response_model=List[MarketPrice])
def get_market_prices(
    market: Optional[str] = Query(None, description="Market symbol (e.g. ^GSPC)"),
    db: Session = Depends(get_db)
):
    try:
        one_month_ago = func.current_date() - timedelta(days=30)

        query = db.query(MarketPriceModel).filter(
            MarketPriceModel.snapshot_time >= one_month_ago
        )

        if market:
            query = query.filter(MarketPriceModel.market == market)

        results = query.order_by(
            MarketPriceModel.market, MarketPriceModel.snapshot_time
        ).all()

        return [
            MarketPrice(
                date=r.snapshot_time,
                market=r.market,
                price=r.price,
                ma_3_days=r.ma_3_days,
                ma_5_days=r.ma_5_days,
                ma_7_days=r.ma_7_days
            ) for r in results
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
@router.get("/list", response_model=List[str])
def list_available_markets(db: Session = Depends(get_db)):
    try:
        results = db.query(MarketPriceModel.market).distinct().order_by(MarketPriceModel.market).all()
        return [r[0] for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")