from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import timedelta
from typing import List, Optional

from async_db import get_db
from models.market_price import MarketPrice as MarketPriceModel
from schemas.market_price import MarketPrice

router = APIRouter(prefix="/market_price", tags=["Market Price"])

@router.get("/", response_model=List[MarketPrice])
async def get_market_prices(
    market: Optional[str] = Query(None, description="Market symbol (e.g. ^GSPC)"),
    db: AsyncSession = Depends(get_db)
):
    try:
        one_month_ago = func.current_date() - timedelta(days=30)

        stmt = select(MarketPriceModel).where(
            MarketPriceModel.snapshot_time >= one_month_ago
        )

        if market:
            stmt = stmt.where(MarketPriceModel.market == market)

        stmt = stmt.order_by(
            MarketPriceModel.market, MarketPriceModel.snapshot_time
        )

        result = await db.execute(stmt)
        records = result.scalars().all()

        return [
            MarketPrice(
                date=r.snapshot_time,
                market=r.market,
                price=r.price,
                ma_3_days=r.ma_3_days,
                ma_5_days=r.ma_5_days,
                ma_7_days=r.ma_7_days
            ) for r in records
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"async_db error: {e}")


@router.get("/list", response_model=List[str])
async def list_available_markets(db: AsyncSession = Depends(get_db)):
    try:
        stmt = select(MarketPriceModel.market).distinct().order_by(MarketPriceModel.market)
        result = await db.execute(stmt)
        return [r for (r,) in result.fetchall()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"async_db error: {e}")