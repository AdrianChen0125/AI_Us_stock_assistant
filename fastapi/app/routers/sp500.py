from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from datetime import date
from typing import Optional, List
from async_db import get_db
from models import SP500Price
from schemas import StockData

router = APIRouter(prefix="/stock_data", tags=["Stock Data"])


@router.get("", response_model=List[StockData])
async def get_stock_data(
    symbol: Optional[List[str]] = Query(None),
    sector: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(SP500Price)

    if symbol:
        stmt = stmt.where(SP500Price.symbol.in_(symbol))
    if sector:
        stmt = stmt.where(SP500Price.sector == sector)
    if start_date:
        stmt = stmt.where(SP500Price.snapshot_date >= start_date)
    if end_date:
        stmt = stmt.where(SP500Price.snapshot_date <= end_date)

    if not any([symbol, sector, start_date, end_date]):
        max_date_result = await db.execute(select(func.max(SP500Price.snapshot_date)))
        max_date = max_date_result.scalar()
        if max_date:
            stmt = stmt.where(SP500Price.snapshot_date == max_date)

    result = await db.execute(stmt)
    data = result.scalars().all()

    if not data:
        raise HTTPException(status_code=404, detail="No data found")

    return data

@router.get("/sector_list", response_model=List[Optional[str]])
async def get_sector(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(SP500Price.sector).distinct().order_by(SP500Price.sector)
    )
    sectors = result.scalars().all()
    return sectors


@router.get("/latest_date", response_model=str)
async def get_latest_date(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(func.max(SP500Price.snapshot_date)))
    max_date = result.scalar()
    if not max_date:
        raise HTTPException(status_code=404, detail="No data found.")
    return str(max_date)