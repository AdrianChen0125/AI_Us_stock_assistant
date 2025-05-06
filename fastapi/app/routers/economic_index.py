# routers/economic_index.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from async_db import get_db
from models.economic_index import EconomicIndex as EconomicIndexModel
from schemas import EconomicIndex
from typing import List, Optional
from datetime import timedelta

router = APIRouter(
    prefix="/economic_index",
    tags=["Economic Index"]
)

@router.get("/", response_model=List[EconomicIndex])
async def get_economic_index(
    index_name: Optional[str] = Query(None, description="Index code (optional)"),
    days: Optional[int] = Query(180, ge=1, le=1000, description="How many days of data to fetch"),
    db: AsyncSession = Depends(get_db)
):
    try:
        stmt = select(EconomicIndexModel).where(
            EconomicIndexModel.month_date >= func.now() - timedelta(days=days)
        )

        if index_name:
            stmt = stmt.where(EconomicIndexModel.series_id == index_name)

        stmt = stmt.order_by(
            EconomicIndexModel.series_id,
            EconomicIndexModel.month_date
        )

        result = await db.execute(stmt)
        records = result.scalars().all()

        return [
            EconomicIndex(
                date=record.month_date,
                index_name=record.series_id,
                value=record.current_month_value
            ) for record in records
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@router.get("/list")
async def list_index_names(db: AsyncSession = Depends(get_db)):
    try:
        stmt = select(EconomicIndexModel.series_id).distinct()
        result = await db.execute(stmt)
        return [r for (r,) in result.fetchall()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")