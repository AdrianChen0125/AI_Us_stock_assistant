from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from async_db import get_db
from services.sentiment_sp500_sector_service import fetch_top_sectors_this_week
from typing import List

router = APIRouter(prefix="/sentiment", tags=["Sentiment Summary"])

@router.get("/top_sectors", response_model=List[dict])
async def get_top_sectors(
    limit: int = Query(5, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    return await fetch_top_sectors_this_week(db=db, limit=limit)