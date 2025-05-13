from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from async_db import async_session
from schemas import TopStockSentiment
from services.sentiment_sp500_top_service import fetch_top_symbols

router = APIRouter(prefix="/sentiment", tags=["Sentiment Summary"])

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

@router.get("/top_stock", response_model=List[TopStockSentiment])
async def get_symbols(
    limit: Optional[int] = Query(None, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    return await fetch_top_symbols(limit=limit, db=db)