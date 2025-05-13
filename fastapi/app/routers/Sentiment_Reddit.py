from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from async_db import get_db
from services.sentiment_reddit_service import fetch_reddit_summary, compare_reddit_sentiment

router = APIRouter(prefix="/sentiment", tags=["Sentiment Summary"])

@router.get("/reddit_summary")
async def get_sentiment_summary(
    days: int = Query(30, ge=1),
    db: AsyncSession = Depends(get_db)
):
    try:
        return await fetch_reddit_summary(days=days, db=db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@router.get("/reddit_summary/compare")
async def compare_sentiment_summary(db: AsyncSession = Depends(get_db)):
    try:
        return await compare_reddit_sentiment(db=db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")