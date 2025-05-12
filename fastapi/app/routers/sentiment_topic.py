from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from async_db import get_db
from models.sentiment_topic import TopSentimentTopic
from schemas.sentiment_topic import SentimentTopic
from sqlalchemy import select,func
from typing import List

router = APIRouter(prefix="/sentiment", tags=["Sentiment Summary"])

@router.get("/topic/latest", response_model=List[SentimentTopic])
async def get_latest_sentiments(db: AsyncSession = Depends(get_db)):
   
    latest_date_result = await db.execute(
        select(func.max(TopSentimentTopic.topic_date))
    )
    latest_date = latest_date_result.scalar()
    if not latest_date:
        return []

    stmt = (
        select(
            TopSentimentTopic.topic_date,
            TopSentimentTopic.topic_summary,
            TopSentimentTopic.comments_count,
            TopSentimentTopic.pos_count,
            TopSentimentTopic.neg_count,
            TopSentimentTopic.source,
        )
        .where(TopSentimentTopic.topic_date == latest_date)
        .order_by(TopSentimentTopic.comments_count.desc())
        .limit(10)
    )
    result = await db.execute(stmt)
    rows = result.fetchall()

    return [dict(row._mapping) for row in rows]