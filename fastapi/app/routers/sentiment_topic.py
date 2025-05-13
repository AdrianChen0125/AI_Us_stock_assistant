from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from async_db import get_db
from schemas.sentiment_topic import SentimentTopic
from services.sentiment_topic_service import fetch_latest_sentiment_topics
from typing import List

router = APIRouter(prefix="/sentiment", tags=["Sentiment Summary"])

@router.get("/topic/latest", response_model=List[SentimentTopic])
async def get_latest_sentiments(db: AsyncSession = Depends(get_db)):
    return await fetch_latest_sentiment_topics(db)