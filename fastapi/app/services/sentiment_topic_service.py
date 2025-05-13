from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from models.sentiment_topic import TopSentimentTopic

async def fetch_latest_sentiment_topics(db: AsyncSession) -> list[dict]:
    
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
            TopSentimentTopic.source
        )
        .where(TopSentimentTopic.topic_date == latest_date)
        .order_by(TopSentimentTopic.comments_count.desc())
        .limit(10)
    )

    result = await db.execute(stmt)
    rows = result.fetchall()

    return [dict(row._mapping) for row in rows]