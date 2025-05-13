from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from models import SP500_Sentiment
from typing import List

async def fetch_top_sectors_this_week(db: AsyncSession, limit: int = 5) -> List[dict]:
    
    latest_date_result = await db.execute(
        select(func.max(SP500_Sentiment.snapshot_date))
    )
    latest_date = latest_date_result.scalar_one_or_none()

    if not latest_date:
        return []

    stmt = (
        select(
            SP500_Sentiment.sector,
            func.sum(SP500_Sentiment.comments_count).label("total_comments"),
            func.sum(SP500_Sentiment.pos_count).label("total_pos"),
            func.sum(SP500_Sentiment.neg_count).label("total_neg"),
        )
        .where(SP500_Sentiment.snapshot_date == latest_date)
        .where(SP500_Sentiment.sector.isnot(None))
        .group_by(SP500_Sentiment.sector)
        .order_by(func.sum(SP500_Sentiment.comments_count).desc())
        .limit(limit)
    )

    result = await db.execute(stmt)
    rows = result.all()

    return [
        {
            "sector": r.sector,
            "total_comments": r.total_comments,
            "total_pos": r.total_pos,
            "total_neg": r.total_neg,
        }
        for r in rows
    ]