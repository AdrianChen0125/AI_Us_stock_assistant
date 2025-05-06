from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import date, timedelta
from async_db import get_db
from models.sentiment import RedditSentimentDaily

router = APIRouter(prefix="/sentiment", tags=["Sentiment Summary"])

@router.get("/reddit_summary")
async def get_sentiment_summary(
    days: int = Query(30, ge=1),
    db: AsyncSession = Depends(get_db)
):
    try:
        cutoff_date = date.today() - timedelta(days=days)
        stmt = (
            select(RedditSentimentDaily)
            .where(RedditSentimentDaily.topic_date >= cutoff_date)
            .order_by(RedditSentimentDaily.topic_date)
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()

        return [
            {
                "published_at": row.topic_date,
                "comments_count": row.comments_count,
                "total_pc": row.pos_count,
                "total_nc": row.neg_count
            }
            for row in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@router.get("/reddit_summary/compare")
async def compare_sentiment_summary(db: AsyncSession = Depends(get_db)):
    try:
        # Get max date
        max_date_stmt = select(func.max(RedditSentimentDaily.topic_date))
        result = await db.execute(max_date_stmt)
        max_date = result.scalar_one_or_none()

        if not max_date:
            return {"message": "No data available"}

        latest_7 = max_date - timedelta(days=7)
        prev_7 = max_date - timedelta(days=14)

        # Get all data in the last 14 days
        stmt = select(RedditSentimentDaily).where(
            RedditSentimentDaily.topic_date >= prev_7
        )
        result = await db.execute(stmt)
        all_rows = result.scalars().all()

        recent = [r for r in all_rows if r.topic_date > latest_7]
        prev = [r for r in all_rows if prev_7 <= r.topic_date <= latest_7]

        def summarize(rows):
            tc = sum(r.comments_count for r in rows)
            pc = sum(r.pos_count for r in rows)
            nc = sum(r.neg_count for r in rows)
            return tc, pc, nc, pc - nc

        tc_recent, pc_recent, nc_recent, net_recent = summarize(recent)
        tc_prev, pc_prev, nc_prev, net_prev = summarize(prev)

        return {
            "recent_7d": {
                "date": max_date,
                "total": tc_recent,
                "positive": pc_recent,
                "negative": nc_recent,
                "net": net_recent,
                "weighted_score": round((pc_recent - nc_recent) / tc_recent, 2) if tc_recent else None
            },
            "prev_7d": {
                "date": latest_7,
                "total": tc_prev,
                "positive": pc_prev,
                "negative": nc_prev,
                "net": net_prev,
                "weighted_score": round((pc_prev - nc_prev) / tc_prev, 2) if tc_prev else None
            },
            "delta_net": net_recent - net_prev,
            "delta_weighted_score": round(
                ((pc_recent - nc_recent) / tc_recent - (pc_prev - nc_prev) / tc_prev), 2
            ) if tc_recent and tc_prev else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")