from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, timedelta
from database import get_db
from models.sentiment import RedditSentimentDaily

router = APIRouter(prefix="/sentiment", tags=["Sentiment Summary"])

@router.get("/reddit_summary")
def get_sentiment_summary(
    days: int = Query(30, ge=1),
    db: Session = Depends(get_db)
):
    try:
        cutoff_date = date.today() - timedelta(days=days)
        results = (
            db.query(RedditSentimentDaily)
            .filter(RedditSentimentDaily.topic_date >= cutoff_date)
            .order_by(RedditSentimentDaily.topic_date)
            .all()
        )

        return [
            {
                "published_at": row.topic_date,
                "comments_count": row.comments_count,
                "total_pc": row.pos_count,
                "total_nc": row.neg_count
            }
            for row in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@router.get("/reddit_summary/compare")
def compare_sentiment_summary(db: Session = Depends(get_db)):
    try:
        max_date = db.query(func.max(RedditSentimentDaily.topic_date)).scalar()
        if not max_date:
            return {"message": "No data available"}

        latest_7 = max_date - timedelta(days=7)
        prev_7 = max_date - timedelta(days=14)

        all_rows = (
            db.query(RedditSentimentDaily)
            .filter(RedditSentimentDaily.topic_date >= prev_7)
            .all()
        )

        recent = [r for r in all_rows if r.topic_date > latest_7]
        prev = [r for r in all_rows if prev_7 <= r.topic_date <= latest_7]

        def summarize(rows):
            tc = sum(r.comments_count for r in rows)
            pc = sum(r.pos_count for r in rows)
            nc = sum(r.neg_count for r in rows)
            return tc,pc, nc, pc - nc

        tc_recent,pc_recent, nc_recent, net_recent = summarize(recent)
        tc_prev, pc_prev, nc_prev, net_prev = summarize(prev)

        return {
    "recent_7d": {
        "date":max_date,
        "total": tc_recent,
        "positive": pc_recent,
        "negative": nc_recent,
        "net": net_recent,
        "weighted_score": round((pc_recent - nc_recent) / tc_recent, 2) if tc_recent else None
    },
    "prev_7d": {
        "date":latest_7,
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