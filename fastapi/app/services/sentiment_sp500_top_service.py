from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from models import SP500_Sentiment

async def fetch_top_symbols(limit: int | None, db: AsyncSession) -> list[dict]:
    # fetch latest day 
    latest_date_stmt = select(func.max(SP500_Sentiment.snapshot_date))
    latest_result = await db.execute(latest_date_stmt)
    target_date = latest_result.scalar_one()

    # query data 
    stmt = (
        select(
            SP500_Sentiment.symbol,
            func.sum(SP500_Sentiment.comments_count).label("total_comments"),
            func.sum(SP500_Sentiment.pos_count).label("total_pos"),
            func.sum(SP500_Sentiment.neg_count).label("total_neg"),
        )
        .where(
            SP500_Sentiment.snapshot_date == target_date,
            SP500_Sentiment.symbol.isnot(None)
        )
        .group_by(SP500_Sentiment.symbol)
        .order_by(func.sum(SP500_Sentiment.comments_count).desc())
    )

    if limit:
        stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    rows = result.all()

    return [
        {
            "symbol": r.symbol,
            "total_comments": r.total_comments,
            "total_pos": r.total_pos,
            "total_neg": r.total_neg
        }
        for r in rows
    ]