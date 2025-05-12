
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional,List
from async_db import async_session
from models import SP500TopStock
from schemas import TopStockSentiment

router = APIRouter(prefix="/sentiment", tags=["Sentiment Summary"])

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

@router.get("/top_stock", response_model=List[TopStockSentiment])
async def get_symbols(
    limit: Optional[int] = Query(None, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    # Get latest snapshot_date
    latest_date_stmt = select(func.max(SP500TopStock.snapshot_date))
    latest_result = await db.execute(latest_date_stmt)
    target_date = latest_result.scalar_one()

    # Query
    stmt = (
        select(
            SP500TopStock.symbol,
            func.sum(SP500TopStock.comments_count).label("total_comments"),
            func.sum(SP500TopStock.pos_count).label("total_pos"),
            func.sum(SP500TopStock.neg_count).label("total_neg"),
        )
        .where(
            SP500TopStock.snapshot_date == target_date,
            SP500TopStock.symbol.isnot(None)
        )
        .group_by(SP500TopStock.symbol)
        .order_by(func.sum(SP500TopStock.comments_count).desc())
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