from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from models.sp500 import SP500Price
from typing import List, Dict

async def fetch_sp500_service(symbols: List[str], db: AsyncSession) -> List[Dict]:
    max_date_result = await db.execute(
        select(func.max(SP500Price.snapshot_date))
    )
    max_date = max_date_result.scalar()

    if not max_date:
        return []

    stmt = (
        select(SP500Price)
        .where(SP500Price.symbol.in_(symbols))
        .where(SP500Price.snapshot_date == max_date)
    )
    result = await db.execute(stmt)
    data = result.scalars().all()

    return [
        {
            "symbol": row.symbol,
            "price": row.price,
            "volume": row.volume,
            "snapshot_date": row.snapshot_date.isoformat(),
        }
        for row in data
    ] if data else []