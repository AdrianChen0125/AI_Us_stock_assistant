from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from models.sp500 import SP500Price
import pandas as pd
from typing import List

async def fetch_sp500_service(symbols: List[str], db: AsyncSession) -> pd.DataFrame:
    max_date_result = await db.execute(
        select(func.max(SP500Price.snapshot_date))
    )
    max_date = max_date_result.scalar()

    if not max_date:
        return pd.DataFrame()

    stmt = (
        select(SP500Price)
        .where(SP500Price.symbol.in_(symbols))
        .where(SP500Price.snapshot_date == max_date)
    )
    result = await db.execute(stmt)
    data = result.scalars().all()

    return pd.DataFrame([row.__dict__ for row in data]) if data else pd.DataFrame()