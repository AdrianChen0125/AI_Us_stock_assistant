from fastapi import APIRouter, Query, Depends
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.logic import get_recommendations
from async_db import get_db

router = APIRouter(
    prefix="/recommend",
    tags=["recommendations"]
)

@router.get("/")
async def recommend_stocks(
    symbols: List[str] = Query(...),
    user_id: str = Query("anonymous"),
    db: AsyncSession = Depends(get_db)
):
    result = await get_recommendations(symbols, db, user_id=user_id)

    if isinstance(result, dict) and "error" in result:
        return {"status": "error", "message": result["error"]}

    return {"status": "ok", "recommendations": result}