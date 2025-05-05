from fastapi import APIRouter, Query
from typing import List
from recommender.logic import get_recommendations

router = APIRouter(
    prefix="/recommend",
    tags=["recommendations"]
)

@router.get("/")
def recommend_stocks(symbols: List[str] = Query(..., description="List of stock symbols")):
    result = get_recommendations(symbols)
    if isinstance(result, dict) and "error" in result:
        return {"status": "error", "message": result["error"]}
    return {"status": "ok", "recommendations": result}