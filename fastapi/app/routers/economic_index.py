from fastapi import APIRouter, Depends, HTTPException,Query
from sqlalchemy.orm import Session
from database import get_db
from models.economic_index import EconomicIndex as EconomicIndexModel
from schemas import EconomicIndex
from typing import List,Optional
from datetime import timedelta
from sqlalchemy import func 

router = APIRouter(
    prefix="/economic_index",
    tags=["Economic Index"]
)

@router.get("/", response_model=List[EconomicIndex])
def get_economic_index(
    index_name: Optional[str] = Query(None, description="指標代碼（可選）"),
    days: Optional[int] = Query(180, ge=1, le=1000, description="查詢過去幾天內的資料"),
    db: Session = Depends(get_db)
):
    try:
        query = db.query(EconomicIndexModel).filter(
            EconomicIndexModel.month_date >= func.now() - timedelta(days=days)
        )

        if index_name:
            query = query.filter(EconomicIndexModel.series_id == index_name)

        results = query.order_by(
            EconomicIndexModel.series_id,
            EconomicIndexModel.month_date
        ).all()

        return [
            EconomicIndex(
                date=record.month_date,
                index_name=record.series_id,
                value=record.current_month_value
            ) for record in results
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
@router.get("/list")
def list_index_names(db: Session = Depends(get_db)):
    results = db.query(EconomicIndexModel.series_id).distinct().all()
    return [r[0] for r in results]