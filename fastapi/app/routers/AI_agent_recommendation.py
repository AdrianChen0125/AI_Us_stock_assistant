from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Literal, Optional
from ai_agent_graph.stock_recommendation_report.graph import build_graph

router = APIRouter(tags=["AI Agent"])

# creata model 
class StockAnalysisInput(BaseModel):
    holdings: List[str]
    recommended: List[str]
    style_preference: List[str] = []
    risk_tolerance: Optional[str] =""

# create schema 
class StockAnalysisOutput(BaseModel):
    analysis: str

@router.post("/stock_recommendation", response_model=StockAnalysisOutput)
async def analyze_stock(input: StockAnalysisInput):
    graph = build_graph()
    result = await graph.ainvoke(input.dict())
    return {"analysis": result["analysis"]}