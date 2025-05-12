from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Literal, Optional
from langgraph_pr.AI_recommender_graph import build_graph

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

@router.post("/ai/stock_recommender", response_model=StockAnalysisOutput)
async def analyze_stock(input: StockAnalysisInput):
    graph = build_graph()
    result = await graph.ainvoke(input.dict())
    return {"analysis": result["analysis"]}