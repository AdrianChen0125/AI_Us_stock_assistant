from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Literal, Optional
from ai_agent_graph.stock_recommendation_report.graph import build_graph

router = APIRouter(prefix="/AI",tags=["AI Agent"])

# creata model 
class StockAnalysisInput(BaseModel):
    holdings: List[str]
    recommended: List[str]
    style_preference: List[str] = []
    risk_tolerance: Optional[str] =""

# create schema 
class StockAnalysisOutput(BaseModel):
    analysis: str

def set_experiment(name: str):
    import mlflow
    mlflow.set_experiment(name)

@router.post("/stock_recommendation", response_model=StockAnalysisOutput)
async def analyze_stock(input: StockAnalysisInput):
    set_experiment("stock_recommendation_v1")
    graph = build_graph()
    result = await graph.ainvoke(input.dict())
    return {"analysis": result["analysis"]}