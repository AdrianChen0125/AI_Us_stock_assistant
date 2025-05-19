from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ai_agent_graph.summerizer.graph import build_report_graph
from ai_agent_graph.summerizer.nodes import AgentState

router = APIRouter(prefix="/AI",tags=["AI Agent"])
# 請求格式
class ReportRequest(BaseModel):
    language: str = "English"
    economic_summary: str
    sentiment_summary: str
    stock_summary: str
    age: str = "18-25"
    experience: str = "Beginner"
    risk: str = "Moderate"

# 回傳格式
class ReportResponse(BaseModel):
    report: str

# 建構報告產生流程圖
graph = build_report_graph()

def set_experiment(name: str):
    import mlflow
    mlflow.set_experiment(name)

@router.post("/summerise_report", response_model=ReportResponse)
async def generate_ai_report(req: ReportRequest):
    try:
        set_experiment("summerise_report_v1")

        state: AgentState = {
            "language": req.language,
            "economic_summary": req.economic_summary,
            "sentiment_summary": req.sentiment_summary,
            "stock_summary": req.stock_summary,
            "user_profile": {
                "age": req.age,
                "experience": req.experience,
                "risk": req.risk
            },
            "report": ""
        }

        result = graph.invoke(state)
        return {"report": result["report"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")