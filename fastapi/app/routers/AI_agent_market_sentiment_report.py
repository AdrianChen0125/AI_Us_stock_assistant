from fastapi import APIRouter
from ai_agent_graph.market_sentiment_report.graph import create_market_report_graph
import logging

router = APIRouter(prefix="/AI",tags=["AI Agent"])
logger = logging.getLogger(__name__) 

def set_experiment(name: str):
    import mlflow
    mlflow.set_experiment(name)

@router.post("/sentiment_report")
async def generate_market_sentiment_report():
    try:
        set_experiment("sentiment_report_v1")
        # Build and compile the LangGraph
        graph = create_market_report_graph()

        # Run the graph asynchronously from an empty state
        final_state = await graph.ainvoke({})  # passing {} instead of MarketState()

        return {
            "report": final_state.get("report", "No report generated.")
        }

    except Exception as e:
        logger.error("Error generating market sentiment report", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }