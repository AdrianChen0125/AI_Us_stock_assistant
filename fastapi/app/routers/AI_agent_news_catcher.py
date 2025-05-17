from fastapi import APIRouter, HTTPException
from ai_agent_graph.news_catcher.graph import build_langgraph
import logging
router = APIRouter(tags=["AI Agent"])
logger = logging.getLogger(__name__)

@router.post("/news_catcher")
async def summarize_topics():
    """
    Triggers the LangGraph AI agent pipeline to fetch, filter, search,
    and summarize sentiment-based trending topics.
    """
    try:
        logger.info("Starting LangGraph execution for summarizing topics...")
        
        graph = build_langgraph()
        initial_state = {}
        final_state = await graph.ainvoke(initial_state)

        results = final_state.get("summarized_results", [])
        logger.info(f"LangGraph execution completed. Found {len(results)} summaries.")

        return {"topics": results}

    except Exception as e:
        logger.error(f"Error during LangGraph execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to summarize topics.")