from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ai_agent_graph.economic_report.graph import get_graph
import re
import logging
import traceback
from typing import List

def set_experiment(name: str):
    import mlflow
    mlflow.set_experiment(name)

router = APIRouter(prefix="/AI",tags=["AI Agent"])
logger = logging.getLogger(__name__)

class ReportRequest(BaseModel):
    language: str = "English"

def split_report_sections(response: dict) -> List[str]:
    try:
        raw = response.get("economic_report", "")

        if isinstance(raw, list) and raw and hasattr(raw[0], "text"):
            text = raw[0].text
        elif isinstance(raw, str):
            text = raw
        else:
            raise ValueError("Invalid report content format.")

        parts = re.split(r"(##+ .+)", text)
        combined = []
        for i in range(1, len(parts), 2):
            title = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            combined.append(f"{title}\n{content}")

        return combined if combined else [text]

    except Exception as e:
        logger.error(f"Error parsing report: {e}")
        return ["Unable to extract report content."]

@router.post("/economic_report/")
async def generate_report(request: ReportRequest):
    try:
        set_experiment("economic_report_v1")
        graph = get_graph()
        result = await graph.ainvoke({"language": request.language})

        sections = split_report_sections(result)
        full_text = "\n\n".join(sections)

        return {"report": full_text}

    except Exception as e:
        logger.error("Report generation failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")