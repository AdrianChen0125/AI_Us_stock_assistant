from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langgraph_pr.graph import get_graph
from pydantic import BaseModel
import traceback
import asyncio
import re
import json

router = APIRouter(prefix="/econamic_report", tags=["AI Agent"])

class ReportRequest(BaseModel):
    language: str = "English"

def split_report_sections(response: dict) -> list[str]:
    try:
        raw = response.get("report", "")

        # case: LangChain TextBlock object
        if isinstance(raw, list) and raw and hasattr(raw[0], "text"):
            text = raw[0].text
        # fallback: raw is already string
        elif isinstance(raw, str):
            text = raw
        else:
            raise ValueError("Invalid report content structure.")

    except Exception as e:
        print(" Parsing report error:", e)
        return [" 無法擷取報告內容"]

    parts = re.split(r"(##+ .+)", text)
    combined = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        combined.append(f"{title}\n{content}")
    return combined if combined else [text]

@router.post("/generate/")
async def generate_streamed_report(request: ReportRequest):
    try:
        graph = get_graph()
        result = await graph.ainvoke({"language": request.language})

        def streamer():
            for section in split_report_sections(result):
                yield section + "\n\n"

        return StreamingResponse(streamer(), media_type="text/plain")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Report generation failed: {e}")