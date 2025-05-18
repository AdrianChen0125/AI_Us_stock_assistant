from fastapi import APIRouter, Depends, HTTPException
from async_db import get_pgvector_conn
from schemas.rag_response import QuestionRequest, RAGResponse
from services.rag_service import process_rag_question

router = APIRouter(prefix="/AI",tags=["AI Agent"])

def set_experiment(name: str):
    import mlflow
    mlflow.set_experiment(name)

@router.post("/rag", response_model=RAGResponse)
async def rag_endpoint(req: QuestionRequest, db=Depends(get_pgvector_conn)):
    
    set_experiment("rag")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="問題不能為空白")
    
    return await process_rag_question(req.question, req.top_k, db)