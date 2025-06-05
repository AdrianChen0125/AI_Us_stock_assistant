from pydantic import BaseModel
from typing import List

class ContextChunk(BaseModel):
    chunk_text: str
    url: str

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class RAGOnlyContextResponse(BaseModel):
    question: str
    context_used: List[ContextChunk]