from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class RAGResponse(BaseModel):
    question: str
    answer: str
    context_used: list[str]