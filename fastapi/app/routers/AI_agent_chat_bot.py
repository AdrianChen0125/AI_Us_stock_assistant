from fastapi import APIRouter, Request
from pydantic import BaseModel
from ai_agent_graph.chat_bot.graph import chat_graph

router = APIRouter(prefix="/AI",tags=["AI Agent"])

class ChatRequest(BaseModel):
    question: str
    history: list = []

def set_experiment(name: str):
    import mlflow
    mlflow.set_experiment(name)

@router.post("/chat")
async def chat(req: ChatRequest, request: Request): 
    set_experiment("chat_bot")
    username = request.state.user  

    inputs = {
        "messages": req.history + [{"role": "user", "content": req.question}]
    }
    config = {"recursion_limit": 10}
    output = chat_graph.invoke(inputs, config=config)
    for msg in output["messages"]:
        print(msg)
    return {"reply": output["messages"][-1].content}