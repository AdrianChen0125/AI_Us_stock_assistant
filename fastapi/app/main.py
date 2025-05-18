from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers - Core Modules
from routers import user_profiles
from routers import economic_index
from routers import market_price
from routers import stock_recommend
from routers import sp500

# Routers - Market Sentiment
from routers import Sentiment_Reddit
from routers import Sentiment_Topic
from routers import Sentiment_Sp500_Top
from routers import Sentiment_Sp500_Sector

# Routers - AI Agent Reports
from routers import AI_agent_recommendation
from routers import AI_agent_economic_report
from routers import AI_agent_market_sentiment_report
from routers import AI_agent_rag
from routers import AI_agent_chat_bot
from routers import AI_agent_summerizer

# Auth
from routers import auth

from middlewares.auth_middleware import AuthMiddleware

import mlflow

# === Initialize FastAPI ===
app = FastAPI()

# Add Auth Middleware
app.add_middleware(AuthMiddleware)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# track AI agent
def setup_experiment(name: str):
    mlflow.set_tracking_uri("http://mlflow:5001")
    mlflow.langchain.autolog()

# === Routers ===
app.include_router(auth.router)
app.include_router(user_profiles.router)
app.include_router(economic_index.router)
app.include_router(market_price.router)
app.include_router(stock_recommend.router)
app.include_router(sp500.router)
app.include_router(Sentiment_Reddit.router)
app.include_router(Sentiment_Topic.router)
app.include_router(Sentiment_Sp500_Top.router)
app.include_router(Sentiment_Sp500_Sector.router)
app.include_router(AI_agent_recommendation.router)
app.include_router(AI_agent_economic_report.router)
app.include_router(AI_agent_market_sentiment_report.router)
app.include_router(AI_agent_rag.router)
app.include_router(AI_agent_chat_bot.router)
app.include_router(AI_agent_summerizer.router)

# === Add Bearer Auth to OpenAPI ===
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="AI Assistant API",
        version="1.0.0",
        description="API with JWT authentication via Bearer token",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }

    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", [{"BearerAuth": []}])

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi