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

# Initialize app
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Routers
# -------------------------------

# User Management
app.include_router(user_profiles.router)

# Economic Data
app.include_router(economic_index.router)
app.include_router(market_price.router)

# Stock Information
app.include_router(stock_recommend.router)
app.include_router(sp500.router)

# Market Sentiment
app.include_router(Sentiment_Reddit.router)
app.include_router(Sentiment_Topic.router)
app.include_router(Sentiment_Sp500_Top.router)
app.include_router(Sentiment_Sp500_Sector.router)

# AI-Generated Reports
app.include_router(AI_agent_recommendation.router)
app.include_router(AI_agent_economic_report.router)
app.include_router(AI_agent_market_sentiment_report.router)
app.include_router(AI_agent_rag.router)