from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import user_profiles
from routers import economic_index
from routers import market_price
from routers import stock_recommend
from routers import sp500 
from routers import Sentiment_Reddit,Sentiment_Topic,Sentiment_Sp500_Top,Sentiment_Sp500_Sector

from routers import AI_recommender
from routers import eco_report


app = FastAPI() 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_profiles.router)

# Econamic report 
app.include_router(economic_index.router)
app.include_router(market_price.router)


app.include_router(stock_recommend.router)
app.include_router(sp500.router)


#  Market Sentiment 
app.include_router(Sentiment_Reddit.router)
app.include_router(Sentiment_Topic.router)
app.include_router(Sentiment_Sp500_Top.router)
app.include_router(Sentiment_Sp500_Sector.router)

# AI agent 
app.include_router(AI_recommender.router)
app.include_router(eco_report.router)