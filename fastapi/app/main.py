from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import user_profiles,economic_index, market_price,eco_report,sentiment,stock_recommend

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_profiles.router)
app.include_router(economic_index.router)
app.include_router(market_price.router)
app.include_router(eco_report.router)
app.include_router(sentiment.router)
app.include_router(stock_recommend.router)
