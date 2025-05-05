from pydantic import BaseModel
from datetime import datetime

class MarketPrice(BaseModel):
    date: datetime
    market: str
    price: float
    ma_3_days: float
    ma_5_days: float
    ma_7_days: float