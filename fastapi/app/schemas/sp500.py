from typing import Optional
from pydantic import BaseModel
from datetime import date

class StockData(BaseModel):
    symbol: str
    company: str
    sector: Optional[str]
    sub_industry: Optional[str]
    market_cap: Optional[int]
    volume: Optional[int]
    previous_close: Optional[float]
    open: Optional[float]
    day_high: Optional[float]
    day_low: Optional[float]
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    high_52w: Optional[float]
    low_52w: Optional[float]
    snapshot_date: date


    class Config:
        from_attributes = True