from pydantic import BaseModel

class TopStockSentiment(BaseModel):
    symbol: str
    total_comments: int
    total_pos: int
    total_neg: int