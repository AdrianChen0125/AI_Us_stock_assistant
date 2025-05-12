from pydantic import BaseModel
from datetime import date

class SentimentTopic(BaseModel):
    topic_date: date
    topic_summary: str
    comments_count: int
    pos_count: int
    neg_count: int
    source: str

    class Config:
        from_attributes = True