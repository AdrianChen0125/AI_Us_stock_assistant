from pydantic import BaseModel
from datetime import date

class RedditSentimentDailySchema(BaseModel):
    topic_date: date
    comments_count: int
    pos_count: int
    neg_count: int

    class Config:
        from_attributes = True