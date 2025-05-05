from sqlalchemy import Column, Date, Integer
from database import Base

class RedditSentimentDaily(Base):
    __tablename__ = "reddit_comment_us_market_daily"
    __table_args__ = {"schema": "dbt_us_stock_data_production"}

    topic_date = Column(Date, primary_key=True)
    comments_count = Column(Integer)
    pos_count = Column(Integer)
    neg_count = Column(Integer)