from sqlalchemy import Column, String, Integer, Date, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class TopSentimentTopic(Base):
    __tablename__ = "top_5_Topic_with_sentiment"
    __table_args__ = {"schema": "dbt_us_stock_data_production"}

    topic_date = Column(Date, primary_key=True)
    topic_summary = Column(Text)
    comments_count = Column(Integer)
    pos_count = Column(Integer)
    neg_count = Column(Integer)
    source = Column(String)