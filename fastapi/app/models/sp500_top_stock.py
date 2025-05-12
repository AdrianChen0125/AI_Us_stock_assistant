from sqlalchemy import Column, String, Date, Integer
from database import Base

class SP500TopStock(Base):
    __tablename__ = "sp500_sentiment_reddit"
    __table_args__ = {"schema": "dbt_us_stock_data_production"}

    id = Column(Integer, primary_key=True, index=True)
    snapshot_date = Column(Date, index=True)
    symbol = Column(String, index=True)
    comments_count = Column(Integer)
    pos_count = Column(Integer)
    neg_count = Column(Integer)