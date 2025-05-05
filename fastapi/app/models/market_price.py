from sqlalchemy import Column, String, Float, TIMESTAMP
from database import Base

class MarketPrice(Base):
    __tablename__ = "market_price"
    __table_args__ = {"schema": "dbt_us_stock_data_production"}

    snapshot_time = Column(TIMESTAMP, primary_key=True)
    market = Column(String, primary_key=True)
    price = Column(Float)
    ma_3_days = Column(Float)
    ma_5_days = Column(Float)
    ma_7_days = Column(Float)