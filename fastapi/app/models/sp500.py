from sqlalchemy import Column, String, Integer, Float, Date
from database import Base

class SP500Price(Base):
    __tablename__ = "sp500_price"
    __table_args__ = {"schema": "dbt_us_stock_data_production"}

    symbol = Column(String, primary_key=True)
    company = Column(String)  
    sector = Column(String)
    sub_industry = Column(String)  
    market_cap = Column(Integer)
    volume = Column(Integer)
    previous_close = Column(Float)
    open = Column(Float)
    day_high = Column(Float)
    day_low = Column(Float)
    pe_ratio = Column(Float)
    forward_pe = Column(Float)
    dividend_yield = Column(Float)
    beta = Column(Float)
    high_52w = Column(Float)
    low_52w = Column(Float)
    snapshot_date = Column(Date, primary_key=True)  