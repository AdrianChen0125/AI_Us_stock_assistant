from sqlalchemy import Column, Date, String, Float
from database import Base

class EconomicIndex(Base):
    __tablename__ = "economic_index"
    __table_args__ = {"schema": "dbt_us_stock_data_production"}

    month_date = Column(Date, primary_key=True)
    series_id = Column(String, primary_key=True)
    current_month_value = Column(Float)