from pydantic import BaseModel
from datetime import date

class EconomicIndex(BaseModel):
    date: date
    index_name: str
    value: float

    class Config:
        from_attributes = True