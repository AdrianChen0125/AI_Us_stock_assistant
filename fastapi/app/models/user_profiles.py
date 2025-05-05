from sqlalchemy import Column, Integer, String, TIMESTAMP, Text, ARRAY
from sqlalchemy.schema import UniqueConstraint
from database import Base

class UserProfile(Base):
    __tablename__ = "user_profiles"
    __table_args__ = {"schema": "raw_data"}

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(TIMESTAMP, server_default="CURRENT_TIMESTAMP")
    age = Column(Text)
    experience = Column(Text)
    interest = Column(ARRAY(Text))
    sources = Column(Text)
    risk = Column(Text)
    language = Column(Text)
    email = Column(Text, unique=True, index=True)