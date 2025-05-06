# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import os
from dotenv import load_dotenv
from contextlib import contextmanager

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# get_db for synchronous usage (generator style, like FastAPI dependencies)
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()