
import os
import asyncpg
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

# 載入環境變數
load_dotenv()

# SQLAlchemy 用的資料庫 URL
DATABASE_URL = os.getenv("DATABASE_URL")

# asyncpg 用的參數（向量查詢等）
PG_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432))
}
# db.py 中自動處理格式（這段你已經有）
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    
# 建立 SQLAlchemy 的 async engine 與 session
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# ORM 的基礎 model 類別
Base = declarative_base()

# ORM 的依賴注入（FastAPI 用）
async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

# asyncpg 的連線（向量查詢用）
async def get_pgvector_conn():
    return await asyncpg.connect(**PG_CONFIG)