-- Create DB
Create ytdb;
Use ytdb;

-- SCHEMA CREATION 
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS processed_data;
CREATE SCHEMA IF NOT EXISTS production_data;
CREATE SCHEMA IF NOT EXISTS rag_docs;

-- ENUM TYPE 
CREATE TYPE sentiment_type AS ENUM ('positive', 'neutral', 'negative');

-- raw_data Schema Tables 
CREATE TABLE IF NOT EXISTS raw_data.market_snapshots (
    id SERIAL PRIMARY KEY,
    market VARCHAR(50) NOT NULL,           -- Market name, e.g., NASDAQ, SP500, BITCOIN
    snapshot_time DATE NOT NULL,           -- Snapshot date
    price NUMERIC,                         -- Market closing price
    UNIQUE (market, snapshot_time)
);

CREATE TABLE raw_data.sp500_snapshots (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    company TEXT,
    sector TEXT,
    sub_industry TEXT,
    market_cap BIGINT,
    volume BIGINT,
    previous_close FLOAT,
    open FLOAT,
    day_high FLOAT,
    day_low FLOAT,
    pe_ratio FLOAT,
    forward_pe FLOAT,
    dividend_yield FLOAT,
    beta FLOAT,
    high_52w FLOAT,
    low_52w FLOAT,
    snapshot_date DATE NOT NULL,
    UNIQUE (symbol, snapshot_date) 
);
CREATE TABLE IF NOT EXISTS raw_data.reddit_comments_sp500
(
    comment_id TEXT PRIMARY KEY,
    symbol TEXT,  
    company TEXT,  
    post_id TEXT,  
	post_title TEXT,
    subreddit TEXT ,
    author TEXT ,
    comment_text TEXT ,
    score integer,
    created_utc timestamp,
    fetched_at timestamp,
)

CREATE TABLE raw_data.youtube_comments (
    comment_id SERIAL PRIMARY KEY, 
    video_id TEXT,
    title TEXT,
    channel TEXT,
    text TEXT NOT NULL,
    author TEXT,
    likes INTEGER,
    published_at TIMESTAMP,
    collected_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE raw_data.reddit_comments (
    comment_id TEXT PRIMARY KEY,  
    subreddit TEXT,
    author TEXT,
    comment_text TEXT,
    created_utc TIMESTAMP,
    fetched_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw_data.economic_indicators (
    indicator     TEXT,        
    series_id     TEXT,        
    value         FLOAT,
    date          DATE,
    fetched_at    TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (series_id, date) 
);

CREATE TABLE IF NOT EXISTS raw_data.user_profiles (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    age TEXT,
    experience TEXT,
    interest TEXT[],       
    sources TEXT,     
    risk TEXT,
    language TEXT,
    email TEXT UNIQUE         
);

-- processed_data Schema Tables 
CREATE TABLE processed_data.youtube_comments (
    processed_id SERIAL PRIMARY KEY,
    comment_id TEXT REFERENCES raw_data.youtube_comments(comment_id),
    sentiment sentiment_type,
    topic_tags TEXT[], 
    keywords TEXT[],    
    processed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE processed_data.reddit_comments (
    processed_id SERIAL PRIMARY KEY,
    comment_id TEXT REFERENCES raw_data.reddit_comments(comment_id),
    sentiment sentiment_type,
    topic_tags TEXT[]
    keywords TEXT[]
    processed_at TIMESTAMP 
);

CREATE TABLE processed_data.reddit_comments_sp500 (
    symbol TEXT NOT NULL,
    post_id TEXT NOT NULL,
    post_title TEXT,
    comment_id TEXT PRIMARY KEY, 
    subreddit TEXT,
    author TEXT,
    comment_text TEXT,
    score INTEGER,
    sentiment sentiment_type,
    created_utc TIMESTAMP,       
    fetched_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS processed_data.reddit_topic (
    topic_date DATE,
    topic_tags TEXT[],
    keywords TEXT[],
    topic_summary TEXT,
    comments_count INTEGER,
    neg_count INTEGER,
    pos_count INTEGER,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS processed_data.youtube_topic (
    topic_date DATE,
    topic_tags TEXT[],
    keywords TEXT[],
    topic_summary TEXT,
    comments_count INTEGER,
    neg_count INTEGER,
    pos_count INTEGER,
    created_at TIMESTAMP
);

--production 

CREATE MATERIALIZED VIEW processed_data.mv_sentiment_by_date AS
SELECT 
    processed_at::date AS published_at,
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) AS positive_cm,
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS negative_cm
FROM 
    processed_data.reddit_comments
GROUP BY 
    processed_at::date
ORDER BY 
    published_at;

-- rag_docs Schema Tables

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;



-- 新聞原文
CREATE TABLE IF NOT EXISTS rag_docs.news_articles (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    query_keyword TEXT,
    url TEXT UNIQUE,  -- 建議這裡也順便加上唯一鍵
    published_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 新聞向量表（MiniLM/BGE 型模型用，384 維）
CREATE TABLE IF NOT EXISTS rag_docs.news_chunks (
    id SERIAL PRIMARY KEY,
    article_id INT REFERENCES rag_docs.news_articles(id) ON DELETE CASCADE,
    chunk_index INTEGER,
    chunk_text TEXT,
    embedding VECTOR(384),
    model_name TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_article_chunk UNIQUE(article_id, chunk_index)
);
CREATE INDEX ON rag_docs.news_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Recommended Indexes 
CREATE INDEX idx_youtube_video_id ON raw_data.youtube_comments(video_id);
CREATE INDEX idx_reddit_subreddit ON raw_data.reddit_comments(subreddit);
CREATE INDEX idx_chunks_document_id ON rag_docs.chunks(document_id);
CREATE INDEX idx_chunks_embedding ON rag_docs.chunks USING ivfflat (embedding vector_cosine_ops);
