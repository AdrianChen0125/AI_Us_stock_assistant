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

CREATE TABLE sp500_snapshots (
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
    post_id TEXT PRIMARY KEY,  -- Reddit 原始 ID 如 "t1_xyz"
    subreddit TEXT,
    author TEXT,
    comment_text TEXT,
    created_utc TIMESTAMP,
    fetched_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw_data.economic_indicators (
    indicator     TEXT,        -- 指標名稱，如 CPI、GDP
    series_id     TEXT,        -- FRED series_id
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
    comment_id INTEGER REFERENCES raw_data.youtube_comments(comment_id),
    sentiment sentiment_type,
    topic_tags TEXT[], 
    keywords TEXT[],    
    processed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE processed_data.reddit_comments (
    processed_id SERIAL PRIMARY KEY,
    post_id TEXT REFERENCES raw_data.reddit_comments(post_id),
    sentiment sentiment_type,
    topic_tags TEXT[]
    keywords TEXT[]
    processed_at TIMESTAMP 
);

CREATE TABLE IF NOT EXISTS processed_data.reddit_topic (
    topic_date DATE,
    topic_tags TEXT[],
    keywords TEXT[],
    topic_summary TEXT,
    comments_count INTEGER,
    post_count INTEGER,
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

CREATE TABLE rag_docs.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT,  -- e.g. 'essay', 'report'
    title TEXT,
    full_text TEXT,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE rag_docs.chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES rag_docs.documents(id),
    chunk_index INTEGER,
    content TEXT,
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Recommended Indexes 
CREATE INDEX idx_youtube_video_id ON raw_data.youtube_comments(video_id);
CREATE INDEX idx_reddit_subreddit ON raw_data.reddit_comments(subreddit);
CREATE INDEX idx_chunks_document_id ON rag_docs.chunks(document_id);
CREATE INDEX idx_chunks_embedding ON rag_docs.chunks USING ivfflat (embedding vector_cosine_ops);
