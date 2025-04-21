

-- Create 
CREATE TABLE videos (
    video_id TEXT PRIMARY KEY,
    title TEXT,
    channel TEXT,
    published_at TIMESTAMP,
    topic_id INTEGER REFERENCES topics(topic_id)
);

CREATE TABLE raw_comments (
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

CREATE TABLE processed_comments (
    processed_id SERIAL PRIMARY KEY,
    comment_id INTEGER REFERENCES raw_comments(comment_id),
    sentiment TEXT CHECK (sentiment IN ('positive', 'neutral', 'negative')),
    topic_tags TEXT[],  -- 例如: ['科技股', '川普']
    keywords TEXT[],     -- 自動關鍵字
    processed_at TIMESTAMP DEFAULT NOW()
);
