{{ config(materialized='view') }}

WITH source_data AS (

    SELECT
        comment_id,
        symbol,
        company,
        post_id,
        post_title,
        subreddit,
        author,
        comment_text,
        score,
        created_utc,
        fetched_at
    FROM 
        {{ ref('raw_reddit_comments_sp500') }}
    WHERE 
        TRIM(comment_text) IS NOT NULL
        AND TRIM(symbol) IS NOT NULL
)

SELECT
    comment_id,
    UPPER(TRIM(symbol)) AS symbol,    -- 股票代號轉大寫
    TRIM(company) AS company,
    post_id,
    post_title,
    subreddit,
    author,
    comment_text,
    score,
    created_utc,
    fetched_at
FROM 
    source_data