{{ config(materialized='view') }}

WITH source_data AS (

    SELECT
        comment_id,
        subreddit,
        author,
        comment_text,
        created_utc,
        fetched_at
    FROM 
        {{ ref('raw_reddit_comments') }}
    WHERE 
        TRIM(comment_text) IS NOT NULL  
)

SELECT
    comment_id,
    LOWER(TRIM(subreddit)) AS subreddit,      
    TRIM(author) AS author,
    TRIM(comment_text) AS comment_text,
    created_utc
FROM 
    source_data