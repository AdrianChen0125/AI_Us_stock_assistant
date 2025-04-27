{{ config(materialized='view') }}

SELECT
    symbol,
    post_id,
    post_title,
    comment_id,
    subreddit,
    author,
    comment_text,
    score,
    sentiment,
    created_utc,
    fetched_at
FROM
    {{ source('processed_data', 'reddit_comments_sp500') }}