{{ config(materialized='view') }}

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
    raw_data.reddit_comments_sp500