{{ config(materialized='view') }}

SELECT
    comment_id,
    subreddit,
    author,
    comment_text,
    created_utc,
    fetched_at
FROM
    raw_data.reddit_comments