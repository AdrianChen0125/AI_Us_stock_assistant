{{ config(
    materialized='table'
) }}

WITH processed_comments AS (

    SELECT
        comment_id,
        sentiment,
        keywords
    FROM {{ source('processed_data', 'reddit_comments') }}
    WHERE comment_id IS NOT NULL

),

raw_comments AS (

    SELECT
        comment_id,
        LOWER(TRIM(subreddit)) AS subreddit,
        TRIM(comment_text) AS comment_text,
        created_utc
    FROM {{ ref('raw_reddit_comments') }}
    WHERE 
        TRIM(comment_text) IS NOT NULL
)

SELECT
    r.created_utc,
    r.subreddit,
    r.comment_id,
    r.comment_text,
    p.sentiment,
    p.keywords
FROM raw_comments r
LEFT JOIN processed_comments p
    ON r.comment_id = p.comment_id
WHERE r.comment_id IS NOT NULL
ORDER BY r.created_utc