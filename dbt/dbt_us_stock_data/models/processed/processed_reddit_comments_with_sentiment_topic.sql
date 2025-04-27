{{ config(materialized='view') }}

SELECT
    processed_id,
    comment_id,
    sentiment,
    topic_tags,
    keywords,
    processed_at
FROM
   {{ source('processed_data', 'reddit_comments') }}