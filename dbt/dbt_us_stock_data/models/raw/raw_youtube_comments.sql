{{ config(materialized='view') }}

SELECT
    comment_id,
    video_id,
    title,
    channel,
    text,
    author,
    likes,
    published_at,
    collected_at
FROM
    {{ source('raw_data', 'youtube_comments') }}