{{ config(materialized='view') }}

WITH source_data AS (

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
        {{ ref('raw_youtube_comments') }}
    WHERE
        TRIM(text) IS NOT NULL
)

SELECT
    comment_id,
    video_id,
    title,
    channel,
    TRIM(text) AS text,
    author,
    COALESCE(likes, 0) AS likes,        
    published_at,
    collected_at
FROM
    source_data