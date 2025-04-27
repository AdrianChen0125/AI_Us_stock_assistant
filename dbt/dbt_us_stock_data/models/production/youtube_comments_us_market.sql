{{ config(
    materialized='view'
) }}

WITH processed_comments AS (

    SELECT
        comment_id,
        sentiment,
        keywords
    FROM {{ source('processed_data', 'youtube_comments') }}
    WHERE 
        comment_id IS NOT NULL

),

raw_comments AS (

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
    FROM {{ ref('raw_youtube_comments') }}
    WHERE
        TRIM(text) IS NOT NULL

)

SELECT
    r.published_at,
    r.channel,
    r.video_id,
    r.title,
    r.comment_id,
    p.sentiment,
    p.keywords,
    r.likes
FROM raw_comments r
LEFT JOIN processed_comments p
    ON r.comment_id = p.comment_id
WHERE r.comment_id IS NOT NULL
ORDER BY r.published_at