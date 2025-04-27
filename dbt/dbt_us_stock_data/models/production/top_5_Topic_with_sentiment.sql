{{ config(
    materialized='view'
) }}

SELECT
    topic_date,
    topic_tags,
    keywords,
    topic_summary,
    comments_count,
    neg_count,
    pos_count,
    created_at,
    'youtube' AS source
FROM {{ source('processed_data', 'youtube_topic') }}

UNION ALL

SELECT
    topic_date,
    topic_tags,
    keywords,
    topic_summary,
    comments_count,
    neg_count,
    pos_count,
    created_at,
    'reddit' AS source
FROM {{ source('processed_data', 'reddit_topic') }}