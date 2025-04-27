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
    created_at
FROM 
    {{ source('processed_data', 'youtube_topic') }}