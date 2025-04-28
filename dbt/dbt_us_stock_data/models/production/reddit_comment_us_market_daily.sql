{{ config(
    materialized='view'
) }}

SELECT
    DATE(created_utc) AS topic_date,
    COUNT(*) AS comments_count,
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS neg_count,
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) AS pos_count
FROM
    {{ ref('reddit_comments_us_market') }}
WHERE
    comment_id IS NOT NULL 
GROUP BY
    DATE(created_utc)
HAVING
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) > 0
ORDER BY
    topic_date