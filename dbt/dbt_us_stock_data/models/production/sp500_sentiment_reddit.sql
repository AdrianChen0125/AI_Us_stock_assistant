{{ config(
    materialized='table'
) }}

WITH comments_aggregated AS (
    SELECT
        symbol,
        DATE(created_utc) AS snapshot_date,
        ROUND(AVG(
            CASE 
                WHEN sentiment = 'positive' THEN 1
                WHEN sentiment = 'negative' THEN -1
                ELSE 0
            END
        ), 4) AS avg_sentiment_score,
        COUNT(*) AS comments_count,
        SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) AS pos_count,
        SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS neg_count
    FROM {{ source('processed_data', 'reddit_comments_sp500') }}
    WHERE 
        symbol IS NOT NULL
        AND created_utc IS NOT NULL
    GROUP BY 
        symbol,
        DATE(created_utc)
),

symbols_with_sector AS (
    SELECT
        symbol,
        sector
    FROM {{ ref('processed_sp500_snapshots') }}
    GROUP BY symbol, sector
)

SELECT
    c.symbol,
    s.sector,
    c.snapshot_date,
    c.avg_sentiment_score,
    c.comments_count,
    c.pos_count,
    c.neg_count
FROM comments_aggregated c
LEFT JOIN symbols_with_sector s
ON c.symbol = s.symbol
ORDER BY 
    c.symbol,
    c.snapshot_date