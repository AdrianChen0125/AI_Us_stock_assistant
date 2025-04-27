{{ config(
    materialized='table'
) }}

WITH base AS (
    SELECT
        market,
        snapshot_time,
        ROUND(price::numeric, 2) AS price
    FROM {{ ref('processed_market_snapshots') }}
)

SELECT
    market,
    snapshot_time,
    price,
    
    ROUND(AVG(price) OVER (
        PARTITION BY market
        ORDER BY snapshot_time
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS ma_3_days, 

    ROUND(AVG(price) OVER (
        PARTITION BY market
        ORDER BY snapshot_time
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ), 2) AS ma_5_days, 

    ROUND(AVG(price) OVER (
        PARTITION BY market
        ORDER BY snapshot_time
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2) AS ma_7_days 

FROM base
ORDER BY
    market, snapshot_time