{{ config(materialized='view') }}

WITH source_data AS (

    SELECT
        market,
        snapshot_time,
        price
    FROM 
        {{ ref('raw_market_snapshots') }}
    WHERE
        price IS NOT NULL
)

SELECT
    LOWER(TRIM(market)) AS market, 
    snapshot_time,
    ROUND(CAST(price AS numeric), 2) AS price
FROM
    source_data