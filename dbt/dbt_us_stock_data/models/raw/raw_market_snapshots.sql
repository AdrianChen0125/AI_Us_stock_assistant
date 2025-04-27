{{ config(materialized='view') }}

SELECT
    id,
    market,
    snapshot_time,
    price
FROM
    {{ source('raw_data', 'market_snapshots') }}