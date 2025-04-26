{{ config(materialized='view') }}

SELECT
    id,
    market,
    snapshot_time,
    price
FROM
    raw_data.market_snapshots