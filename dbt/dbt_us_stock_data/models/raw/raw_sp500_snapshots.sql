{{ config(materialized='view') }}

SELECT
    id,
    symbol,
    company,
    sector,
    sub_industry,
    market_cap,
    volume,
    previous_close,
    open,
    day_high,
    day_low,
    pe_ratio,
    forward_pe,
    dividend_yield,
    beta,
    high_52w,
    low_52w,
    snapshot_date
FROM
    {{ source('raw_data', 'sp500_snapshots') }}