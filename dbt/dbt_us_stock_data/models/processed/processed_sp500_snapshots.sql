{{ config(materialized='view') }}

WITH source_data AS (

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
        {{ ref('raw_sp500_snapshots') }}
    WHERE 
        TRIM(symbol) IS NOT NULL      
        AND snapshot_date IS NOT NULL 
)

SELECT
    id,
    UPPER(TRIM(symbol)) AS symbol,       
    TRIM(company) AS company,
    TRIM(sector) AS sector,
    TRIM(sub_industry) AS sub_industry,
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
    source_data