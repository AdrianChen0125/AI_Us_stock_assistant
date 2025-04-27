{{ config(materialized='view') }}

SELECT
    indicator,
    series_id,
    value,
    date,
    fetched_at
FROM
    {{ source('raw_data', 'economic_indicators') }}