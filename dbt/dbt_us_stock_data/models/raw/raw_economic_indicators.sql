{{ config(materialized='view') }}

SELECT
    indicator,
    series_id,
    value,
    date,
    fetched_at
FROM
    raw_data.economic_indicators