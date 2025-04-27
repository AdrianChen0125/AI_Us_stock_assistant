{{ config(materialized='view') }}

WITH source_data AS (

    SELECT
        indicator,
        series_id,
        value,
        date,
        fetched_at
    FROM 
        {{ ref('raw_economic_indicators') }}
    WHERE
        value IS NOT NULL                  
        AND TRIM(series_id) IS NOT NULL     
        AND TRIM(indicator) IS NOT NULL     
)

SELECT
    TRIM(indicator) AS indicator,
    TRIM(series_id) AS series_id,
    ROUND(CAST(value AS numeric), 2) AS value,
    date
FROM
    source_data