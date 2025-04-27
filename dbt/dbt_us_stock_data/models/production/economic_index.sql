{{ config(
    materialized='table'
) }}

WITH base AS (
    SELECT
        series_id,
        date_trunc('month', date) AS month_date,
        ROUND(value::numeric, 2) AS value
    FROM {{ ref('processed_economic_indicators') }}
),

ranked AS (
    SELECT
        *,
        LAG(value) OVER (PARTITION BY series_id ORDER BY month_date) AS prev_value
    FROM base
)

SELECT
    series_id,
    month_date,
    value AS current_month_value,
    prev_value AS previous_month_value,
    ROUND(value - prev_value, 2) AS change_value,
    CASE 
        WHEN prev_value IS NULL OR prev_value = 0 THEN NULL
        ELSE ROUND((value - prev_value) / prev_value * 100, 2)
    END AS change_percent
FROM ranked
ORDER BY
    series_id,
    month_date