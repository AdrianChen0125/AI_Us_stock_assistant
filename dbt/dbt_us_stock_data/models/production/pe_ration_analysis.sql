{{ config(
    materialized='table'
) }}

WITH latest_snapshot AS (

    SELECT MAX(snapshot_date) AS max_snapshot_date
    FROM {{ ref('processed_sp500_snapshots') }}
),

clean_data AS (

    SELECT
        sector,
        symbol,
        company,
        pe_ratio
    FROM {{ ref('processed_sp500_snapshots') }}, latest_snapshot
    WHERE
        sector IS NOT NULL
        AND pe_ratio IS NOT NULL
        AND pe_ratio > 0
        AND snapshot_date = max_snapshot_date
),

ranked_max_pe AS (
    SELECT
        sector,
        symbol AS max_pe_symbol,
        company AS max_pe_company,
        pe_ratio AS max_pe_ratio,
        ROW_NUMBER() OVER (PARTITION BY sector ORDER BY pe_ratio DESC) AS rnk_max
    FROM clean_data
),

ranked_min_pe AS (
    SELECT
        sector,
        symbol AS min_pe_symbol,
        company AS min_pe_company,
        pe_ratio AS min_pe_ratio,
        ROW_NUMBER() OVER (PARTITION BY sector ORDER BY pe_ratio ASC) AS rnk_min
    FROM clean_data
)

SELECT
    d.sector,
    ROUND(AVG(d.pe_ratio)::numeric, 2) AS avg_pe_ratio,
    COUNT(*) AS company_count,
    max_pe.max_pe_company,
    max_pe.max_pe_ratio,
    min_pe.min_pe_company,
    min_pe.min_pe_ratio
FROM clean_data d
LEFT JOIN ranked_max_pe max_pe
    ON d.sector = max_pe.sector AND max_pe.rnk_max = 1
LEFT JOIN ranked_min_pe min_pe
    ON d.sector = min_pe.sector AND min_pe.rnk_min = 1
GROUP BY
    d.sector,
    max_pe.max_pe_company,
    max_pe.max_pe_ratio,
    min_pe.min_pe_company,
    min_pe.min_pe_ratio
ORDER BY avg_pe_ratio DESC