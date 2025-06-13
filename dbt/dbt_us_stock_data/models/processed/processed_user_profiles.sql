{{ config(materialized='view') }}

WITH source_data AS (

    SELECT
        id,
        created_at,                      
        interest,
        NULLIF(TRIM(risk), '') AS risk, 
        NULLIF(TRIM(holdings), '') AS holdings,            
        NULLIF(TRIM(language), '') AS language,          
        LOWER(NULLIF(TRIM(email), '')) AS email          
    FROM 
        {{ ref('raw_user_profiles') }}

)

SELECT
    *
FROM 
    source_data