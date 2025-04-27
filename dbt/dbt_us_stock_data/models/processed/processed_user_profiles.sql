{{ config(materialized='view') }}

WITH source_data AS (

    SELECT
        id,
        created_at,
        NULLIF(TRIM(age), '') AS age,                   
        NULLIF(TRIM(experience), '') AS experience,      
        interest,
        NULLIF(TRIM(sources), '') AS sources,            
        NULLIF(TRIM(risk), '') AS risk,                  
        NULLIF(TRIM(language), '') AS language,          
        LOWER(NULLIF(TRIM(email), '')) AS email          
    FROM 
        {{ ref('raw_user_profiles') }}

)

SELECT
    *
FROM 
    source_data