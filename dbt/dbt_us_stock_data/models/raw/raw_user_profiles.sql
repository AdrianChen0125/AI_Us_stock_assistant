{{ config(materialized='view') }}

SELECT
    id,
    created_at,
    age,
    experience,
    interest,
    sources,
    risk,
    language,
    email
FROM
    {{ source('raw_data', 'user_profiles') }}