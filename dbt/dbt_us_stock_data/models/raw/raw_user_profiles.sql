{{ config(materialized='view') }}

SELECT
    id,
    created_at,
    risk,
    interest,
    holdings,
    language,
    email
FROM
    {{ source('raw_data', 'user_profiles') }}