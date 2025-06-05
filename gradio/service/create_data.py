import psycopg2
from config import DB_CONFIG

def save_to_db(risk,interest,holdings,language, email):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()


        interest_array = "{" + ",".join(interest) + "}"

        upsert_query = """
        INSERT INTO raw_data.user_profiles (risk, interest, holdings,language, email)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (email)
        DO UPDATE SET
        risk = EXCLUDED.risk,
        interest = EXCLUDED.interest,
        holdings = EXCLUDED.holdings,
        language = EXCLUDED.language
        """

        cur.execute(upsert_query, (risk, interest_array, holdings, language, email
        ))

        conn.commit()
        cur.close()
        conn.close()

        user_profile = {
            "risk": risk,
            "interest": interest_array,
            "holdings": holdings,
            "language": language,
            "email": email
        }
        return " Your response has been saved!", user_profile
    except Exception as e:
        return f" Database error: {e}"
