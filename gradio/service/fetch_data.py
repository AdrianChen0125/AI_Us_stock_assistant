import time
import os
import psycopg2
import pandas as pd
from datetime import date
import requests

DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "port": os.environ.get("DB_PORT"),
    "dbname": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD")}

API_BASE = "http://fastapi:8000"

# ----- Market data ------


def fetch_economic_index_summary(index_name: str = None, days: int = 180):
    try:
        params = {"days": days}
        if index_name:
            params["index_name"] = index_name

        response = requests.get(f"{API_BASE}/economic_index", params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            return pd.DataFrame(), "No data found"

        df = pd.DataFrame(data)
        summary = "\n".join([f"{row['date']} | {row['index_name']} | {row['value']}" for row in data])
        return df, summary

    except Exception as e:
        print("API ERROR:", e)
        return pd.DataFrame(), f"API error: {e}"
    
def fetch_market_price_summary(market):
    try:
        res = requests.get(
            f"{API_BASE}/market_price", 
            params={"market": market}
        )
        res.raise_for_status()
        data = res.json()

        if not data:
            return pd.DataFrame(), "No data"

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])

        df = df[["date", "market", "price"]]

        summary = "\n".join([
            f"{row['date'].date()} | {row['market']} | Price: {row['price']}"
            for _, row in df.iterrows()
        ])

        return df, summary

    except Exception as e:
        print("API error:", e)
        return pd.DataFrame(), f"API error: {e}"
        return pd.DataFrame(), f"API error: {e}"

def fetch_market_price_last_7_days():
    conn, cursor = None, None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
            SELECT 
                market,
                snapshot_time,
                price,
                ma_3_days,
                ma_5_days,
                ma_7_days
            FROM dbt_us_stock_data_production.market_price
            WHERE snapshot_time >= (SELECT MAX(snapshot_time) FROM dbt_us_stock_data_production.market_price) - INTERVAL '7 days'
            ORDER BY snapshot_time DESC;
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            return pd.DataFrame(), "No data found."

        df = pd.DataFrame(rows, columns=[
            "market", "snapshot_time", "price", "ma_3_days", "ma_5_days", "ma_7_days"
        ])

        # Summary for ai 
        summary = "\n".join([
            f"{row[1]} | {row[0]} | price: {row[2]} | MA3: {row[3]} | MA5: {row[4]} | MA7: {row[5]}"
            for row in rows
        ])

        return df, summary

    except Exception as e:
        return pd.DataFrame(), f"Database error: {e}"

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_index_list():
                try:
                    res = requests.get(f"{API_BASE}/economic_index/list")
                    return res.json() if res.status_code == 200 else []
                except Exception as e:
                    print("Error loading index list:", e)
                    return []
def get_market_list():
    try:
        res = requests.get(f"{API_BASE}/market_price/list")
        return res.json() if res.status_code == 200 else []
    except Exception as e:
        print("Market list error:", e)
        return []
# ----- Sentiment data ------

def fetch_sentiment_data():
    try:
        res = requests.get(f"{API_BASE}/sentiment/reddit_summary/compare")
        res.raise_for_status()
        data = res.json()

        df = pd.DataFrame([
            {
                "label": "This Week",
                "date": data["recent_7d"]["date"],
                "total": data["recent_7d"]["total"],
                "positive": data["recent_7d"]["positive"],
                "negative": data["recent_7d"]["negative"]
            },
            {
                "label": "Last Week",
                "date": data["prev_7d"]["date"],
                "total": data["prev_7d"]["total"],
                "positive": data["prev_7d"]["positive"],
                "negative": data["prev_7d"]["negative"]
            }
        ])
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"❌ Failed to fetch: {e}"
    
def fetch_sentiment_topic_summary():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                topic_date,
                topic_summary,
                keywords,
                comments_count,
                pos_count,
                neg_count,
                source
            FROM dbt_us_stock_data_production."top_5_Topic_with_sentiment"
            WHERE topic_date = (
                SELECT MAX(topic_date) FROM dbt_us_stock_data_production."top_5_Topic_with_sentiment"
            )
            ORDER BY comments_count DESC, topic_date DESC
            LIMIT 10;
        """)

        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame(), "No data found"
        df = pd.DataFrame(rows, columns=["date", "title", "keywords", "comment_count", "positive", "negative","source"])
        summary = "\n".join([f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]}" for r in rows])
        return df, summary
    
    except Exception as e:
        return pd.DataFrame(), f"Database error: {e}"
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

def fetch_overall_sentiment_summary():
    conn, cursor = None, None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
            topic_date, 
            pos_count,
            neg_count
            FROM dbt_us_stock_data_production.reddit_comment_us_market_daily
        """)
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=["published_at", "total_pc", "total_nc"])
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

def fetch_top10_symbols_df():
            df, _ = fetch_top10_symbols_this_week()
            return df
            
# ----- Stock Data -----
def fetch_top10_symbols_this_week():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
            WITH this_week AS (
                SELECT 
                    symbol,
                    SUM(comments_count) AS total_comments,
                    SUM(pos_count) AS total_pos,
                    SUM(neg_count) AS total_neg
                FROM dbt_us_stock_data_production.sp500_sentiment_reddit
                WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM dbt_us_stock_data_production.sp500_sentiment_reddit)
                  AND symbol IS NOT NULL
                GROUP BY symbol
            )
            SELECT 
                symbol,
                total_comments,
                total_pos,
                total_neg
            FROM this_week
            ORDER BY total_comments DESC
            LIMIT 10
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            return pd.DataFrame(), "No data found"

        df = pd.DataFrame(rows, columns=["🔥 Symbol", "💬 Comments", "👍 Positive", "👎 Negative"])
        summary = "\n".join([f"{r[0]} | {r[1]} | {r[2]} | {r[3]}" for r in rows])
        return df,summary

    except Exception as e:
        return pd.DataFrame(), f"Database error: {e}"

    finally:
        if cursor: cursor.close()
        if conn: conn.close()

def fetch_top5_sectors_this_week():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
            WITH this_week AS (
                SELECT 
                    sector,
                    SUM(comments_count) AS total_comments,
                    SUM(pos_count) AS total_pos,
                    SUM(neg_count) AS total_neg
                FROM dbt_us_stock_data_production.sp500_sentiment_reddit
                WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM dbt_us_stock_data_production.sp500_sentiment_reddit)
                GROUP BY sector
            )
            SELECT 
                sector,
                total_comments,
                total_pos,
                total_neg
            FROM this_week
            ORDER BY total_comments DESC
            LIMIT 5
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            return pd.DataFrame(), "No this week data"

        df = pd.DataFrame(rows, columns=["sector", "total_comments", "total_pos", "total_neg"])

        summary = "\n".join([f"{r[0]} | {r[1]} | {r[2]} | {r[3]}" for r in rows])
        return df, summary

    except Exception as e:
        return pd.DataFrame(), f"Database error: {e}"

    finally:
        if cursor: cursor.close()
        if conn: conn.close()