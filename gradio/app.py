from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import smtplib
import os
import gradio as gr
import psycopg2
import pandas as pd
import plotly.graph_objects as go
from datetime import date
import requests
from PIL import Image
from auth import login
from chat_bot import call_chatbot_api
import requests


# ---------- Global Variable ----------
last_generated_report = ""  # Store the latest generated report for email sending
# ---------- Config ----------
DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "port": os.environ.get("DB_PORT"),
    "dbname": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD")}

API_BASE = "http://fastapi:8000"

# ---------- Email Sending ----------
def send_email_report(to_email, subject, report_content):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(report_content, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        return f"Report sent to {to_email}"
    except Exception as e:
        return f"Failed to send email: {e}"

# ---------- Data Fetching ----------

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


# ----------- Save back to DB---------------

def save_to_db(age, experience, interest, sources, risk,language, email):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()


        interest_array = "{" + ",".join(interest) + "}"

        upsert_query = """
        INSERT INTO raw_data.user_profiles (age, experience, interest, sources, risk,language, email)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (email)
        DO UPDATE SET
        age = EXCLUDED.age,
        experience = EXCLUDED.experience,
        interest = EXCLUDED.interest,
        sources = EXCLUDED.sources,
        risk = EXCLUDED.risk,
        language = EXCLUDED.language
        """

        cur.execute(upsert_query, (
            age, experience, interest_array, sources, risk, language, email
        ))

        conn.commit()
        cur.close()
        conn.close()

        
        user_profile = {
            "age": age,
            "experience": experience,
            "interest": interest,
            "sources": sources,
            "risk": risk,
            "language": language,
            "email": email
        }
        return "✅ Your response has been saved!", user_profile
    except Exception as e:
        return f"❌ Database error: {e}"

# ---------- plot ---------
def plot_sentiment_line_chart():
    df = fetch_overall_sentiment_summary()
    if df.empty or "error" in df.columns:
        return go.Figure().update_layout(title="❌ Failed to load data")
    
    df["total_pc"] = pd.to_numeric(df["total_pc"], errors="coerce")
    df["total_nc"] = pd.to_numeric(df["total_nc"], errors="coerce")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["published_at"], y=df["total_pc"], mode="lines+markers", name="Positive"))
    fig.add_trace(go.Scatter(x=df["published_at"], y=df["total_nc"], mode="lines+markers", name="Negative"))
    fig.update_layout(title="📊 Reddit Daily Market Sentiment Trend ",height=515 ,xaxis_title="Date", yaxis_title="Comment Count")
    return fig

def plot_sentiment_pie(df):
    if df.empty:
        return go.Figure().update_layout(title="No data"), go.Figure()

    def make_pie_row(row):
        neutral = max(row["total"] - row["positive"] - row["negative"], 0)
        labels = ["Positive", "Neutral", "Negative"]
        values = [row["positive"], neutral, row["negative"]]
        title = f"{row['label']} ({row['date']})"
        fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=0.4)])
        fig.update_layout(title_text=title,height=250,margin=dict(t=10, b=10, l=10, r=10))
        return fig

    this_week_fig = make_pie_row(df.iloc[0])
    last_week_fig = make_pie_row(df.iloc[1])

    return this_week_fig, last_week_fig

def plot_index_chart(df: pd.DataFrame, title: str = ""):
    if df.empty:
        return go.Figure().update_layout(title="Failed to load data")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["value"],
        mode="lines+markers",
        name=title or "Index"
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45, dtick="M1"),
        yaxis_title="Value",
        xaxis_title="Date",
        margin=dict(l=20, r=20, t=30, b=30),
        height=350
    )
    return fig

def plot_price_chart(market):
    df, summary = fetch_market_price_summary(market)
    if df.empty:
        return go.Figure().update_layout(title="Failed to load data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price"],
        mode="lines+markers", name=market
    ))
    fig.update_layout(
        title=f"{market} - Price Trend (30 Days)",
        xaxis_title="Date",
        yaxis_title="Price",
        height=350
    )
    return fig

def get_sentiment_table():
    df, _ = fetch_sentiment_topic_summary()
    last_time = df['date'].max().strftime("%Y-%m-%d %H:%M:%S")
    df1 = df[["source","positive", "negative","title"]]
    
    return df1, last_time 

def plot_sector_chart():
    df, _ = fetch_top5_sectors_this_week()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["sector"],
        y=df["total_comments"],
        name="Total Comments",
        text=df["total_comments"],
        textposition="auto"
    ))

    fig.update_layout(
        title="Top 5 Sectors by Discussions",
        xaxis_title="Sector",
        yaxis_title="Total Comments",
        height=400,
        margin=dict(l=20, r=20, t=50, b=30),
    )

    return fig
# ---------- image ----------

def show_image(filename):
    image_path = os.path.join("/app/assets", filename)
    return Image.open(image_path)

# ---------- stocke recommendation

def g4_recommend_multi(symbols_raw, email):
    # 處理輸入：切逗號、去空格、轉大寫
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    print(f"[INPUT] 使用者輸入股票：{symbols}")

    if not symbols:
        return "請至少輸入一個股票代碼", ""

    url = "http://fastapi:8000/recommend/"
    # 組合查詢參數：symbols 多值 + user_id
    params = [("symbols", s) for s in symbols]
    if email:
        params.append(("user_id", email))  # ✅ 傳入 email 當作 user_id
    else:
        params.append(("user_id", "anonymous"))

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        print("[API 回傳資料]", data)

        if data.get("status") == "ok":
            recs = data.get("recommendations", [])
            return "", "\n".join(recs) if recs else "沒有推薦結果"
        else:
            return "推薦失敗，API 回傳非 ok 狀態", ""
    except Exception as e:
        return f"錯誤：{str(e)}", ""

def fetch_industries():
    try:
        resp = requests.get(f"{API_BASE}/stock_data/sector_list")
        return [""] + [i for i in resp.json() if i]
    except Exception:
        return []

def fetch_max_date():
    try:
        resp = requests.get(f"{API_BASE}/stock_data/latest_date")
        return pd.to_datetime(resp.json()).date()
    except Exception:
        return date.today()
    
def query_stock_data(symbol, sector, start_date, end_date):
    try:
        params = []

        if symbol:
            symbol_list = [s.strip() for s in symbol.split(",") if s.strip()]
            for s in symbol_list:
                params.append(("symbol", s))

        if sector:
            params.append(("sector", sector))
        if start_date:
            params.append(("start_date", start_date))
        if end_date:
            params.append(("end_date", end_date))

        
        resp = requests.get(f"{API_BASE}/stock_data", params=params)

        if resp.status_code != 200:
            return pd.DataFrame([{"錯誤": f"HTTP {resp.status_code}", "訊息": resp.text}])

        try:
            data = resp.json()
        except Exception as e:
            return pd.DataFrame([{"錯誤": "JSON decode error", "訊息": str(e)}])

        if not data:
            return pd.DataFrame([{"訊息": "查無資料"}])

        return pd.DataFrame(data)

    except Exception as e:
        return pd.DataFrame([{"錯誤": "查詢失敗", "訊息": str(e)}])

# ---------- Report Generation ----------

def get_economic_report(language='English', token=None):
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        payload = {"language": language}

        res = requests.post("http://fastapi:8000/AI/economic_report", json=payload, headers=headers)
        res.raise_for_status()

        data = res.json()
        report = data.get("report", "")

        
        clean_report = report.strip()

        return clean_report, report  # 一個顯示用、一個原始 markdown 備用

    except Exception as e:
        print("❌ Error fetching report:", e)
        return "⚠️ Failed to fetch report.", None

def get_market_sentiment_report(token):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        res = requests.post(f"{API_BASE}/AI/sentiment_report", headers=headers)
        res.raise_for_status()
        report = res.json().get("report", "No content")
        return report, report  
    except Exception as e:
        return f"Error: {e}", None
   
def generate_overall_report(economic_report,sentiment_report,recommedation_report,user_profile, token):
    payload = {
        "language": user_profile.get("language", "English"),
        "economic_summary": economic_report,
        "sentiment_summary": sentiment_report,
        "stock_summary": recommedation_report,
        "age": user_profile.get("age", "18-25"),
        "experience": user_profile.get("experience", "Beginner"),
        "risk": user_profile.get("risk", "Moderate"),
    }

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        print("Sending payload:", payload)
        res = requests.post(
            "http://fastapi:8000/AI/summerise_report",
            json=payload,
            headers=headers
        )
        res.raise_for_status()
        return res.json().get("report", "No report generated.")
    except Exception as e:
        return f"❌ Error generating report: {e}"
    

# ----------send email-----------

def send_latest_report(email):
    return send_email_report(email, "Your Personalized Investment Report", last_generated_report)


# ---------- dropdown 
def get_index_list():
                try:
                    res = requests.get("http://fastapi:8000/economic_index/list")
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
# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown("## 📊 Adrian's US Stock Weekly Report")
    user_state = gr.State() 

    with gr.Tabs():

        # --- Tab 1: Questionnaire ---
        with gr.Tab(" Main page (首頁) "):
            
            with gr.Row():    
                with gr.Column(scale=1):
                    image_output = gr.Image(show_image("Intro_page.png"),width=900,height=850,type="pil")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 🔐 Login First")
                    
                    username = gr.Textbox(label="Username")
                    password = gr.Textbox(label="Password", type="password")
                    login_btn = gr.Button("Login")
                    login_status = gr.Markdown()
                    access_token = gr.Textbox(visible=False)
                    
                    with gr.Column(visible=False) as form_section:
                        user_profile_state = gr.State()
                        age = gr.Dropdown(["18-25", "26-35", "36-45", "46-60", "60+"], label="Your Age Group")
                        experience = gr.Radio(["Beginner", "Intermediate", "Advanced"], value="Intermediate", label="Investment Experience")
                        interest = gr.CheckboxGroup(["Tech Stocks", "ETF", "High Dividend", "US Bonds", "Crypto"],value="High Dividend",label="Investment Preferences")
                        sources = gr.Textbox(
                            label="Stock You Have",
                            placeholder="e.g., Nvidia, Tesla..."
                        )
                        risk = gr.Radio(["Conservative", "Moderate", "Aggressive"], value="Moderate", label="Risk Tolerance")
                        language = gr.Radio(["English", "chinese"], value="English", label="Language")
                        email = gr.Textbox(label="Your Email (for report delivery)", placeholder="example@email.com")
                        submit_btn = gr.Button("Submit")
                        output = gr.Textbox(label="Submission Status", interactive=False)

                    
                    login_btn.click(
                        fn=login,
                        inputs=[username, password],
                        outputs=[login_status, login_btn, form_section, access_token]
                    )


                    submit_btn.click(
                        fn=save_to_db,
                        inputs=[age, experience, interest, sources, risk, language, email],
                        outputs=[output,user_profile_state]
                    )


        with gr.Tab(" Economic & Market Trends（經濟與市場趨勢）"):
            with gr.Row():
                 with gr.Column(scale=1):
                    gr.Markdown("## Economic Indicator 經濟指數")
                    
                    with gr.Row():
                        print("Index list:", get_index_list())
                        index_dropdown = gr.Dropdown(label="Select Index", choices= get_index_list(), value=None, scale=2)
                        days_input = gr.Number(label="Days Range", value=180, precision=0, scale=1)

                    chart_output = gr.Plot()

                    # 移除 token 參數
                    def update_single_chart(index_name, days):
                        if not index_name:
                            return go.Figure().update_layout(title="Please select an index.")
                        df, _ = fetch_economic_index_summary(index_name=index_name, days=int(days))
                        return plot_index_chart(df, title=f"{index_name} Trend")

                
                    index_dropdown.change(
                        fn=update_single_chart,
                        inputs=[index_dropdown, days_input],
                        outputs=chart_output,
                        queue=True,
                    )

                    days_input.change(
                        fn=update_single_chart,
                        inputs=[index_dropdown, days_input],
                        outputs=chart_output,
                        queue=True,
)

                 with gr.Column(scale=1):
                    gr.Markdown("## Market Price 市場走勢")

                    market_dropdown = gr.Dropdown(label="Select Market", choices = get_market_list(), value=None)
                    market_chart_output = gr.Plot()
                    
                    def update_market_chart(market):
                        if not market:
                            return go.Figure().update_layout(title="Please select a market.")
                        df, _ = fetch_market_price_summary(market)
                        return plot_price_chart(df, title=f"{market} Price Trend")


                    market_dropdown.change(fn=plot_price_chart, inputs=[market_dropdown], outputs=[market_chart_output])
            
            gr.Markdown("###  AI Agent 趨勢分析小幫手 ")

            ai_generated_report = gr.TextArea(label="📄 AI Analysis Report", lines=25)
            economic_report_state = gr.State()
            generate_btn = gr.Button(" Generate Trend Report")

            
            generate_btn.click(
            fn=get_economic_report,
            inputs=[language,access_token],
            outputs=[ai_generated_report,economic_report_state]
        )
            
        with gr.Tab(" Market Sentiments (市場情緒) "):
            with gr.Row():
                with gr.Column(scale=1): 
                    gr.Markdown("###  Daily Sentiment about Us Stock 美股市場情緒(30日)") 
                    sentiment_chart = gr.Plot(label="Sentiment Line Chart")
                    chart_btn = gr.Button(" Refresh Trend")
                
                with gr.Column(scale=1):
                    gr.Markdown("###  Weekly Reddit Sentiment about Us Stock 美股市場情緒(每週)")  
                    pie1 = gr.Plot(label="This Week Sentiment")
                    pie2 = gr.Plot(label="Last Week Sentiment")
                    pie_btn = gr.Button("Update Weekly Comparison")

            def update_sentiment_pie():
                df, err = fetch_sentiment_data()
                if err:
                    return go.Figure().update_layout(title=err), go.Figure(), err
                fig1, fig2 = plot_sentiment_pie(df)
                return fig1, fig2

            gr.Markdown("###  Top 10 Topic about Us Stock 美股熱門討論話題") 

            sentiment_table = gr.Dataframe(label=" Sentiment Topic Summary",wrap=True)
            last_time_text = gr.Markdown()
            table_btn = gr.Button("🔄 Refresh")

            def get_top_10_topic():
                df,_ = get_sentiment_table()
                return df
                
            table_btn.click(
                fn = get_sentiment_table,
                outputs = sentiment_table
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(" ### Pooular Sector and Ticker (熱門產業) ")

                    chart_output = gr.Plot(label="Weekly Top 5 Sector")
                    last_time_text = gr.Markdown()
                    refresh_btn1 = gr.Button("🔄 Refresh")

                    refresh_btn1.click(
                    fn= plot_sector_chart,
                    outputs= chart_output
                )
                
                with gr.Column(scale=1):
                    gr.Markdown(" ### Top 10 Stock Discussions (熱門標地) ")    
                    symbol_table = gr.Dataframe(wrap = True,interactive=False)

                    refresh_btn2 = gr.Button("🔄 Refresh")

            def fetch_top10_symbols_df():
                df,_ = fetch_top10_symbols_this_week()
                return df
                
            refresh_btn2.click(
                fn= fetch_top10_symbols_df,
                outputs=symbol_table
                )


            gr.Markdown("###  AI Agent 市場情緒分析小幫手 ")

            ai_sentiment_report = gr.TextArea(label="📄 AI Analysis Report", lines=25)
            sentiment_report_state = gr.State() 
            sentiment_report_generate_btn = gr.Button(" Generate Trend Report")

            
            sentiment_report_generate_btn.click(
                fn=get_market_sentiment_report,
                inputs=[access_token],
                outputs=[ai_sentiment_report, sentiment_report_state]
            )

            
           
            
            chart_btn.click(
                plot_sentiment_line_chart,
                outputs = sentiment_chart
                )


            pie_btn.click(
                fn=update_sentiment_pie, 
                outputs=[pie1, pie2]
                )

            demo.load(fn=plot_sentiment_line_chart, outputs=sentiment_chart)
            demo.load(fn=get_sentiment_table, outputs=[sentiment_table,last_time_text])
            demo.load(fn=plot_sector_chart, outputs=chart_output)
            demo.load(fn=fetch_top10_symbols_df, outputs=symbol_table)
            demo.load(fn=update_sentiment_pie, outputs=[pie1, pie2])


        max_snapshot_date = fetch_max_date()
        industries = fetch_industries()
        with gr.Tab(" SP500 美股糾察隊"):
            
            
            gr.Markdown("### 股票推薦系統")

            with gr.Row():
                with gr.Column(scale=2):
                    stock_input = gr.Textbox(
                        label="輸入股票代碼（例如：GOOG, AAPL, MSFT）", lines=2, placeholder="GOOG, AAPL, MSFT"
                    )
                    submit_button = gr.Button("查詢推薦")
                with gr.Column(scale=3):
                    error_output = gr.Textbox(label="錯誤訊息", visible=False)
                    recommend_output = gr.Textbox(label="推薦股票", lines=6)

            submit_button.click(
                fn=g4_recommend_multi,
                inputs=[stock_input,email],
                outputs=[error_output, recommend_output]
            )
            

            gr.Markdown("### 🤖 AI 股票推薦分析")

            with gr.Row():
                holdings_input = gr.Textbox(label="目前持有股票 (逗號分隔)", placeholder="AAPL, TSLA")
                recommended_input = gr.Textbox(label="推薦股票 (逗號分隔)", placeholder="MSFT, GOOG")

            ai_output = gr.Textbox(label="AI 分析建議", lines=20)
            stock_recommendation_state = gr.State()
            analyze_button = gr.Button("分析推薦")

            def call_ai_analysis(holdings, recommended, user_profile, token):
                payload = {
                    "holdings": [s.strip() for s in holdings.split(",") if s.strip()],
                    "recommended": [s.strip() for s in recommended.split(",") if s.strip()],
                    "style_preference": user_profile.get("interest", []),
                    "risk_tolerance": user_profile.get("risk", "Moderate")
                }

                headers = {"Authorization": f"Bearer {token}"} if token else {}

                try:
                    res = requests.post("http://fastapi:8000/AI/stock_recommendation", json=payload, headers=headers)
                    res.raise_for_status()
                    result = res.json()["analysis"]
                    return "\n\n".join(result.split("\n\n")), result
                except Exception as e:
                    return f"❌ Error: {str(e)}", ""

            analyze_button.click(
                fn=call_ai_analysis,
                inputs=[sources, recommended_input, user_profile_state, access_token],
                outputs=[ai_output, stock_recommendation_state]
            )
            
            gr.Markdown("### 📊 S&P 500 Stock Data")

            with gr.Row():
                symbol_input = gr.Textbox(label="Symbol")
                sector_input = gr.Dropdown(label="sector", choices = industries,value="")
                start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", value=str(max_snapshot_date))
                end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", value=str(max_snapshot_date)) 

            output = gr.Dataframe(label="查詢結果", interactive=False)
            search_btn = gr.Button("查詢")

            search_btn.click(
                query_stock_data,
                inputs=[symbol_input, sector_input, start_date, end_date],
                outputs=output
            )

        with gr.Tab(" View Report"):
            with gr.Row():
                with gr.Column(scale=2): 
                    output_text = gr.TextArea(label=" Report Content", lines=20) 
                    send_status = gr.Textbox(label=" Email Status", interactive=False)

                    with gr.Row():
                        submit_btn = gr.Button(" Generate Report")
                        send_btn = gr.Button(" Send Report via Email")

                                    
                        submit_btn.click(
                            fn=generate_overall_report,
                            inputs=[
                                economic_report_state,
                                sentiment_report_state,
                                stock_recommendation_state,
                                user_profile_state,
                                access_token 
                            ],
                            outputs=output_text
                        )
                        
                        send_btn.click(
                        fn=send_latest_report,
                        inputs=email,
                        outputs=send_status
                        )
  
                with gr.Column(scale=1):
                    image_output = gr.Image(
                        value=show_image("fin_page.png"), 
                        width= 800,
                        height= 500,
                        type="pil"
                    )

        with gr.Tab("華爾街Small Talk "):
            chatbox = gr.Chatbot(label="AI Assistant", type="messages")
            msg_input = gr.Textbox(placeholder="輸入你的問題", label="問題")
            state = gr.State([])
            chat_btn = gr.Button("送出")

            chat_btn.click(
                fn=call_chatbot_api,
                inputs=[msg_input, state, access_token],
                outputs=[chatbox, state],
                scroll_to_output=True
                )
        

if __name__ == '__main__':
    print("Gradio version in use:", gr.__version__)
    demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    root_path="/gradio"  )
    
