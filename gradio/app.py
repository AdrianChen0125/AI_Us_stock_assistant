from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import smtplib
import os
import gradio as gr
import psycopg2
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from datetime import datetime, timedelta
import requests
from PIL import Image
from io import BytesIO

from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# ---------- Global Variable ----------
last_generated_report = ""  # Store the latest generated report for email sending
# ---------- Config ----------
DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "port": os.environ.get("DB_PORT"),
    "dbname": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD")}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# langchain
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    streaming=True, 
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0.7)

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
API_BASE = "http://fastapi:8000"
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
        return df

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
        return df, None

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

def search_news_for_keywords(keywords, max_articles=3):
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    query = " OR ".join(keywords[:5])
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&from={from_date}&pageSize={max_articles}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        articles = response.json().get("articles", [])
        if not articles:
            return "No news found."
        return "\n\n".join([f"【{a['title']}】\n{a['description']}\nLink: {a['url']}" for a in articles])
    except Exception as e:
        return f"News search error: {e}"

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
    df, error = fetch_top5_sectors_this_week()
    if error or df.empty:
        fig = go.Figure()
        fig.update_layout(title="❌ Failed to load data", height=300)
        return fig

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

# ---------- Report Generation ----------
def generate_personal_report(age, experience, interest, sources, risk, langauge, email):
    
    profile_info = f"""
    User Profile:
    - Age: {age}
    - Experience: {experience}
    - Preferences: {", ".join(interest)}
    - Sources: {sources}
    - Risk Tolerance: {risk}
    """

    df, sentiment_summary = fetch_sentiment_topic_summary()
    df['news'] = df["keywords"].apply(search_news_for_keywords)
    index_df, eco_index = fetch_economic_index_summary()
    interested_news = search_news_for_keywords(sources)
    _, market_summary = fetch_market_price_last_7_days()
    top5_sector_df = fetch_top5_sectors_this_week()

    if langauge == 'chinese':
        prompt = f"""
        你是一位專業的投資顧問。以下是客戶資訊，請用**繁體中文**撰寫一份清晰、專業的投資分析與建議。
        【客戶資料】
        {profile_info}
        ---
        一、【經濟分析】
        （a）分析過去六個月的經濟指標，找出重要轉折點並簡述宏觀趨勢：
        {eco_index}
        （b）最近一週市場價格總結：
        {market_summary}
        ---
        二、【市場情緒與新聞】
        （a）根據情緒摘要與新聞，判斷市場情緒（多頭/空頭），並指出受影響的產業或個股：
        {sentiment_summary}
        （b）分析 Reddit 討論最熱前五大產業及原因。
        {top5_sector_df}
        新聞摘要：
        {df['news']}
        ---
        三、【使用者關注主題】
        整理使用者關注標的的新聞與分析：
        {interested_news}
        ---
        請以以下格式撰寫並根據 用戶 {profile_info} 訂製報告
        - 經濟概覽
        - 市場情緒分析
        - 個股/主題推薦 等等
        """
    else:
        prompt = f"""
        You are a professional investment advisor. Based on the following information, provide a clear and structured investment analysis and suggestion :
        【User Profile】{profile_info}
        ---
        1. 【Economic Analysis】
        (a) Analyze the economic indicators over the past 6 months, identify key turning points, and briefly describe the macroeconomic trends:
        {eco_index} 
        (b) market price in last week {market_summary}
        ---
        2. 【Market Sentiment & News】
        (a) Based on the sentiment summary below and related news, determine market sentiment (bullish/bearish), and highlight impacted sectors or stocks also mention the news you found:
        {sentiment_summary}
        (b) Analyze why this top 5 popular sector discussed in reddit {top5_sector_df} 
        News:
        {df['news']}
        ---
        3. 【User's Focused Stocks or Topics】
        News related to user's interest and mention the news you found:
        {interested_news}
        Provide insights, risk/opportunity assessment, and give a recommendation based on the above context.
        ---
        Present your response in this structure:
        - Macroeconomic Overview
        - Market Sentiment
        - Stock/Topic Recommendation
        """

    try:
        res = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial advisor specialized in macroeconomic analysis and personalized investment advice."},
                {"role": "user", "content": prompt}
            ]
        )
        global last_generated_report
        last_generated_report = res.choices[0].message.content
        return last_generated_report
    except Exception as e:
        return f"Error generating report: {e}"

def get_economic_report(language):

    try:
        res = requests.post("http://fastapi:8000/report/generate/", json={"language": language})
        res.raise_for_status()
        return res.text
    except Exception as e:
        print("Error fetching report:", e)
        return "Failed to fetch report."


# ----------send email-----------

def update_report_and_return(age, experience, interest, sources, risk,language, email):
    return generate_personal_report(age, experience, interest, sources, risk,language, email)

def send_latest_report(email):
    return send_email_report(email, "Your Personalized Investment Report", last_generated_report)

# ---------- LangChain Agent ----------

news_tool = Tool(name="SearchNews", func=lambda q: search_news_for_keywords([q]), description="search news")
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="en"))

agent = initialize_agent(
    [news_tool, 
    Tool.from_function(wiki_tool, name="Wikipedia", description="search wikipedia")],
    llm=llm,
    agent="conversational-react-description",
    verbose=True,
    handle_parsing_errors=True  
)

def convert_to_langchain_chat_history(gradio_history):
    
    chat_history = []
    for msg in gradio_history:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                chat_history.append(("user", content))
            elif role == "assistant":
                chat_history.append(("assistant", content))
    return chat_history

def continue_chat_with_agent(history, user_input):
    try:
        chat_history_lc = convert_to_langchain_chat_history(history)

        agent_reply = agent.invoke({
            "input": user_input,
            "chat_history": chat_history_lc
        })

        if hasattr(agent_reply, "content"):
            agent_reply = agent_reply.content
        elif isinstance(agent_reply, dict) and "output" in agent_reply:
            agent_reply = agent_reply["output"]
        else:
            agent_reply = str(agent_reply)

    except Exception as e:
        agent_reply = f"Agent 錯誤：{e}"

    history = history + [
        {"role": "user", "content": str(user_input)},
        {"role": "assistant", "content": str(agent_reply)}
    ]
    return history, history, ""

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
        with gr.Tab("Introduction page"):
            
            with gr.Row():    
                with gr.Column(scale=1):
                    image_output = gr.Image(show_image("Intro_page.png"),width=900,height=850,type="pil")
                    
                with gr.Column(scale=1):
                    gr.Markdown("""
            ## 💸 Welcome to Your AI U.S. Stock Investment Assistant

            I am Money, your smart investment assistant.I'm here to help you build a smarter U.S. stock strategy.  
            By understanding your background, experience, and preferences, we can tailor investment recommendations just for you.  

            👉 Take a moment to complete the short survey on the right to get started.
            """)
                    age = gr.Dropdown(["18-25", "26-35", "36-45", "46-60", "60+"], label="Your Age Group")
                    experience = gr.Radio(["Beginner", "Intermediate", "Advanced"], label="Investment Experience")
                    interest = gr.CheckboxGroup(
                        ["Tech Stocks", "ETF", "High Dividend", "US Bonds", "Forex", "Crypto"],
                        label="Investment Preferences"
                    )
                    sources = gr.Textbox(
                        label="Stock or Keywords You Follow",
                        placeholder="e.g., Nvidia, Tesla......"
                    )
                    risk = gr.Radio(["Conservative", "Moderate", "Aggressive"], label="Risk Tolerance")
                    language = gr.Radio(["English", "chinese"], label = "language")
                    email = gr.Textbox(label = "Your Email (for report delivery)", placeholder="example@email.com")
                    submit_btn = gr.Button("Submit")
                    output = gr.Textbox(label="Submission Status", interactive=False)

                    submit_btn.click(
                        fn = save_to_db,
                        inputs=[age, experience, interest, sources, risk, language, email],
                        outputs=[output, user_state]
                        )

        with gr.Tab(" Economic Indicators"):
            with gr.Row():
                 with gr.Column(scale=1):
                    gr.Markdown("### Economic Indicator 經濟指數")
                    with gr.Row():
                        index_dropdown = gr.Dropdown(label="Select Index",choices=get_index_list(),value=None,scale=2)
                        days_input = gr.Number(label="Days Range",value=180,precision=0,scale=1)
    
                    chart_output = gr.Plot()

                    def update_single_chart(index_name, days):
                        if not index_name:
                            return go.Figure().update_layout(title="Please select an index.")
                        df, _ = fetch_economic_index_summary(index_name=index_name, days=int(days))
                        return plot_index_chart(df, title=f"{index_name} Trend")
                
                    index_dropdown.change(fn=update_single_chart, inputs=[index_dropdown, days_input], outputs=chart_output)
                    days_input.change(fn=update_single_chart, inputs=[index_dropdown, days_input], outputs=chart_output)

                 with gr.Column(scale=1):
                    gr.Markdown("### Market Price 市場走勢")

                    market_dropdown = gr.Dropdown(label="Select Market", choices=get_market_list(), value=None)
                    market_chart_output = gr.Plot()
                    
                    def update_market_chart(market):
                        if not market:
                            return go.Figure().update_layout(title="Please select a market.")
                        df, _ = fetch_market_price_summary(market)
                        return plot_price_chart(df, title=f"{market} Price Trend")


                    market_dropdown.change(fn=plot_price_chart, inputs=[market_dropdown], outputs=[market_chart_output])
            
            gr.Markdown("###  AI Trend Insight")

            ai_generated_report = gr.TextArea(label="📄 AI Analysis Report", lines=25)
            generate_btn = gr.Button(" Generate Trend Report")

            
            generate_btn.click(
            fn=get_economic_report,
            inputs=[language],
            outputs=[ai_generated_report]
        )
            
        with gr.Tab(" Market sentiments"):
            with gr.Row():
                with gr.Column(scale=1):  
                    sentiment_chart = gr.Plot(label="Sentiment Line Chart")
                    chart_btn = gr.Button(" Refresh Trend")
                
                with gr.Column(scale=1):  
                    pie1 = gr.Plot(label="This Week Sentiment")
                    pie2 = gr.Plot(label="Last Week Sentiment")
                    pie_btn = gr.Button("Update Weekly Comparison")

            def update_sentiment_pie():
                df, err = fetch_sentiment_data()
                if err:
                    return go.Figure().update_layout(title=err), go.Figure(), err
                fig1, fig2 = plot_sentiment_pie(df)
                return fig1, fig2

            sentiment_table = gr.Dataframe(label=" Sentiment Topic Summary",wrap=True)
            last_time_text = gr.Markdown()
            table_btn = gr.Button("🔄 Refresh")

            gr.Markdown(" 📈 Pooular Sector and Ticker ")

            chart_output = gr.Plot(label="Weekly Top 5 Sector")
            last_time_text = gr.Markdown()
            refresh_btn1 = gr.Button("🔄 Refresh")

            symbol_table = gr.Dataframe(wrap=True,label=" Top 10 Stock Discussions",interactive=False)

            refresh_btn2 = gr.Button("🔄 Refresh")

            refresh_btn1.click(
                fn= plot_sector_chart,
                outputs=chart_output
            )
            refresh_btn2.click(
                fn= fetch_top10_symbols_this_week,
                 outputs=symbol_table
                 )
            
            chart_btn.click(
                plot_sentiment_line_chart,
                outputs = sentiment_chart
                )

            table_btn.click(
                fn = get_sentiment_table,
                outputs = sentiment_table
            )
            pie_btn.click(
                fn=update_sentiment_pie, 
                outputs=[pie1, pie2]
                )

            demo.load(fn=plot_sentiment_line_chart, outputs=sentiment_chart)
            demo.load(fn=get_sentiment_table, outputs=[sentiment_table,last_time_text])
            demo.load(fn=plot_sector_chart, outputs=chart_output)
            demo.load(fn=fetch_top10_symbols_this_week, outputs=symbol_table)
            demo.load(fn=update_sentiment_pie, outputs=[pie1, pie2])
            
        with gr.Tab("📄 View Report"):
                
            output_text = gr.TextArea(label="📄 Report Content", lines=20) 
            send_status = gr.Textbox(label="📬 Email Status", interactive=False)

            with gr.Row():
                submit_btn = gr.Button("📝 Generate Report")
                send_btn = gr.Button("📤 Send Report via Email")

            with gr.Row():  
                with gr.Column(scale=2):
                    chatbox = gr.Chatbot(type="messages")
                    with gr.Row():
                        msg_input = gr.Textbox(placeholder="輸入你的問題", label="問題")
                        state = gr.State([]
                                         )
                with gr.Column(scale=1):
                    image_output = gr.Image(
                        value=show_image("fin_page.png"), 
                        width= 800,
                        height= 500,
                        type="pil"
                    )

                    
            # Function bindings

            submit_btn.click(
                fn = update_report_and_return,
                inputs=[age, experience, interest, sources, risk,language, email],
                outputs=output_text
            )

            send_btn.click(
                fn=send_latest_report,
                inputs=email,
                outputs=send_status
            )

            msg_input.submit(
                fn=continue_chat_with_agent,
                inputs=[state, msg_input],  
                outputs=[chatbox, state, msg_input]
            )

if __name__ == '__main__':
    print("Gradio version in use:", gr.__version__)
    demo.launch(server_name="0.0.0.0", server_port=7860)
    
