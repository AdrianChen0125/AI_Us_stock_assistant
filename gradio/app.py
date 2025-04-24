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
    "password": os.environ.get("DB_PASSWORD")
}

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
def fetch_economic_index_summary():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT date, series_id, value
            FROM raw_data.economic_indicators
            WHERE date >= NOW()::date - INTERVAL '1 year'
            ORDER BY series_id, date;
        """)
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame(), "No data found"
        
        df = pd.DataFrame(rows, columns=["date", "index_name", "value"])
        summary = "\n".join([f"{r[0]} | {r[1]} | {r[2]}" for r in rows])
        return df,summary
    
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
            SELECT published_at, SUM(positive_cm), SUM(negative_cm)
            FROM processed_data.mv_sentiment_by_date
            GROUP BY published_at ORDER BY published_at;
        """)
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=["published_at", "total_pc", "total_nc"])
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

def fetch_sentiment_topic_summary():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT topic_date, topic_summary, keywords, comments_count, pos_count, neg_count
            FROM processed_data.reddit_topic
            WHERE topic_date = (
                SELECT MAX(topic_date) FROM processed_data.reddit_topic
            )
            ORDER BY comments_count DESC, topic_date DESC
            LIMIT 5;
        """)
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame(), "No data found"
        df = pd.DataFrame(rows, columns=["date", "title", "keywords", "comment_count", "positive", "negative"])
        summary = "\n".join([f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]}" for r in rows])
        return df, summary
    
    except Exception as e:
        return pd.DataFrame(), f"Database error: {e}"
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

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
    fig.update_layout(title="📊 Reddit Daily Market Sentiment Trend ",height=350 ,xaxis_title="Date", yaxis_title="Comment Count")
    return fig

def plot_index_chart(index_name, title):
    df,summary = fetch_economic_index_summary()
    if df.empty:
        return go.Figure().update_layout(title="❌ Failed to load data")

    df = df[df["index_name"] == index_name].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["value"],
        mode="lines+markers", name=index_name
    ))
    fig.update_layout(
        xaxis=dict(tickangle=-45,dtick="M3"),
        margin=dict(l=15, r=15, t=30, b=30),
        height=250
        )
    return fig

def get_sentiment_table():
                    df, _ = fetch_sentiment_topic_summary()
                    df1 = df[["date", "title","positive", "negative"]]
                    return df1
# ---------- image ----------

def show_image(filename):
    image_path = os.path.join("/app/assets", filename)
    return Image.open(image_path)

# ---------- Report Generation ----------
def generate_personal_report(age, experience, interest, sources, risk,langauge , email):
    
    profile_info = f"""
    User Profile:
    - Age: {age}
    - Experience: {experience}
    - Preferences: {", ".join(interest)}
    - Sources: {sources}
    - Risk Tolerance: {risk}
    """

    df, summary = fetch_sentiment_topic_summary()
    df['news'] = df["keywords"].apply(search_news_for_keywords)
    index_df, eco_index = fetch_economic_index_summary()
    interested_news = search_news_for_keywords(sources)

    prompt = f"""
You are a professional investment advisor. Based on the following information, provide a clear and structured investment analysis and suggestion in {language}:

【User Profile】
{profile_info}

---

1. 【Economic Analysis】
Analyze the economic indicators over the past 6 months, identify key turning points, and briefly describe the macroeconomic trends:
{eco_index}

---

2. 【Market Sentiment & News】
Based on the sentiment summary below and related news, determine market sentiment (bullish/bearish), and highlight impacted sectors or stocks also mention the news you found:
{summary}

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

def generate_report_from_economic_index(language):
    df, _ = fetch_economic_index_summary()
    if df.empty:
        return "❌ 查無經濟數據資料，無法產生分析報告。"

    summary_lines = []
    for index in df["index_name"].unique():
        sub_df = df[df["index_name"] == index]
        recent = sub_df.sort_values("date").tail(6)
        values = ", ".join([f"{d}={round(v,2)}" for d, v in zip(recent["date"], recent["value"])])
        summary_lines.append(f"{index}：{values}")

    summary_text = "\n".join(summary_lines)

    # prompt
    prompt = f"""
You are a professional economic and investment advisor. Based on the recent six-month trends of key economic indicators, write a clear and insightful **macroeconomic trend report ** in {language}.

Please structure your analysis in bullet points across the following three areas:

### 📈 Inflation and Interest Rate Trends
- Analyze trends in CPI, FEDFUNDS, and GS10, and discuss potential drivers behind the changes
- Explain the implications for inflation control and monetary policy

### 🛍️ Consumer Activity and Employment
- Observe trends and fluctuations in RSAFS, UMCSENT, and UNRATE
- Infer the likely phase of the current business cycle

### 📊 Potential Impact on Financial Markets
- Discuss how macroeconomic developments may influence equities, bonds, and investor risk appetite

Here is the summarized data:
{summary_text}
"""

    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位總體經濟與投資分析師，擅長針對資料撰寫趨勢報告與專業建議。"},
                {"role": "user", "content": prompt}
            ]
        )
        return res.choices[0].message.content

    except Exception as e:
        return f"❌ 報告產生錯誤：{e}"

# ----------send email-----------

def update_report_and_return(age, experience, interest, sources, risk, email):
    return generate_personal_report(age, experience, interest, sources, risk,language, email)

def send_latest_report(email):
    return send_email_report(email, "Your Personalized Investment Report", last_generated_report)

# ---------- LangChain Agent ----------
news_tool = Tool(name="SearchNews", func=lambda q: search_news_for_keywords([q]), description="search news")
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="en"))

agent = initialize_agent(
    [news_tool, Tool.from_function(wiki_tool, name="Wikipedia", description="search wikipedia")],
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
                    language = gr.Radio(["English", "Chinese"], label = "language")
                    email = gr.Textbox(label = "Your Email (for report delivery)", placeholder="example@email.com")
                    submit_btn = gr.Button("Submit")
                    output = gr.Textbox(label="Submission Status", interactive=False)

                    submit_btn.click(
                        fn = save_to_db,
                        inputs=[age, experience, interest, sources, risk, language, email],
                        outputs=[output, user_state]
                        )

        with gr.Tab(" Economic Indicators"):

            gr.Markdown("### 💸 Inflation & Interest Rates")
            with gr.Row():
                cpi_chart = gr.Plot(label="CPI (CPIAUCSL)")
                fedfunds_chart = gr.Plot(label="Fed Funds Rate (FEDFUNDS)")
                gs10_chart = gr.Plot(label="10Y Treasury Yield (GS10)")

            gr.Markdown("### 🛍️ Consumer Activity & Employment")
            # 第二排：消費 + 就業
            with gr.Row():
                rsafs_chart = gr.Plot(label="Retail Sales (RSAFS)")
                umcsent_chart = gr.Plot(label="Consumer Sentiment (UMCSENT)")
                unrate_chart = gr.Plot(label="Unemployment Rate (UNRATE)")

            update_all_btn = gr.Button("🔄 Update All Charts")

            def update_all_economic_charts():
                return (
                    plot_index_chart("CPIAUCSL", "CPI"),
                    plot_index_chart("FEDFUNDS", "Fed Funds Rate"),
                    plot_index_chart("GS10", "10Y Treasury Yield"),
                    plot_index_chart("RSAFS", "Retail Sales"),
                    plot_index_chart("UMCSENT", "Consumer Sentiment"),
                    plot_index_chart("UNRATE", "Unemployment Rate")
                )

            update_all_btn.click(
                fn = update_all_economic_charts,
                outputs=[
                    cpi_chart, fedfunds_chart, gs10_chart,
                    rsafs_chart, umcsent_chart, unrate_chart
                ]
            )
            gr.Markdown("###  AI Trend Insight")

            ai_generated_report = gr.TextArea(label="📄 AI Analysis Report", lines=20)
            generate_btn = gr.Button(" Generate Trend Report")

            def generate_report_eco(state):
                lang = state["language"]
                return generate_report_from_economic_index(lang)
            
            generate_btn.click(
                fn = generate_report_eco,
                inputs = [user_state],
                outputs = ai_generated_report
                )

            demo.load(fn = update_all_economic_charts, outputs=[
                    cpi_chart, fedfunds_chart, gs10_chart,
                    rsafs_chart, umcsent_chart, unrate_chart
                ])
            
        with gr.Tab(" Market sentiments"):
                
            sentiment_chart = gr.Plot(label="📈 Sentiment Line Chart")
            chart_btn = gr.Button("🔄 Refresh Chart")

            sentiment_table = gr.Dataframe(label="📊 Sentiment Topic Summary")
            table_btn = gr.Button("🔄 Refresh Table")
                            
            
            chart_btn.click(
                plot_sentiment_line_chart,
                outputs=sentiment_chart
                )

            table_btn.click(
                fn=get_sentiment_table,
                outputs=sentiment_table
            )

            demo.load(fn=plot_sentiment_line_chart, outputs=sentiment_chart)
            demo.load(fn=get_sentiment_table, outputs=sentiment_table)
            
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
    
