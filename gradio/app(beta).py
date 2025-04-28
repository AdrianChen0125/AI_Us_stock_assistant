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

from utils import (
    fetch_sentiment_topic_summary,
    fetch_economic_index_summary,
    fetch_market_price_summary,
    fetch_market_price_last_7_days,
    fetch_overall_sentiment_summary,
    fetch_top10_symbols_this_week,
    fetch_top5_sectors_this_week,
    search_news_for_keywords,
)
from langgraph_logic import generate_personal_report_via_langgraph

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
    print(df)
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
        margin=dict(l = 20, r = 20, t = 30, b = 30),
        height=250
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

def call_langgraph_report(age, experience, interest, sources, risk, language, email):
    try:
        result = generate_personal_report_via_langgraph(
            age = age,
            experience = experience,
            interest = interest, 
            sources = sources,
            risk = risk,
            language = language,
            email = email
        )
        return result
    except Exception as e:
        return f"❌ Error: {e}"

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

def update_report_and_return(age, experience, interest, sources, risk,language, email):
    return call_langgraph_report(age, experience, interest, sources, risk,language, email)

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
                    interest = gr.CheckboxGroup(["Tech Stocks", "ETF", "High Dividend", "US Bonds", "Forex", "Crypto"],label="Investment Preferences")
                    sources = gr.Textbox(
                        label='Stock You have (Use "," to separate)',
                        placeholder="e.g., Nvidia, Tesla......"
                    )
                    risk = gr.Radio(["Conservative", "Moderate", "Aggressive"], label="Risk Tolerance")
                    language = gr.Radio(["English", "Traditional_Chinese(繁體中文)"], label = "language")
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
            gr.Markdown(" 📈 Sentiment and Topic ")
            sentiment_chart = gr.Plot(label="Sentiment Line Chart")
            chart_btn = gr.Button("🔄 Refresh")
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
                fn=plot_sector_chart,
                outputs=chart_output
            )
            refresh_btn2.click(
                fn=fetch_top10_symbols_this_week,
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

            demo.load(fn=plot_sentiment_line_chart, outputs=sentiment_chart)
            demo.load(fn=get_sentiment_table, outputs=[sentiment_table,last_time_text])
            demo.load(fn=plot_sector_chart, outputs=chart_output)
            demo.load(fn=fetch_top10_symbols_this_week, outputs=symbol_table)
            
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
    
