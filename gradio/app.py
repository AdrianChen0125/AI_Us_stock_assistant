# US Stock Sentiment Analysis Web App with GPT, DALL-E, LangChain

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

# LangChain
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun


# ---------- Config ----------
DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "port": os.environ.get("DB_PORT"),
    "dbname": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD")
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-4", temperature=0.5)

# ---------- Utility Functions ----------
def fetch_overall_sentiment_summary():
    conn, cursor = None, None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT published_at, SUM(positive_cm), SUM(negative_cm)
            FROM mv_sentiment_by_date
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
    conn, cursor = None, None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM (
                SELECT processed_date, keywords, total_comments, total_pc, total_nc,
                ROW_NUMBER() OVER (PARTITION BY processed_date ORDER BY total_comments DESC) AS row_number
                FROM mv_sentiment_by_topic 
                WHERE processed_date > NOW() - INTERVAL '14 days'
            ) ranked
            WHERE row_number <= 5 ORDER BY processed_date DESC, total_comments DESC;
        """)
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame(), "查無資料"
        df = pd.DataFrame(rows, columns=["日期", "關鍵字", "總留言數", "正面留言", "負面留言", "排名"])
        summary = "日期 | 關鍵字 | 總留言數 | 正面留言 | 負面留言 | 排名\n" + "-"*60 + "\n"
        for r in rows:
            summary += f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]}\n"
        return df, summary
    except Exception as e:
        return pd.DataFrame(), f"資料庫錯誤：{e}"
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

def search_news_for_keywords(keywords: list, max_articles=5):
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    query = " OR ".join(keywords[:6])
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&from={from_date}&pageSize={max_articles}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        articles = response.json().get("articles", [])
        if not articles:
            return "查無新聞資料。"
        return "\n\n".join([f"【{a['title']}】\n{a['description']}\n連結：{a['url']}" for a in articles])
    except Exception as e:
        return f"新聞查詢錯誤：{e}"

# ---------- GPT Analysis ----------
def analyze_with_news_via_openai():
    df, summary = fetch_sentiment_topic_summary()
    if df.empty:
        return "查無主題情緒資料", "", df, [], []

    keywords_set = set()
    for kw_set in df["關鍵字"].head(3):
        keywords_set.update(kw_set)
    keywords = list(keywords_set)
    keywords_for_news = keywords[:5] + ['us stock']

    news_text = search_news_for_keywords(keywords_for_news)
    if not news_text.strip():
        news_text = "查無新聞資料，請僅依據社群情緒資料進行分析。"

    prompt = f"""
你是一位投資分析顧問，任務是結合社群熱門主題的情緒資料與最新新聞，幫助投資人掌握市場趨勢與潛在風險。

請根據以下資料回答以下問題，並列點輸出分析結果（用繁體中文）：

1. 市場目前最關注的議題是什麼？對應哪些新聞？
2. 情緒資料中哪些主題呈現正向／負向趨勢？
3. 可能影響情緒波動的股市相關新聞？
4. 根據上述觀察，提出 5 點值得注意的市場動向或投資建議

---

社群情緒摘要：
{summary}

新聞內容：
{news_text}

請分析市場關注重點、情緒變化背後的可能原因，以及可能的投資趨勢或警訊，條列式用繁體中文回覆。
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是金融顧問，擅長結合新聞與社群數據做市場分析。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        result = res.choices[0].message.content
        chat_init = [
            {"role": "system", "content": "你是市場分析助理，請根據以下報告持續回答使用者問題。"},
            {"role": "user", "content": result}
        ]
        return result, f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", df, chat_init, chat_init
    except Exception as e:
        return f"OpenAI 錯誤：{e}", "", df, [], []

# ---------- LangChain Agent ----------
news_tool = Tool(name="SearchNews", func=lambda q: search_news_for_keywords([q]), description="查詢最新的新聞")
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="zh"))
agent = initialize_agent(
    [news_tool, Tool.from_function(wiki_tool, name="Wikipedia", description="查詢維基百科")],
    llm=llm,
    agent="conversational-react-description",
    verbose=True,
    handle_parsing_errors=True  
)

def convert_to_langchain_chat_history(gradio_history):
    """將 Gradio 的 messages 格式轉成 LangChain Agent 所需格式"""
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
    return history, history







# ---------- Image Gen ----------
def create_prompt_from_data(sentiment_df, topic_df, keywords_per_row=2, max_total_keywords=6):
    if sentiment_df.empty or "total_pc" not in sentiment_df.columns or "total_nc" not in sentiment_df.columns:
        return "Data is invalid. Cannot generate image prompt."

    # Get latest sentiment
    latest_row = sentiment_df.iloc[-1]
    net_sentiment = latest_row["total_pc"] - latest_row["total_nc"]

    # Determine sentiment direction
    if net_sentiment > 0:
        sentiment_trend = "The market is optimistic and investor confidence is high."
        color_tone = "Use warm, bright tones like sunrise orange and soft yellow."
        character_description = (
            "Include a cute, cheerful cartoon character (like a young investor ) smiling and celebrating, "
            "looking at rising charts or positive news headlines, with happy energy and soft expressions."
        )
    else:
        sentiment_trend = "The market is tense and investors are concerned."
        color_tone = "Use cool, moody tones like bluish gray, navy, and soft purple."
        character_description = (
            "Include a worried cartoon character (like a nervous investor) looking concerned, "
            "holding a newspaper with falling stocks, or staring at a downtrend chart, with expressive eyes and subtle humor."
        )

    # Extract keywords
    keywords = []
    for kw_set in topic_df["關鍵字"]:
        if isinstance(kw_set, set):
            keywords.extend(list(kw_set)[:keywords_per_row])
        if len(keywords) >= max_total_keywords:
            break
    topics_text = "、".join(keywords[:max_total_keywords])

    # Final image prompt
    image_prompt = f""" 
    Visualize the current market sentiment: "{sentiment_trend}" with focus on topics: {topics_text}.
    {character_description}Use a Ghibli-style illustration with emotional atmosphere. Incorporate financial visuals like charts, tickers.
    {color_tone}The composition should be clean, balanced. Avoid overly complex backgrounds.
    """.strip()

    return image_prompt

def generate_sentiment_image(prompt):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        print(f"[generate_sentiment_image] 錯誤：{e}")
        return None

def generate_image_on_click():
    sentiment_df = fetch_overall_sentiment_summary()
    topic_df, _ = fetch_sentiment_topic_summary()
    if sentiment_df.empty or topic_df.empty:
        return None

    prompt = create_prompt_from_data(sentiment_df, topic_df)
    image_url = generate_sentiment_image(prompt)
    if not image_url:
        return None

    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        file_path = "/tmp/sentiment_image.png"
        image.save(file_path)
        return file_path
    except Exception as e:
        print(f"[Image Download Error] {e}")
        return None

# ---------- Chart ----------
def plot_sentiment_line_chart():
    df = fetch_overall_sentiment_summary()
    if df.empty or "error" in df.columns:
        return go.Figure().update_layout(title="資料載入錯誤")
    df["total_pc"] = pd.to_numeric(df["total_pc"], errors="coerce")
    df["total_nc"] = pd.to_numeric(df["total_nc"], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["published_at"], y=df["total_pc"], mode="lines+markers", name="正面留言"))
    fig.add_trace(go.Scatter(x=df["published_at"], y=df["total_nc"], mode="lines+markers", name="負面留言"))
    fig.update_layout(title="每日市場情緒趨勢", xaxis_title="日期", yaxis_title="留言數")
    return fig

# ---------- UI ----------
with gr.Blocks() as demo:
    history_state = gr.State([])
    gr.Markdown("## 🧠 美股留言情緒觀察平台")

    chart = gr.Plot()
    chart_btn = gr.Button("更新情緒圖")

    table = gr.Dataframe(label="主題情緒統計")

    with gr.Row():
        with gr.Column():
            output = gr.TextArea(label="分析報告")
            timestamp = gr.Markdown()
            update_btn = gr.Button("更新 AI 分析")
        with gr.Column():
            image_output = gr.Image(type="filepath", label="AI 圖片")
            generate_btn = gr.Button("生成圖片")


    with gr.Blocks():
        chatbox = gr.Chatbot(type="messages")
        msg_input = gr.Textbox(placeholder="輸入你的問題", label="問題")

        state = gr.State([])  # holds chat history
        msg_input.submit(
        fn=continue_chat_with_agent,
        inputs=[state, msg_input],  # pass both history and user input
        outputs=[chatbox, state]    # update chatbox and chat history
        )
    # 綁定互動
    chart_btn.click(plot_sentiment_line_chart, outputs=chart)
    generate_btn.click(generate_image_on_click, outputs=image_output)
    update_btn.click(analyze_with_news_via_openai, outputs=[output, timestamp, table, chatbox, history_state])
   
# 啟動
if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)
