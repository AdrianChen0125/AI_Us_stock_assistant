import time
import os
import gradio as gr
import psycopg2
import pandas as pd
import plotly.graph_objects as go
from datetime import date
import requests
from PIL import Image

from service.fetch_data import *


# ----- plots ----- #
def plot_sentiment_line_chart():
    df = fetch_overall_sentiment_summary()
    if df.empty or "error" in df.columns:
        return go.Figure().update_layout(title=" Failed to load data")
    
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

def plot_price_chart(df, title="Price Trend"):
    if df.empty:
        return go.Figure().update_layout(title="Failed to load data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price"],
        mode="lines+markers", name=title
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        height=350
    )
    return fig

def update_sentiment_pie():
            df, err = fetch_sentiment_data()
            if err:
                return go.Figure().update_layout(title=err), go.Figure()
            fig1, fig2 = plot_sentiment_pie(df)
            return fig1, fig2

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

# ----- images ----- #

def show_image(filename):
    image_path = os.path.join("/app/assets", filename)
    return Image.open(image_path)