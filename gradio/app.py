import time
import smtplib
import os
import gradio as gr
import psycopg2
import pandas as pd
from datetime import date
import requests

# ---------- Import Services ---------- #
from service.auth import login
from service.email_service import send_email_report, send_latest_report
from service.fetch_data import *
from service.plots import *
from service.stock_recommender import g4_recommend_multi
from service.report_generator import *
from service.chat_bot import call_chatbot_api

# ---------- Import UI Components ---------- #
from ui.render_main_page import main_tab
from ui.render_economic_market import economic_market_trends_tab 
from ui.render_market_sentiment import  market_sentiments_tab


# ---------- Config ----------
DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "port": os.environ.get("DB_PORT"),
    "dbname": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD")}

API_BASE = "http://fastapi:8000"

# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown("# AI Investment Assistant")

    # global states to hold user profile and access token
    user_profile_state = gr.State()
    access_token = gr.State()
    economic_report_state = gr.State()
    sentiment_report_state = gr.State() 
    stock_recommendation_state = gr.State()
    

    with gr.Tabs():
        ## Main Page 
        main_tab(user_profile_state, access_token)

        ## Economic and Market Trends 
        economic_market_trends_tab(user_profile_state, economic_report_state, access_token)

        ## Market Sentiments 
        
        elements = market_sentiments_tab(sentiment_report_state, access_token)

        demo.load(fn=plot_sentiment_line_chart, outputs=elements["sentiment_chart"])
        demo.load(fn=get_sentiment_table, outputs=[elements["sentiment_table"], elements["last_time_text"]])
        demo.load(fn=plot_sector_chart, outputs=elements["chart_output"])
        demo.load(fn=fetch_top10_symbols_df, outputs=elements["symbol_table"])
        demo.load(fn=update_sentiment_pie, outputs=[elements["pie1"], elements["pie2"]])

        with gr.Tab(" Stock Recommendation"):

            gr.Markdown("## Stock Recommendation System")

            sectors_list = [
                "basic materials", "communication services", 
                "consumer cyclical","consumer defensive",
                "energy", "financial services", "healthcare",
                "industrials", "real estate", "technology"
            ]

            def enforce_sector_limit(selected):
                return selected[:2] if len(selected) > 2 else selected

            with gr.Row():
                with gr.Column(scale=2):    
                    sectors = gr.CheckboxGroup(
                        label="Select sectors (limit to 2)",
                        choices=sectors_list,
                        value=["technology"],
                        interactive=True
                    )
                    submit_button = gr.Button("Search Stocks")
                    recommend_output = gr.Textbox(label="Result of Recommendation", lines=6)

                with gr.Column(scale=3):
                    ai_output = gr.Textbox(label="", lines=35)
                    analyze_button = gr.Button("Analyze Recommendation")

            # limit the number of selected sectors to 2
            sectors.change(
                fn = enforce_sector_limit,
                inputs = sectors, 
                outputs = sectors
                )


            submit_button.click(
                fn= g4_recommend_multi,
                inputs=[user_profile_state, sectors],
                outputs=[recommend_output]
            )
                        
            
            analyze_button.click(
                fn = get_stock_recommendation_report,
                inputs=[user_profile_state, recommend_output, access_token],
                outputs=[ai_output, stock_recommendation_state]
            )
            
        with gr.Tab("View Report"):
            with gr.Row():
                with gr.Column(scale=2): 
                    output_text = gr.TextArea(label=" Report Content", lines=20) 
                    send_status = gr.Textbox(label=" Email Status", interactive=False)

                    with gr.Row():
                        submit_btn = gr.Button(" Generate Report")
                        send_btn = gr.Button(" Send Report via Email")

                                    
                        submit_btn.click(
                            fn= generate_overall_report,
                            inputs=[
                                economic_report_state,
                                sentiment_report_state,
                                stock_recommendation_state,
                                user_profile_state,
                                access_token 
                            ],
                            outputs = output_text
                        )
                        
                        send_btn.click(
                        fn = send_latest_report,
                        inputs = [user_profile_state, output_text],
                        outputs = send_status
                        )
  
                with gr.Column(scale=1):
                    image_output = gr.Image(
                        value=show_image("fin_page.png"), 
                        width= 800,
                        height= 500,
                        type="pil"
                    )

        with gr.Tab("Small Talk on Wall Street"):
            gr.ChatInterface(
                fn=call_chatbot_api,
                title="AI Assistant",
                chatbot=gr.Chatbot(type="messages"),
                textbox=gr.Textbox(placeholder="輸入你的問題...", label="問題"),
                additional_inputs=[access_token],
                flagging_mode="manual",  
                flagging_options=["👍 like", "👎 bad"],
                theme="default" 
            )


if __name__ == '__main__':
    print("Gradio version in use:", gr.__version__)
    demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    root_path="/gradio"  )
    
