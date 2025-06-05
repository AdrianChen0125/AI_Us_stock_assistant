import gradio as gr
import plotly.graph_objects as go
from service.plots import *
from service.fetch_data import *
from service.report_generator import get_market_sentiment_report

def market_sentiments_tab(sentiment_report_state, access_token):
    with gr.Tab("Market Sentiments"):
        with gr.Row():
            with gr.Column(scale=1): 
                gr.Markdown("### Daily Sentiment about US Stock") 
                sentiment_chart = gr.Plot(label="Sentiment Line Chart")
                chart_btn = gr.Button("Refresh Trend")
            
            with gr.Column(scale=1):
                gr.Markdown("### Weekly Reddit Sentiment about US Stock")  
                pie1 = gr.Plot(label="This Week Sentiment")
                pie2 = gr.Plot(label="Last Week Sentiment")
                pie_btn = gr.Button("Update Weekly Comparison")

        def update_sentiment_pie():
            df, err = fetch_sentiment_data()
            if err:
                return go.Figure().update_layout(title=err), go.Figure()
            fig1, fig2 = plot_sentiment_pie(df)
            return fig1, fig2

        gr.Markdown("### Top 10 Topics about US Stock") 
        sentiment_table = gr.Dataframe(label="Sentiment Topic Summary", wrap=True)
        last_time_text = gr.Markdown()
        table_btn = gr.Button("Refresh")

        def get_top_10_topic():
            df, _ = get_sentiment_table()
            return df

        table_btn.click(
            fn = get_sentiment_table,
            outputs=[sentiment_table, last_time_text]
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Popular Sectors and Tickers")
                chart_output = gr.Plot(label="Weekly Top 5 Sector")
                last_time_sector_text = gr.Markdown()
                refresh_btn1 = gr.Button("🔄 Refresh")

                refresh_btn1.click(
                    fn = plot_sector_chart,
                    outputs = chart_output
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Top 10 Stock Discussions")    
                symbol_table = gr.Dataframe(wrap=True, interactive=False)
                refresh_btn2 = gr.Button("🔄 Refresh")

        def fetch_top10_symbols_df():
            df, _ = fetch_top10_symbols_this_week()
            return df

        refresh_btn2.click(
            fn = fetch_top10_symbols_df,
            outputs = symbol_table
        )

        gr.Markdown("### AI Agent 市場情緒分析小幫手")

        ai_sentiment_report = gr.TextArea(label="📄 AI Analysis Report", lines=25)
        
        sentiment_report_generate_btn = gr.Button("Generate Trend Report")

        sentiment_report_generate_btn.click(
            fn = get_market_sentiment_report,
            inputs=[access_token],
            outputs=[ai_sentiment_report, sentiment_report_state]
        )

        chart_btn.click(
            plot_sentiment_line_chart,
            outputs = sentiment_chart
        )

        pie_btn.click(
            fn = update_sentiment_pie, 
            outputs = [pie1, pie2]
        )
        
    return {
    "sentiment_chart": sentiment_chart,
    "sentiment_table": sentiment_table,
    "last_time_text": last_time_text,
    "chart_output": chart_output,
    "symbol_table": symbol_table,
    "pie1": pie1,
    "pie2": pie2,
}