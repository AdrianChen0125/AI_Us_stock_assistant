import gradio as gr
import plotly.graph_objects as go
from service.fetch_data import fetch_economic_index_summary
from service.fetch_data import fetch_market_price_summary
from service.fetch_data import get_index_list, get_market_list
from service.report_generator import get_economic_report
from service.plots import plot_index_chart, plot_price_chart


def economic_market_trends_tab(user_profile_state, economic_report_state, access_token):
    with gr.Tab("Economic & Market Trends"):
        with gr.Row():
            # Economic Indicators Section
            with gr.Column(scale=1):
                gr.Markdown("## Economic Indicator")

                with gr.Row():
                    index_dropdown = gr.Dropdown(
                        label="Select Index",
                        choices=get_index_list(),
                        value=None,
                        scale=2
                    )
                    days_input = gr.Number(
                        label="Days Range",
                        value=180,
                        precision=0,
                        scale=1
                    )

                chart_output = gr.Plot()

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

            # Market Trends Section
            with gr.Column(scale=1):
                gr.Markdown("## Market Trends")

                market_dropdown = gr.Dropdown(
                    label="Select Market",
                    choices=get_market_list(),
                    value=None
                )
                market_chart_output = gr.Plot()

                def update_market_chart(market):
                    if not market:
                        return go.Figure().update_layout(title="Please select a market.")
                    df, _ = fetch_market_price_summary(market)
                    return plot_price_chart(df, title=f"{market} Price Trend")

                market_dropdown.change(
                    fn=update_market_chart,
                    inputs=[market_dropdown],
                    outputs=[market_chart_output],
                    queue=True,
                )

        # AI Trend Report Section
        gr.Markdown("### AI Agent 趨勢分析小幫手 ")

        ai_generated_report = gr.TextArea(label="📄 AI Analysis Report", lines=25)
        generate_btn = gr.Button("Generate Trend Report")

        generate_btn.click(
            fn = get_economic_report,
            inputs=[user_profile_state, access_token],
            outputs=[ai_generated_report, economic_report_state]
        )