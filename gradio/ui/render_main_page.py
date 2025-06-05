import gradio as gr
from service.auth import login
from service.plots import show_image  
from service.create_data import *

video_path = "assets/intro_video.mp4"
def main_tab(user_profile_state, access_token):
    with gr.Tab("Main Page"):
        with gr.Row():
            with gr.Column(scale=1):
                # gr.Image(show_image("Intro_page.png"), width=900, height=850, type="pil")
                 gr.Video(video_path, autoplay=True, width=900, height=850)
            with gr.Column(scale=1):
                gr.Markdown("### 🔐 Login First")

                username = gr.Textbox(label="Username")
                password = gr.Textbox(label="Password", type="password")
                login_btn = gr.Button("Login")
                login_status = gr.Markdown()
                
                access_token_box = gr.Textbox(visible=False)

                with gr.Column(visible=False) as form_section:
                    risk = gr.Radio(["low", "moderate", "high"], value="moderate", label="Risk Tolerance")
                    interest = gr.CheckboxGroup(["stock", "etf"], value=["stock"], label="Investment Preferences")
                    holdings = gr.Textbox(label="Stock You Have", placeholder="e.g., NVDA, TSLA...")
                    language = gr.Radio(["English", "Chinese"], value="English", label="Language")
                    email = gr.Textbox(label="Your Email", placeholder="example@email.com")
                    submit_btn = gr.Button("Submit")
                    output = gr.Textbox(label="Submission Status", interactive=False)

                login_btn.click(
                    fn=login,
                    inputs=[username, password],
                    outputs=[login_status, login_btn, form_section, access_token_box]
                )

                access_token_box.change(
                    lambda x: x,
                    inputs = access_token_box,
                    outputs = access_token
                )

                submit_btn.click(
                    fn = save_to_db,
                    inputs = [risk, interest, holdings, language, email],
                    outputs = [output, user_profile_state]
                )