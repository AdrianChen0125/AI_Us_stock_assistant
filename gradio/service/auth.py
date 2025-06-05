import requests
import gradio as gr

API_URL = "http://fastapi:8000/auth/token"

def login(username, password):
    try:
        r = requests.post(
            API_URL,
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        if r.status_code == 200:
            token = r.json()["access_token"]
            return (
                "Login successfully!! Now you can use the app",
                gr.update(visible=False),
                gr.update(visible=True),
                token
            )
        return (
            " Login failed. Please check your username and password.",
            gr.update(visible=True),
            gr.update(visible=False),
            ""
        )
    except Exception as e:
        return (f"error：{str(e)}", gr.update(), gr.update(visible=False), "")
