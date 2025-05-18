import gradio as gr
import requests


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
                "登入成功，您可以開始填寫資料",
                gr.update(visible=False),
                gr.update(visible=True),
                token
            )
        return (
            " 帳號或密碼錯誤",
            gr.update(visible=True),
            gr.update(visible=False),
            ""
        )
    except Exception as e:
        return (f"錯誤：{str(e)}", gr.update(), gr.update(visible=False), "")
