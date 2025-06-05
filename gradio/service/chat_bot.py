import requests

API_CHAT_URL = "http://fastapi:8000/AI/chat"

def call_chatbot_api(msg, history, token):
    if not token:
        return [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": "請先登入取得 token"}
        ]

    try:
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "question": msg,
            "history": history
        }

        res = requests.post(API_CHAT_URL, json=payload, headers=headers)
        res.raise_for_status()
        reply = res.json().get("reply", "[No reply received]")

        return [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": reply}
        ]

    except Exception as e:
        return [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": f"Error: {e}"}
        ]
