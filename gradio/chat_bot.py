import requests

API_CHAT_URL = "http://fastapi:8000/AI/chat"

def call_chatbot_api(msg, history, token):
    if not token:
        return history + [{"role": "user", "content": msg},
                          {"role": "assistant", "content": "請先登入取得 token"}], history

    try:
        headers = {"Authorization": f"Bearer {token}"}
        messages = history + [{"role": "user", "content": msg}]
        payload = {
            "question": msg,
            "history": history  # 這裡的 history 應該已經是 [{"role":..., "content":...}]
        }

        res = requests.post(API_CHAT_URL, json=payload, headers=headers)
        res.raise_for_status()
        reply = res.json().get("reply", "[No reply received]")
        messages.append({"role": "assistant", "content": reply})
        return messages, messages

    except Exception as e:
        messages = history + [{"role": "user", "content": msg},
                              {"role": "assistant", "content": f"Error: {e}"}]
        return messages, history