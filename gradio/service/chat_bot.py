# import requests

# API_CHAT_URL = "http://fastapi:8000/AI/chat"

# def call_chatbot_api(msg, history, token):
#     if not token:
#         return [
#             {"role": "user", "content": msg},
#             {"role": "assistant", "content": "請先登入取得 token"}
#         ]

#     try:
#         headers = {"Authorization": f"Bearer {token}"}
#         payload = {
#             "question": msg,
#             "history": history
#         }

#         res = requests.post(API_CHAT_URL, json=payload, headers=headers)
#         res.raise_for_status()
#         reply = res.json().get("reply", "[No reply received]")

#         return [
#             {"role": "user", "content": msg},
#             {"role": "assistant", "content": reply}
#         ]

#     except Exception as e:
#         return [
#             {"role": "user", "content": msg},
#             {"role": "assistant", "content": f"Error: {e}"}
#         ]

import httpx
import asyncio

API_CHAT_URL = "http://fastapi:8000/AI/chat"

async def call_chatbot_api(msg, history, token):
    if not token:
        yield {"role": "user", "content": msg}
        yield {"role": "assistant", "content": "請先登入取得 token"}
        return

    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "question": msg,
        "history": history
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", API_CHAT_URL, json=payload, headers=headers) as response:
                response.raise_for_status()

                yield {"role": "user", "content": msg}

                async for line in response.aiter_lines():
                    if line.strip():
                        # 假設後端用的是 SSE 或類似 newline 分段
                        yield {"role": "assistant", "content": line}

    except Exception as e:
        yield {"role": "assistant", "content": f"⚠️ Error: {e}"}