from openai import OpenAI
import os

# 初始化 OpenAI 客戶端，使用環境變數中的 API 金鑰
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate_text(text: str, target_lang: str = "zh") -> str:
    """
    使用 OpenAI GPT 模型翻譯輸入文字為指定語言。
    
    參數：
    - text: 要翻譯的文字
    - target_lang: 目標語言代碼（"zh" 表示中文，"en" 表示英文）

    回傳：
    - 翻譯後的文字（或錯誤訊息）
    """
    # 定義系統提示語（可依語言擴充）
    lang_map = {
        "zh": "請將以下內容翻譯成繁體中文：",
        "en": "Please translate the following into English:"
    }

    # 取對應語言的提示，找不到則預設英文
    system_prompt = lang_map.get(target_lang, "Translate the following:")

    try:
        # 發送翻譯請求給 OpenAI
        res = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
        )
        # 回傳翻譯結果（去除首尾空白）
        return res.choices[0].message.content.strip()
    except Exception as e:
        # 發生錯誤時回傳錯誤訊息
        return f"[Translation Error: {str(e)}]"