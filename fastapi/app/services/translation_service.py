from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 語言別名映射
LANGUAGE_MAP = {
    "chinese": "zh",
    "zh": "zh",
    "zh-hant": "zh",
    "中文": "zh",
    "english": "en",
    "en": "en"
}

def translate_text(text: str, target_lang: str = "zh") -> str:
    """
    使用 OpenAI GPT 模型翻譯文字為指定語言。
    支援的目標語言：zh（繁體中文）、en（英文）
    """
    normalized_lang = target_lang.strip().lower()
    lang_code = LANGUAGE_MAP.get(normalized_lang)

    if lang_code not in ["zh", "en"]:
        return f"[Translation Error: Unsupported language '{target_lang}']"

    prompts = {
        "zh": "請將以下內容翻譯成繁體中文：",
        "en": "Please translate the following into English:"
    }

    system_prompt = prompts[lang_code]

    try:
        res = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"[Translation Error: {str(e)}]"