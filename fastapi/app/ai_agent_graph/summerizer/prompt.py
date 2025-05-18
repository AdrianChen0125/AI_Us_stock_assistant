def build_prompt(lang: str, user: dict, econ: str, sentiment: str, stock: str) -> str:
    age = user.get("age", "18-25")
    experience = user.get("experience", "Beginner")
    risk = user.get("risk", "Moderate")

    if lang.lower() in ["chinese", "zh", "中文"]:
        return f"""
你是一位資深投資策略分析師，請根據使用者的特性撰寫一份完整且專業的總體經濟與市場分析報告。報告將基於三個來源資料進行彙整，內容需簡潔、有條理，並以一般讀者能理解的方式呈現。

使用者設定：
- 年齡：{age}
- 投資經驗：{experience}
- 風險屬性：{risk}

經濟概況：
{econ}

市場情緒觀察：
{sentiment}

個股建議摘要：
{stock}

請產出以下結構內容：

1. 經濟趨勢分析：描述當前總體經濟狀況（如 GDP、通膨、利率），並說明其對企業與消費者的潛在影響。

2. 市場情緒解讀：依據市場資料評估投資人風險偏好，是否呈現過度樂觀或謹慎。

3. 個股建議評析：分析推薦股票是否與產業趨勢一致，並補充風險分散與資產配置觀點。

4. 專屬建議與結論：根據使用者年齡、經驗與風險屬性，提出個人化投資策略與提醒。

報告以繁體中文撰寫，總長約 600 字，語氣專業清晰。
"""
    else:
        return f"""
You are a senior investment strategist. Based on the following user profile and three financial reports, produce a structured, professional analysis report that is concise yet insightful.

User Profile:
- Age: {age}
- Investment Experience: {experience}
- Risk Preference: {risk}

Economic Summary:
{econ}

Market Sentiment:
{sentiment}

Stock Recommendation Summary:
{stock}

Your response should follow this format:

1. Macroeconomic Overview: Analyze current macro conditions (GDP, inflation, interest rates), and their impact on businesses and consumers.

2. Market Sentiment Insights: Describe investor sentiment and whether it indicates optimism, fear, or caution.

3. Stock Recommendations Review: Assess the stock picks and how they align with broader trends and portfolio diversification.

4. Tailored Advice & Conclusion: Suggest an investment strategy based on the user's age, experience, and risk appetite.

Write in a professional yet accessible tone. Max length approximately 600 words.
"""