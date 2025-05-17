def build_economic_report_prompt(summary: str, language: str = "English") -> str:
    prompt = f"""
    You are a professional macroeconomic and investment analyst.
    Based on the following summarized data of macroeconomic indicators and market movements, please generate a concise economic report in **{language}**. The report will be used for investment briefings and strategy sessions.

    Structure your analysis into the following sections:

    ### 📈 1. Inflation and Interest Rate Outlook
    - Discuss trends in CPI, FEDFUNDS, and GS10.
    - Comment on inflation direction and monetary policy stance.
    - Mention potential risks ahead.

    ---

    ### 🛍️ 2. Consumer Activity & Labor Market
    - Analyze RSAFS, UMCSENT, and UNRATE data.
    - Describe household sentiment and job market trends.

    ---

    ### 📊 3. Market Trend Implications
    - Summarize impact on stocks, bonds, and investor outlook.
    - Optional: Mention crypto/commodities if relevant.
    - Give forward-looking investment insight

    ---

    Here is the summarized data to reference:
    {summary}
    """.strip()
    return prompt