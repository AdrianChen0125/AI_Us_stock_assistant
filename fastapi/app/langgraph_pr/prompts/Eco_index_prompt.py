import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def call_llm_report(summary: str, language: str) -> str:
    prompt = f"""
        You are a professional macroeconomic and investment analyst.
        Based on the following summarized data of macroeconomic indicators and market movements, please generate a concise economic report in **{language}**. The report will be used for investment briefings and strategy sessions.
        Structure your analysis into the following sections:
        ---
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
        """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()

