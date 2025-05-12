import os
import openai
import anthropic

USE_CLAUDE = os.getenv("USE_CLAUDE", "true").lower() == "true"
openai.api_key = os.getenv("OPENAI_API_KEY")
claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

async def call_llm_report(summary: str,language: str) -> str:

    prompt = f"""
    You are a professional macroeconomic and investment analyst.
    Based on the following summarized data of macroeconomic indicators and market movements, please generate a comprehensive yet concise economic report in {language}. Your report should be insightful, logically structured, and suitable for use in investment briefings or portfolio strategy discussions.

    Structure your analysis into the following sections:

    ---

    ### 📈 1. Inflation and Interest Rate Outlook
    - Analyze the recent CPI, Federal Funds Rate (FEDFUNDS), and 10-Year Treasury Yield (GS10) data.
    - Comment on inflation trends and monetary policy direction (e.g., rate hikes, pauses, easing).
    - Highlight potential inflationary or disinflationary risks ahead.

    ---

    ### 🛍️ 2. Consumer Activity & Labor Market
    - Interpret trends in retail sales (RSAFS), consumer sentiment (UMCSENT), and unemployment rate (UNRATE).
    - Assess household confidence, spending power, and job market resilience or weakness.
    - Indicate whether the economy appears to be expanding, slowing, or in transition.

    ---

    ### 📊 3. Market Trend Implications
    - Summarize how macroeconomic trends may affect equity indices, fixed income markets, and investor risk appetite.
    - Optionally include insights on crypto assets or commodities if market price data suggests momentum or reversal.
    - Provide clear investment insights or forward-looking implications.

    ---

    Here is the summarized data to reference:
    {summary}
    """

    if USE_CLAUDE:
        res = claude_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.content

    else:
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return res["choices"][0]["message"]["content"]