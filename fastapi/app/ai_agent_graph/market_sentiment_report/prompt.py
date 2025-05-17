def build_market_sentiment_prompt(state: dict) -> str:
    reddit = state.get("reddit_df", "")
    sentiment = state.get("sentiment_summary", "")
    sector = state.get("sector_summary", "")
    symbol = state.get("symbol_summary", "")

    prompt = f"""
You are a professional market analyst. Based on the following sentiment data, write an insightful U.S. stock market sentiment report.

Reddit Sentiment (30 days): {reddit}
Topic Summary: {sentiment}
Top Sectors: {sector}
Top Symbols: {symbol}

Please organize your analysis into 4–5 concise paragraphs (~400–600 words) and include:
1. Market sentiment trends and key emotional drivers
2. Sector and symbol highlights
3. Emerging themes
4. Practical investment implications
5. A brief forward-looking conclusion
"""
    return prompt