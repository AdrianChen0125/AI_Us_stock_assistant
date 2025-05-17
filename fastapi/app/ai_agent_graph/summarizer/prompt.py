def build_overall_market_prompt(state: dict) -> str:
    economic = state.get("economic_summary", "")
    sentiment = state.get("market_sentiment_summary", "")
    stock = state.get("stock_recommender_summary", "")

    return f"""
    Based on the following three reports, generate a well-structured, professional market and economic outlook:
    📘 Economic Summary:
    {economic}
    📊 Market Sentiment Report:
    {sentiment}
    📈 Stock Recommendation Summary:
    {stock}
    Your report (max ~600 words) should include:
    1. Macroeconomic Overview
    2. Market Sentiment Analysis
    3. Equity Recommendation Review
    4. Conclusion and Strategic Suggestions
    Use a clear, concise tone appropriate for investors with moderate financial knowledge.
    """ 