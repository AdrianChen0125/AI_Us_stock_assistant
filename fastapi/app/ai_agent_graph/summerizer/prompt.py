def build_prompt(lang: str, user: dict, econ: str, sentiment: str, stock: str) -> str:
    age = user.get("age", "18-25")
    experience = user.get("experience", "Beginner")
    risk = user.get("risk", "Moderate")

    return (
        f"You are a seasoned investment strategist. Based on the user's profile and the following financial insights, write a well-structured, insightful, and forward-looking investment report. The style should reflect Warren Buffett's long-term value investing philosophy.\n\n"
        f"User Profile:\n"
        f"- Age: {age}\n"
        f"- Investment Experience: {experience}\n"
        f"- Risk Appetite: {risk}\n\n"
        f"Source Summaries:\n"
        f"- Macroeconomic Overview:\n{econ}\n\n"
        f"- Market Sentiment:\n{sentiment}\n\n"
        f"- Stock Recommendations:\n{stock}\n\n"
        f"Your report (around 600 words) should:\n"
        f"1. Integrate all sources — analyze how economic trends, investor sentiment, and stock selection relate to each other.\n"
        f"2. Highlight potential contradictions or opportunities — e.g., positive sentiment vs. weak economic signals.\n"
        f"3. Personalize recommendations based on the user's profile — tailor tone and strategy for their risk tolerance and experience.\n"
        f"4. Reinforce long-term, value-driven investing principles — quoting or paraphrasing Buffett where relevant (e.g., 'Be fearful when others are greedy').\n\n"
        f"Avoid simply summarizing. Deliver an analysis that flows logically, shows clear thinking, and helps the reader understand the implications for their investment strategy."
    )