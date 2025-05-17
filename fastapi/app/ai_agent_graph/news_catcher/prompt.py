# prompt.py

FILTER_TOPICS_PROMPT = """
You are an intelligent assistant tasked with analyzing trending discussion topics.

Your job has two steps:
1. Filter out overlapping, redundant, or very similar topics from the list below. Keep only the most distinct and representative ones.
2. For each selected topic, generate a concise, keyword-friendly version suitable for Google or Wikipedia search. Avoid special characters, emojis, hashtags, or full sentences.

Do NOT add any dates or years (like 2021 or 2025) unless they are explicitly mentioned in the original topic.

Respond ONLY with raw JSON, using this format:
[
  {{
    "topic": "Why is Apple Stock Surging?",
    "search_query": "apple stock price surge"
  }},
  ...
]

Topics:
{topics}
"""

SUMMARIZE_RESULT_PROMPT = """
You are an AI research assistant. Your job is to summarize real-world information from search results.

Topic: "{topic}"

Content:
{content}

Summarize the most important facts and insights from the content above in 3–5 sentences. Be specific and informative — highlight what this topic is about, why it matters, and what key details were found.

- Focus on real data, quotes, or specific points mentioned in the content.
- Avoid vague language or generalizations.
- Do NOT speculate or add assumptions.
- Write clearly and objectively as if preparing a briefing for an analyst.

Return a single, concise paragraph.
"""