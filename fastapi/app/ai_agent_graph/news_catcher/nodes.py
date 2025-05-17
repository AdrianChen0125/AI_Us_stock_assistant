import logging
import json
from typing import Dict, Any, List
from openai import AsyncOpenAI
from langsmith.wrappers import wrap_openai
from .prompt import FILTER_TOPICS_PROMPT, SUMMARIZE_RESULT_PROMPT
from services.sentiment_topic_service import fetch_latest_sentiment_topics
from services.google_search import google_custom_search
from async_db import get_db
import asyncio
import re

logger = logging.getLogger(__name__)
openai_client = wrap_openai(AsyncOpenAI())

# Step 1: Fetch sentiment topics
async def get_sentiment_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    db_gen = get_db()
    db = await db_gen.__anext__()

    try:
        raw_topics = await fetch_latest_sentiment_topics(db)

        filtered_topic_names = [
            t["topic_summary"]
            for t in raw_topics
            if t.get("comments_count", 0) > 200
        ]

        logger.info(f"Found {len(filtered_topic_names)} topics with > 200 comments")
        return {**state, "sentiment_summary": filtered_topic_names}
    finally:
        await db.aclose()


# Step 2: AI-based topic filtering
def sanitize_topic(text: str) -> str:
    # Remove surrounding or embedded quotes
    text = text.replace('"', '').replace("'", "")
    # Optionally: remove emojis or hashtags
    text = re.sub(r"[#@][\w]+", "", text)  # removes hashtags/mentions
    text = re.sub(r"[^\w\s\-:]", "", text)  # removes emojis/symbols except dash/colon
    return text.strip()

async def filter_similar_topic(state: Dict[str, Any]) -> Dict[str, Any]:
    topic_names = state.get("sentiment_summary", [])
    prompt = FILTER_TOPICS_PROMPT.format(topics="\n".join(topic_names))

    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    try:
        raw = response.choices[0].message.content.strip()
        logger.debug(f"Raw LLM response: {raw}")

        json_start = raw.find("[")
        json_data = raw[json_start:] if json_start != -1 else raw

        parsed = json.loads(json_data)

        cleaned = []
        for entry in parsed:
            cleaned.append({
                "topic": sanitize_topic(entry.get("topic", "")),
                "search_query": sanitize_topic(entry.get("search_query", entry.get("topic", "")))
            })

        return {**state, "filtered_topics": cleaned}

    except Exception as e:
        logger.warning(f"Failed to parse or clean LLM response: {e}")
        fallback = [{"topic": sanitize_topic(t), "search_query": sanitize_topic(t)} for t in topic_names]
        return {**state, "filtered_topics": fallback}


async def multiple_search_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    topics = state.get("filtered_topics", [])[:5]
    results = []

    for topic in topics:
        try:
            search_results = await asyncio.to_thread(google_custom_search, topic)
        except Exception as e:
            logger.warning(f"Search failed for topic '{topic}': {e}")
            search_results = []

        results.append({
            "topic": topic,
            "search_results": search_results  
        })

    return {**state, "search_results": results}


# Step 4: Summarize search results
async def summarize_results(state: Dict[str, Any]) -> Dict[str, Any]:
    results = state.get("search_results", [])
    summaries = []

    for result in results:
        topic = result["topic"]
        entries = result["search_results"]

        if not entries:
            summary = f"No search results found for topic '{topic}'."
        else:
            combined_content = "\n\n".join(
                f"{e['title']}\n{e['snippet']}\n{e['link']}" for e in entries
            )

            prompt = SUMMARIZE_RESULT_PROMPT.format(topic=topic, content=combined_content)
            response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            summary = response.choices[0].message.content.strip()

        summaries.append({
            "topic": topic,
            "summary": summary,
            "search_count": len(entries)
        })

    logger.info("Summarized all search results")
    return {**state, "summarized_results": summaries}