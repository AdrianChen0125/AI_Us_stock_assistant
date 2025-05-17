# services/google_search.py

import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

API_KEY = os.getenv("GOOGLE_API_KEY")
CX = os.getenv("GOOGLE_CSE_ID")

MIN_SNIPPET_LENGTH = 10  # Skip junk results

def google_custom_search(query: str, max_results: int = 10) -> list:
    """
    Performs a Google Custom Search and returns clean, usable results.
    Each result includes: title, snippet, and link.
    """

    if not API_KEY or not CX:
        raise ValueError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID.")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CX,
        "q": query,
        "num": max_results
    }

    logger.info(f"Performing Google search for query: {query}")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        logger.error(f"Search failed: {response.status_code} - {response.text}")
        raise RuntimeError(f"Search failed: {response.status_code} {response.text}")

    data = response.json()
    if "items" not in data:
        logger.warning(f"No items found for query: {query}")
        return []

    results = []
    for item in data["items"]:
        title = item.get("title", "").strip()
        snippet = item.get("snippet", "").strip()
        link = item.get("link", "").strip()

        if len(snippet) < MIN_SNIPPET_LENGTH:
            continue  

        results.append({
            "title": title,
            "snippet": snippet,
            "link": link
        })

    logger.info(f"Found {len(results)} cleaned search results for query: {query}")
    return results