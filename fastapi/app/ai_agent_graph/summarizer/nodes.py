from openai import OpenAI
from .prompt import build_overall_market_prompt
from async_db import get_db
from typing import Dict

client = OpenAI()

async def get_inputs(state: Dict) -> Dict:
    # Inputs are expected from the frontend already populated
    return state

def generate_overall_report(state: Dict) -> Dict:
    prompt = build_overall_market_prompt(state)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a senior investment strategist who synthesizes macroeconomic, market sentiment, and stock analysis."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content
    return {**state, "final_report_en": content}

def translate_to_zh(state: Dict) -> Dict:
    if state.get("language", "").lower() not in ["chinese", "zh", "中文"]:
        return state  # Skip translation if not Chinese

    prompt = f"Translate the following investment report into Traditional Chinese:\n\n{state['final_report_en']}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a professional financial translator."},
            {"role": "user", "content": prompt}
        ]
    )

    return {**state, "final_report": response.choices[0].message.content}