# prompt.py
from typing import Dict, Any

# Node to construct the prompt based on user input and stock data
def build_prompt(state: Dict[str, Any]) -> Dict[str, Any]:
    holdings = state.get("holdings", [])
    recommended = state.get("recommended", [])
    style = ", ".join(state.get("style_preference", []))
    risk = state.get("risk_tolerance", "")
    df = state.get("stock_df")

    table = df.to_string(index=False) if not df.empty else "(No data available)"

    prompt = (
        "You are a senior investment advisor. Based on the following information, provide a clear and professional investment analysis:\n\n"
        f"[Current Holdings] {', '.join(holdings)}\n"
        f"[Recommended Stocks] {', '.join(recommended)}\n"
        f"[Recommended Stock Data]\n{table}\n\n"
        f"[Investment Preferences] {style}\n[Risk Tolerance] {risk}\n\n"
        "Please provide:\n"
        "- Correlation between the current holdings and the recommended stocks\n"
        "- A brief assessment of the current holdings and their strengths/weaknesses\n"
        "- A detailed explanation for each recommended stock, including why it fits the investor's profile\n"
        "- Suggest similar or alternative stocks within the same sector or with related characteristics\n"
        "- Include a few additional S&P 500 stocks from different sectors for diversification\n"
        "- Based on risk tolerance, recommend portfolio allocation and highlight key risks\n"
        "- Conclude with a clear recommendation strategy aligned with the investor’s profile"
    )

    return {**state, "prompt": prompt}