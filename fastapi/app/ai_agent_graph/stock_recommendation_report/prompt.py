from typing import Dict, Any

def build_prompt(state: Dict[str, Any]) -> Dict[str, Any]:
    holdings = state.get("holdings", [])
    recommended = state.get("recommended", [])
    style = ", ".join(state.get("style_preference", []))
    risk = state.get("risk_tolerance", "")
    df = state.get("stock_df")

    table = df.to_string(index=False) if not df.empty else "(No data available)"

    prompt = (
    "You are a senior investment advisor. Based on the following investor profile and stock information, provide a professional and tailored investment analysis.\n\n"
    f"[Current Holdings] {', '.join(holdings)}\n"
    f"[Recommended Stocks] {', '.join(recommended)}\n"
    f"[Recommended Stock Data]\n{table}\n\n"
    f"[Investment Style] {style}\n[Risk Tolerance] {risk}\n\n"
    "Your analysis should follow this structure:\n\n"
    "1. **Risk Compatibility Check**: Evaluate whether the current holdings align with the investor's risk tolerance. If any holding is too aggressive or too conservative, explain why and suggest more suitable alternatives from related sectors or styles.\n\n"
    "2. **Current Holdings Review**: Briefly assess the strengths and weaknesses of the current holdings in terms of risk, return potential, and diversification.\n\n"
    "3. **Recommended Stocks Analysis**: For each recommended stock, explain why it fits the investor’s profile. Focus on fundamentals, sector outlook, and risk-adjusted return.\n\n"
    "4. **Alternatives & Similar Stocks**: Suggest other stocks that are either similar in profile or from the same sector, especially if they offer a better risk fit.\n\n"
    "5. **Diversification Tips**: Recommend 2–3 additional S&P 500 stocks from different sectors to improve portfolio diversification.\n\n"
    "6. **Portfolio Allocation Strategy**: Based on the investor’s risk tolerance, propose a portfolio breakdown (e.g., % in growth vs. defensive), and highlight any key risks.\n\n"
    "Conclude with a clear investment strategy recommendation aligned with the investor’s goals and risk capacity."
)

    return {**state, "prompt": prompt}