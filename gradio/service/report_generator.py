import requests
import json 

API_BASE = "http://fastapi:8000"

def get_economic_report(user_profile_state, token=None):
    try:
        language = user_profile_state.get("language", "English")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        payload = {"language": language}

        res = requests.post(f"{API_BASE}/AI/economic_report", json=payload, headers=headers)
        res.raise_for_status()

        data = res.json()
        report = data.get("report", "")

        
        clean_report = report.strip()

        return clean_report, report 

    except Exception as e:
        print(" Error fetching report:", e)
        return " Failed to fetch report.", None

def get_market_sentiment_report(token):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        res = requests.post(f"{API_BASE}/AI/sentiment_report", headers=headers)
        res.raise_for_status()
        report = res.json().get("report", "No content")
        return report, report  
    except Exception as e:
        return f"Error: {e}", None

def get_stock_recommendation_report(user_profile_state, recommended, token):
                
    # Turn holdings and recommended into list
    holdings  = user_profile_state.get("holdings")

    parsed_holdings = [s.strip().upper() for s in holdings.split(",") if s.strip()]
    parsed_recommended = [s.strip().upper() for s in recommended.split(",") if s.strip()]
    
    
    style_preference = user_profile_state.get("interest")
    if isinstance(style_preference, str):
        style_preference = [style_preference]
    elif not isinstance(style_preference, list):
        style_preference = []

    
    risk_tolerance = user_profile_state.get("risk", "Moderate")

    
    payload = {
        "holdings": parsed_holdings,
        "recommended": parsed_recommended,
        "style_preference": style_preference,
        "risk_tolerance": risk_tolerance
    }

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        res = requests.post(f"{API_BASE}/AI/stock_recommendation", json=payload, headers=headers)
        res.raise_for_status()
        result = res.json()["analysis"]
        return "\n\n".join(result.split("\n\n")), result
    except Exception as e:
        return f" Error: {str(e)}", ""
   
def generate_overall_report(economic_report,sentiment_report,recommedation_report,user_profile, token):
    payload = {
        "language": user_profile.get("language", "English"),
        "economic_summary": economic_report,
        "sentiment_summary": sentiment_report,
        "stock_summary": recommedation_report,
        "risk": user_profile.get("risk", "Moderate"),
    }

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        print("Sending payload:", payload)
        res = requests.post(
            f"{API_BASE}/AI/summerise_report",
            json=payload,
            headers=headers
        )
        res.raise_for_status()
        return res.json().get("report", "No report generated.")
    except Exception as e:
        return f" Error generating report: {e}"
    
