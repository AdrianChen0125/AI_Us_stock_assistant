import requests

API_BASE = "http://fastapi:8000"

def g4_recommend_multi(user_profile_state, interest_sectors=None, interest_asset_types=None, top_n=15):

    risk = user_profile_state.get("risk", "moderate")
    raw_interest = user_profile_state.get("interest")

    if interest_sectors is None:
        interest_sectors = ["technology"]


    if isinstance(raw_interest, str):
        interest_asset_types = [i.strip().lower() for i in raw_interest.strip("{}").split(",")]
    elif isinstance(raw_interest, list):
        interest_asset_types = [i.lower() for i in raw_interest]
    else:
        interest_asset_types = ["stock", "etf"]

    payload = {
        "risk": risk.lower(),
        "interest_sectors": [s.lower() for s in interest_sectors],
        "interest_asset_types": [a.lower() for a in interest_asset_types],
        "top_n": top_n
    }

    try:
        response = requests.post(f"{API_BASE}/recommend/", json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        recs = data.get("recommendations", [])
        return ",".join(recs) if recs else "沒有推薦結果"
        
    except Exception as e:
        return f"error：{str(e)}", ""