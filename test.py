import requests

EIA_API_KEY = "keugXTLVkckSYikGBemgeOOtlYrt3oINtuVVgW2O"

url = f"https://api.eia.gov/series/?api_key={EIA_API_KEY}&series_id=PET.RBRTE.D"
response = requests.get(url)
print(response.status_code)
print(response.json())