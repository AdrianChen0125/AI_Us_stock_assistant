import pandas as pd
import csv
import os
import time
from playwright.sync_api import sync_playwright

INPUT_CSV = "etoro_investors.csv"
OUTPUT_CSV = "all_investor_holdings.csv"

# 讀取 username 清單
df = pd.read_csv(INPUT_CSV)
usernames = df["username"].dropna().unique()

# 檢查已儲存的資料，避開重複抓取
if os.path.exists(OUTPUT_CSV):
    done_df = pd.read_csv(OUTPUT_CSV)
    done_users = set(done_df["username"].unique())
    print(f" 已有 {len(done_users)} 位使用者資料，將略過")
else:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "username", "symbol", "symbol_name", "direction",
            "invested", "p/l(%)", "value"
        ])
        writer.writeheader()
    done_users = set()

def get_portfolio_holdings(username):
    url = f"https://www.etoro.com/people/{username}/portfolio"
    data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        try:
            page.goto(url, timeout=60000)
            page.wait_for_selector("[automation-id='cd-public-portfolio-table-item-title']", timeout=15000)
        except:
            print(f" 無法載入 {username} 的 portfolio")
            browser.close()
            return []

        symbols = page.query_selector_all("[automation-id='cd-public-portfolio-table-item-title']")
        symbol_names = page.query_selector_all("div.et-font-xxs.et-color-dark-grey.ellipsis")
        directions = page.query_selector_all("div.et-font-weight-normal:has-text('Long'), div.et-font-weight-normal:has-text('Short')")
        invested = page.query_selector_all("div.et-font-weight-normal.et-flex.justify-end.et-font-s.ng-star-inserted:nth-of-type(1)")
        profit_loss = page.query_selector_all("div.et-font-weight-normal.et-flex.justify-end.et-font-s.ng-star-inserted:nth-of-type(2)")
        value = page.query_selector_all("div.et-font-weight-normal.et-flex.justify-end.et-font-s.ng-star-inserted:nth-of-type(3)")

        for i in range(len(symbols)):
            row = {
                "username": username,
                "symbol": symbols[i].inner_text().strip() if i < len(symbols) else "",
                "symbol_name": symbol_names[i].inner_text().strip() if i < len(symbol_names) else "",
                "direction": directions[i].inner_text().strip() if i < len(directions) else "",
                "invested": invested[i].inner_text().strip() if i < len(invested) else "",
                "p/l(%)": profit_loss[i].inner_text().strip() if i < len(profit_loss) else "",
                "value": value[i].inner_text().strip() if i < len(value) else ""
            }
            data.append(row)

        browser.close()
    return data

# === 主迴圈：抓取並存檔 ===
for username in usernames:
    if username in done_users:
        print(f"⏩ 已抓過 {username}，略過")
        continue

    print(f"🔍 抓取 {username} 的持倉...")
    try:
        holdings = get_portfolio_holdings(username)
        if holdings:
            with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=holdings[0].keys())
                writer.writerows(holdings)
            print(f" {username} 已儲存，共 {len(holdings)} 筆持倉")
        else:
            print(f" {username} 無持倉資料")
    except Exception as e:
        print(f" 發生錯誤：{e}")
        continue

    time.sleep(2)  # 建議等待，避免被封鎖