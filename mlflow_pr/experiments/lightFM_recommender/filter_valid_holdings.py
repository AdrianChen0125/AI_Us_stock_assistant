import pandas as pd

# === 1. 載入資料 ===

# 使用者持倉資料（原始）
holdings_df = pd.read_csv("all_investor_holdings.csv")

# 美股股票資料
stocks_df = pd.read_csv("nasdaq_tickers.csv")
stock_symbols = set(stocks_df["Symbol"].str.upper().str.strip())

# ETF 資料
etf_df = pd.read_csv("nasdaq_etf.csv")
etf_symbols = set(etf_df["SYMBOL"].str.upper().str.strip())

# 合併所有合法 symbol
valid_symbols = stock_symbols.union(etf_symbols)

# === 2. 過濾持倉 ===

# 處理 symbol 欄位，轉大寫與去空白
holdings_df["symbol"] = holdings_df["symbol"].str.upper().str.strip()

# 保留 symbol 存在於股票或 ETF 清單的項目
filtered_df = holdings_df[holdings_df["symbol"].isin(valid_symbols)]

# === 3. 輸出結果 ===
filtered_df.to_csv("filtered_user_holdings.csv", index=False)
print(f"✅ 篩選完成，共 {len(filtered_df)} 筆資料符合條件。")