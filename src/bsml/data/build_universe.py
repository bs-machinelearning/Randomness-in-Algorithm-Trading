#!/usr/bin/env python3
"""
Download 1Y daily OHLCV for 10 large ETFs using yfinance.
- Saves one CSV per ETF under data/etf_1y/
- Also writes a combined CSV: data/etf_1y/etfs_1y_combined.csv
Edit TICKERS if you want a different set.
"""

import os
from datetime import date
import pandas as pd
import yfinance as yf

# Approximate top 10 mega ETFs (by AUM/liquidity). Adjust as you like.
TICKERS = [
    "SPY",   # SPDR S&P 500
    "IVV",   # iShares Core S&P 500
    "VOO",   # Vanguard S&P 500
    "QQQ",   # Invesco QQQ
    "VTI",   # Vanguard Total Stock Market
    "VEA",   # Vanguard FTSE Developed Markets
    "VWO",   # Vanguard FTSE Emerging Markets
    "AGG",   # iShares Core U.S. Aggregate Bond
    "IEMG",  # iShares Core MSCI Emerging Markets
    "IJR",   # iShares Core S&P Small-Cap
]

OUT_DIR = os.path.join("data", "etf_1y")
os.makedirs(OUT_DIR, exist_ok=True)

def fetch_one(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df.rename(columns=str.lower).reset_index()
    df.insert(0, "ticker", ticker)
    # add simple daily return
    if "close" in df.columns:
        df["ret_1d"] = df["close"].pct_change()
    return df

def main():
    frames = []
    for t in TICKERS:
        print(f"Downloading {t} …")
        try:
            df = fetch_one(t)
        except Exception as e:
            print(f"  ERROR fetching {t}: {e}")
            continue
        if df.empty:
            print(f"  WARN: no data for {t}")
            continue
        out_path = os.path.join(OUT_DIR, f"{t}_1y.csv")
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({len(df)} rows)")
        frames.append(df)

    if frames:
        combo = pd.concat(frames, ignore_index=True)
        combo_path = os.path.join(OUT_DIR, "etfs_1y_combined.csv")
        combo.to_csv(combo_path, index=False)
        print(f"Combined CSV written: {combo_path} ({len(combo)} rows))")
    else:
        print("No data downloaded. Check internet/yfinance and retry.")

if __name__ == "__main__":
    main()