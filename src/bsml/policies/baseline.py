import numpy as np
import pandas as pd


def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline trading policy.

    - Input: prices with columns ['date', 'symbol', 'price']
    - For each symbol & day:
        * compute daily return
        * if return >= 0 → BUY (qty = +1)
        * if return < 0  → SELL (qty = -1)
    - Output trades DataFrame with:
        ['date', 'symbol', 'side', 'qty', 'price', 'ref_price']
    """

    # Work on a copy and keep only the needed columns
    df = prices.copy()
    df = df[["date", "symbol", "price"]].copy()

    # Ensure proper types
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Drop rows with bad / missing prices
    df = df.dropna(subset=["price"])

    # Sort by symbol then date
    df = df.sort_values(["symbol", "date"])

    # Compute daily returns per symbol
    df["ret"] = df.groupby("symbol")["price"].pct_change()

    # Simple signal:
    # - First day per symbol: treat as ret = 0 → BUY
    # - ret >= 0 → BUY (qty = +1)
    # - ret < 0  → SELL (qty = -1)
    ret_filled = df["ret"].fillna(0.0)
    qty = np.where(ret_filled >= 0.0, 1.0, -1.0)

    df["qty"] = qty
    df["side"] = np.where(df["qty"] > 0, "BUY", "SELL")

    # Build final trades DataFrame
    trades = pd.DataFrame(
        {
            "date": df["date"],
            "symbol": df["symbol"],
            "side": df["side"],
            "qty": df["qty"].astype(float),
            "price": df["price"].astype(float),       # used by cost model
            "ref_price": df["price"].astype(float),   # policies can override this
        }
    )

    return trades
