"""
P7 Bridge Module - FOR ADAPTIVE_LOOP_V1

Converts trades to adversary dataset format.
Prediction: "Will a trade occur tomorrow?"

Owner: P7
"""

import numpy as np
import pandas as pd


def prepare_adversary_data(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Prepare adversary training data.
    
    For each day, features = price data, label = 1 if trade tomorrow, 0 otherwise
    """
    
    if len(trades) == 0 or len(prices) == 0:
        return pd.DataFrame()
    
    trades = trades.copy()
    prices = prices.copy()
    
    # Ensure datetime
    trades['date'] = pd.to_datetime(trades['date'], dayfirst=True)
    prices['date'] = pd.to_datetime(prices['date'], dayfirst=True)
    
    # Sort
    trades = trades.sort_values(['symbol', 'date']).reset_index(drop=True)
    prices = prices.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    all_obs = []
    
    for symbol in prices['symbol'].unique():
        symbol_trades = trades[trades['symbol'] == symbol].copy()
        symbol_prices = prices[prices['symbol'] == symbol].copy()
        
        if len(symbol_prices) < 20:
            continue
        
        # Engineer features
        symbol_prices = symbol_prices.copy()
        symbol_prices['returns_1d'] = symbol_prices['price'].pct_change()
        symbol_prices['returns_5d'] = symbol_prices['price'].pct_change(5)
        symbol_prices['returns_20d'] = symbol_prices['price'].pct_change(20)
        symbol_prices['vol_5d'] = symbol_prices['returns_1d'].rolling(5).std()
        symbol_prices['vol_20d'] = symbol_prices['returns_1d'].rolling(20).std()
        symbol_prices['sma_5'] = symbol_prices['price'].rolling(5).mean()
        symbol_prices['sma_20'] = symbol_prices['price'].rolling(20).mean()
        symbol_prices['price_to_sma5'] = symbol_prices['price'] / (symbol_prices['sma_5'] + 1e-8)
        symbol_prices['price_to_sma20'] = symbol_prices['price'] / (symbol_prices['sma_20'] + 1e-8)
        symbol_prices['momentum_5'] = symbol_prices['returns_5d'] / (symbol_prices['vol_5d'] + 1e-8)
        
        # Create labels: 1 if trade occurs on THIS day, 0 otherwise
        trade_dates = set(symbol_trades['date'].dt.date)
        symbol_prices['label'] = symbol_prices['date'].dt.date.isin(trade_dates).astype(int)
        symbol_prices['signal'] = symbol_prices['label']
        
        all_obs.append(symbol_prices)
    
    if len(all_obs) == 0:
        return pd.DataFrame()
    
    result = pd.concat(all_obs, ignore_index=True).dropna()
    
    if verbose:
        n_positive = result['label'].sum()
        n_total = len(result)
        feature_cols = [c for c in result.columns 
                       if c not in ['date', 'symbol', 'label', 'signal', 'price']]
        
        print(f"[Bridge] Total observations: {n_total}")
        print(f"[Bridge] Trade days: {n_positive} ({n_positive/n_total*100:.1f}%)")
        print(f"[Bridge] No-trade days: {n_total - n_positive} ({(n_total-n_positive)/n_total*100:.1f}%)")
        print(f"[Bridge] Features: {len(feature_cols)}")
    
    return result
