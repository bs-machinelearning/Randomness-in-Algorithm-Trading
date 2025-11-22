"""
P7 Bridge Module - FIXED VERSION

Owner: P7
Week: 4
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def prepare_adversary_data(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    prediction_window_hours: float = 4.0,
    auto_detect_window: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Convert trades to adversary prediction format with proper features.
    
    Prediction task: "Will a trade occur in the next X hours?"
    """
    
    if len(trades) == 0:
        return pd.DataFrame()
    
    # Ensure datetime
    trades = trades.copy()
    prices = prices.copy()
    trades['date'] = pd.to_datetime(trades['date'])
    prices['date'] = pd.to_datetime(prices['date'])
    
    # Sort
    trades = trades.sort_values('date').reset_index(drop=True)
    prices = prices.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Get all unique times from prices (these are our observation points)
    all_dates = prices[['date']].drop_duplicates().sort_values('date').reset_index(drop=True)
    
    # Create observation dataset
    all_obs = []
    
    for symbol in prices['symbol'].unique():
        symbol_trades = trades[trades['symbol'] == symbol].copy()
        symbol_prices = prices[prices['symbol'] == symbol].copy()
        
        if len(symbol_prices) == 0:
            continue
        
        # For each price observation, create features and label
        symbol_obs = symbol_prices.copy()
        symbol_obs['symbol'] = symbol
        
        # Label: will trade occur in next X hours?
        symbol_obs['label'] = 0
        
        for idx, row in symbol_obs.iterrows():
            obs_time = row['date']
            window_end = obs_time + pd.Timedelta(hours=prediction_window_hours)
            
            # Check if any trade occurs in (obs_time, window_end]
            # Note: EXCLUSIVE of obs_time, so we look forward
            trades_in_window = symbol_trades[
                (symbol_trades['date'] > obs_time) & 
                (symbol_trades['date'] <= window_end)
            ]
            
            if len(trades_in_window) > 0:
                symbol_obs.at[idx, 'label'] = 1
        
        symbol_obs['signal'] = symbol_obs['label']
        all_obs.append(symbol_obs)
    
    if len(all_obs) == 0:
        return pd.DataFrame()
    
    result = pd.concat(all_obs, ignore_index=True)
    
    if verbose:
        n_positive = result['label'].sum()
        n_total = len(result)
        print(f"[Bridge] Total observations: {n_total}")
        print(f"[Bridge] Positive (trade within {prediction_window_hours:.1f}h): {n_positive} ({n_positive/n_total*100:.1f}%)")
        print(f"[Bridge] Negative (no trade): {n_total - n_positive} ({(n_total-n_positive)/n_total*100:.1f}%)")
        
        # Show available features
        feature_cols = [c for c in result.columns if c not in ['date', 'symbol', 'label', 'signal', 'price']]
        if len(feature_cols) > 0:
            print(f"[Bridge] Available features: {len(feature_cols)} → {feature_cols[:5]}...")
        else:
            print(f"[Bridge] ⚠️ WARNING: No features found! Only columns: {list(result.columns)}")
    
    return result
