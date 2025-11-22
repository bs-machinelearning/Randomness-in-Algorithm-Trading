"""
P7 Bridge Module - FINAL FIX

Owner: P7
Week: 4
"""

import numpy as np
import pandas as pd


def prepare_adversary_data(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    prediction_window_hours: float = 4.0,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Convert trades to adversary prediction format.
    
    Key insight: Look FORWARD from each observation point.
    We sample observation points at regular intervals (1 hour) to avoid
    the issue of having a trade at every price timestamp.
    """
    
    if len(trades) == 0 or len(prices) == 0:
        return pd.DataFrame()
    
    # Ensure datetime
    trades = trades.copy()
    prices = prices.copy()
    trades['date'] = pd.to_datetime(trades['date'])
    prices['date'] = pd.to_datetime(prices['date'])
    
    trades = trades.sort_values('date').reset_index(drop=True)
    prices = prices.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    all_obs = []
    
    for symbol in prices['symbol'].unique():
        symbol_trades = trades[trades['symbol'] == symbol].copy()
        symbol_prices = prices[prices['symbol'] == symbol].copy()
        
        if len(symbol_prices) == 0:
            continue
        
        # Create hourly observation grid
        min_date = symbol_prices['date'].min()
        max_date = symbol_prices['date'].max()
        
        # Sample observations every hour
        obs_dates = pd.date_range(start=min_date, end=max_date, freq='1H')
        
        # For each observation time, get nearest price data for features
        obs_data = []
        for obs_time in obs_dates:
            # Get nearest price before or at obs_time
            prior_prices = symbol_prices[symbol_prices['date'] <= obs_time]
            if len(prior_prices) == 0:
                continue
            
            nearest_price = prior_prices.iloc[-1].copy()
            
            # Label: will trade occur in (obs_time, obs_time + window]?
            window_end = obs_time + pd.Timedelta(hours=prediction_window_hours)
            trades_in_window = symbol_trades[
                (symbol_trades['date'] > obs_time) & 
                (symbol_trades['date'] <= window_end)
            ]
            
            obs_row = nearest_price.to_dict()
            obs_row['date'] = obs_time
            obs_row['label'] = 1 if len(trades_in_window) > 0 else 0
            obs_row['signal'] = obs_row['label']
            
            obs_data.append(obs_row)
        
        if obs_data:
            symbol_df = pd.DataFrame(obs_data)
            all_obs.append(symbol_df)
    
    if len(all_obs) == 0:
        return pd.DataFrame()
    
    result = pd.concat(all_obs, ignore_index=True)
    
    if verbose:
        n_positive = result['label'].sum()
        n_total = len(result)
        print(f"[Bridge] Total observations: {n_total}")
        print(f"[Bridge] Positive (trade within {prediction_window_hours:.1f}h): {n_positive} ({n_positive/n_total*100:.1f}%)")
        print(f"[Bridge] Negative (no trade): {n_total - n_positive} ({(n_total-n_positive)/n_total*100:.1f}%)")
        
        feature_cols = [c for c in result.columns if c not in ['date', 'symbol', 'label', 'signal', 'price']]
        print(f"[Bridge] Features: {len(feature_cols)}")
    
    return result
