"""
Master Bridge: Baseline trades -> Master Adversary format

Converts baseline trades to daily signal format for training master adversary.

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np
from typing import Tuple


def trades_to_daily_signals(trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Convert trades to daily signal format"""
    trades_clean = trades_df.copy()
    
    if 'timestamp' in trades_clean.columns:
        trades_clean['date'] = pd.to_datetime(trades_clean['timestamp']).dt.date
    elif 'date' in trades_clean.columns:
        trades_clean['date'] = pd.to_datetime(trades_clean['date']).dt.date
    else:
        raise ValueError("No date/timestamp column")
    
    trades_clean['date'] = pd.to_datetime(trades_clean['date'])
    trades_clean['signal'] = 1
    trade_signals = trades_clean.groupby(['date', 'symbol'])['signal'].max().reset_index()
    
    prices_clean = prices_df.copy()
    prices_clean['date'] = pd.to_datetime(prices_clean['date']).dt.date
    prices_clean['date'] = pd.to_datetime(prices_clean['date'])
    
    price_dates = sorted(prices_clean['date'].unique())
    symbols = sorted(trades_clean['symbol'].unique())
    
    full_grid = pd.MultiIndex.from_product(
        [price_dates, symbols],
        names=['date', 'symbol']
    ).to_frame(index=False)
    
    signals_df = full_grid.merge(trade_signals, on=['date', 'symbol'], how='left')
    signals_df['signal'] = signals_df['signal'].fillna(0).astype(int)
    
    return signals_df


def enrich_with_price_features(signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Add price-based features"""
    prices_clean = prices_df.copy()
    prices_clean['date'] = pd.to_datetime(prices_clean['date']).dt.date
    prices_clean['date'] = pd.to_datetime(prices_clean['date'])
    
    price_col = None
    for col in ['close', 'price', 'adj_close', 'Close']:
        if col in prices_clean.columns:
            price_col = col
            break
    
    if price_col is None:
        raise ValueError(f"No price column found")
    
    enriched_list = []
    
    for symbol in signals_df['symbol'].unique():
        symbol_signals = signals_df[signals_df['symbol'] == symbol].copy()
        symbol_prices = prices_clean[prices_clean['symbol'] == symbol].copy()
        
        symbol_data = symbol_signals.merge(
            symbol_prices[['date', price_col]], 
            on='date', 
            how='left'
        )
        
        symbol_data.rename(columns={price_col: 'price'}, inplace=True)
        symbol_data = symbol_data.sort_values('date')
        symbol_data['price'] = symbol_data['price'].fillna(method='ffill')
        
        # Price momentum
        symbol_data['mom_5d'] = symbol_data['price'].pct_change(5)
        symbol_data['mom_10d'] = symbol_data['price'].pct_change(10)
        symbol_data['mom_20d'] = symbol_data['price'].pct_change(20)
        
        # Volatility
        returns = symbol_data['price'].pct_change()
        symbol_data['vol_10d'] = returns.rolling(10).std() * np.sqrt(252)
        symbol_data['vol_30d'] = returns.rolling(30).std() * np.sqrt(252)
        symbol_data['vol_60d'] = returns.rolling(60).std() * np.sqrt(252)
        
        # Price levels
        symbol_data['price_ma_20'] = symbol_data['price'].rolling(20).mean()
        symbol_data['price_ma_50'] = symbol_data['price'].rolling(50).mean()
        symbol_data['price_vs_ma20'] = symbol_data['price'] / (symbol_data['price_ma_20'] + 1e-8)
        
        # Signal history
        symbol_data['signal_lag1'] = symbol_data['signal'].shift(1)
        symbol_data['signal_lag2'] = symbol_data['signal'].shift(2)
        symbol_data['trades_last_5d'] = symbol_data['signal'].rolling(5).sum()
        
        enriched_list.append(symbol_data)
    
    enriched_df = pd.concat(enriched_list, ignore_index=True)
    
    # Time features
    enriched_df['day_of_week'] = enriched_df['date'].dt.dayofweek
    enriched_df['month'] = enriched_df['date'].dt.month
    enriched_df['is_monday'] = (enriched_df['day_of_week'] == 0).astype(int)
    enriched_df['is_friday'] = (enriched_df['day_of_week'] == 4).astype(int)
    
    return enriched_df


def prepare_adversary_data(trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Main pipeline: trades -> adversary-ready format"""
    signals_df = trades_to_daily_signals(trades_df, prices_df)
    enriched_df = enrich_with_price_features(signals_df, prices_df)
    
    # Create labels (predict next day)
    enriched_df = enriched_df.sort_values(['symbol', 'date'])
    enriched_df['label'] = enriched_df.groupby('symbol')['signal'].shift(-1)
    
    enriched_df = enriched_df.dropna(subset=['label'])
    enriched_df['label'] = enriched_df['label'].astype(int)
    enriched_df = enriched_df.dropna()
    
    return enriched_df


def time_split_data(data_df: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple:
    """Chronological split"""
    df = data_df.sort_values('date').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    return train, val, test
