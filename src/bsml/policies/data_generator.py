"""
Data Generator Module
Generates realistic synthetic ETF price data with proper correlations
"""

import numpy as np
import pandas as pd

def generate_etf_prices(config):
    """
    Generate synthetic ETF prices with realistic correlations and characteristics
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with universe, dates, and parameters
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, SPY, QQQ, IVV, VOO, VTI, EEM, GLD, TLT, XLF, EFA
    """
    
    # ETF characteristics (annual return %, annual volatility %, correlation to SPY)
    etf_params = {
        'SPY':  {'mu': 0.12, 'sigma': 0.16, 'corr_spy': 1.00},
        'QQQ':  {'mu': 0.15, 'sigma': 0.22, 'corr_spy': 0.85},
        'IVV':  {'mu': 0.12, 'sigma': 0.16, 'corr_spy': 0.99},
        'VOO':  {'mu': 0.12, 'sigma': 0.16, 'corr_spy': 0.99},
        'VTI':  {'mu': 0.13, 'sigma': 0.17, 'corr_spy': 0.95},
        'EEM':  {'mu': 0.06, 'sigma': 0.24, 'corr_spy': 0.65},
        'GLD':  {'mu': 0.08, 'sigma': 0.15, 'corr_spy': -0.10},
        'TLT':  {'mu': 0.02, 'sigma': 0.14, 'corr_spy': -0.30},
        'XLF':  {'mu': 0.10, 'sigma': 0.20, 'corr_spy': 0.90},
        'EFA':  {'mu': 0.09, 'sigma': 0.18, 'corr_spy': 0.85},
    }
    
    # Generate business day date range
    date_range = pd.date_range(
        start=config['start_date'], 
        end=config['end_date'], 
        freq='B'
    )
    n_days = len(date_range)
    
    # Create correlation matrix
    universe = config['universe']
    n_etfs = len(universe)
    corr_matrix = np.ones((n_etfs, n_etfs))
    
    for i, etf_i in enumerate(universe):
        for j, etf_j in enumerate(universe):
            if i != j:
                # Correlation through SPY
                corr_matrix[i, j] = (
                    etf_params[etf_i]['corr_spy'] * 
                    etf_params[etf_j]['corr_spy']
                )
    
    # Cholesky decomposition for correlation
    L = np.linalg.cholesky(corr_matrix)
    
    # Generate uncorrelated standard normal returns
    np.random.seed(42)
    raw_returns = np.random.standard_normal((n_days, n_etfs))
    
    # Apply correlation structure
    correlated_returns = raw_returns @ L.T
    
    # Build price series for each ETF
    prices_df = pd.DataFrame(index=date_range, columns=universe)
    
    for i, etf in enumerate(universe):
        params = etf_params[etf]
        
        # Convert annual to daily
        daily_mu = params['mu'] / 252
        daily_sigma = params['sigma'] / np.sqrt(252)
        
        # Scale returns
        returns = daily_mu + daily_sigma * correlated_returns[:, i]
        
        # Generate price series starting at 100
        price_series = 100 * np.exp(np.cumsum(returns))
        prices_df[etf] = price_series
    
    # Add date column
    prices_df['date'] = prices_df.index
    prices_df = prices_df.reset_index(drop=True)
    
    # Reorder columns: date first, then ETFs
    prices_df = prices_df[['date'] + universe]
    
    return prices_df


if __name__ == '__main__':
    # Test the generator
    import json
    
    with open('../config.json', 'r') as f:
        config = json.load(f)[0]
    
    prices = generate_etf_prices(config)
    print(prices.head())
    print(f"\nGenerated {len(prices)} days of data")
    print(f"\nCorrelation matrix:")
    print(prices[config['universe']].corr().round(2))
