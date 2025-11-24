"""
Export Trade Strategies - Generate CSV Files

This script generates trades for each policy (Baseline, Uniform, Pink Noise, OU)
and exports them as separate CSV files for analysis.

Run with:
    cd ~/Randomness-in-Algorithm-Trading---BSML
    export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
    python src/bsml/adaptive/export_strategy_trades.py

Owner: P7
Week: 3
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.policies.pink_policy import PinkNoisePolicy, DEFAULT_PINK_PARAMS
from bsml.policies.ou_policy import OUPolicy, DEFAULT_OU_PARAMS
from bsml.policies.baseline import generate_trades as baseline_generate
from bsml.data.loader import load_prices


def export_baseline_trades(prices: pd.DataFrame, output_dir: Path, verbose: bool = True):
    """Generate and export baseline trades"""
    
    if verbose:
        print("\n" + "="*80)
        print("BASELINE POLICY")
        print("="*80)
    
    trades = baseline_generate(prices)
    
    if verbose:
        print(f"Generated {len(trades)} trades")
        print(f"Date range: {trades['date'].min()} to {trades['date'].max()}")
        print(f"Symbols: {trades['symbol'].nunique()}")
        print(f"\nFirst 5 trades:")
        print(trades.head())
    
    # Save to CSV
    output_file = output_dir / "baseline_trades.csv"
    trades.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\n✓ Saved to: {output_file}")
    
    return trades


def export_uniform_trades(prices: pd.DataFrame, output_dir: Path, 
                          price_noise: float = 0.03, time_noise: float = 30.0,
                          verbose: bool = True):
    """Generate and export uniform randomization trades"""
    
    if verbose:
        print("\n" + "="*80)
        print("UNIFORM POLICY")
        print("="*80)
        print(f"Parameters: price_noise={price_noise}, time_noise={time_noise}min")
    
    # Create policy
    uniform_params = DEFAULT_UNIFORM_PARAMS.copy()
    uniform_params['price_noise'] = price_noise
    uniform_params['time_noise_minutes'] = time_noise
    uniform_policy = UniformPolicy(params=uniform_params)
    
    # Generate trades
    trades = uniform_policy.generate_trades(prices)
    
    if verbose:
        print(f"Generated {len(trades)} trades")
        print(f"Date range: {trades['date'].min()} to {trades['date'].max()}")
        print(f"Symbols: {trades['symbol'].nunique()}")
        
        # Calculate statistics
        if 'ref_price' in trades.columns and 'price' in trades.columns:
            price_diff = (trades['ref_price'] - trades['price']).abs()
            print(f"\nPrice deviation stats:")
            print(f"  Mean: {price_diff.mean():.4f}")
            print(f"  Std:  {price_diff.std():.4f}")
            print(f"  Max:  {price_diff.max():.4f}")
        
        print(f"\nFirst 5 trades:")
        print(trades.head())
    
    # Save to CSV
    output_file = output_dir / "uniform_trades.csv"
    trades.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\n✓ Saved to: {output_file}")
    
    return trades


def export_pink_noise_trades(prices: pd.DataFrame, output_dir: Path,
                             alpha: float = 1.0, price_scale: float = 0.04,
                             verbose: bool = True):
    """Generate and export pink noise trades"""
    
    if verbose:
        print("\n" + "="*80)
        print("PINK NOISE POLICY")
        print("="*80)
        print(f"Parameters: alpha={alpha}, price_scale={price_scale}")
    
    # Create policy
    pink_policy = PinkNoisePolicy(alpha=alpha, price_scale=price_scale, seed=42)
    
    # Generate trades
    trades = pink_policy.generate_trades(prices)
    
    if verbose:
        print(f"Generated {len(trades)} trades")
        print(f"Date range: {trades['date'].min()} to {trades['date'].max()}")
        print(f"Symbols: {trades['symbol'].nunique()}")
        
        # Calculate statistics
        if 'ref_price' in trades.columns and 'price' in trades.columns:
            price_diff = (trades['ref_price'] - trades['price']).abs()
            print(f"\nPrice deviation stats:")
            print(f"  Mean: {price_diff.mean():.4f}")
            print(f"  Std:  {price_diff.std():.4f}")
            print(f"  Max:  {price_diff.max():.4f}")
        
        print(f"\nFirst 5 trades:")
        print(trades.head())
    
    # Save to CSV
    output_file = output_dir / "pink_noise_trades.csv"
    trades.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\n✓ Saved to: {output_file}")
    
    return trades


def export_ou_trades(prices: pd.DataFrame, output_dir: Path,
                    theta: float = 0.15, sigma: float = 0.02, price_scale: float = 1.0,
                    verbose: bool = True):
    """Generate and export Ornstein-Uhlenbeck trades"""
    
    if verbose:
        print("\n" + "="*80)
        print("ORNSTEIN-UHLENBECK POLICY")
        print("="*80)
        print(f"Parameters: theta={theta}, sigma={sigma}, price_scale={price_scale}")
    
    # Create policy
    ou_policy = OUPolicy(theta=theta, sigma=sigma, price_scale=price_scale, seed=42)
    
    # Generate trades
    trades = ou_policy.generate_trades(prices)
    
    if verbose:
        print(f"Generated {len(trades)} trades")
        print(f"Date range: {trades['date'].min()} to {trades['date'].max()}")
        print(f"Symbols: {trades['symbol'].nunique()}")
        
        # Calculate statistics
        if 'ref_price' in trades.columns and 'price' in trades.columns:
            price_diff = (trades['ref_price'] - trades['price']).abs()
            print(f"\nPrice deviation stats:")
            print(f"  Mean: {price_diff.mean():.4f}")
            print(f"  Std:  {price_diff.std():.4f}")
            print(f"  Max:  {price_diff.max():.4f}")
        
        print(f"\nFirst 5 trades:")
        print(trades.head())
    
    # Save to CSV
    output_file = output_dir / "ou_trades.csv"
    trades.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\n✓ Saved to: {output_file}")
    
    return trades


def generate_comparison_summary(baseline_trades, uniform_trades, pink_trades, ou_trades, output_dir: Path):
    """Generate a summary CSV comparing all strategies"""
    
    summary_data = []
    
    for strategy_name, trades in [
        ('Baseline', baseline_trades),
        ('Uniform', uniform_trades),
        ('Pink Noise', pink_trades),
        ('Ornstein-Uhlenbeck', ou_trades)
    ]:
        
        stats = {
            'Strategy': strategy_name,
            'Total_Trades': len(trades),
            'Unique_Symbols': trades['symbol'].nunique(),
            'Date_Start': trades['date'].min(),
            'Date_End': trades['date'].max(),
            'Avg_Quantity': trades['qty'].abs().mean() if 'qty' in trades.columns else 0,
            'Total_Buy': (trades['side'] == 'BUY').sum() if 'side' in trades.columns else 0,
            'Total_Sell': (trades['side'] == 'SELL').sum() if 'side' in trades.columns else 0,
        }
        
        # Add price statistics if available
        if 'ref_price' in trades.columns and 'price' in trades.columns:
            price_diff = (trades['ref_price'] - trades['price']).abs()
            stats['Avg_Price_Deviation'] = price_diff.mean()
            stats['Max_Price_Deviation'] = price_diff.max()
        else:
            stats['Avg_Price_Deviation'] = 0
            stats['Max_Price_Deviation'] = 0
        
        summary_data.append(stats)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    output_file = output_dir / "strategy_comparison_summary.csv"
    summary_df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print(f"\n✓ Saved to: {output_file}")
    
    return summary_df


def main():
    """Main execution"""
    
    print("="*80)
    print("TRADE STRATEGY EXPORT TOOL")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create output directory
    output_dir = Path("outputs/strategy_trades")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load price data
    print("\n[1/5] Loading price data...")
    prices = load_prices("data/ALL_backtest.csv")
    print(f"  ✓ Loaded {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    
    # Generate trades for each strategy
    print("\n[2/5] Generating Baseline trades...")
    baseline_trades = export_baseline_trades(prices, output_dir, verbose=True)
    
    print("\n[3/5] Generating Uniform trades...")
    uniform_trades = export_uniform_trades(
        prices, output_dir, 
        price_noise=0.047, 
        time_noise=47.0,  # Optimized parameters
        verbose=True
    )
    
    print("\n[4/5] Generating Pink Noise trades...")
    pink_trades = export_pink_noise_trades(
        prices, output_dir,
        alpha=1.0,
        price_scale=0.04,
        verbose=True
    )
    
    print("\n[5/5] Generating OU trades...")
    ou_trades = export_ou_trades(
        prices, output_dir,
        theta=0.15,
        sigma=0.02,
        price_scale=1.0,
        verbose=True
    )
    
    # Generate comparison summary
    print("\n[Summary] Generating comparison...")
    generate_comparison_summary(baseline_trades, uniform_trades, pink_trades, ou_trades, output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ EXPORT COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Generated files:")
    print(f"  1. {output_dir}/baseline_trades.csv")
    print(f"  2. {output_dir}/uniform_trades.csv")
    print(f"  3. {output_dir}/pink_noise_trades.csv")
    print(f"  4. {output_dir}/ou_trades.csv")
    print(f"  5. {output_dir}/strategy_comparison_summary.csv")
    print()
    print("Use these CSVs to:")
    print("  - Analyze trade patterns")
    print("  - Visualize price deviations")
    print("  - Compare execution timing")
    print("  - Validate randomization effects")


if __name__ == "__main__":
    main()
