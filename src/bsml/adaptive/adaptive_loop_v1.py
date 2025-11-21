"""
P7 Week 3: Adaptive Adversary v1.0 - Pilot on Uniform Policy
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / 'tests'))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.policies.utils import generate_policy_seed

from Skeleton_Adversary_Model import (
    extract_features,
    generate_labels,
    train_adversary_classifier,
    compute_auc,
    make_time_splits,
    ROLLING_WINDOWS
)

from bsml.data.loader import load_prices
from .bridge import enrich_trades_for_adversary


AUC_HIGH_THRESHOLD = 0.75
AUC_LOW_THRESHOLD = 0.55
AUC_TARGET_MIN = 0.60
AUC_TARGET_MAX = 0.70
AUC_TARGET_MIDPOINT = 0.65

ADJUSTMENT_FACTOR_INCREASE = 1.20
ADJUSTMENT_FACTOR_DECREASE = 0.80

DEFAULT_CONVERGENCE_PATIENCE = 5
DEFAULT_MAX_ITERATIONS = 20


def decide_adjustment(auc_score: float) -> Tuple[str, float]:
    if auc_score > AUC_HIGH_THRESHOLD:
        return 'INCREASE', ADJUSTMENT_FACTOR_INCREASE
    elif auc_score < AUC_LOW_THRESHOLD:
        return 'DECREASE', ADJUSTMENT_FACTOR_DECREASE
    elif AUC_TARGET_MIN <= auc_score <= AUC_TARGET_MAX:
        return 'HOLD', 1.0
    else:
        if auc_score > AUC_TARGET_MIDPOINT:
            return 'NUDGE_UP', ADJUSTMENT_FACTOR_INCREASE
        else:
            return 'NUDGE_DOWN', ADJUSTMENT_FACTOR_DECREASE


def adaptive_training_loop(
    prices_df: pd.DataFrame,
    initial_params: Optional[Dict] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    convergence_patience: int = DEFAULT_CONVERGENCE_PATIENCE,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    if initial_params is None:
        initial_params = DEFAULT_UNIFORM_PARAMS.copy()
    
    current_params = initial_params.copy()
    policy = UniformPolicy(params=current_params, seed=seed)
    
    auc_history = []
    param_history = []
    action_history = []
    
    in_target_count = 0
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
            print(f"Current params: {current_params}")
        
        if verbose:
            print("\n[1/7] Generating trades...")
        
        trades_df = policy.generate_trades(prices_df)
        
        if verbose:
            print(f"  Generated {len(trades_df)} trades")
        
        if len(trades_df) == 0:
            print("  ERROR: No trades generated. Stopping.")
            break
        
        if verbose:
            print("\n[2/7] Enriching trades for adversary...")
        
        enriched_trades = enrich_trades_for_adversary(
            trades_df, 
            prices_df,
            policy_id=f'adaptive_uniform_iter{iteration+1}'
        )
        
        if verbose:
            print(f"  Enriched to {len(enriched_trades)} rows with {len(enriched_trades.columns)} columns")
        
        if verbose:
            print("\n[3/7] Extracting features...")
        
        features_df = extract_features(enriched_trades, rolling_windows=ROLLING_WINDOWS)
        
        if verbose:
            print("\n[4/7] Generating labels...")
        
        labels = generate_labels(features_df, delta_steps=1)
        features_df['label'] = labels
        
        features_df = features_df.dropna(subset=['label'])
        
        if len(features_df) < 100:
            print(f"  WARNING: Only {len(features_df)} valid samples after cleaning. Stopping.")
            break
        
        if verbose:
            print("\n[5/7] Creating time splits...")
        
        train_df, val_df, test_df = make_time_splits(
            features_df,
            train_end="2024-06-30",
            val_end="2024-09-30"
        )
        
        if verbose:
            print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        if len(val_df) < 50:
            if verbose:
                print("  WARNING: Validation set too small. Using test set instead.")
            val_df = test_df
        
        if len(val_df) < 50:
            print("  ERROR: Not enough validation data. Stopping.")
            break
        
        exclude = {
            "timestamp", "symbol", "policy_id", "label", "pnl", 
            "side", "qty", "ref_price", "date", "exec_flag",
            "action_side", "action_size", "is_market_order"
        }
        
        feature_cols = [
            c for c in features_df.columns
            if c not in exclude and np.issubdtype(features_df[c].dtype, np.number)
        ]
        
        if verbose:
            print(f"  Using {len(feature_cols)} feature columns")
        
        if verbose:
            print("\n[6/7] Training adversary...")
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['label'].values
        
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            print("  WARNING: Labels are all 0 or all 1. Cannot train classifier.")
            print("  Using dummy AUC = 0.50 (random guessing)")
            auc_score = 0.50
        else:
            model = train_adversary_classifier(
                features=X_train,
                labels=y_train,
                model='histgb',
                params={'max_depth': 6, 'learning_rate': 0.05, 'max_iter': 300}
            )
            
            if verbose:
                print("\n[7/7] Computing validation AUC...")
            
            X_val = val_df[feature_cols].fillna(0)
            y_val = val_df['label'].values
            
            if y_val.sum() == 0 or y_val.sum() == len(y_val):
                print("  WARNING: Validation labels are all 0 or all 1.")
                print("  Using dummy AUC = 0.50")
                auc_score = 0.50
            else:
                auc_score = compute_auc(model, X_val, y_val)
        
        if verbose:
            print(f"\n>>> Validation AUC: {auc_score:.4f}")
        
        auc_history.append(auc_score)
        param_history.append(current_params.copy())
        
        action, multiplier = decide_adjustment(auc_score)
        
        action_history.append({
            'iteration': iteration + 1,
            'action': action,
            'auc': auc_score,
            'multiplier': multiplier,
            'params': current_params.copy()
        })
        
        if verbose:
            print(f">>> Action: {action} (multiplier: {multiplier:.2f})")
        
        if action == 'HOLD':
            in_target_count += 1
            if verbose:
                print(f">>> In target range ({in_target_count}/{convergence_patience})")
            
            if in_target_count >= convergence_patience:
                if verbose:
                    print(f"\n🎉 CONVERGED after {iteration + 1} iterations!")
                break
        else:
            in_target_count = 0
            
            current_params['price_noise'] *= multiplier
            current_params['time_noise_minutes'] *= multiplier
            
            current_params['price_noise'] = np.clip(
                current_params['price_noise'], 
                0.001,
                0.20
            )
            current_params['time_noise_minutes'] = np.clip(
                current_params['time_noise_minutes'],
                1,
                180
            )
            
            policy = UniformPolicy(params=current_params, seed=seed)
            
            if verbose:
                print(f">>> New params: {current_params}")
    
    results = {
        'final_params': current_params,
        'auc_history': auc_history,
        'param_history': param_history,
        'action_history': action_history,
        'converged': in_target_count >= convergence_patience,
        'n_iterations': len(auc_history)
    }
    
    return results


def main():
    print("="*70)
    print("P7 WEEK 3 PILOT: Adaptive Adversary on Uniform Policy")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n[Setup] Loading price data...")
    try:
        prices_df = load_prices("data/ALL_backtest.csv")
        print(f"  Loaded {len(prices_df)} price rows")
        print(f"  Symbols: {prices_df['symbol'].nunique()}")
        print(f"  Date range: {prices_df['date'].min()} to {prices_df['date'].max()}")
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return
    
    print("\n[Execution] Running adaptive training loop...")
    print(f"  Initial params: {DEFAULT_UNIFORM_PARAMS}")
    print(f"  Max iterations: 10")
    print(f"  Convergence patience: 3")
    
    try:
        results = adaptive_training_loop(
            prices_df,
            initial_params=DEFAULT_UNIFORM_PARAMS,
            max_iterations=10,
            convergence_patience=3,
            seed=42,
            verbose=True
        )
    except Exception as e:
        print(f"\n  ERROR in adaptive loop: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Converged: {results['converged']}")
    print(f"Iterations: {results['n_iterations']}")
    print(f"Final AUC: {results['auc_history'][-1]:.4f}")
    print(f"Final params: {results['final_params']}")
    print(f"\nAUC trajectory: {[f'{x:.3f}' for x in results['auc_history']]}")
    
    print("\n[Output] Saving results...")
    output_dir = Path("outputs/adaptive_runs/uniform_pilot")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    auc_df = pd.DataFrame({
        'iteration': range(1, len(results['auc_history']) + 1),
        'auc': results['auc_history']
    })
    auc_path = output_dir / "auc_trajectory.csv"
    auc_df.to_csv(auc_path, index=False)
    print(f"  Saved: {auc_path}")
    
    action_df = pd.DataFrame(results['action_history'])
    action_path = output_dir / "action_history.csv"
    action_df.to_csv(action_path, index=False)
    print(f"  Saved: {action_path}")
    
    param_df = pd.DataFrame(results['param_history'])
    param_df.insert(0, 'iteration', range(1, len(param_df) + 1))
    param_path = output_dir / "param_history.csv"
    param_df.to_csv(param_path, index=False)
    print(f"  Saved: {param_path}")
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Week 3 pilot complete!")


if __name__ == "__main__":
    main()
