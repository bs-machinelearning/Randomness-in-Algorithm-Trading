"""
P7 Adaptive Adversary Framework V2 - Policy Distinguishability

Task: "Can adversary distinguish Baseline vs Uniform trades?"
Metric: AUC for binary classification

INVERTED LOGIC:
- High AUC (>0.65) → Policies distinguishable → INCREASE randomization
- Low AUC (<0.55) → Policies indistinguishable (too random?) → DECREASE randomization
- Target AUC: [0.55, 0.65] → Sweet spot

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import json

from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.policies.baseline import generate_trades as baseline_generate
from bsml.data.loader import load_prices

# Import V2 modules (relative imports)
from .bridge_v2 import prepare_adversary_data_v2, time_split_data
from .adversary_classifier_v2 import P7AdversaryV2


class AdaptiveConfigV2:
    """Configuration for adaptive training loop V2"""
    
    # INVERTED THRESHOLDS: Lower AUC = Better
    AUC_HIGH_THRESHOLD = 0.65  # If above this, policies too distinguishable
    AUC_LOW_THRESHOLD = 0.55   # If below this, too indistinguishable
    AUC_TARGET_MIN = 0.55
    AUC_TARGET_MAX = 0.65
    
    # Adjustment factors
    FACTOR_INCREASE = 1.25     # Increase noise when AUC too high
    FACTOR_DECREASE = 0.80     # Decrease noise when AUC too low
    FACTOR_NUDGE = 1.10        # Small adjustments
    
    # Loop parameters
    MAX_ITERATIONS = 10
    CONVERGENCE_PATIENCE = 3
    MIN_VAL_SAMPLES = 100
    
    # Classifier parameters
    USE_CV = True
    N_CV_FOLDS = 5
    
    # Data split
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    # Parameter bounds
    PRICE_NOISE_MIN = 0.005
    PRICE_NOISE_MAX = 0.25
    TIME_NOISE_MIN = 5
    TIME_NOISE_MAX = 240
    
    OUTPUT_DIR = Path("outputs/adaptive_runs/uniform_v2_pilot")


def decide_adjustment_v2(auc: float, config: AdaptiveConfigV2 = None):
    """
    Decide parameter adjustment based on AUC.
    
    INVERTED LOGIC: Lower AUC = Better randomization
    
    Args:
        auc: AUC score from adversary
        config: Configuration object
    
    Returns:
        (action, multiplier, reason)
    """
    if config is None:
        config = AdaptiveConfigV2()
    
    if auc > config.AUC_HIGH_THRESHOLD:
        # Policies are too distinguishable - need MORE randomization
        return 'INCREASE', config.FACTOR_INCREASE, f'Too distinguishable (AUC={auc:.3f})'
    
    elif auc < config.AUC_LOW_THRESHOLD:
        # Policies are too indistinguishable - might be hurting returns
        return 'DECREASE', config.FACTOR_DECREASE, f'Too indistinguishable (AUC={auc:.3f})'
    
    elif config.AUC_TARGET_MIN <= auc <= config.AUC_TARGET_MAX:
        # In target range - hold
        return 'HOLD', 1.0, f'In target range (AUC={auc:.3f})'
    
    elif auc > config.AUC_TARGET_MAX:
        # Slightly above target
        return 'NUDGE_UP', config.FACTOR_NUDGE, f'Slightly distinguishable (AUC={auc:.3f})'
    
    else:
        # Slightly below target
        return 'NUDGE_DOWN', 1.0 / config.FACTOR_NUDGE, f'Slightly indistinguishable (AUC={auc:.3f})'


def adjust_parameters(params: Dict, multiplier: float, config: AdaptiveConfigV2 = None) -> Dict:
    """
    Adjust randomization parameters with safety bounds.
    
    Args:
        params: Current parameters dict
        multiplier: Adjustment multiplier
        config: Configuration object
    
    Returns:
        New parameters dict
    """
    if config is None:
        config = AdaptiveConfigV2()
    
    new_params = params.copy()
    
    new_params['price_noise'] = np.clip(
        new_params['price_noise'] * multiplier,
        config.PRICE_NOISE_MIN,
        config.PRICE_NOISE_MAX
    )
    
    new_params['time_noise_minutes'] = np.clip(
        new_params['time_noise_minutes'] * multiplier,
        config.TIME_NOISE_MIN,
        config.TIME_NOISE_MAX
    )
    
    return new_params


class IterationLoggerV2:
    """Track metrics across iterations for V2"""
    
    def __init__(self):
        self.iterations = []
        self.start_time = datetime.now()
    
    def log_iteration(self, iteration, params, auc, action, multiplier, reason,
                     train_metrics, val_metrics, cv_scores=None):
        """Log iteration metrics"""
        
        cv_mean = None
        cv_std = None
        if cv_scores is not None:
            if isinstance(cv_scores, (list, np.ndarray)) and len(cv_scores) > 0:
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))
        
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'price_noise': float(params['price_noise']),
            'time_noise_minutes': float(params['time_noise_minutes']),
            'auc_val': float(auc),
            'auc_cv_mean': cv_mean,
            'auc_cv_std': cv_std,
            'action': action,
            'multiplier': float(multiplier),
            'reason': reason,
            'n_train': train_metrics.get('n_samples', 0),
            'n_val': val_metrics.get('n_samples', 0),
            'n_features': train_metrics.get('n_features', 0),
            'train_baseline_pct': (
                train_metrics['label_distribution']['baseline'] / train_metrics['n_samples'] * 100
                if train_metrics.get('n_samples') else 0
            ),
            'val_baseline_pct': (
                val_metrics['label_distribution']['baseline'] / val_metrics['n_samples'] * 100
                if val_metrics.get('n_samples') else 0
            ),
            'val_accuracy': val_metrics.get('accuracy', None),
            'val_f1': val_metrics.get('f1', None),
        }
        self.iterations.append(entry)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        return pd.DataFrame(self.iterations)
    
    def print_summary(self):
        """Print summary table"""
        if not self.iterations:
            print("\n[No iterations completed]")
            return
        
        df = self.to_dataframe()
        summary_cols = ['iteration', 'auc_val', 'action', 'price_noise', 'time_noise_minutes']
        
        print("\n" + "="*80)
        print("ITERATION SUMMARY")
        print("="*80)
        print(df[summary_cols].to_string(index=False, float_format='%.4f'))
    
    def save_results(self, output_dir: Path):
        """Save results to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        df.to_csv(output_dir / "adaptive_results_v2.csv", index=False)
        
        with open(output_dir / "adaptive_results_v2.json", 'w') as f:
            json.dump(self.iterations, f, indent=2)
        
        summary = {
            'metadata': {
                'task': 'policy_distinguishability',
                'classifier': 'gradient_boosting',
                'approach': 'baseline_vs_uniform',
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            },
            'results': {
                'total_iterations': len(self.iterations),
                'final_auc': float(self.iterations[-1]['auc_val']) if self.iterations else None,
                'final_params': {
                    'price_noise': float(self.iterations[-1]['price_noise']),
                    'time_noise_minutes': float(self.iterations[-1]['time_noise_minutes'])
                } if self.iterations else None,
                'auc_trajectory': [float(x['auc_val']) for x in self.iterations],
                'actions': [x['action'] for x in self.iterations],
            }
        }
        
        with open(output_dir / "summary_v2.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  ✓ Results saved to {output_dir}")


def adaptive_training_loop_v2(
    prices_df: pd.DataFrame,
    initial_params: Optional[Dict] = None,
    config: Optional[AdaptiveConfigV2] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Adaptive training loop V2 - Policy Distinguishability
    
    Args:
        prices_df: Price data
        initial_params: Initial Uniform policy parameters
        config: Configuration object
        seed: Random seed
        verbose: Print diagnostics
    
    Returns:
        Dict with results
    """
    
    if config is None:
        config = AdaptiveConfigV2()
    
    if initial_params is None:
        initial_params = DEFAULT_UNIFORM_PARAMS.copy()
    
    params = initial_params.copy()
    logger = IterationLoggerV2()
    
    hold_count = 0
    converged = False
    
    if verbose:
        print("\n" + "="*80)
        print("P7 ADAPTIVE ADVERSARY V2 - POLICY DISTINGUISHABILITY")
        print("="*80)
        print(f"Task: Can adversary distinguish Baseline vs Uniform trades?")
        print(f"Classifier: Gradient Boosting")
        print(f"Max iterations: {config.MAX_ITERATIONS}")
        print(f"Convergence patience: {config.CONVERGENCE_PATIENCE}")
        print(f"AUC target: [{config.AUC_TARGET_MIN}, {config.AUC_TARGET_MAX}]")
        print(f"Initial params: price_noise={params['price_noise']:.4f}, time_noise={params['time_noise_minutes']:.1f}min")
        print(f"\nNote: Lower AUC = Better randomization (policies harder to distinguish)")
    
    for iteration in range(config.MAX_ITERATIONS):
        iter_num = iteration + 1
        
        if verbose:
            print("\n" + "="*80)
            print(f"ITERATION {iter_num}/{config.MAX_ITERATIONS}")
            print("="*80)
        
        try:
            # Create uniform policy with current params
            uniform_policy = UniformPolicy(params=params, seed=seed)
            
            if verbose:
                print("[1/5] Generating trades from both policies...")
            
            # Generate adversary data (baseline + uniform)
            adversary_data = prepare_adversary_data_v2(
                prices_df,
                baseline_generate,
                uniform_policy,
                verbose=verbose
            )
            
            if len(adversary_data) < 100:
                print(f"  ✗ Too few observations ({len(adversary_data)})")
                break
            
            if verbose:
                print("[2/5] Splitting data chronologically...")
            
            train, val, test = time_split_data(adversary_data, config.TRAIN_RATIO, config.VAL_RATIO)
            
            if verbose:
                print(f"  → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
            
            if len(val) < config.MIN_VAL_SAMPLES:
                print(f"  ✗ Val set too small ({len(val)})")
                break
            
            if verbose:
                print("[3/5] Training adversary classifier...")
            
            adversary = P7AdversaryV2(
                use_cv=config.USE_CV,
                n_cv_folds=config.N_CV_FOLDS,
                random_state=seed
            )
            
            train_metrics = adversary.train(train, verbose=verbose)
            
            if not train_metrics.get('success', False):
                print(f"  ✗ Training failed: {train_metrics.get('reason')}")
                break
            
            if verbose:
                print("[4/5] Evaluating distinguishability...")
            
            val_metrics = adversary.evaluate(val, verbose=verbose)
            
            if not val_metrics.get('success', False):
                print(f"  ✗ Evaluation failed")
                break
            
            auc = val_metrics['auc']
            
            if verbose:
                print("[5/5] Deciding adjustment...")
            
            action, multiplier, reason = decide_adjustment_v2(auc, config)
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"RESULT: AUC = {auc:.4f}")
                print(f"ACTION: {action}")
                print(f"REASON: {reason}")
            
            logger.log_iteration(
                iter_num, params, auc, action, multiplier, reason,
                train_metrics, val_metrics,
                adversary.cv_scores if hasattr(adversary, 'cv_scores') else None
            )
            
            if action == 'HOLD':
                hold_count += 1
                if verbose:
                    print(f"✓ In target range ({hold_count}/{config.CONVERGENCE_PATIENCE})")
                
                if hold_count >= config.CONVERGENCE_PATIENCE:
                    converged = True
                    if verbose:
                        print(f"\n🎉 CONVERGED after {iter_num} iterations!")
                    break
            
            else:
                hold_count = 0
                params = adjust_parameters(params, multiplier, config)
                
                if verbose:
                    print(f"\nAdjustment:")
                    print(f"  price_noise: {params['price_noise']:.4f}")
                    print(f"  time_noise: {params['time_noise_minutes']:.1f} min")
        
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if verbose:
        logger.print_summary()
        
        print("\n" + "="*80)
        print("FINAL STATUS")
        print("="*80)
        print(f"Converged: {converged}")
        print(f"Iterations: {len(logger.iterations)}")
        if logger.iterations:
            final_auc = logger.iterations[-1]['auc_val']
            print(f"Final AUC: {final_auc:.4f}")
            print(f"Final params: price_noise={params['price_noise']:.4f}, time_noise={params['time_noise_minutes']:.1f}min")
            
            if final_auc < 0.55:
                print("\n✓ Excellent: Policies are indistinguishable")
            elif final_auc < 0.65:
                print("\n✓ Good: Acceptable distinguishability level")
            else:
                print("\n⚠️  Warning: Policies still too distinguishable")
    
    return {
        'logger': logger,
        'final_params': params,
        'converged': converged,
        'n_iterations': len(logger.iterations)
    }


def main():
    """Main entry point for V2 pilot"""
    print("="*80)
    print("P7 WEEK 3 PILOT: ADAPTIVE ADVERSARY V2")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nApproach: Policy Distinguishability (Baseline vs Uniform)")
    print("Task: Train adversary to distinguish which policy generated trades")
    print("Goal: Lower AUC = Better randomization")
    
    config = AdaptiveConfigV2()
    
    print("\n[1/3] Loading data...")
    try:
        prices = load_prices("data/ALL_backtest.csv")
        print(f"  ✓ {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return
    
    print("\n[2/3] Running adaptive loop...")
    try:
        results = adaptive_training_loop_v2(prices, config=config, seed=42, verbose=True)
    except Exception as e:
        print(f"\n✗ FATAL: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[3/3] Saving results...")
    try:
        results['logger'].save_results(config.OUTPUT_DIR)
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
    
    print("\n" + "="*80)
    if results['converged']:
        print("✅ CONVERGED")
    else:
        print("✅ MAX ITERATIONS")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if results['logger'].iterations:
        final_auc = results['logger'].iterations[-1]['auc_val']
        print(f"\nFinal AUC: {final_auc:.4f}")
        
        if final_auc < 0.55:
            print("🎉 Excellent: Randomization makes policies indistinguishable!")
        elif final_auc < 0.65:
            print("✓ Good: Acceptable level of randomization")
        else:
            print("⚠️  Needs more work: Policies still distinguishable")


if __name__ == "__main__":
    main()
