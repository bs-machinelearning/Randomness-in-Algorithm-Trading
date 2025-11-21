"""
P7 Adaptive Adversary Framework - Week 3 Production Version

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.data.loader import load_prices
from bsml.adaptive.bridge import enrich_trades_for_adversary
from bsml.adaptive.adversary_classifier import P7AdaptiveAdversary, time_split_trades


class AdaptiveConfig:
    """Configuration for adaptive training loop"""
    
    AUC_HIGH_THRESHOLD = 0.75
    AUC_LOW_THRESHOLD = 0.55
    AUC_TARGET_MIN = 0.60
    AUC_TARGET_MAX = 0.70
    AUC_TARGET_MID = 0.65
    
    FACTOR_INCREASE = 1.20
    FACTOR_DECREASE = 0.80
    FACTOR_NUDGE = 1.10
    
    MAX_ITERATIONS = 10
    CONVERGENCE_PATIENCE = 3
    MIN_VAL_SAMPLES = 100
    
    PREDICTION_WINDOW_MINUTES = 10
    USE_SMOTE = True
    USE_CV = True
    N_CV_FOLDS = 3
    
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    PRICE_NOISE_MIN = 0.001
    PRICE_NOISE_MAX = 0.25
    TIME_NOISE_MIN = 1
    TIME_NOISE_MAX = 240
    
    OUTPUT_DIR = Path("outputs/adaptive_runs/uniform_pilot")
    SAVE_DETAILED_LOGS = True


def decide_adjustment(auc: float, config: AdaptiveConfig = None) -> tuple:
    """Decide parameter adjustment based on AUC"""
    if config is None:
        config = AdaptiveConfig()
    
    if auc > config.AUC_HIGH_THRESHOLD:
        return 'INCREASE', config.FACTOR_INCREASE, 'Too predictable'
    elif auc < config.AUC_LOW_THRESHOLD:
        return 'DECREASE', config.FACTOR_DECREASE, 'Too random'
    elif config.AUC_TARGET_MIN <= auc <= config.AUC_TARGET_MAX:
        return 'HOLD', 1.0, 'In target range'
    elif auc > config.AUC_TARGET_MAX:
        return 'NUDGE_UP', config.FACTOR_NUDGE, 'Slightly high'
    else:
        return 'NUDGE_DOWN', 1.0 / config.FACTOR_NUDGE, 'Slightly low'


def adjust_parameters(params: Dict, multiplier: float, config: AdaptiveConfig = None) -> Dict:
    """Adjust parameters with safety bounds"""
    if config is None:
        config = AdaptiveConfig()
    
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


class IterationLogger:
    """Track metrics across iterations"""
    
    def __init__(self):
        self.iterations = []
        self.start_time = datetime.now()
    
    def log_iteration(self, iteration, params, auc, action, multiplier, reason, train_metrics, val_metrics, cv_scores=None):
        """Log complete iteration metrics"""
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'price_noise': float(params['price_noise']),
            'time_noise_minutes': float(params['time_noise_minutes']),
            'auc_val': float(auc),
            'auc_cv_mean': float(np.mean(cv_scores)) if cv_scores else None,
            'auc_cv_std': float(np.std(cv_scores)) if cv_scores else None,
            'action': action,
            'multiplier': float(multiplier),
            'reason': reason,
            'n_train': train_metrics.get('n_samples', 0),
            'n_val': val_metrics.get('n_samples', 0),
            'n_features': train_metrics.get('n_features', 0),
            'train_pos_rate': (
                train_metrics['label_distribution']['positive'] / train_metrics['n_samples'] * 100
                if train_metrics.get('n_samples') else 0
            ),
            'val_pos_rate': (
                val_metrics['label_distribution']['positive'] / val_metrics['n_samples'] * 100
                if val_metrics.get('n_samples') else 0
            ),
            'confusion_matrix': val_metrics.get('confusion_matrix'),
        }
        self.iterations.append(entry)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert logs to DataFrame"""
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
        """Save all results to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        df.to_csv(output_dir / "adaptive_results.csv", index=False)
        
        with open(output_dir / "adaptive_results.json", 'w') as f:
            json.dump(self.iterations, f, indent=2)
        
        summary = {
            'total_iterations': len(self.iterations),
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'final_auc': float(self.iterations[-1]['auc_val']) if self.iterations else None,
            'final_params': {
                'price_noise': float(self.iterations[-1]['price_noise']),
                'time_noise_minutes': float(self.iterations[-1]['time_noise_minutes'])
            } if self.iterations else None,
            'auc_trajectory': [float(x['auc_val']) for x in self.iterations],
            'actions': [x['action'] for x in self.iterations]
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  ✓ Results saved to {output_dir}")


def adaptive_training_loop(
    prices_df: pd.DataFrame,
    initial_params: Optional[Dict] = None,
    config: Optional[AdaptiveConfig] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """Production-grade adaptive training loop"""
    
    if config is None:
        config = AdaptiveConfig()
    
    if initial_params is None:
        initial_params = DEFAULT_UNIFORM_PARAMS.copy()
    
    params = initial_params.copy()
    policy = UniformPolicy(params=params, seed=seed)
    logger = IterationLogger()
    
    hold_count = 0
    converged = False
    
    if verbose:
        print("\n" + "="*80)
        print("P7 ADAPTIVE ADVERSARY TRAINING LOOP")
        print("="*80)
        print(f"Prediction window: {config.PREDICTION_WINDOW_MINUTES} min")
        print(f"Max iterations: {config.MAX_ITERATIONS}")
        print(f"Initial params: price_noise={params['price_noise']:.4f}, time_noise={params['time_noise_minutes']:.1f}min")
    
    for iteration in range(config.MAX_ITERATIONS):
        iter_num = iteration + 1
        
        if verbose:
            print("\n" + "="*80)
            print(f"ITERATION {iter_num}/{config.MAX_ITERATIONS}")
            print("="*80)
        
        try:
            # Generate trades
            trades = policy.generate_trades(prices_df)
            if len(trades) == 0:
                print("No trades generated!")
                break
            
            # Enrich
            enriched = enrich_trades_for_adversary(trades, prices_df, f'uniform_iter{iter_num}')
            
            # Split
            train, val, test = time_split_trades(enriched, config.TRAIN_RATIO, config.VAL_RATIO)
            
            if len(val) < config.MIN_VAL_SAMPLES:
                print(f"Validation set too small ({len(val)})")
                break
            
            # Train adversary
            adversary = P7AdaptiveAdversary(
                window_threshold_minutes=config.PREDICTION_WINDOW_MINUTES,
                use_smote=config.USE_SMOTE,
                use_cv=config.USE_CV,
                n_cv_folds=config.N_CV_FOLDS,
                random_state=seed
            )
            
            train_metrics = adversary.train(train, verbose=verbose)
            if not train_metrics.get('success', False):
                print(f"Training failed!")
                break
            
            # Evaluate
            val_metrics = adversary.evaluate(val, verbose=verbose)
            if not val_metrics.get('success', False):
                print(f"Evaluation failed!")
                break
            
            auc = val_metrics['auc']
            
            # Decision
            action, multiplier, reason = decide_adjustment(auc, config)
            
            if verbose:
                print(f"\nRESULT: AUC={auc:.4f}, ACTION={action}, REASON={reason}")
            
            # Log
            logger.log_iteration(
                iter_num, params, auc, action, multiplier, reason,
                train_metrics, val_metrics,
                adversary.cv_scores if hasattr(adversary, 'cv_scores') else None
            )
            
            # Convergence check
            if action == 'HOLD':
                hold_count += 1
                if hold_count >= config.CONVERGENCE_PATIENCE:
                    converged = True
                    if verbose:
                        print(f"\n🎉 CONVERGED after {iter_num} iterations!")
                    break
            else:
                hold_count = 0
                params = adjust_parameters(params, multiplier, config)
                policy = UniformPolicy(params=params, seed=seed)
                if verbose:
                    print(f"New params: price_noise={params['price_noise']:.4f}, time_noise={params['time_noise_minutes']:.1f}min")
        
        except Exception as e:
            print(f"ERROR in iteration: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if verbose:
        logger.print_summary()
    
    return {
        'logger': logger,
        'final_params': params,
        'converged': converged,
        'n_iterations': len(logger.iterations)
    }


def main():
    """Main entry point"""
    print("="*80)
    print("P7 WEEK 3 PILOT: ADAPTIVE ADVERSARY")
    print("="*80)
    
    config = AdaptiveConfig()
    
    print("\nLoading data...")
    try:
        prices = load_prices("data/ALL_backtest.csv")
        print(f"✓ Loaded {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return
    
    print("\nRunning adaptive loop...")
    try:
        results = adaptive_training_loop(prices, config=config, seed=42, verbose=True)
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nSaving results...")
    try:
        results['logger'].save_results(config.OUTPUT_DIR)
    except Exception as e:
        print(f"✗ ERROR saving: {e}")
    
    print("\n" + "="*80)
    print("✅ WEEK 3 PILOT COMPLETE")
    print("="*80)
    
    if results['logger'].iterations:
        final_auc = results['logger'].iterations[-1]['auc_val']
        print(f"\nFinal AUC: {final_auc:.4f}")
        if final_auc > 0.75:
            print("⚠️  High predictability - need more randomization")
        elif final_auc < 0.55:
            print("⚠️  Too random - may be too noisy")
        else:
            print("✓ Good balance - in target range")


if __name__ == "__main__":
    main()
