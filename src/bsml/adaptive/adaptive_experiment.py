"""
Adaptive Adversary Experiment - Main Loop

Implements the complete adaptive framework:
1. Generate trades for all 4 policies (Baseline, Pink, OU, Uniform)
2. Extract features for each binary classifier
3. Train 3 adversaries (Baseline vs Pink/OU/Uniform)
4. Evaluate AUC scores
5. Adapt policies if AUC > 0.7 (increase randomization)

Owner: P7
Week: 3
"""

import sys
from pathlib import Path

# Add src to path for imports
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from bsml.data.loader import load_prices
from bsml.policies.baseline import generate_trades as baseline_generate
from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.policies.pink_policy import PinkNoisePolicy, DEFAULT_PINK_PARAMS
from bsml.policies.ou_policy import OUPolicy, DEFAULT_OU_PARAMS

# Import our new modules
from adaptive_feature_extraction import (
    extract_pink_noise_features,
    extract_ou_features,
    extract_uniform_features,
    combine_features_for_binary_classification
)
from adaptive_classifiers import (
    BaselineVsPinkClassifier,
    BaselineVsOUClassifier,
    BaselineVsUniformClassifier,
    train_and_evaluate_classifier
)


class AdaptiveExperiment:
    """
    Main experiment class for adaptive adversary framework.
    """
    
    def __init__(self, prices: pd.DataFrame, auc_threshold: float = 0.7, random_state: int = 42):
        """
        Initialize experiment.
        
        Args:
            prices: Market price data
            auc_threshold: AUC threshold for triggering adaptation
            random_state: Random seed
        """
        self.prices = prices
        self.auc_threshold = auc_threshold
        self.random_state = random_state
        
        # Initialize policies with default parameters
        self.pink_params = {'alpha': 1.0, 'price_scale': 0.04}
        self.ou_params = {'theta': 0.15, 'sigma': 0.02, 'price_scale': 1.0}
        self.uniform_params = {'price_noise': 0.03, 'time_noise_minutes': 30.0}
        
        # Track AUC history
        self.auc_history = {
            'pink': [],
            'ou': [],
            'uniform': []
        }
        
        # Track parameter history
        self.param_history = {
            'pink': [],
            'ou': [],
            'uniform': []
        }
    
    def generate_all_trades(self) -> Dict[str, pd.DataFrame]:
        """
        Generate trades for all 4 policies.
        
        Returns:
            Dictionary mapping policy name to trades DataFrame
        """
        
        print("\n" + "="*80)
        print("GENERATING TRADES FOR ALL POLICIES")
        print("="*80)
        
        # Baseline
        print("\n[1/4] Baseline...")
        baseline_trades = baseline_generate(self.prices)
        print(f"  ✓ Generated {len(baseline_trades)} trades")
        
        # Pink Noise
        print("\n[2/4] Pink Noise...")
        print(f"  Parameters: alpha={self.pink_params['alpha']}, price_scale={self.pink_params['price_scale']}")
        pink_policy = PinkNoisePolicy(
            alpha=self.pink_params['alpha'],
            price_scale=self.pink_params['price_scale'],
            seed=self.random_state
        )
        pink_trades = pink_policy.generate_trades(self.prices)
        print(f"  ✓ Generated {len(pink_trades)} trades")
        
        # OU
        print("\n[3/4] Ornstein-Uhlenbeck...")
        print(f"  Parameters: theta={self.ou_params['theta']}, sigma={self.ou_params['sigma']}, price_scale={self.ou_params['price_scale']}")
        ou_policy = OUPolicy(
            theta=self.ou_params['theta'],
            sigma=self.ou_params['sigma'],
            price_scale=self.ou_params['price_scale'],
            seed=self.random_state
        )
        ou_trades = ou_policy.generate_trades(self.prices)
        print(f"  ✓ Generated {len(ou_trades)} trades")
        
        # Uniform
        print("\n[4/4] Uniform...")
        print(f"  Parameters: price_noise={self.uniform_params['price_noise']}, time_noise={self.uniform_params['time_noise_minutes']}min")
        uniform_params = DEFAULT_UNIFORM_PARAMS.copy()
        uniform_params.update(self.uniform_params)
        uniform_policy = UniformPolicy(params=uniform_params)
        uniform_trades = uniform_policy.generate_trades(self.prices)
        print(f"  ✓ Generated {len(uniform_trades)} trades")
        
        return {
            'baseline': baseline_trades,
            'pink': pink_trades,
            'ou': ou_trades,
            'uniform': uniform_trades
        }
    
    def train_adversaries(self, trades: Dict[str, pd.DataFrame], verbose: bool = True) -> Dict[str, float]:
        """
        Train all three binary adversaries and return AUC scores.
        
        Args:
            trades: Dictionary of trades for each policy
            verbose: Print detailed results
        
        Returns:
            Dictionary of AUC scores for each adversary
        """
        
        baseline_trades = trades['baseline']
        pink_trades = trades['pink']
        ou_trades = trades['ou']
        uniform_trades = trades['uniform']
        
        results = {}
        
        # ========== ADVERSARY 1: Baseline vs Pink ==========
        if verbose:
            print("\n" + "="*80)
            print("ADVERSARY 1: BASELINE vs PINK NOISE")
            print("="*80)
        
        # Extract features
        pink_features = extract_pink_noise_features(baseline_trades, pink_trades)
        baseline_features_pink = extract_pink_noise_features(baseline_trades, baseline_trades)
        
        # Combine and create labels
        X_pink, y_pink = combine_features_for_binary_classification(
            baseline_features_pink, pink_features
        )
        
        # Train classifier
        pink_classifier = BaselineVsPinkClassifier(random_state=self.random_state)
        _, test_auc_pink = train_and_evaluate_classifier(
            pink_classifier, X_pink, y_pink, test_size=0.3, verbose=verbose
        )
        
        results['pink'] = test_auc_pink
        
        # ========== ADVERSARY 2: Baseline vs OU ==========
        if verbose:
            print("\n" + "="*80)
            print("ADVERSARY 2: BASELINE vs OU")
            print("="*80)
        
        # Extract features
        ou_features = extract_ou_features(baseline_trades, ou_trades)
        baseline_features_ou = extract_ou_features(baseline_trades, baseline_trades)
        
        # Combine and create labels
        X_ou, y_ou = combine_features_for_binary_classification(
            baseline_features_ou, ou_features
        )
        
        # Train classifier
        ou_classifier = BaselineVsOUClassifier(random_state=self.random_state)
        _, test_auc_ou = train_and_evaluate_classifier(
            ou_classifier, X_ou, y_ou, test_size=0.3, verbose=verbose
        )
        
        results['ou'] = test_auc_ou
        
        # ========== ADVERSARY 3: Baseline vs Uniform ==========
        if verbose:
            print("\n" + "="*80)
            print("ADVERSARY 3: BASELINE vs UNIFORM")
            print("="*80)
        
        # Extract features
        uniform_features = extract_uniform_features(baseline_trades, uniform_trades)
        baseline_features_uniform = extract_uniform_features(baseline_trades, baseline_trades)
        
        # Combine and create labels
        X_uniform, y_uniform = combine_features_for_binary_classification(
            baseline_features_uniform, uniform_features
        )
        
        # Train classifier
        uniform_classifier = BaselineVsUniformClassifier(random_state=self.random_state)
        _, test_auc_uniform = train_and_evaluate_classifier(
            uniform_classifier, X_uniform, y_uniform, test_size=0.3, verbose=verbose
        )
        
        results['uniform'] = test_auc_uniform
        
        return results
    
    def adapt_policies(self, auc_scores: Dict[str, float]) -> Dict[str, bool]:
        """
        Adapt policies if AUC > threshold.
        
        Args:
            auc_scores: Dictionary of AUC scores
        
        Returns:
            Dictionary indicating which policies were adapted
        """
        
        adaptations = {}
        
        print("\n" + "="*80)
        print("ADAPTATION DECISIONS")
        print("="*80)
        
        # Pink Noise adaptation
        print(f"\nPink Noise AUC: {auc_scores['pink']:.4f}")
        if auc_scores['pink'] > self.auc_threshold:
            print(f"  ⚠️  AUC > {self.auc_threshold} → ADAPTING")
            self.pink_params['price_scale'] *= 1.5
            print(f"  New price_scale: {self.pink_params['price_scale']:.4f}")
            adaptations['pink'] = True
        else:
            print(f"  ✓ AUC ≤ {self.auc_threshold} → NO ADAPTATION NEEDED")
            adaptations['pink'] = False
        
        # OU adaptation
        print(f"\nOU AUC: {auc_scores['ou']:.4f}")
        if auc_scores['ou'] > self.auc_threshold:
            print(f"  ⚠️  AUC > {self.auc_threshold} → ADAPTING")
            self.ou_params['sigma'] *= 1.3
            self.ou_params['theta'] *= 1.2
            self.ou_params['price_scale'] *= 1.5
            print(f"  New sigma: {self.ou_params['sigma']:.4f}")
            print(f"  New theta: {self.ou_params['theta']:.4f}")
            print(f"  New price_scale: {self.ou_params['price_scale']:.4f}")
            adaptations['ou'] = True
        else:
            print(f"  ✓ AUC ≤ {self.auc_threshold} → NO ADAPTATION NEEDED")
            adaptations['ou'] = False
        
        # Uniform adaptation
        print(f"\nUniform AUC: {auc_scores['uniform']:.4f}")
        if auc_scores['uniform'] > self.auc_threshold:
            print(f"  ⚠️  AUC > {self.auc_threshold} → ADAPTING")
            self.uniform_params['price_noise'] *= 1.3
            self.uniform_params['time_noise_minutes'] *= 1.3
            print(f"  New price_noise: {self.uniform_params['price_noise']:.4f}")
            print(f"  New time_noise: {self.uniform_params['time_noise_minutes']:.1f}min")
            adaptations['uniform'] = True
        else:
            print(f"  ✓ AUC ≤ {self.auc_threshold} → NO ADAPTATION NEEDED")
            adaptations['uniform'] = False
        
        return adaptations
    
    def run_single_iteration(self, iteration: int = 0) -> Dict[str, float]:
        """
        Run a single iteration of the experiment.
        
        Args:
            iteration: Iteration number
        
        Returns:
            AUC scores for this iteration
        """
        
        print("\n" + "="*80)
        print(f"ITERATION {iteration}")
        print("="*80)
        
        # Generate trades
        trades = self.generate_all_trades()
        
        # Train adversaries
        auc_scores = self.train_adversaries(trades, verbose=True)
        
        # Record history
        self.auc_history['pink'].append(auc_scores['pink'])
        self.auc_history['ou'].append(auc_scores['ou'])
        self.auc_history['uniform'].append(auc_scores['uniform'])
        
        self.param_history['pink'].append(self.pink_params.copy())
        self.param_history['ou'].append(self.ou_params.copy())
        self.param_history['uniform'].append(self.uniform_params.copy())
        
        # Adapt policies
        adaptations = self.adapt_policies(auc_scores)
        
        return auc_scores
    
    def run_adaptive_loop(self, n_iterations: int = 5) -> pd.DataFrame:
        """
        Run multiple iterations with adaptation.
        
        Args:
            n_iterations: Number of iterations
        
        Returns:
            DataFrame with AUC history
        """
        
        print("\n" + "="*80)
        print(f"ADAPTIVE EXPERIMENT: {n_iterations} ITERATIONS")
        print("="*80)
        print(f"AUC Threshold: {self.auc_threshold}")
        print(f"Random State: {self.random_state}")
        
        for i in range(n_iterations):
            self.run_single_iteration(iteration=i)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'iteration': range(n_iterations),
            'pink_auc': self.auc_history['pink'],
            'ou_auc': self.auc_history['ou'],
            'uniform_auc': self.auc_history['uniform']
        })
        
        return results_df


def main():
    """Main execution"""
    
    print("="*80)
    print("ADAPTIVE ADVERSARY EXPERIMENT")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load price data
    print("\n[Step 1] Loading price data...")
    prices = load_prices("data/ALL_backtest.csv")
    print(f"  ✓ Loaded {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    
    # Initialize experiment
    print("\n[Step 2] Initializing experiment...")
    experiment = AdaptiveExperiment(
        prices=prices,
        auc_threshold=0.7,
        random_state=42
    )
    print("  ✓ Experiment initialized")
    
    # Run single iteration (no adaptation loop for now)
    print("\n[Step 3] Running experiment...")
    auc_scores = experiment.run_single_iteration(iteration=0)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nPink Noise AUC: {auc_scores['pink']:.4f}")
    print(f"OU AUC: {auc_scores['ou']:.4f}")
    print(f"Uniform AUC: {auc_scores['uniform']:.4f}")
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
