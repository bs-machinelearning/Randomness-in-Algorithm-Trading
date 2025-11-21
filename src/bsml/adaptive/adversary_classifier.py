"""
P7 Adaptive Adversary Classifier

Predicts trade timing patterns to measure policy predictability.
Used in adaptive feedback loop to tune randomization parameters.

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from typing import Tuple


class P7AdaptiveAdversary:
    """
    Lightweight adversary for adaptive parameter tuning.
    
    Predicts: "Will the next trade happen soon?" based on time gaps.
    """
    
    def __init__(self, window_threshold_days: float = 1.5):
        """
        Args:
            window_threshold_days: Threshold for "soon" prediction
        """
        self.threshold = window_threshold_days
        self.model = None
        self.feature_cols = []
        
    def extract_features(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract predictive features from trade history.
        
        Features:
        - Price momentum (returns over multiple windows)
        - Price volatility (rolling std)
        - Trade frequency patterns
        - Time-based patterns (hour, day of week)
        """
        df = trades_df.sort_values(['symbol', 'timestamp']).copy()
        
        # === PRICE FEATURES ===
        df['price_return_1'] = df.groupby('symbol')['mid_price'].pct_change(1)
        df['price_return_5'] = df.groupby('symbol')['mid_price'].pct_change(5)
        
        # Rolling volatility
        for window in [3, 5, 10]:
            df[f'price_vol_{window}'] = (
                df.groupby('symbol')['price_return_1']
                .transform(lambda x: x.rolling(window, min_periods=2).std())
            )
        
        # Price levels (normalized)
        for window in [5, 10, 20]:
            df[f'price_ma_{window}'] = (
                df.groupby('symbol')['mid_price']
                .transform(lambda x: x.rolling(window, min_periods=2).mean())
            )
            df[f'price_vs_ma_{window}'] = (
                df['mid_price'] / (df[f'price_ma_{window}'] + 1e-8)
            )
        
        # === TIME FEATURES ===
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # === TRADE FREQUENCY ===
        for window in [3, 5, 10]:
            df[f'trades_last_{window}'] = (
                df.groupby('symbol')['exec_flag']
                .transform(lambda x: x.rolling(window, min_periods=1).sum())
            )
        
        # === QUANTITY PATTERNS ===
        if 'qty' in df.columns:
            df['qty_abs'] = df['qty'].abs()
            df['qty_ma_5'] = (
                df.groupby('symbol')['qty_abs']
                .transform(lambda x: x.rolling(5, min_periods=1).mean())
            )
        
        return df
    
    def create_labels(self, trades_df: pd.DataFrame) -> pd.Series:
        """
        Create labels based on time to next trade.
        
        Label = 1 if next trade happens within threshold days
        Label = 0 otherwise (weekend, holiday, or last trade)
        
        This creates natural variation:
        - Normal trading days: next trade in 1 day → label=1
        - Weekends/holidays: next trade in 3+ days → label=0
        """
        df = trades_df.sort_values(['symbol', 'timestamp']).copy()
        
        # Calculate days until next trade (per symbol)
        df['days_to_next'] = (
            df.groupby('symbol')['timestamp']
            .shift(-1) - df['timestamp']
        ).dt.total_seconds() / 86400.0
        
        # Create binary labels
        df['label'] = (df['days_to_next'] <= self.threshold).astype(int)
        
        # Last trade per symbol has no "next" → label as 0
        df['label'] = df['label'].fillna(0).astype(int)
        
        return df['label']
    
    def train(self, trades_df: pd.DataFrame, verbose: bool = True) -> bool:
        """
        Train adversary on trade history.
        
        Returns:
            True if training succeeded, False if failed
        """
        if verbose:
            print(f"  [Adversary] Training on {len(trades_df)} samples...")
        
        # Extract features
        features_df = self.extract_features(trades_df)
        
        # Generate labels
        labels = self.create_labels(features_df)
        features_df['label'] = labels
        
        # Debug: label distribution
        n_pos = labels.sum()
        n_total = len(labels)
        if verbose:
            print(f"  [Adversary] Labels → 0: {n_total - n_pos}, 1: {n_pos} "
                  f"({n_pos/n_total*100:.1f}% positive)")
        
        # Check if we can train
        if n_pos == 0:
            print("  [ERROR] No positive labels! Cannot train.")
            return False
        if n_pos == n_total:
            print("  [ERROR] All labels positive! Cannot train.")
            return False
        if n_pos < 10 or (n_total - n_pos) < 10:
            print("  [ERROR] Too few samples in one class!")
            return False
        
        # Select feature columns
        exclude = {
            'timestamp', 'symbol', 'policy_id', 'label', 'side', 
            'date', 'exec_flag', 'pnl', 'ref_price', 'days_to_next',
            'qty', 'volume', 'price'
        }
        self.feature_cols = [
            c for c in features_df.columns
            if c not in exclude and np.issubdtype(features_df[c].dtype, np.number)
        ]
        
        if verbose:
            print(f"  [Adversary] Using {len(self.feature_cols)} features")
        
        # Prepare training data
        X = features_df[self.feature_cols].fillna(0).values
        y = labels.values
        
        # Train classifier
        self.model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.1,
            max_iter=100,
            min_samples_leaf=20,
            random_state=42
        )
        
        self.model.fit(X, y)
        
        if verbose:
            print(f"  [Adversary] Training complete ✓")
        
        return True
    
    def evaluate(self, trades_df: pd.DataFrame, verbose: bool = True) -> float:
        """
        Evaluate adversary on validation data.
        
        Returns:
            AUC score (0.5 = random, 1.0 = perfect)
        """
        if self.model is None:
            if verbose:
                print("  [ERROR] Model not trained!")
            return 0.50
        
        if verbose:
            print(f"  [Adversary] Evaluating on {len(trades_df)} samples...")
        
        # Extract features
        features_df = self.extract_features(trades_df)
        
        # Generate labels
        labels = self.create_labels(features_df)
        features_df['label'] = labels
        
        # Debug: label distribution
        n_pos = labels.sum()
        n_total = len(labels)
        if verbose:
            print(f"  [Adversary] Val labels → 0: {n_total - n_pos}, 1: {n_pos} "
                  f"({n_pos/n_total*100:.1f}% positive)")
        
        # Check if we can evaluate
        if n_pos == 0 or n_pos == n_total:
            if verbose:
                print("  [WARNING] All labels same class! Returning AUC=0.50")
            return 0.50
        
        # Prepare data
        X = features_df[self.feature_cols].fillna(0).values
        y = labels.values
        
        # Predict probabilities
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Compute AUC
        auc = roc_auc_score(y, y_pred_proba)
        
        if verbose:
            print(f"  [Adversary] AUC = {auc:.4f}")
        
        return float(auc)


def time_split_trades(
    trades_df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split trades chronologically for training/validation/testing.
    
    Args:
        trades_df: Trade records
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (rest goes to test)
    
    Returns:
        (train_df, val_df, test_df)
    """
    df = trades_df.sort_values('timestamp').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    return train, val, test
