"""
P7 Adaptive Adversary Classifier

Lightweight adversary specifically for adaptive parameter tuning.
Predicts trade occurrence based on policy-generated trades.

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
    Adversary classifier for P7's adaptive loop.
    
    Predicts: "Will this policy make a trade in the next time window?"
    """
    
    def __init__(self, window_size: int = 1):
        """
        Args:
            window_size: How many bars ahead to predict trades
        """
        self.window_size = window_size
        self.model = None
        
    def extract_features(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from trade data for adversary.
        
        Features focus on:
        - Price momentum patterns
        - Trade timing patterns
        - Trade size patterns
        - Volatility indicators
        """
        df = trades_df.sort_values(['symbol', 'timestamp']).copy()
        
        # Price features
        df['price_return'] = df.groupby('symbol')['mid_price'].pct_change()
        df['price_return_abs'] = df['price_return'].abs()
        
        # Rolling statistics (per symbol)
        for window in [3, 5, 10]:
            df[f'price_mean_{window}'] = (
                df.groupby('symbol')['mid_price']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[f'price_std_{window}'] = (
                df.groupby('symbol')['mid_price']
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
            df[f'price_zscore_{window}'] = (
                (df['mid_price'] - df[f'price_mean_{window}']) / 
                (df[f'price_std_{window}'] + 1e-6)
            )
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Trade size patterns
        if 'qty' in df.columns:
            df['qty_abs'] = df['qty'].abs()
            df['log_qty'] = np.log1p(df['qty_abs'])
        
        # Volume patterns
        if 'volume' in df.columns:
            df['log_volume'] = np.log1p(df['volume'])
            for window in [3, 5]:
                df[f'volume_mean_{window}'] = (
                    df.groupby('symbol')['volume']
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
        
        # Recent trade frequency (using exec_flag)
        for window in [3, 5, 10]:
            df[f'trades_last_{window}'] = (
                df.groupby('symbol')['exec_flag']
                .transform(lambda x: x.rolling(window, min_periods=1).sum())
            )
        
        # Distance from recent prices
        df['price_vs_mean3'] = df['mid_price'] / (df['price_mean_3'] + 1e-6)
        df['price_vs_mean5'] = df['mid_price'] / (df['price_mean_5'] + 1e-6)
        
        return df
    
    def create_labels(self, trades_df: pd.DataFrame) -> pd.Series:
        """
        Create binary labels: Will a trade occur in next window_size bars?
        
        Uses vectorized approach for speed.
        """
        df = trades_df.sort_values(['symbol', 'timestamp']).copy()
        
        # Shift exec_flag forward by window_size and check if any execution happens
        df['label'] = 0
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df.loc[mask, 'exec_flag'].values
            
            # For each position, check if there's a trade in next window_size steps
            labels = np.zeros(len(symbol_data), dtype=int)
            for i in range(len(symbol_data)):
                window_end = min(i + self.window_size + 1, len(symbol_data))
                # Check if any exec_flag=1 in the forward window
                if symbol_data[i:window_end].sum() > 0:
                    labels[i] = 1
            
            df.loc[mask, 'label'] = labels
        
        return df['label']
    
    def train(self, trades_df: pd.DataFrame) -> None:
        """
        Train adversary classifier on trades.
        """
        print(f"  [Adversary] Training on {len(trades_df)} samples...")
        
        # Extract features
        features_df = self.extract_features(trades_df)
        
        # Create labels
        labels = self.create_labels(features_df)
        features_df['label'] = labels
        
        # Debug: print label distribution
        print(f"  [Adversary] Labels - 0: {(labels==0).sum()}, 1: {(labels==1).sum()}")
        print(f"  [Adversary] Positive rate: {labels.mean()*100:.1f}%")
        
        # Select numeric feature columns
        exclude = {'timestamp', 'symbol', 'policy_id', 'label', 'side', 
                   'date', 'exec_flag', 'pnl', 'ref_price'}
        feature_cols = [
            c for c in features_df.columns 
            if c not in exclude and np.issubdtype(features_df[c].dtype, np.number)
        ]
        
        print(f"  [Adversary] Using {len(feature_cols)} features")
        
        # Prepare data
        X = features_df[feature_cols].fillna(0).values
        y = features_df['label'].values
        
        # Check if we can train
        if y.sum() == 0:
            print(f"  [WARNING] No positive labels! Cannot train.")
            return
        if y.sum() == len(y):
            print(f"  [WARNING] All labels are positive! Cannot train.")
            return
        
        # Train classifier
        self.model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.1,
            max_iter=50,
            random_state=42
        )
        
        self.model.fit(X, y)
        print(f"  [Adversary] Training complete")
    
    def evaluate(self, trades_df: pd.DataFrame) -> float:
        """
        Evaluate adversary on new trades, return AUC score.
        
        Returns:
            AUC score (0.5 = random, 1.0 = perfect prediction)
        """
        if self.model is None:
            print(f"  [WARNING] Model not trained, returning 0.50")
            return 0.50
        
        print(f"  [Adversary] Evaluating on {len(trades_df)} samples...")
        
        # Extract features
        features_df = self.extract_features(trades_df)
        
        # Create labels
        labels = self.create_labels(features_df)
        features_df['label'] = labels
        
        # Debug: print label distribution
        print(f"  [Adversary] Val Labels - 0: {(labels==0).sum()}, 1: {(labels==1).sum()}")
        print(f"  [Adversary] Val Positive rate: {labels.mean()*100:.1f}%")
        
        # Select feature columns
        exclude = {'timestamp', 'symbol', 'policy_id', 'label', 'side',
                   'date', 'exec_flag', 'pnl', 'ref_price'}
        feature_cols = [
            c for c in features_df.columns 
            if c not in exclude and np.issubdtype(features_df[c].dtype, np.number)
        ]
        
        # Prepare data
        X = features_df[feature_cols].fillna(0).values
        y = features_df['label'].values
        
        # Check if we can evaluate
        if y.sum() == 0 or y.sum() == len(y):
            print(f"  [WARNING] All labels same class! Returning 0.50")
            return 0.50
        
        # Predict
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Calculate AUC
        auc = roc_auc_score(y, y_pred_proba)
        
        print(f"  [Adversary] AUC = {auc:.4f}")
        
        return auc


def time_split_trades(
    trades_df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split trades by time for training/validation/testing.
    
    Args:
        trades_df: DataFrame of trades
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    
    Returns:
        (train_df, val_df, test_df)
    """
    df = trades_df.sort_values('timestamp').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    print(f"  [Split] Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    return train, val, test
