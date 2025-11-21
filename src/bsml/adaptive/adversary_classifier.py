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
from typing import Tuple, Optional


class P7AdaptiveAdversary:
    """
    Adversary classifier for P7's adaptive loop.
    
    Predicts: "Will this policy make a trade in the next time window?"
    """
    
    def __init__(self, window_size: int = 5):
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
        for window in [5, 10, 20]:
            df[f'price_mean_{window}'] = (
                df.groupby('symbol')['mid_price']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[f'price_std_{window}'] = (
                df.groupby('symbol')['mid_price']
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Trade size patterns (if available)
        if 'qty' in df.columns:
            df['qty_abs'] = df['qty'].abs()
            df['qty_mean_5'] = (
                df.groupby('symbol')['qty_abs']
                .transform(lambda x: x.rolling(5, min_periods=1).mean())
            )
        
        # Recent trade frequency
        df['trades_last_5'] = (
            df.groupby('symbol')['exec_flag']
            .transform(lambda x: x.rolling(5, min_periods=1).sum())
        )
        
        return df
    
    def create_labels(self, trades_df: pd.DataFrame) -> pd.Series:
        """
        Create binary labels: Will a trade occur in next window_size bars?
        
        Label = 1 if exec_flag = 1 within next window_size bars
        Label = 0 otherwise
        """
        df = trades_df.sort_values(['symbol', 'timestamp']).copy()
        
        # For each row, check if there's an execution in next window_size bars
        labels = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df = symbol_df.reset_index(drop=True)
            
            symbol_labels = []
            for i in range(len(symbol_df)):
                # Look ahead window_size bars
                window_end = min(i + self.window_size, len(symbol_df))
                has_trade = symbol_df.iloc[i:window_end]['exec_flag'].sum() > 0
                symbol_labels.append(1 if has_trade else 0)
            
            labels.extend(symbol_labels)
        
        return pd.Series(labels, index=df.index)
    
    def train(self, trades_df: pd.DataFrame) -> None:
        """
        Train adversary classifier on trades.
        """
        # Extract features
        features_df = self.extract_features(trades_df)
        
        # Create labels
        labels = self.create_labels(features_df)
        features_df['label'] = labels
        
        # Select numeric feature columns
        exclude = {'timestamp', 'symbol', 'policy_id', 'label', 'side', 
                   'date', 'exec_flag', 'pnl'}
        feature_cols = [
            c for c in features_df.columns 
            if c not in exclude and np.issubdtype(features_df[c].dtype, np.number)
        ]
        
        # Prepare data
        X = features_df[feature_cols].fillna(0).values
        y = features_df['label'].values
        
        # Train classifier
        self.model = HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.1,
            max_iter=100,
            random_state=42
        )
        
        self.model.fit(X, y)
    
    def evaluate(self, trades_df: pd.DataFrame) -> float:
        """
        Evaluate adversary on new trades, return AUC score.
        
        Returns:
            AUC score (0.5 = random, 1.0 = perfect prediction)
        """
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        # Extract features
        features_df = self.extract_features(trades_df)
        
        # Create labels
        labels = self.create_labels(features_df)
        features_df['label'] = labels
        
        # Select feature columns
        exclude = {'timestamp', 'symbol', 'policy_id', 'label', 'side',
                   'date', 'exec_flag', 'pnl'}
        feature_cols = [
            c for c in features_df.columns 
            if c not in exclude and np.issubdtype(features_df[c].dtype, np.number)
        ]
        
        # Prepare data
        X = features_df[feature_cols].fillna(0).values
        y = features_df['label'].values
        
        # Check if labels are valid
        if y.sum() == 0 or y.sum() == len(y):
            # All same label - return random guess
            return 0.50
        
        # Predict
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Calculate AUC
        auc = roc_auc_score(y, y_pred_proba)
        
        return auc
    
    def train_and_evaluate(
        self, 
        train_trades: pd.DataFrame, 
        test_trades: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Train on train set, evaluate on both train and test sets.
        
        Returns:
            (train_auc, test_auc)
        """
        self.train(train_trades)
        
        train_auc = self.evaluate(train_trades)
        test_auc = self.evaluate(test_trades)
        
        return train_auc, test_auc


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
    
    return train, val, test
