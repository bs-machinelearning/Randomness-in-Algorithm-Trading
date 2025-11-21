"""
P7 Adaptive Adversary Classifier - Production Grade

Advanced adversary with:
- Dynamic prediction window (auto-detected from data)
- Cross-validation
- SMOTE resampling for imbalanced data
- Enhanced feature engineering
- Proper train/val/test splits

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class P7AdaptiveAdversary:
    """
    Production-grade adversary with dynamic prediction window.
    
    Predicts: "Will the next trade happen within X minutes?"
    where X is determined by analyzing actual trade frequency.
    """
    
    def __init__(
        self, 
        window_threshold_minutes: float,
        use_smote: bool = True,
        use_cv: bool = True,
        n_cv_folds: int = 3,
        random_state: int = 42
    ):
        """
        Args:
            window_threshold_minutes: Prediction window in minutes (REQUIRED)
            use_smote: Apply SMOTE for label balancing
            use_cv: Use cross-validation during training
            n_cv_folds: Number of CV folds
            random_state: Random seed for reproducibility
        """
        # Validate window
        if window_threshold_minutes is None:
            raise ValueError("window_threshold_minutes must be provided!")
        if window_threshold_minutes <= 0:
            raise ValueError(f"window_threshold_minutes must be > 0, got {window_threshold_minutes}")
        if window_threshold_minutes > 10080:  # 1 week
            warnings.warn(f"Large prediction window: {window_threshold_minutes:.0f} min ({window_threshold_minutes/1440:.1f} days)")
        
        self.threshold_minutes = float(window_threshold_minutes)
        self.use_smote = use_smote
        self.use_cv = use_cv
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        
        # Models
        self.model_gb = None
        self.model_rf = None
        self.feature_cols = []
        self.cv_scores = []
        self.feature_importances = {}
        
        # Training metadata
        self.training_label_stats = None
        
    def extract_features(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rich features for timing prediction.
        
        Features engineered for multiple timescales:
        - Works for 10-minute predictions (HFT)
        - Works for 24-hour predictions (daily trading)
        - Works for multi-day predictions (weekly trading)
        """
        df = trades_df.sort_values(['symbol', 'timestamp']).copy()
        
        # =====================================================================
        # PRICE MOMENTUM (Multi-scale)
        # =====================================================================
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_return_{lag}'] = (
                df.groupby('symbol')['mid_price'].pct_change(lag)
            )
        
        df['return_accel'] = (
            df['price_return_1'] - df.groupby('symbol')['price_return_1'].shift(1)
        )
        
        # =====================================================================
        # VOLATILITY (Multi-horizon)
        # =====================================================================
        for window in [3, 5, 10, 20]:
            df[f'price_vol_{window}'] = (
                df.groupby('symbol')['price_return_1']
                .transform(lambda x: x.rolling(window, min_periods=2).std())
            )
            
            if 'volume' in df.columns:
                df[f'volume_vol_{window}'] = (
                    df.groupby('symbol')['volume']
                    .transform(lambda x: x.rolling(window, min_periods=2).std())
                )
        
        df['vol_of_vol'] = (
            df.groupby('symbol')['price_vol_5']
            .transform(lambda x: x.rolling(5, min_periods=2).std())
        )
        
        # =====================================================================
        # PRICE LEVELS (Normalized)
        # =====================================================================
        for window in [5, 10, 20]:
            # Moving average
            df[f'price_ma_{window}'] = (
                df.groupby('symbol')['mid_price']
                .transform(lambda x: x.rolling(window, min_periods=2).mean())
            )
            
            # Z-score (inline volatility calculation)
            rolling_vol = (
                df.groupby('symbol')['price_return_1']
                .transform(lambda x: x.rolling(window, min_periods=2).std())
            )
            df[f'price_zscore_{window}'] = (
                (df['mid_price'] - df[f'price_ma_{window}']) / (rolling_vol + 1e-8)
            )
            
            # High/Low range
            df[f'price_high_{window}'] = (
                df.groupby('symbol')['mid_price']
                .transform(lambda x: x.rolling(window, min_periods=1).max())
            )
            df[f'price_low_{window}'] = (
                df.groupby('symbol')['mid_price']
                .transform(lambda x: x.rolling(window, min_periods=1).min())
            )
            df[f'price_range_{window}'] = (
                df[f'price_high_{window}'] - df[f'price_low_{window}']
            )
        
        # =====================================================================
        # TIME FEATURES (Intraday + Weekly patterns)
        # =====================================================================
        dt = pd.to_datetime(df['timestamp'])
        df['hour'] = dt.dt.hour
        df['minute'] = dt.dt.minute
        df['day_of_week'] = dt.dt.dayofweek
        
        # Intraday patterns
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        df['is_morning'] = ((df['hour'] >= 9) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)
        df['is_close'] = ((df['hour'] >= 15) & (df['hour'] < 16)).astype(int)
        df['minutes_since_open'] = (df['hour'] - 9) * 60 + df['minute']
        
        # Weekly patterns
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_midweek'] = ((df['day_of_week'] >= 1) & (df['day_of_week'] <= 3)).astype(int)
        
        # =====================================================================
        # TRADE INTENSITY (Critical for timing prediction)
        # =====================================================================
        for window in [3, 5, 10, 20]:
            df[f'trades_last_{window}'] = (
                df.groupby('symbol')['exec_flag']
                .transform(lambda x: x.rolling(window, min_periods=1).sum())
            )
            df[f'trade_velocity_{window}'] = df[f'trades_last_{window}'] / window
        
        # Time since last trade (KEY FEATURE)
        df['minutes_since_last'] = (
            df.groupby('symbol')['timestamp'].diff().dt.total_seconds() / 60.0
        )
        df['minutes_since_last'] = df['minutes_since_last'].fillna(1440)
        
        # Normalized by prediction window (helps model generalize)
        df['time_since_ratio'] = df['minutes_since_last'] / self.threshold_minutes
        
        # =====================================================================
        # QUANTITY PATTERNS
        # =====================================================================
        if 'qty' in df.columns:
            df['qty_abs'] = df['qty'].abs()
            
            for window in [3, 5, 10]:
                df[f'qty_mean_{window}'] = (
                    df.groupby('symbol')['qty_abs']
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
                df[f'qty_std_{window}'] = (
                    df.groupby('symbol')['qty_abs']
                    .transform(lambda x: x.rolling(window, min_periods=1).std())
                )
            
            df['qty_change'] = df.groupby('symbol')['qty_abs'].diff()
        
        # =====================================================================
        # VOLUME PATTERNS
        # =====================================================================
        if 'volume' in df.columns:
            df['log_volume'] = np.log1p(df['volume'])
            
            for window in [3, 5, 10]:
                df[f'volume_ma_{window}'] = (
                    df.groupby('symbol')['volume']
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
                df[f'volume_ratio_{window}'] = (
                    df['volume'] / (df[f'volume_ma_{window}'] + 1)
                )
        
        return df
    
    def create_labels(self, trades_df: pd.DataFrame) -> pd.Series:
        """
        Create labels: Will next trade happen within threshold minutes?
        
        Label distribution will depend on threshold:
        - Small threshold (10 min) → Few positives
        - Medium threshold (6 hours) → Balanced
        - Large threshold (24 hours) → Many positives
        """
        df = trades_df.sort_values(['symbol', 'timestamp']).copy()
        
        # Calculate minutes until next trade
        df['minutes_to_next'] = (
            df.groupby('symbol')['timestamp']
            .shift(-1) - df['timestamp']
        ).dt.total_seconds() / 60.0
        
        # Binary label
        df['label'] = (df['minutes_to_next'] <= self.threshold_minutes).astype(int)
        df['label'] = df['label'].fillna(0).astype(int)
        
        return df['label']
    
    def _select_features(self, features_df: pd.DataFrame) -> list:
        """Select numeric features, excluding metadata"""
        exclude = {
            'timestamp', 'symbol', 'policy_id', 'label', 'side', 
            'date', 'exec_flag', 'pnl', 'ref_price', 'minutes_to_next',
            'qty', 'volume', 'price', 'mid_price'
        }
        
        feature_cols = [
            c for c in features_df.columns
            if c not in exclude and np.issubdtype(features_df[c].dtype, np.number)
        ]
        
        return feature_cols
    
    def _validate_labels(self, y: np.ndarray, min_samples: int = 10) -> Tuple[bool, str]:
        """
        Validate label distribution before training.
        
        Returns:
            (is_valid, reason)
        """
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        
        if n_pos < min_samples:
            return False, f'Too few positive samples ({n_pos} < {min_samples})'
        if n_neg < min_samples:
            return False, f'Too few negative samples ({n_neg} < {min_samples})'
        if n_pos == 0 or n_neg == 0:
            return False, 'Single class detected'
        
        # Check extreme imbalance
        pos_rate = n_pos / len(y)
        if pos_rate < 0.01:
            return False, f'Extreme imbalance: only {pos_rate*100:.2f}% positive'
        if pos_rate > 0.99:
            return False, f'Extreme imbalance: {pos_rate*100:.2f}% positive'
        
        return True, 'OK'
    
    def train(self, trades_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """Train adversary with validation"""
        if verbose:
            print(f"  [Adversary] Training on {len(trades_df)} samples...")
            print(f"  [Adversary] Prediction window: {self.threshold_minutes:.1f} min ({self.threshold_minutes/60:.2f} hrs)")
        
        # Extract features
        features_df = self.extract_features(trades_df)
        labels = self.create_labels(features_df)
        features_df['label'] = labels
        
        # Select features
        self.feature_cols = self._select_features(features_df)
        
        if verbose:
            print(f"  [Adversary] Extracted {len(self.feature_cols)} features")
        
        # Prepare data
        X = features_df[self.feature_cols].fillna(0).values
        y = labels.values
        
        # Check distribution
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_rate = n_pos / len(y)
        
        if verbose:
            print(f"  [Adversary] Labels → 0: {n_neg}, 1: {n_pos} ({pos_rate*100:.1f}% positive)")
        
        # Store label stats
        self.training_label_stats = {
            'n_positive': int(n_pos),
            'n_negative': int(n_neg),
            'positive_rate': float(pos_rate)
        }
        
        # Validate
        is_valid, reason = self._validate_labels(y, min_samples=10)
        if not is_valid:
            print(f"  [ERROR] Invalid labels: {reason}")
            return {
                'success': False, 
                'reason': reason,
                'n_samples': len(X),
                'n_features': len(self.feature_cols),
                'label_distribution': {'positive': int(n_pos), 'negative': int(n_neg)}
            }
        
        # Apply SMOTE if needed (only if significantly imbalanced)
        if self.use_smote and (pos_rate < 0.25 or pos_rate > 0.75):
            if verbose:
                print(f"  [Adversary] Applying SMOTE (imbalance: {pos_rate*100:.1f}%)...")
            
            try:
                target_ratio = 0.5 if pos_rate < 0.5 else 0.5
                k_neighbors = min(5, min(n_pos, n_neg) - 1)
                
                smote = SMOTE(
                    sampling_strategy=target_ratio,
                    random_state=self.random_state,
                    k_neighbors=max(1, k_neighbors)
                )
                X, y = smote.fit_resample(X, y)
                
                if verbose:
                    n_pos_new = y.sum()
                    print(f"  [Adversary] After SMOTE → 1: {n_pos_new} ({n_pos_new/len(y)*100:.1f}%)")
            except Exception as e:
                if verbose:
                    print(f"  [Warning] SMOTE failed: {e}. Continuing without.")
        
        # Train Gradient Boosting
        if verbose:
            print(f"  [Adversary] Training Gradient Boosting...")
        
        self.model_gb = HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=self.random_state
        )
        self.model_gb.fit(X, y)
        
        # Train Random Forest
        if verbose:
            print(f"  [Adversary] Training Random Forest...")
        
        self.model_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=20,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model_rf.fit(X, y)
        
        # Cross-validation
        cv_results = {}
        if self.use_cv:
            if verbose:
                print(f"  [Adversary] Running {self.n_cv_folds}-fold CV...")
            
            X_cv = features_df[self.feature_cols].fillna(0).values
            y_cv = features_df['label'].values
            
            try:
                cv_scores = cross_val_score(
                    self.model_gb, X_cv, y_cv,
                    cv=self.n_cv_folds,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                self.cv_scores = cv_scores
                cv_results = {
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'cv_scores': cv_scores.tolist()
                }
                
                if verbose:
                    print(f"  [Adversary] CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            except Exception as e:
                if verbose:
                    print(f"  [Warning] CV failed: {e}")
        
        # Feature importances
        if hasattr(self.model_gb, 'feature_importances_'):
            importances = self.model_gb.feature_importances_
            self.feature_importances = dict(zip(self.feature_cols, importances))
            
            if verbose:
                top_features = sorted(
                    self.feature_importances.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                print(f"  [Adversary] Top 5 features:")
                for feat, imp in top_features:
                    print(f"    - {feat}: {imp:.4f}")
        
        if verbose:
            print(f"  [Adversary] Training complete ✓")
        
        return {
            'success': True,
            'n_samples': len(X),
            'n_features': len(self.feature_cols),
            'label_distribution': {'positive': int(y.sum()), 'negative': int(len(y) - y.sum())},
            **cv_results
        }
    
    def evaluate(self, trades_df: pd.DataFrame, verbose: bool = True, 
                 return_predictions: bool = False) -> Dict:
        """Evaluate on validation/test data"""
        if self.model_gb is None or self.model_rf is None:
            if verbose:
                print("  [ERROR] Models not trained!")
            return {
                'auc': 0.50, 'success': False,
                'n_samples': 0,
                'label_distribution': {'positive': 0, 'negative': 0}
            }
        
        if verbose:
            print(f"  [Adversary] Evaluating on {len(trades_df)} samples...")
        
        # Extract features
        features_df = self.extract_features(trades_df)
        labels = self.create_labels(features_df)
        features_df['label'] = labels
        
        # Prepare data
        X = features_df[self.feature_cols].fillna(0).values
        y = labels.values
        
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        
        if verbose:
            print(f"  [Adversary] Val labels → 0: {n_neg}, 1: {n_pos} ({n_pos/len(y)*100:.1f}%)")
        
        if n_pos == 0 or n_neg == 0:
            if verbose:
                print(f"  [WARNING] Single class! Returning AUC=0.50")
            return {
                'auc': 0.50, 'success': False, 'reason': 'single_class',
                'n_samples': len(y),
                'label_distribution': {'positive': int(n_pos), 'negative': int(n_neg)}
            }
        
        # Ensemble prediction
        y_pred_gb = self.model_gb.predict_proba(X)[:, 1]
        y_pred_rf = self.model_rf.predict_proba(X)[:, 1]
        y_pred_proba = (y_pred_gb + y_pred_rf) / 2.0
        
        # AUC
        auc = roc_auc_score(y, y_pred_proba)
        
        # Confusion matrix
        y_pred = (y_pred_proba >= 0.5).astype(int)
        cm = confusion_matrix(y, y_pred)
        
        if verbose:
            print(f"  [Adversary] AUC = {auc:.4f}")
            print(f"  [Adversary] Confusion Matrix:")
            print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
            print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        result = {
            'auc': float(auc),
            'success': True,
            'confusion_matrix': cm.tolist(),
            'n_samples': len(y),
            'label_distribution': {'positive': int(n_pos), 'negative': int(n_neg)}
        }
        
        if return_predictions:
            result['predictions'] = y_pred_proba
            result['labels'] = y
        
        return result


def time_split_trades(
    trades_df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split to avoid lookahead bias"""
    df = trades_df.sort_values('timestamp').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    return train, val, test
