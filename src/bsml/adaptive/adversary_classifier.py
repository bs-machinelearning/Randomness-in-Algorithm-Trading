"""
P7 Adaptive Adversary Classifier - STRENGTHENED VERSION

Prediction task: "Will a trade occur tomorrow?"
Improvements: 3 models, stronger hyperparameters, weighted ensemble

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class P7AdaptiveAdversary:
    """
    STRENGTHENED adversary with:
    - 3 models (GB + RF + ExtraTrees)
    - Better hyperparameters
    - Weighted ensemble
    - 5-fold CV
    """
    
    def __init__(
        self,
        use_smote: bool = True,
        use_cv: bool = True,
        n_cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            use_smote: Apply SMOTE for label balancing
            use_cv: Use cross-validation
            n_cv_folds: Number of CV folds
            random_state: Random seed
        """
        self.use_smote = use_smote
        self.use_cv = use_cv
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        
        self.model_gb = None
        self.model_rf = None
        self.model_et = None
        self.feature_cols = None
        self.cv_scores = []
        
    def _select_features(self, df: pd.DataFrame) -> list:
        """Select feature columns"""
        exclude = {
            'date', 'symbol', 'signal', 'label', 'price'
        }
        
        feature_cols = [
            c for c in df.columns
            if c not in exclude and np.issubdtype(df[c].dtype, np.number)
        ]
        
        return feature_cols
    
    def train(self, data_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """Train STRENGTHENED adversary with 3 models"""
        if verbose:
            print(f"  [Adversary] Training STRENGTHENED classifier on {len(data_df)} samples...")
        
        self.feature_cols = self._select_features(data_df)
        
        if verbose:
            print(f"  [Adversary] Using {len(self.feature_cols)} features")
        
        X = data_df[self.feature_cols].fillna(0).values
        y = data_df['label'].values
        
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_rate = n_pos / len(y)
        
        if verbose:
            print(f"  [Adversary] Labels → 0: {n_neg}, 1: {n_pos} ({pos_rate*100:.1f}% positive)")
        
        if n_pos < 10 or n_neg < 10:
            return {
                'success': False,
                'reason': 'too_few_samples',
                'n_samples': len(X),
                'n_features': len(self.feature_cols),
                'label_distribution': {'positive': int(n_pos), 'negative': int(n_neg)}
            }
        
        # SMOTE
        if self.use_smote and (pos_rate < 0.4 or pos_rate > 0.6):
            if verbose:
                print(f"  [Adversary] Applying SMOTE...")
            
            try:
                smote = SMOTE(
                    sampling_strategy=0.5,
                    random_state=self.random_state,
                    k_neighbors=min(5, min(n_pos, n_neg) - 1)
                )
                X, y = smote.fit_resample(X, y)
                
                if verbose:
                    print(f"  [Adversary] After SMOTE → {y.sum()} positive ({y.sum()/len(y)*100:.1f}%)")
            except Exception as e:
                if verbose:
                    print(f"  [Warning] SMOTE failed: {e}")
        
        # Model 1: Gradient Boosting (STRONGER)
        if verbose:
            print(f"  [Adversary] Training Gradient Boosting (200 trees, depth 6)...")
        
        self.model_gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=self.random_state
        )
        self.model_gb.fit(X, y)
        
        # Model 2: Random Forest (STRONGER)
        if verbose:
            print(f"  [Adversary] Training Random Forest (200 trees, depth 10)...")
        
        self.model_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model_rf.fit(X, y)
        
        # Model 3: Extra Trees (NEW)
        if verbose:
            print(f"  [Adversary] Training Extra Trees (200 trees, depth 10)...")
        
        self.model_et = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            max_features='sqrt',
            bootstrap=False,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model_et.fit(X, y)
        
        # Cross-validation
        cv_results = {}
        if self.use_cv:
            if verbose:
                print(f"  [Adversary] Running {self.n_cv_folds}-fold CV...")
            
            X_cv = data_df[self.feature_cols].fillna(0).values
            y_cv = data_df['label'].values
            
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
        
        if verbose:
            print(f"  [Adversary] Training complete ✓")
        
        return {
            'success': True,
            'n_samples': len(X),
            'n_features': len(self.feature_cols),
            'label_distribution': {'positive': int(y.sum()), 'negative': int(len(y) - y.sum())},
            **cv_results
        }
    
    def evaluate(self, data_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """Evaluate with 3-model weighted ensemble"""
        if self.model_gb is None or self.model_rf is None or self.model_et is None:
            if verbose:
                print("  [ERROR] Models not trained!")
            return {
                'auc': 0.50,
                'success': False,
                'n_samples': 0,
                'label_distribution': {'positive': 0, 'negative': 0}
            }
        
        if verbose:
            print(f"  [Adversary] Evaluating (3-model ensemble)...")
        
        X = data_df[self.feature_cols].fillna(0).values
        y = data_df['label'].values
        
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        
        if verbose:
            print(f"  [Adversary] Val labels → 0: {n_neg}, 1: {n_pos} ({n_pos/len(y)*100:.1f}%)")
        
        if n_pos == 0 or n_neg == 0:
            if verbose:
                print(f"  [WARNING] Single class!")
            return {
                'auc': 0.50,
                'success': False,
                'n_samples': len(y),
                'label_distribution': {'positive': int(n_pos), 'negative': int(n_neg)}
            }
        
        # Ensemble prediction
        y_pred_gb = self.model_gb.predict_proba(X)[:, 1]
        y_pred_rf = self.model_rf.predict_proba(X)[:, 1]
        y_pred_et = self.model_et.predict_proba(X)[:, 1]
        
        # Weighted average (GB=40%, RF=30%, ET=30%)
        y_pred_proba = 0.4 * y_pred_gb + 0.3 * y_pred_rf + 0.3 * y_pred_et
        
        auc_score = roc_auc_score(y, y_pred_proba)
        
        if verbose:
            auc_gb = roc_auc_score(y, y_pred_gb)
            auc_rf = roc_auc_score(y, y_pred_rf)
            auc_et = roc_auc_score(y, y_pred_et)
            print(f"  [Adversary] Individual: GB={auc_gb:.4f}, RF={auc_rf:.4f}, ET={auc_et:.4f}")
            print(f"  [Adversary] Ensemble AUC = {auc_score:.4f}")
        
        return {
            'auc': float(auc_score),
            'success': True,
            'n_samples': len(y),
            'label_distribution': {'positive': int(n_pos), 'negative': int(n_neg)}
        }


def time_split_data(data_df: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple:
    """Chronological split"""
    df = data_df.sort_values('date').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    return train, val, test
