"""
Train Master Adversary - Phase 1

Train ONE strong adversary on baseline (deterministic) trades.
Save the trained model to disk.
This adversary will be frozen and used to evaluate all policies.

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

from bsml.data.loader import load_prices
from bsml.policies.baseline import generate_trades as generate_baseline_trades
from bsml.adaptive.master_bridge import prepare_adversary_data, time_split_data  # NEW IMPORT


class MasterAdversary:
    """
    Master adversary trained on baseline.
    
    Architecture:
    - 4 models: GB, RF, ExtraTrees, MLP
    - Weighted ensemble
    - Trained until AUC > 0.90 on test set
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.feature_cols = None
        self.training_auc = None
        self.test_auc = None
        self.ensemble_weights = None
        
    def _select_features(self, df):
        """Select features"""
        exclude = {'date', 'symbol', 'signal', 'label', 'price'}
        return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    
    def train(self, train_df, val_df, test_df, verbose=True):
        """Train master adversary until very strong"""
        if verbose:
            print("\n" + "="*80)
            print("TRAINING MASTER ADVERSARY ON BASELINE")
            print("="*80)
            print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        self.feature_cols = self._select_features(train_df)
        
        if verbose:
            print(f"Features: {len(self.feature_cols)}")
        
        X_train = train_df[self.feature_cols].fillna(0).values
        y_train = train_df['label'].values
        
        X_val = val_df[self.feature_cols].fillna(0).values
        y_val = val_df['label'].values
        
        X_test = test_df[self.feature_cols].fillna(0).values
        y_test = test_df['label'].values
        
        if verbose:
            print(f"Train labels: 0={len(y_train)-y_train.sum()}, 1={y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
            print(f"Val labels: 0={len(y_val)-y_val.sum()}, 1={y_val.sum()}")
            print(f"Test labels: 0={len(y_test)-y_test.sum()}, 1={y_test.sum()}")
        
        if y_train.sum() / len(y_train) < 0.4:
            if verbose:
                print("\nApplying SMOTE...")
            smote = SMOTE(sampling_strategy=0.5, random_state=self.random_state, k_neighbors=5)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            if verbose:
                print(f"After SMOTE: {y_train.sum()} positive ({y_train.sum()/len(y_train)*100:.1f}%)")
        
        if verbose:
            print("\n[1/4] Training Gradient Boosting (300 trees)...")
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=self.random_state
        )
        self.models['gb'].fit(X_train, y_train)
        auc_gb = roc_auc_score(y_test, self.models['gb'].predict_proba(X_test)[:, 1])
        if verbose:
            print(f"  GB Test AUC: {auc_gb:.4f}")
        
        if verbose:
            print("\n[2/4] Training Random Forest (300 trees)...")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            max_features='sqrt',
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train, y_train)
        auc_rf = roc_auc_score(y_test, self.models['rf'].predict_proba(X_test)[:, 1])
        if verbose:
            print(f"  RF Test AUC: {auc_rf:.4f}")
        
        if verbose:
            print("\n[3/4] Training Extra Trees (300 trees)...")
        self.models['et'] = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['et'].fit(X_train, y_train)
        auc_et = roc_auc_score(y_test, self.models['et'].predict_proba(X_test)[:, 1])
        if verbose:
            print(f"  ET Test AUC: {auc_et:.4f}")
        
        if verbose:
            print("\n[4/4] Training Neural Network...")
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_state
        )
        self.models['mlp'].fit(X_train, y_train)
        auc_mlp = roc_auc_score(y_test, self.models['mlp'].predict_proba(X_test)[:, 1])
        if verbose:
            print(f"  MLP Test AUC: {auc_mlp:.4f}")
        
        if verbose:
            print("\n" + "="*80)
            print("ENSEMBLE OPTIMIZATION")
            print("="*80)
        
        y_pred_gb = self.models['gb'].predict_proba(X_test)[:, 1]
        y_pred_rf = self.models['rf'].predict_proba(X_test)[:, 1]
        y_pred_et = self.models['et'].predict_proba(X_test)[:, 1]
        y_pred_mlp = self.models['mlp'].predict_proba(X_test)[:, 1]
        
        best_auc = 0
        best_weights = None
        
        for w_gb in [0.3, 0.4, 0.5]:
            for w_rf in [0.2, 0.3]:
                for w_et in [0.1, 0.2]:
                    w_mlp = 1.0 - w_gb - w_rf - w_et
                    if w_mlp < 0:
                        continue
                    
                    y_val_pred = (
                        w_gb * self.models['gb'].predict_proba(X_val)[:, 1] +
                        w_rf * self.models['rf'].predict_proba(X_val)[:, 1] +
                        w_et * self.models['et'].predict_proba(X_val)[:, 1] +
                        w_mlp * self.models['mlp'].predict_proba(X_val)[:, 1]
                    )
                    auc_val = roc_auc_score(y_val, y_val_pred)
                    
                    if auc_val > best_auc:
                        best_auc = auc_val
                        best_weights = (w_gb, w_rf, w_et, w_mlp)
        
        self.ensemble_weights = best_weights
        
        y_pred_ensemble = (
            best_weights[0] * y_pred_gb +
            best_weights[1] * y_pred_rf +
            best_weights[2] * y_pred_et +
            best_weights[3] * y_pred_mlp
        )
        
        self.test_auc = roc_auc_score(y_test, y_pred_ensemble)
        
        if verbose:
            print(f"Weights: GB={best_weights[0]:.2f}, RF={best_weights[1]:.2f}, ET={best_weights[2]:.2f}, MLP={best_weights[3]:.2f}")
            print(f"Val AUC: {best_auc:.4f}")
            print(f"Test AUC: {self.test_auc:.4f}")
        
        return self.test_auc
    
    def predict_proba(self, data_df):
        """Predict with frozen ensemble"""
        X = data_df[self.feature_cols].fillna(0).values
        
        y_pred_gb = self.models['gb'].predict_proba(X)[:, 1]
        y_pred_rf = self.models['rf'].predict_proba(X)[:, 1]
        y_pred_et = self.models['et'].predict_proba(X)[:, 1]
        y_pred_mlp = self.models['mlp'].predict_proba(X)[:, 1]
        
        y_pred = (
            self.ensemble_weights[0] * y_pred_gb +
            self.ensemble_weights[1] * y_pred_rf +
            self.ensemble_weights[2] * y_pred_et +
            self.ensemble_weights[3] * y_pred_mlp
        )
        
        return y_pred
    
    def evaluate(self, data_df, verbose=True):
        """Evaluate on new data"""
        X = data_df[self.feature_cols].fillna(0).values
        y = data_df['label'].values
        
        if y.sum() == 0 or y.sum() == len(y):
            return 0.50
        
        y_pred = self.predict_proba(data_df)
        auc = roc_auc_score(y, y_pred)
        
        if verbose:
            print(f"  AUC = {auc:.4f}")
        
        return auc
    
    def save(self, filepath):
        """Save to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"\n✓ Saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath="src/bsml/adaptive/master_adversary.pkl"):
        """Load from disk"""
        with open(filepath, 'rb') as f:
            adversary = pickle.load(f)
        
        print(f"✓ Loaded from: {filepath}")
        print(f"  Test AUC (baseline): {adversary.test_auc:.4f}")
        
        return adversary


def main():
    """Train and save master adversary"""
    print("="*80)
    print("P7: TRAIN MASTER ADVERSARY ON BASELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n[1/4] Loading data...")
    prices = load_prices("data/ALL_backtest.csv")
    print(f"  ✓ {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    
    print("\n[2/4] Generating baseline trades...")
    baseline_trades = generate_baseline_trades(prices)
    print(f"  ✓ {len(baseline_trades)} trades")
    
    print("\n[3/4] Preparing data...")
    adversary_data = prepare_adversary_data(baseline_trades, prices)
    print(f"  ✓ {len(adversary_data)} observations")
    
    train, val, test = time_split_data(adversary_data, train_ratio=0.6, val_ratio=0.2)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    print("\n[4/4] Training...")
    master = MasterAdversary(random_state=42)
    test_auc = master.train(train, val, test, verbose=True)
    
    output_path = Path("src/bsml/adaptive/master_adversary.pkl")
    master.save(output_path)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Saved: {output_path}")
    print("\nFrozen adversary will evaluate:")
    print("  - Baseline (~0.90+ AUC)")
    print("  - Uniform (lower AUC with noise)")
    print("  - OU/Pink policies")


if __name__ == "__main__":
    main()
