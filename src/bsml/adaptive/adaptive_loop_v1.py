"""P7 Week 3: Adaptive Adversary v1.0"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / 'tests'))

import numpy as np
import pandas as pd
from bsml.policies import UniformPolicy, DEFAULT_UNIFORM_PARAMS
from bsml.data.loader import load_prices
from bsml.adaptive.bridge import enrich_trades_for_adversary
import Skeleton_Adversary_Model as sadv

AUC_HIGH = 0.75
AUC_LOW = 0.55
AUC_TARGET_MIN = 0.60
AUC_TARGET_MAX = 0.70

def decide_adjustment(auc):
    if auc > AUC_HIGH: return 'INCREASE', 1.20
    elif auc < AUC_LOW: return 'DECREASE', 0.80
    elif AUC_TARGET_MIN <= auc <= AUC_TARGET_MAX: return 'HOLD', 1.0
    else: return ('NUDGE_UP', 1.20) if auc > 0.65 else ('NUDGE_DOWN', 0.80)

def adaptive_training_loop(prices_df, max_iter=5):
    params = DEFAULT_UNIFORM_PARAMS.copy()
    policy = UniformPolicy(params=params, seed=42)
    results = []
    
    for i in range(max_iter):
        print(f"\nITER {i+1}: params={params}")
        trades = policy.generate_trades(prices_df)
        enriched = enrich_trades_for_adversary(trades, prices_df)
        features = sadv.extract_features(enriched, [10, 50])
        features['label'] = sadv.generate_labels(features, 1)
        features = features.dropna(subset=['label'])
        
        train, val, _ = sadv.make_time_splits(features, "2024-06-30", "2024-09-30")
        if len(val) < 50: val = train
        
        fcols = [c for c in features.columns if np.issubdtype(features[c].dtype, np.number) and c not in {'label','pnl'}]
        X_tr, y_tr = train[fcols].fillna(0), train['label'].values
        
        if y_tr.sum() in [0, len(y_tr)]:
            auc = 0.50
        else:
            model = sadv.train_adversary_classifier(X_tr, y_tr)
            X_val, y_val = val[fcols].fillna(0), val['label'].values
            auc = sadv.compute_auc(model, X_val, y_val) if y_val.sum() not in [0, len(y_val)] else 0.50
        
        print(f"AUC: {auc:.4f}")
        action, mult = decide_adjustment(auc)
        print(f"Action: {action}")
        results.append({'iter': i+1, 'auc': auc, 'action': action})
        
        if action != 'HOLD':
            params['price_noise'] = np.clip(params['price_noise'] * mult, 0.001, 0.20)
            params['time_noise_minutes'] = np.clip(params['time_noise_minutes'] * mult, 1, 180)
            policy = UniformPolicy(params=params, seed=42)
    
    return results

def main():
    print("P7 WEEK 3 PILOT")
    prices = load_prices("data/ALL_backtest.csv")
    print(f"Loaded {len(prices)} rows")
    results = adaptive_training_loop(prices, max_iter=5)
    print(f"\nResults: {results}")
    Path("outputs/adaptive_runs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv("outputs/adaptive_runs/results.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    main()
