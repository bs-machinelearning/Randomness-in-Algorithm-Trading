"""
P7 Adaptive Adversary Framework - Week 3 Production Version

Features:
- Auto-detection of prediction window based on actual trade gaps
- SMOTE resampling for balanced training
- Cross-validation
- Comprehensive logging
- Convergence detection

Owner: P7
Week: 3
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
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
    
    PREDICTION_WINDOW_MINUTES = None
    USE_
