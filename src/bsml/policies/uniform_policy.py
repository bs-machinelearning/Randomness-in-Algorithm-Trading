"""
Uniform randomization policy.

Adds IID (independent, identically distributed) uniform noise to timing
and thresholds. This is the simplest randomization strategy with no
temporal correlation.

Critical Design:
- Uniform distribution: all values in range equally likely
- No autocorrelation: each perturbation is independent
- Configurable market hours clamping

Owner: P4
Week: 2
"""

from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional
import numpy as np

from .base_policy import RandomizationPolicy
from .utils import (
    clamp_to_market_hours,
    check_market_hours,
    validate_parameter_bounds
)


class UniformPolicy(RandomizationPolicy):
    """
    Uniform randomization policy.
    
    Adds uniform random noise to timing and thresholds with no temporal
    correlation. Each perturbation is independent.
    
    Parameters:
        timing_range_hours (float): Max shift in hours (e.g., 2.0 = ±2 hours)
        threshold_pct (float): Max threshold shift as % (e.g., 0.10 = ±10%)
        respect_market_hours (bool): If True, clamp to 9:30 AM - 4:00 PM
    
    Statistical Properties:
        For uniform distribution on [-a, a]:
        - Mean = 0
        - Std = a / √3 ≈ 0.577 * a
        - All values in [-a, a] equally likely
    
    Example:
        >>> policy = UniformPolicy(
        ...     seed=42,
        ...     params={
        ...         'timing_range_hours': 2.0,
        ...         'threshold_pct': 0.10,
        ...         'respect_market_hours': True
        ...     }
        ... )
        >>> 
        >>> # Perturb timing
        >>> original = datetime(2025, 7, 15, 10, 30)
        >>> perturbed = policy.perturb_timing(original)
        >>> print(perturbed)  # e.g., 2025-07-15 11:47:00
        >>> 
        >>> # Perturb threshold
        >>> base_price = 150.00
        >>> perturbed_price = policy.perturb_threshold(base_price)
        >>> print(perturbed_price)  # e.g., 151.50
    """
    
    # Parameter bounds (soft limits with warnings)
    PARAMETER_BOUNDS = {
        'timing_range_hours': {'min': 0.1, 'max': 6.5, 'default': 2.0},
        'threshold_pct': {'min': 0.01, 'max': 0.30, 'default': 0.10},
    }
    
    def __init__(self, seed: int, params: Dict[str, Any]):
        """
        Initialize Uniform policy.
        
        Args:
            seed: Master seed for reproducibility
            params: Must contain 'timing_range_hours' and 'threshold_pct'
                    Optional: 'respect_market_hours' (default True)
        
        Raises:
            ValueError: If required parameters are missing
        """
        # Validate required parameters
        required = ['timing_range_hours', 'threshold_pct']
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(
                f"UniformPolicy requires parameters: {', '.join(missing)}. "
                f"Got: {list(params.keys())}"
            )
        
        # Set defaults for optional parameters
        if 'respect_market_hours' not in params:
            params['respect_market_hours'] = True
        
        # Initialize base class
        super().__init__(seed, params)
        
        # Validate parameter bounds
        self._validate_params()
        
        # Track statistics for diagnostics
        self._timing_shifts = []
        self._threshold_shifts = []
        
        # Market hours
        self._market_open = time(9, 30)
        self._market_close = time(16, 0)
    
    def _validate_params(self) -> None:
        """Validate parameters are within reasonable bounds."""
        for key, bounds in self.PARAMETER_BOUNDS.items():
            if key in self.params:
                self.params[key] = validate_parameter_bounds(
                    key, 
                    self.params[key],
                    bounds['min'],
                    bounds['max'],
                    warn_only=True
                )
    
    def perturb_timing(
        self, 
        timestamp: datetime, 
        signal_strength: float = 1.0
    ) -> datetime:
        """
        Perturb timing with uniform random noise.
        
        Critical Logic:
        1. Generate uniform random shift in [-range, +range]
        2. Scale by signal strength (optional)
        3. Apply shift to timestamp
        4. Optionally clamp to market hours
        
        Args:
            timestamp: Original signal timestamp
            signal_strength: Optional scaling factor [0, 1]
            
        Returns:
            Perturbed timestamp
        
        Example:
            >>> policy = UniformPolicy(seed=42, params={
            ...     'timing_range_hours': 2.0,
            ...     'threshold_pct': 0.10
            ... })
            >>> 
            >>> ts = datetime(2025, 7, 15, 10, 30)
            >>> perturbed = policy.perturb_timing(ts)
            >>> 
            >>> # Shift is within ±2 hours
            >>> delta = (perturbed - ts).total_seconds() / 3600
            >>> assert -2.0 <= delta <= 2.0
        """
        # Generate uniform random shift in hours
        shift_hours = self.rng.uniform(
            -self.params['timing_range_hours'],
            self.params['timing_range_hours']
        )
        
        # Scale by signal strength if provided
        if signal_strength != 1.0:
            shift_hours *= signal_strength
        
        # Apply shift
        perturbed = timestamp + timedelta(hours=shift_hours)
        
        # Optionally clamp to market hours
        if self.params.get('respect_market_hours', True):
            perturbed = clamp_to_market_hours(
                perturbed,
                self._market_open,
                self._market_close
            )
        
        # Log for diagnostics
        self._timing_shifts.append(shift_hours)
        
        return perturbed
    
    def perturb_threshold(
        self, 
        base_threshold: float, 
        current_price: Optional[float] = None
    ) -> float:
        """
        Perturb threshold with uniform random noise.
        
        Critical Logic:
        1. Generate uniform random shift as percentage
        2. Apply multiplicative shift (1 + pct_shift)
        3. Ensure positive price
        
        Args:
            base_threshold: Original threshold value
            current_price: Not used in Uniform policy (kept for interface)
            
        Returns:
            Perturbed threshold (always positive)
        
        Example:
            >>> policy = UniformPolicy(seed=42, params={
            ...     'timing_range_hours': 2.0,
            ...     'threshold_pct': 0.10
            ... })
            >>> 
            >>> base = 150.00
            >>> perturbed = policy.perturb_threshold(base)
            >>> 
            >>> # Shift is within ±10%
            >>> pct_change = (perturbed - base) / base
            >>> assert -0.10 <= pct_change <= 0.10
        """
        # Generate uniform random shift as percentage
        shift_pct = self.rng.uniform(
            -self.params['threshold_pct'],
            self.params['threshold_pct']
        )
        
        # Apply shift (multiplicative)
        perturbed = base_threshold * (1.0 + shift_pct)
        
        # Ensure positive price (critical for financial validity)
        perturbed = max(perturbed, 0.01)
        
        # Log for diagnostics
        self._threshold_shifts.append(shift_pct)
        
        return perturbed
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return Uniform policy diagnostics.
        
        Includes:
        - Policy metadata (name, seed, params)
        - Timing shift statistics (mean, std, min, max)
        - Threshold shift statistics (mean, std, min, max)
        - Exposure violation statistics
        
        Returns:
            Dictionary with comprehensive diagnostics
        
        Example:
            >>> policy = UniformPolicy(seed=42, params={...})
            >>> 
            >>> # Generate perturbations
            >>> for _ in range(100):
            ...     policy.perturb_timing(datetime(2025, 7, 15, 10, 30))
            ...     policy.perturb_threshold(150.0)
            >>> 
            >>> diag = policy.get_diagnostics()
            >>> print(diag['timing']['mean_shift_hours'])  # Should be ~0
            >>> print(diag['timing']['std_shift_hours'])   # Should be range/√3
        """
        diagnostics = {
            'policy': 'Uniform',
            'seed': self.seed,
            'params': self.params.copy(),
            'n_perturbations': len(self._timing_shifts) + len(self._threshold_shifts),  # FIXED: Use + instead of max()
        }
        
        # Timing shift statistics
        if self._timing_shifts:
            timing_array = np.array(self._timing_shifts)
            diagnostics['timing'] = {
                'mean_shift_hours': float(np.mean(timing_array)),
                'std_shift_hours': float(np.std(timing_array)),
                'min_shift_hours': float(np.min(timing_array)),
                'max_shift_hours': float(np.max(timing_array)),
                'expected_std': self.params['timing_range_hours'] / np.sqrt(3),
                'n_samples': len(timing_array)
            }
        
        # Threshold shift statistics
        if self._threshold_shifts:
            threshold_array = np.array(self._threshold_shifts)
            diagnostics['threshold'] = {
                'mean_shift_pct': float(np.mean(threshold_array)),
                'std_shift_pct': float(np.std(threshold_array)),
                'min_shift_pct': float(np.min(threshold_array)),
                'max_shift_pct': float(np.max(threshold_array)),
                'expected_std': self.params['threshold_pct'] / np.sqrt(3),
                'n_samples': len(threshold_array)
            }
        
        # Exposure violation statistics
        exposure_log = self.get_exposure_log()
        if exposure_log:
            violations = [log for log in exposure_log if not log['valid']]
            diagnostics['exposure'] = {
                'n_checks': len(exposure_log),
                'n_violations': len(violations),
                'violation_rate': len(violations) / len(exposure_log) if exposure_log else 0.0,
                'mean_delta': float(np.mean([log['abs_delta'] for log in exposure_log])),
                'max_delta': float(np.max([log['abs_delta'] for log in exposure_log])),
            }
        
        # Adjustment history
        adjustment_log = self.get_adjustment_log()
        if adjustment_log:
            diagnostics['adjustments'] = {
                'n_adjustments': len(adjustment_log),
                'last_auc': adjustment_log[-1]['auc_score'],
                'last_direction': adjustment_log[-1]['direction'],
            }
        
        return diagnostics
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"UniformPolicy(seed={self.seed}, "
            f"timing_range={self.params['timing_range_hours']}h, "
            f"threshold_pct={self.params['threshold_pct']*100:.1f}%)"
        )

    # --- Runner interface shim (required by P3) ---
    def generate_trades(prices: "pd.DataFrame") -> "pd.DataFrame":
        """
        Standard entry point required by the runner (P3).
        Must return a DataFrame with columns:
          ['date','symbol','side','qty','ref_price'].
        P4 will implement the actual uniform policy logic here.
        """
        raise NotImplementedError("Uniform policy: implement generate_trades(prices).")



# ============================================================================
# Predefined Parameter Sets
# ============================================================================

DEFAULT_UNIFORM_PARAMS = {
    'timing_range_hours': 2.0,   # ±2 hours
    'threshold_pct': 0.10,        # ±10%
    'respect_market_hours': True
}

CONSERVATIVE_UNIFORM_PARAMS = {
    'timing_range_hours': 1.0,   # ±1 hour
    'threshold_pct': 0.05,        # ±5%
    'respect_market_hours': True
}

AGGRESSIVE_UNIFORM_PARAMS = {
    'timing_range_hours': 4.0,   # ±4 hours
    'threshold_pct': 0.20,        # ±20%
    'respect_market_hours': True
}

NOCLAMPING_UNIFORM_PARAMS = {
    'timing_range_hours': 2.0,   # ±2 hours
    'threshold_pct': 0.10,        # ±10%
    'respect_market_hours': False  # Allow perturbations outside market hours
}
