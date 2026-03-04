import numpy as np
import pandas as pd
from .baseline import generate_trades as baseline_generate

# Hard bounds enforced regardless of instance params
_MAX_ABS_PRICE_JITTER = 0.0005   # ±$0.0005 absolute per Section 6 of paper
_MAX_TIME_MINUTES = 120          # ±120 minutes per Section 6 of paper

# Parameter presets expected by bsml.policies.__init__
DEFAULT_UNIFORM_PARAMS = {
    "price_noise": 0.0005,        # ±$0.0005 absolute (not multiplicative ±3%)
    "time_noise_minutes": 120,    # ±120 minutes (not ±30 min)
}

CONSERVATIVE_UNIFORM_PARAMS = {
    "price_noise": 0.0002,
    "time_noise_minutes": 60,
}

AGGRESSIVE_UNIFORM_PARAMS = {
    "price_noise": 0.0005,
    "time_noise_minutes": 120,
}

NOCLAMPING_UNIFORM_PARAMS = {
    "price_noise": 0.0005,
    "time_noise_minutes": 120,
}


class UniformPolicy:
    """
    Wraps the baseline schedule but applies uniform random perturbations
    to ref_price (additive absolute jitter) and timing (±minutes).

    Section 6: Δp_i ~ U(-0.0005, 0.0005), Δt_i ~ U(-120, 120) minutes.
    """

    def __init__(self, params=None, seed=None):
        if params is None:
            params = DEFAULT_UNIFORM_PARAMS
        self.price_noise = min(abs(params["price_noise"]), _MAX_ABS_PRICE_JITTER)
        self.time_noise_minutes = min(abs(params["time_noise_minutes"]), _MAX_TIME_MINUTES)
        self.rng = np.random.default_rng(seed)

    def perturb_price(self, price: float) -> float:
        """Additive absolute jitter: price + U(-price_noise, price_noise)."""
        delta = self.rng.uniform(-self.price_noise, self.price_noise)
        return price + delta

    def perturb_time(self, timestamp) -> pd.Timestamp:
        """Shift timestamp by uniform minutes within ±time_noise_minutes."""
        delta = self.rng.uniform(-self.time_noise_minutes, self.time_noise_minutes)
        return timestamp + pd.Timedelta(minutes=float(delta))

    def generate_trades(self, prices: pd.DataFrame) -> pd.DataFrame:
        trades = baseline_generate(prices).copy()
        n = len(trades)
        if n == 0:
            return trades

        # Additive absolute price jitter on ref_price only; 'price' stays as execution cost basis
        delta = self.rng.uniform(-self.price_noise, self.price_noise, size=n)
        delta = np.clip(delta, -_MAX_ABS_PRICE_JITTER, _MAX_ABS_PRICE_JITTER)
        trades["ref_price"] = trades["ref_price"] + delta

        # Timing jitter: shift date column
        dates = pd.to_datetime(trades["date"])
        trades["date"] = [
            self.perturb_time(ts.to_pydatetime()) for ts in dates
        ]

        return trades


def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Module-level entrypoint expected by the runner for policy='uniform_policy'.
    """
    policy = UniformPolicy(seed=42)
    return policy.generate_trades(prices)


__all__ = [
    "UniformPolicy",
    "DEFAULT_UNIFORM_PARAMS",
    "CONSERVATIVE_UNIFORM_PARAMS",
    "AGGRESSIVE_UNIFORM_PARAMS",
    "NOCLAMPING_UNIFORM_PARAMS",
    "generate_trades",
]
