"""
Ornstein-Uhlenbeck execution policy (Section 7 of paper).

Discretisation (Euler-Maruyama, Δt = 1 trading day):
    X_{n+1} = X_n + θ(μ - X_n)Δt + σ√Δt · ε_n,   ε_n ~ N(0,1)

With Δt = 1 this simplifies to:
    X_{n+1} = X_n + θ(μ - X_n) + σ · ε_n

Stationary distribution: N(μ, σ²/(2θ))
Expected autocorrelations at paper defaults (θ=0.5, σ=0.5):
    ρ(1) ≈ exp(−θ) ≈ 0.606,  ρ(5) ≈ 0.082,  ρ(10) ≈ 0.007
"""

import numpy as np
import pandas as pd
from .baseline import generate_trades as baseline_generate

# ── Paper-specified defaults (Section 7) ─────────────────────────────────────
DEFAULT_OU_PARAMS = {
    "theta":       0.5,    # mean-reversion speed (paper §7 default)
    "sigma":       0.5,    # noise magnitude      (paper §7 default)
    "price_scale": 0.04,   # maps OU state → price deviation fraction
}

FAST_REVERSION_OU_PARAMS = {
    "theta":       1.0,
    "sigma":       0.5,
    "price_scale": 0.04,
}

SLOW_REVERSION_OU_PARAMS = {
    "theta":       0.1,
    "sigma":       0.5,
    "price_scale": 0.04,
}


class OUPolicy:
    """
    Ornstein-Uhlenbeck mean-reverting noise policy.

    Applies OU-correlated noise to ref_price:
        ref_price_t = price_t * (1 + price_scale * X_t)
    """

    def __init__(
        self,
        theta: float = DEFAULT_OU_PARAMS["theta"],
        sigma: float = DEFAULT_OU_PARAMS["sigma"],
        price_scale: float = DEFAULT_OU_PARAMS["price_scale"],
        mu: float = 0.0,
        seed: int = None,
    ):
        self.theta = theta
        self.sigma = sigma
        self.price_scale = price_scale
        self.mu = mu
        self.rng = np.random.default_rng(seed)

    def _ou_noise(self, n: int) -> np.ndarray:
        """
        Simulate n steps of OU via Euler-Maruyama with Δt=1.

        X_0 is drawn from the stationary distribution N(μ, σ²/(2θ))
        to avoid a burn-in transient.
        """
        stat_std = self.sigma / np.sqrt(max(2.0 * self.theta, 1e-8))
        x = np.empty(n)
        x[0] = self.rng.normal(self.mu, stat_std)   # stationary initialisation

        for t in range(1, n):
            # Euler-Maruyama: ΔX = θ(μ - X_t)Δt + σ√Δt · ε,  Δt=1 → √Δt=1
            x[t] = (
                x[t - 1]
                + self.theta * (self.mu - x[t - 1])
                + self.sigma * self.rng.normal()
            )
        return x

    def generate_trades(self, prices: pd.DataFrame) -> pd.DataFrame:
        trades = baseline_generate(prices).copy()
        n = len(trades)
        if n == 0:
            return trades

        if "price" not in trades.columns:
            trades["price"] = trades.get("ref_price", prices["price"].values)

        if "ref_price" not in trades.columns:
            trades["ref_price"] = trades["price"]

        noise = self._ou_noise(n)
        trades["ref_price"] = trades["ref_price"] * (1.0 + self.price_scale * noise)

        return trades


# ── Module-level runner entrypoint for policy='ou' ───────────────────────────

def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """Runner entrypoint: uses paper-default OU parameters."""
    policy = OUPolicy(
        theta=DEFAULT_OU_PARAMS["theta"],
        sigma=DEFAULT_OU_PARAMS["sigma"],
        price_scale=DEFAULT_OU_PARAMS["price_scale"],
        seed=42,
    )
    return policy.generate_trades(prices)


__all__ = [
    "OUPolicy",
    "DEFAULT_OU_PARAMS",
    "FAST_REVERSION_OU_PARAMS",
    "SLOW_REVERSION_OU_PARAMS",
    "generate_trades",
]
