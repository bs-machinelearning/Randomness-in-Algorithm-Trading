import numpy as np
import pandas as pd
from .baseline import generate_trades as baseline_generate

class OUPolicy:
    """
    Ornstein–Uhlenbeck mean-reverting noise policy.
    Adds correlated noise to prices.
    """

    def __init__(self, theta=0.15, sigma=0.02, price_scale=1.0, seed=None):
        self.theta = theta       # mean reversion strength
        self.sigma = sigma       # noise volatility
        self.price_scale = price_scale
        self.rng = np.random.default_rng(seed)

    def generate_ou_noise(self, n):
        x = np.zeros(n)
        for t in range(1, n):
            dx = self.theta * (0 - x[t-1]) + self.sigma * self.rng.normal()
            x[t] = x[t-1] + dx
        return x

    def generate_trades(self, prices: pd.DataFrame) -> pd.DataFrame:
        trades = baseline_generate(prices).copy()
        n = len(trades)

        noise = self.generate_ou_noise(n)
        trades["ref_price"] = trades["ref_price"] * (1 + self.price_scale * noise)

        return trades
