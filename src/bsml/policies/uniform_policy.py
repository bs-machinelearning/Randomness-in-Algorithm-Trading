import numpy as np
import pandas as pd
from .baseline import generate_trades as baseline_generate

class UniformPolicy:
    """
    Wraps the baseline schedule but applies *uniform* random perturbations
    to ref_price and optionally timing.
    """

    def __init__(self, price_noise=0.03, time_noise_minutes=30, seed=None):
        self.price_noise = price_noise
        self.time_noise_minutes = time_noise_minutes
        self.rng = np.random.default_rng(seed)

    def perturb_price(self, price):
        # price * (1 + U(-α, α))
        eps = self.rng.uniform(-self.price_noise, self.price_noise)
        return price * (1 + eps)

    def perturb_time(self, timestamp):
        # Shift timestamp by uniform minutes
        delta = self.rng.uniform(-self.time_noise_minutes, self.time_noise_minutes)
        return timestamp + pd.Timedelta(minutes=float(delta))

    def generate_trades(self, prices: pd.DataFrame) -> pd.DataFrame:
        trades = baseline_generate(prices)

        trades["ref_price"] = trades["ref_price"].apply(self.perturb_price)
        trades["date"] = trades["date"].apply(self.perturb_time)

        return trades
