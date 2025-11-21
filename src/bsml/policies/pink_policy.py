import numpy as np
import pandas as pd
from .baseline import generate_trades as baseline_generate

class PinkPolicy:
    """
    Pink-noise policy: low-frequency noise → persistent drifts.
    """

    def __init__(self, alpha=1.0, price_scale=0.04, seed=None):
        self.alpha = alpha
        self.price_scale = price_scale
        self.rng = np.random.default_rng(seed)

    def generate_pink_noise(self, n):
        # 1/f noise via FFT method
        freqs = np.fft.rfftfreq(n)
        phases = self.rng.normal(size=freqs.shape) + 1j * self.rng.normal(size=freqs.shape)

        # Avoid division by zero at freq 0
        spectrum = phases / np.where(freqs == 0, 1, freqs ** self.alpha)

        noise = np.fft.irfft(spectrum, n)
        noise = (noise - noise.mean()) / noise.std()

        return noise

    def generate_trades(self, prices: pd.DataFrame) -> pd.DataFrame:
        trades = baseline_generate(prices).copy()
        n = len(trades)

        noise = self.generate_pink_noise(n)
        trades["ref_price"] = trades["ref_price"] * (1 + self.price_scale * noise)

        return trades
