"""
Pink-noise (1/f) execution policy (Section 8 of paper).

FFT pipeline (Section 8):
    1. Draw white noise:  z ~ N(0, I)
    2. FFT:               Z = FFT(z)
    3. Apply filter:      Z_f = Z * f^(-α/2)   (DC component = 1e-10)
    4. IFFT and take real part
    5. Standardise to mean=0, std=1

Power spectrum: S(f) = |Z_f|^2 ∝ 1/f^α  (pink noise when α=1)

Expected autocorrelations at α=1.0:
    ρ(1) ≈ 0.45,  ρ(5) ≈ 0.20,  ρ(20) ≈ 0.10
"""

import numpy as np
import pandas as pd
from .baseline import generate_trades as baseline_generate

# ── Paper-specified defaults (Section 8) ─────────────────────────────────────
DEFAULT_PINK_PARAMS = {
    "alpha":       1.0,    # spectral exponent (1.0 = pink / 1/f noise)
    "price_scale": 0.04,   # maps noise → price deviation fraction
}

WHITE_NOISE_PARAMS = {
    "alpha":       0.0,    # flat spectrum (white noise)
    "price_scale": 0.04,
}

BROWN_NOISE_PARAMS = {
    "alpha":       2.0,    # 1/f² spectrum (Brownian motion / brown noise)
    "price_scale": 0.04,
}


class PinkPolicy:
    """
    Pink-noise (1/f^α) execution policy.

    Applies spectrally-coloured noise to ref_price:
        ref_price_t = price_t * (1 + price_scale * noise_t)
    where noise is standardised to mean=0, std=1.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_PINK_PARAMS["alpha"],
        price_scale: float = DEFAULT_PINK_PARAMS["price_scale"],
        seed: int = None,
    ):
        self.alpha = alpha
        self.price_scale = price_scale
        self.rng = np.random.default_rng(seed)

    def generate_pink_noise(self, n: int) -> np.ndarray:
        """
        Generate n samples of 1/f^alpha noise via the FFT method.

        Pipeline:
            z ~ N(0, I)  →  FFT  →  multiply by f^(-alpha/2)
            (DC = 1e-10)  →  IFFT  →  real part  →  standardise
        """
        # Step 1: white noise in frequency domain
        z_real = self.rng.normal(size=n)
        Z = np.fft.rfft(z_real)

        # Step 2: frequency array; replace DC (f=0) with small positive value
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1e-10   # DC component per Section 8 (never divide by zero)

        # Step 3: apply f^(-alpha/2) filter — gives power spectrum ∝ 1/f^alpha
        filt = freqs ** (-self.alpha / 2.0)
        Z_filtered = Z * filt

        # Step 4: IFFT, take real part
        noise = np.fft.irfft(Z_filtered, n)

        # Step 5: standardise to zero mean, unit variance
        std = noise.std()
        if std < 1e-8:
            return np.zeros(n)
        noise = (noise - noise.mean()) / std

        return noise

    def generate_trades(self, prices: pd.DataFrame) -> pd.DataFrame:
        trades = baseline_generate(prices).copy()
        n = len(trades)
        if n == 0:
            return trades

        if "price" not in trades.columns:
            trades["price"] = trades.get("ref_price", prices["price"].values)

        if "ref_price" not in trades.columns:
            trades["ref_price"] = trades["price"]

        noise = self.generate_pink_noise(n)
        trades["ref_price"] = trades["ref_price"] * (1.0 + self.price_scale * noise)

        return trades


# Alias expected by bsml.policies.__init__
PinkNoisePolicy = PinkPolicy


# ── Module-level runner entrypoint for policy='pink' ─────────────────────────

def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """Runner entrypoint: uses paper-default pink-noise parameters."""
    policy = PinkPolicy(
        alpha=DEFAULT_PINK_PARAMS["alpha"],
        price_scale=DEFAULT_PINK_PARAMS["price_scale"],
        seed=42,
    )
    return policy.generate_trades(prices)


__all__ = [
    "PinkPolicy",
    "PinkNoisePolicy",
    "DEFAULT_PINK_PARAMS",
    "WHITE_NOISE_PARAMS",
    "BROWN_NOISE_PARAMS",
    "generate_trades",
]
