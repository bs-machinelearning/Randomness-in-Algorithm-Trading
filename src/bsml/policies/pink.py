import pandas as pd
from .pink_policy import PinkPolicy


def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Runner entrypoint for policy='pink'.

    Delegates to PinkPolicy, which:
    - Starts from the baseline schedule
    - Adds pink-noise (1/f) perturbations to ref_price
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame(columns=["date", "symbol", "side", "qty", "ref_price"])

    # You can tweak alpha/price_scale to change noise character
    policy = PinkPolicy(alpha=1.0, price_scale=0.04, seed=42)
    return policy.generate_trades(prices)
