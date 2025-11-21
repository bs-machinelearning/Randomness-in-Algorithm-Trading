import pandas as pd
from pathlib import Path
import yaml

from .baseline import generate_trades as baseline_generate_trades
from .ou_policy import OUPolicy, DEFAULT_OU_PARAMS

def generate_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Runner entrypoint for policy='ou'.

    1) Build deterministic baseline schedule.
    2) Use OUPolicy to add mean-reverting, autocorrelated noise to timing
       and ref_price.
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame(columns=["date", "symbol", "side", "qty", "ref_price"])

    required = {"date", "symbol", "price"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"ou.generate_trades: missing columns {missing}")

    # seed from config
    try:
        cfg = yaml.safe_load(Path("configs/run.yaml").read_text())
        seed = int(cfg.get("seed", 42))
    except Exception:
        seed = 42

    trades = baseline_generate_trades(prices).copy()

    policy = OUPolicy(seed=seed, params=DEFAULT_OU_PARAMS)

    # perturb timing
    perturbed_dates = []
    for ts in trades["date"]:
        ts_py = pd.to_datetime(ts).to_pydatetime()
        perturbed_dates.append(policy.perturb_timing(ts_py))
    trades["date"] = perturbed_dates

    # perturb thresholds via ref_price
    perturbed_ref = []
    for p in trades["ref_price"]:
        perturbed_ref.append(policy.perturb_threshold(float(p)))
    trades["ref_price"] = perturbed_ref

    return trades
