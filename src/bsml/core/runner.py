from pathlib import Path
from datetime import datetime
import csv
import importlib
import yaml
import numpy as np
import pandas as pd

# P3 components you already own
from bsml.data.loader import load_prices
from bsml.utils.logging import run_id_from_cfg, prepare_outdir, snapshot
from bsml.cost.models import load_cost_config, apply_costs


# ── Metric helpers ────────────────────────────────────────────────────────────

def _compute_sharpe(trades_costed: pd.DataFrame) -> float:
    """Annualised Sharpe ratio from signed daily P&L."""
    if trades_costed.empty:
        return 0.0
    tc = trades_costed.copy()
    tc["date"] = pd.to_datetime(tc["date"])
    # signed weight: positive=long, negative=short
    sign = tc["side"].map({"BUY": 1.0, "SELL": -1.0}).fillna(0.0)
    tc["pnl"] = sign * tc["qty"] * tc.get("net_price", tc["price"])
    daily = tc.groupby("date")["pnl"].sum()
    if daily.std() == 0 or len(daily) < 2:
        return 0.0
    return float(np.sqrt(252) * daily.mean() / daily.std())


def _compute_maxdd(trades_costed: pd.DataFrame) -> float:
    """Maximum peak-to-trough drawdown on cumulative P&L series."""
    if trades_costed.empty:
        return 0.0
    tc = trades_costed.copy()
    tc["date"] = pd.to_datetime(tc["date"])
    sign = tc["side"].map({"BUY": 1.0, "SELL": -1.0}).fillna(0.0)
    tc["pnl"] = sign * tc["qty"] * tc.get("net_price", tc["price"])
    cumulative = tc.groupby("date")["pnl"].sum().cumsum()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / (rolling_max.abs() + 1e-8)
    return float(drawdown.min())  # negative number; more negative = larger drawdown


def _compute_is_bps(trades_costed: pd.DataFrame) -> float:
    """
    Mean implementation shortfall in basis points.
    IS = (net_price - ref_price) / ref_price * 10_000
    Positive = execution was worse than arrival price.
    """
    if trades_costed.empty or "ref_price" not in trades_costed.columns:
        return 0.0
    tc = trades_costed.dropna(subset=["ref_price", "price"])
    if tc.empty:
        return 0.0
    net_col = "net_price" if "net_price" in tc.columns else "price"
    is_bps = ((tc[net_col] - tc["ref_price"]) / tc["ref_price"].clip(lower=1e-8)) * 10_000
    return float(is_bps.mean())


def _compute_auc(trades_costed: pd.DataFrame) -> float:
    """
    AUC-ROC from adversary classifier on the run's trades.
    Returns 0.5 on any failure (graceful degradation).
    """
    try:
        from bsml.policies.adversary import AdversaryClassifier
        clf = AdversaryClassifier()
        return clf.train_and_evaluate(trades_costed)
    except Exception:
        return 0.5


# ── Main runner ───────────────────────────────────────────────────────────────

def main():
    """
    P3 runner: orchestrates a run in a reproducible, config-driven way.

    Order of operations (and why):
    1) read config       -> single source of truth for all parameters
    2) prepare out dir   -> where this run's files will live
    3) snapshot configs  -> reproducibility (log exact YAML + timestamp)
    4) load prices       -> validated input table (schema checked by loader)
    5) load costs cfg    -> numbers used later by cost wiring
    6) call P2 policy    -> get intended trades (P2 responsibility)
    7) apply costs       -> attach execution placeholders (P3 wiring)
    8) compute metrics   -> Sharpe, MaxDD, IS, AUC
    9) write CSVs        -> tidy outputs other roles will consume
    """

    # 1) Read main configuration (config-driven pipeline)
    cfg = yaml.safe_load(Path("configs/run.yaml").read_text())

    # 2) Derive a stable run folder from the config
    run_id = run_id_from_cfg(cfg)
    out_dir = prepare_outdir(cfg["output_dir"], run_id)

    # 3) Snapshot configs early (so even preflight runs are logged)
    snapshot(out_dir)

    # 4) Load input prices (schema: date, symbol, price)
    prices_path = Path("data/ALL_backtest.csv")
    prices = load_prices(prices_path)

    # 5) Load cost parameters from YAML
    costs_cfg = load_cost_config(cfg["costs"])

    # 6) Import policy dynamically and generate trades
    policy_name = cfg.get("policy", "baseline")

    try:
        policy_mod = importlib.import_module(f"bsml.policies.{policy_name}")
    except ModuleNotFoundError as e:
        (out_dir / "STATUS.txt").write_text(
            f"Runner preflight complete, but policy module '{policy_name}' was not found.\n"
            f"Error: {e}\n"
        )
        print(f"Policy module '{policy_name}' not found. Runner preflight is complete.")
        return

    try:
        trades = policy_mod.generate_trades(prices)
    except NotImplementedError as e:
        (out_dir / "STATUS.txt").write_text(
            f"Runner preflight complete: policy '{policy_name}' not implemented yet.\n{e}\n"
        )
        print(f"Policy '{policy_name}' not implemented yet. Runner preflight is complete.")
        return

    # 7) Apply cost wiring
    trades_costed = apply_costs(trades, costs_cfg)

    # 8) Compute real metrics
    sharpe = _compute_sharpe(trades_costed)
    maxdd = _compute_maxdd(trades_costed)
    delta_is_bps = _compute_is_bps(trades_costed)
    auc = _compute_auc(trades_costed)

    # 9) Write tidy CSVs for downstream roles
    trades.to_csv(out_dir / "trades_raw.csv", index=False)
    trades_costed.to_csv(out_dir / "trades_costed.csv", index=False)

    print(f"Run completed. Outputs in: {out_dir}")
    print(f"  Sharpe={sharpe:.4f}  MaxDD={maxdd:.4f}  IS={delta_is_bps:.2f}bps  AUC={auc:.4f}")

    # Append to results/seed_sweep.csv
    results_path = Path("results/seed_sweep.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "policy", "seed", "split",
        "sharpe", "delta_is_bps", "maxdd", "auc",
        "timestamp",
    ]

    exists = results_path.exists()
    with results_path.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            cfg.get("policy"), cfg.get("seed"), "all",
            round(sharpe, 6), round(delta_is_bps, 4), round(maxdd, 6), round(auc, 6),
            datetime.utcnow().isoformat(),
        ])


if __name__ == "__main__":
    main()
