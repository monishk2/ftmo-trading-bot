"""
Walk-Forward Optimization
=========================

Prevents curve-fitting by splitting historical data into rolling
in-sample / out-of-sample windows, optimising parameters only on
in-sample data, then recording performance on never-seen out-of-sample
data.

Window design
-------------
  in_sample_months  = 6   (optimise parameters here)
  out_of_sample_months = 3 (test the winning parameters here, untouched)
  step_months          = 3 (advance both windows by 3 months each iteration)

  Window 1: IS = Jan–Jun, OOS = Jul–Sep
  Window 2: IS = Apr–Sep, OOS = Oct–Dec
  …and so on until the dataset is exhausted.

Parameter grid (London Breakout)
----------------------------------
  min_asian_range_pips : [10, 15, 20, 25, 30, 35, 40, 45, 50]
  entry_buffer_pips    : [1, 2, 3, 4, 5]
  risk_reward_ratio    : [1.5, 2.0, 2.5, 3.0]
  risk_per_trade_pct   : [0.5, 0.75, 1.0]

Selection criterion
--------------------
  Best Sharpe ratio among all parameter combinations that produced
  >= min_trades (20) during the in-sample window.

Critical: ALL test runs include dynamic slippage, realistic spread,
and round-trip commission as configured in instruments.json.
Parameters that only work without execution costs are not real edges.

Robustness metrics
------------------
  degradation_ratio = oos_sharpe / is_sharpe
  Target: > 0.6 per window, average > 0.4 (otherwise WARNING).

Output
------
  WalkForwardResult dataclass + console table + optional JSON dump.

CLI
---
  python -m backtesting.walk_forward \\
      --data data/historical/EURUSD_15m.parquet \\
      --strategy london_breakout \\
      [--output reports/wf_result.json] \\
      [--initial-balance 10000] \\
      [--in-sample-months 6] \\
      [--oos-months 3] \\
      [--step-months 3] \\
      [--min-trades 20]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.backtester import Backtester, BacktestResult
from backtesting.metrics import _risk_adjusted   # private but stable enough

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent / "config"

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

_LONDON_GRID = {
    "min_asian_range_pips": [10, 15, 20, 25, 30, 35, 40, 45, 50],
    "entry_buffer_pips":    [1, 2, 3, 4, 5],
    "risk_per_trade_pct":   [0.5, 0.75, 1.0],
    # risk_reward_ratio fixed at base-config value (2.0) — not part of this search
}

_FVG_GRID = {
    # Prompt 13 refined grid: 4×3×4×4 = 192 combos (~60–90 min on EURUSD 15m)
    "min_fvg_size_pips":        [5, 10, 15, 20],           # pip size of the gap
    "entry_buffer_pips":        [1, 3, 5],                  # SL pad beyond zone edge
    "risk_per_trade_pct":       [0.25, 0.5, 0.75, 1.0],    # account % risked
    "max_candles_until_cancel": [8, 12, 16, 20],            # patience before cancel
    # risk_reward_ratio fixed at base-config value (2.0)
    # entry_window_start/end fixed (09:30–12:00 ET)
}

_NY_GRID = {
    # NY Session breakout — vary range width and buffer only
    # (range window 02:00–09:15 ET is fixed; entry window 09:30–11:00 ET is fixed)
    "min_asian_range_pips": [20, 25, 30, 35, 40, 50],
    "entry_buffer_pips":    [1, 2, 3, 4, 5],
    "risk_per_trade_pct":   [0.5, 0.7, 1.0],
}

_M15_GRID = {
    # M15 Momentum Scalping — 2×2×3×3×3×2 = 216 combinations
    "ema_fast":           [9, 12],
    "ema_slow":           [21, 26],
    "rsi_long_min":       [40, 45, 50],
    "rsi_long_max":       [60, 65, 70],
    "atr_tp_mult":        [1.5, 2.0, 2.5],
    "risk_per_trade_pct": [0.5, 0.7],
}

_H1_GRID = {
    # H1 Trend Following — 2×2×3×3×2 = 72 combinations
    "ema_fast":           [20, 26],
    "ema_slow":           [50, 60],
    "atr_tp_mult":        [1.5, 2.0, 2.5],
    "atr_sl_mult":        [1.0, 1.5, 2.0],
    "risk_per_trade_pct": [0.5, 0.7],
}

_GOLD_GRID = {
    # Gold Session Breakout — 3×3×3×3×3×3×3 = 2187 combinations
    # Reduce to manageable size: 3×3×2×2×3×3×2 = 324 combinations
    "buffer_pips":          [30, 50, 75],
    "min_range_pips":       [300, 500, 700],
    "max_range_pips":       [2000, 2500],
    "sl_cap_pips":          [1000, 1500],
    "rr_ratio":             [1.5, 2.0, 2.5],
    "risk_per_trade_pct":   [0.5, 0.8, 1.0],
    "vpa_volume_mult":      [1.2, 1.5],
}

_SWEEP_GRID = {
    # Gold Sweep Reversal — 2×1×2×2×2×2×2×2 = 256 combos
    "sweep_min_pips":        [15, 25],
    "sl_buffer_pips":        [10],
    "sl_cap_pips":           [600, 1000],
    "rr_ratio":              [2.0, 2.5],
    "risk_per_trade_pct":    [1.0, 1.5],
    "candle_body_atr_min":   [0.4, 0.6],
    "volume_mult":           [1.0, 1.3],
    "regime_atr_percentile": [35, 45],
}

_M1_SWEEP_GRID = {
    # Gold M1 Sweep Reversal — 3×2×2×3×2×2×2 = 288 combos
    # trend_guard_threshold fixed at 500 (design hypothesis, not searched)
    "sweep_min_pips":       [20, 30, 40],
    "sl_buffer_pips":       [5, 8],
    "sl_cap_pips":          [300, 500],
    "rr_ratio":             [2.0, 2.5, 3.0],
    "risk_per_trade_pct":   [1.0, 1.5],
    "candle_body_atr_min":  [0.2, 0.4],
    "volume_mult":          [1.0, 1.5],
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    window_idx:       int
    is_start:         str          # YYYY-MM
    is_end:           str
    oos_start:        str
    oos_end:          str
    best_params:      Dict[str, Any]
    is_sharpe:        float
    is_return_pct:    float
    is_trades:        int
    oos_sharpe:       float
    oos_return_pct:   float
    oos_trades:       int
    degradation_ratio: float       # oos_sharpe / is_sharpe  (NaN if is_sharpe == 0)


@dataclass
class WalkForwardResult:
    strategy_name:         str
    instrument:            str
    windows:               List[WindowResult]
    avg_degradation_ratio: float
    recommended_params:    Dict[str, Any]
    is_robust:             bool          # avg degradation > 0.4
    overfitting_warning:   bool          # avg degradation < 0.4
    grid_used:             Dict[str, List]
    initial_balance:       float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    instrument: str = "EURUSD",
    initial_balance: float = 10_000.0,
    in_sample_months: int = 6,
    oos_months: int = 3,
    step_months: int = 3,
    min_trades: int = 20,
    phase: str = "challenge",
    verbose: bool = True,
    param_grid: Optional[Dict[str, List]] = None,
    base_config_overrides: Optional[Dict] = None,
    use_fast_runner: bool = False,
) -> WalkForwardResult:
    """
    Execute walk-forward optimisation.

    Parameters
    ----------
    df :
        Full OHLCV history (15-min bars, any tz).  Will be sliced per window.
    strategy_name :
        "london_breakout" or "fvg_retracement".
    instrument :
        Key in instruments.json.
    initial_balance :
        Starting account balance for every window's backtest.
    in_sample_months :
        Number of months for the optimisation window.
    oos_months :
        Number of months for the validation window.
    step_months :
        Months to advance both windows at each iteration.
    min_trades :
        Minimum trades an IS run must produce to be considered valid.
    phase :
        FTMO phase passed to Backtester.
    verbose :
        If True, print progress to stdout.
    """
    strat_lower = strategy_name.lower().replace(" ", "_")
    if strat_lower in ("london_breakout", "london_open_breakout"):
        grid = _LONDON_GRID
        _build_strategy = _build_london_strategy
        _base_config    = _load_london_base_config()
    elif strat_lower in ("fvg_retracement", "fvg"):
        grid = _FVG_GRID
        _build_strategy = _build_fvg_strategy
        _base_config    = _load_fvg_base_config()
    elif strat_lower in ("ny_session_breakout", "ny_session", "ny_breakout"):
        grid = _NY_GRID
        _build_strategy = _build_ny_strategy
        _base_config    = _load_ny_base_config()
    elif strat_lower in ("m15_momentum_scalping", "m15_momentum", "m15"):
        grid = _M15_GRID
        _build_strategy = _build_m15_strategy
        _base_config    = _load_m15_base_config()
    elif strat_lower in ("h1_trend_following", "h1_trend", "h1"):
        grid = _H1_GRID
        _build_strategy = _build_h1_strategy
        _base_config    = _load_h1_base_config()
    elif strat_lower in ("gold_session_breakout", "gold_session", "gold"):
        grid = _GOLD_GRID
        _build_strategy = _build_gold_strategy
        _base_config    = _load_gold_base_config()
    elif strat_lower in ("gold_sweep_reversal", "gold_sweep", "sweep_reversal"):
        grid = _SWEEP_GRID
        _build_strategy = _build_sweep_strategy
        _base_config    = _load_sweep_base_config()
    elif strat_lower in ("gold_m1_sweep_reversal", "gold_m1_sweep", "m1_sweep"):
        grid = _M1_SWEEP_GRID
        _build_strategy = _build_m1_sweep_strategy
        _base_config    = _load_m1_sweep_base_config()
        use_fast_runner = True   # M1 data: always use sparse runner
    else:
        raise ValueError(f"Unknown strategy: {strategy_name!r}. "
                         "Use 'london_breakout', 'fvg_retracement', 'ny_session_breakout', "
                         "'m15_momentum_scalping', 'h1_trend_following', 'gold_session_breakout', "
                         "'gold_sweep_reversal', or 'gold_m1_sweep_reversal'.")

    # Caller-supplied grid takes precedence over the strategy-default grid
    if param_grid is not None:
        grid = param_grid

    # Caller-supplied config overrides (e.g. max_asian_range_pips floor for JPY pairs)
    if base_config_overrides:
        _base_config = {**_base_config, **base_config_overrides}

    instrument_cfg = _load_instrument_config(instrument)
    param_combos   = list(_expand_grid(grid))
    total_combos   = len(param_combos)

    if verbose:
        print(f"\nWalk-Forward Optimisation — {strategy_name} on {instrument}")
        print(f"  Grid: {total_combos} parameter combinations")
        print(f"  Windows: IS={in_sample_months}m  OOS={oos_months}m  step={step_months}m")
        print(f"  Min trades for valid IS run: {min_trades}")
        print(f"  Costs: spread + dynamic slippage + commission (ALWAYS applied)")
        print()

    # Normalise df index to UTC
    df = _normalise_df(df)
    windows_iter = list(_generate_windows(df, in_sample_months, oos_months, step_months))

    if not windows_iter:
        raise ValueError("Not enough data to form a single walk-forward window. "
                         f"Need at least {in_sample_months + oos_months} months.")

    window_results: List[WindowResult] = []

    for w_idx, (is_df, oos_df, is_start, is_end, oos_start, oos_end) in enumerate(windows_iter):
        if verbose:
            print(f"Window {w_idx+1}/{len(windows_iter)} "
                  f"IS: {is_start:%Y-%m} → {is_end:%Y-%m}  "
                  f"OOS: {oos_start:%Y-%m} → {oos_end:%Y-%m}", flush=True)

        # ── In-sample grid search ─────────────────────────────────────
        best_params: Optional[Dict]  = None
        best_sharpe:                 float = -np.inf

        _run = (lambda bt: bt.run_fast()) if use_fast_runner else (lambda bt: bt.run())

        for combo_idx, params in enumerate(param_combos):
            try:
                strategy = _build_strategy(_base_config, params, instrument_cfg)
                result   = _run(Backtester(
                    strategy=strategy,
                    df=is_df,
                    instrument=instrument,
                    initial_balance=initial_balance,
                    phase=phase,
                    seed=combo_idx,        # deterministic per combo
                ))

                if len(result.trades) < min_trades:
                    continue

                sharpe, _ = _risk_adjusted(result.equity_curve)
                if sharpe > best_sharpe:
                    best_sharpe  = sharpe
                    best_params  = params

            except Exception as exc:
                logger.debug("Combo %d failed: %s", combo_idx, exc)
                continue

        if best_params is None:
            if verbose:
                print(f"  ⚠ No valid IS parameter set found (all < {min_trades} trades). Skipping.")
            continue

        # ── Re-run IS with best params to get return % ────────────────
        is_strat   = _build_strategy(_base_config, best_params, instrument_cfg)
        is_result  = _run(Backtester(
            strategy=is_strat, df=is_df, instrument=instrument,
            initial_balance=initial_balance, phase=phase, seed=0,
        ))
        is_return  = (is_result.final_balance - initial_balance) / initial_balance * 100.0
        is_trades  = len(is_result.trades)

        # ── Out-of-sample run with frozen best params ─────────────────
        oos_strat  = _build_strategy(_base_config, best_params, instrument_cfg)
        oos_result = _run(Backtester(
            strategy=oos_strat, df=oos_df, instrument=instrument,
            initial_balance=initial_balance, phase=phase, seed=0,
        ))
        oos_sharpe, _ = _risk_adjusted(oos_result.equity_curve)
        oos_return    = (oos_result.final_balance - initial_balance) / initial_balance * 100.0
        oos_trades    = len(oos_result.trades)

        # ── Degradation ratio ─────────────────────────────────────────
        if best_sharpe != 0 and not np.isnan(best_sharpe):
            deg_ratio = oos_sharpe / best_sharpe
        else:
            deg_ratio = float("nan")

        wr = WindowResult(
            window_idx=w_idx,
            is_start=is_start.strftime("%Y-%m"),
            is_end=is_end.strftime("%Y-%m"),
            oos_start=oos_start.strftime("%Y-%m"),
            oos_end=oos_end.strftime("%Y-%m"),
            best_params=best_params,
            is_sharpe=round(best_sharpe, 4),
            is_return_pct=round(is_return, 4),
            is_trades=is_trades,
            oos_sharpe=round(oos_sharpe, 4),
            oos_return_pct=round(oos_return, 4),
            oos_trades=oos_trades,
            degradation_ratio=round(deg_ratio, 4) if not np.isnan(deg_ratio) else float("nan"),
        )
        window_results.append(wr)

        if verbose:
            deg_str = f"{deg_ratio:.2f}" if not np.isnan(deg_ratio) else "N/A"
            print(f"  IS  Sharpe={best_sharpe:.2f}  Return={is_return:+.1f}%  Trades={is_trades}")
            print(f"  OOS Sharpe={oos_sharpe:.2f}  Return={oos_return:+.1f}%  Trades={oos_trades}")
            print(f"  Degradation ratio: {deg_str}  Best params: {best_params}")
            print()

    if not window_results:
        raise RuntimeError("Walk-forward produced no valid windows.")

    # ── Summary ───────────────────────────────────────────────────────
    valid_deg = [w.degradation_ratio for w in window_results
                 if not np.isnan(w.degradation_ratio)]
    avg_deg   = float(np.mean(valid_deg)) if valid_deg else float("nan")
    is_robust            = avg_deg > 0.4 if not np.isnan(avg_deg) else False
    overfitting_warning  = avg_deg < 0.4 if not np.isnan(avg_deg) else True

    recommended = _recommend_params(window_results)

    wf_result = WalkForwardResult(
        strategy_name=strategy_name,
        instrument=instrument,
        windows=window_results,
        avg_degradation_ratio=round(avg_deg, 4) if not np.isnan(avg_deg) else float("nan"),
        recommended_params=recommended,
        is_robust=is_robust,
        overfitting_warning=overfitting_warning,
        grid_used=grid,
        initial_balance=initial_balance,
    )

    _print_summary(wf_result, verbose)
    return wf_result


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------

def _generate_windows(
    df: pd.DataFrame,
    is_months: int,
    oos_months: int,
    step_months: int,
) -> List[Tuple]:
    """
    Yield (is_df, oos_df, is_start, is_end, oos_start, oos_end) tuples.
    Uses pandas DateOffset so month boundaries are calendar-correct.
    df must have a tz-aware DatetimeIndex.
    """
    start = df.index[0].to_pydatetime()
    # Floor to first day of that month
    start = pd.Timestamp(year=start.year, month=start.month, day=1, tz=df.index.tz)
    data_end = df.index[-1]

    windows = []
    cursor = start
    while True:
        is_start  = cursor
        is_end    = cursor + pd.DateOffset(months=is_months)
        oos_start = is_end
        oos_end   = oos_start + pd.DateOffset(months=oos_months)

        if oos_end > data_end + pd.Timedelta(days=1):
            break

        is_slice  = df[(df.index >= is_start)  & (df.index < is_end)]
        oos_slice = df[(df.index >= oos_start) & (df.index < oos_end)]

        if len(is_slice) < 1000 or len(oos_slice) < 200:
            cursor += pd.DateOffset(months=step_months)
            continue

        windows.append((is_slice, oos_slice, is_start, is_end, oos_start, oos_end))
        cursor += pd.DateOffset(months=step_months)

    return windows


# ---------------------------------------------------------------------------
# Parameter grid helpers
# ---------------------------------------------------------------------------

def _expand_grid(grid: Dict[str, List]) -> List[Dict]:
    """Cartesian product of grid values → list of param dicts."""
    keys   = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ---------------------------------------------------------------------------
# Strategy builders (inject grid params into base config)
# ---------------------------------------------------------------------------

def _load_london_base_config() -> Dict:
    with open(_CONFIG_DIR / "strategy_params.json") as fh:
        return json.load(fh)["london_open_breakout"]


def _load_fvg_base_config() -> Dict:
    with open(_CONFIG_DIR / "strategy_params.json") as fh:
        return json.load(fh)["fvg_retracement"]


def _load_ny_base_config() -> Dict:
    with open(_CONFIG_DIR / "strategy_params.json") as fh:
        return json.load(fh)["ny_session_breakout"]


def _load_instrument_config(instrument: str) -> Dict:
    with open(_CONFIG_DIR / "instruments.json") as fh:
        return json.load(fh)[instrument]


def _build_london_strategy(base_cfg: Dict, params: Dict, instrument_cfg: Dict):
    from strategies.london_open_breakout import LondonOpenBreakout
    cfg = {**base_cfg, **params}
    # Derive max_asian_range_pips as 2× min to keep the filter meaningful
    cfg["max_asian_range_pips"] = max(
        params["min_asian_range_pips"] * 2,
        float(base_cfg.get("max_asian_range_pips", 40)),
    )
    s = LondonOpenBreakout()
    s.setup(cfg, instrument_cfg)
    return s


def _build_fvg_strategy(base_cfg: Dict, params: Dict, instrument_cfg: Dict):
    from strategies.fvg_retracement import FVGRetracement
    cfg = {**base_cfg, **params}
    s = FVGRetracement()
    s.setup(cfg, instrument_cfg)
    return s


def _build_ny_strategy(base_cfg: Dict, params: Dict, instrument_cfg: Dict):
    from strategies.ny_session_breakout import NYSessionBreakout
    cfg = {**base_cfg, **params}
    cfg["max_asian_range_pips"] = max(
        params["min_asian_range_pips"] * 2,
        float(base_cfg.get("max_asian_range_pips", 100)),
    )
    s = NYSessionBreakout()
    s.setup(cfg, instrument_cfg)
    return s


def _load_gold_base_config() -> Dict:
    with open(_CONFIG_DIR / "strategy_params.json") as fh:
        return json.load(fh)["gold_session_breakout"]


def _build_gold_strategy(base_cfg: Dict, params: Dict, instrument_cfg: Dict):
    from strategies.gold_session_breakout import GoldSessionBreakout
    cfg = {**base_cfg, **params}
    s = GoldSessionBreakout()
    s.setup(cfg, instrument_cfg)
    return s


def _load_h1_base_config() -> Dict:
    with open(_CONFIG_DIR / "strategy_params.json") as fh:
        return json.load(fh)["h1_trend_following"]


def _build_h1_strategy(base_cfg: Dict, params: Dict, instrument_cfg: Dict):
    from strategies.h1_trend_following import H1TrendFollowing
    cfg = {**base_cfg, **params}
    s = H1TrendFollowing()
    s.setup(cfg, instrument_cfg)
    return s


def _load_sweep_base_config() -> Dict:
    with open(_CONFIG_DIR / "strategy_params.json") as fh:
        return json.load(fh)["gold_sweep_reversal"]


def _build_sweep_strategy(base_cfg: Dict, params: Dict, instrument_cfg: Dict):
    from strategies.gold_sweep_reversal import GoldSweepReversal
    cfg = {**base_cfg, **params}
    s = GoldSweepReversal()
    s.setup(cfg, instrument_cfg)
    return s


def _load_m1_sweep_base_config() -> Dict:
    with open(_CONFIG_DIR / "strategy_params.json") as fh:
        return json.load(fh)["gold_m1_sweep_reversal"]


def _build_m1_sweep_strategy(base_cfg: Dict, params: Dict, instrument_cfg: Dict):
    from strategies.gold_m1_sweep_reversal import GoldM1SweepReversal
    cfg = {**base_cfg, **params}
    s = GoldM1SweepReversal()
    s.setup(cfg, instrument_cfg)
    return s


def _load_m15_base_config() -> Dict:
    with open(_CONFIG_DIR / "strategy_params.json") as fh:
        return json.load(fh)["m15_momentum_scalping"]


def _build_m15_strategy(base_cfg: Dict, params: Dict, instrument_cfg: Dict):
    from strategies.m15_momentum_scalping import M15MomentumScalping
    cfg = {**base_cfg, **params}
    # Enforce RSI short = mirror of RSI long (not in grid, derived)
    cfg["rsi_short_min"] = 100.0 - cfg.get("rsi_long_max", 65)
    cfg["rsi_short_max"] = 100.0 - cfg.get("rsi_long_min", 45)
    s = M15MomentumScalping()
    s.setup(cfg, instrument_cfg)
    return s


# ---------------------------------------------------------------------------
# Data normalisation
# ---------------------------------------------------------------------------

def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.columns = [c.lower() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Recommended parameters
# ---------------------------------------------------------------------------

def _recommend_params(windows: List[WindowResult]) -> Dict[str, Any]:
    """
    Return the parameter set most consistently good in OOS.

    Method: among windows with positive OOS Sharpe, find which parameter
    *combination* appears most often.  Ties broken by highest average
    OOS Sharpe across all windows where that combo was used.
    """
    # Score each window's best_params by OOS Sharpe (positive only)
    param_scores: Dict[str, List[float]] = {}

    for w in windows:
        key = json.dumps(w.best_params, sort_keys=True)
        param_scores.setdefault(key, []).append(w.oos_sharpe)

    if not param_scores:
        return {}

    # Rank by: (appearances with positive OOS Sharpe, then avg OOS Sharpe)
    def _rank(item):
        k, scores = item
        positive  = [s for s in scores if s > 0]
        avg       = float(np.mean(scores)) if scores else 0.0
        return (len(positive), avg)

    best_key = max(param_scores.items(), key=_rank)[0]
    return json.loads(best_key)


# ---------------------------------------------------------------------------
# Output / display
# ---------------------------------------------------------------------------

def _print_summary(result: WalkForwardResult, verbose: bool) -> None:
    if not verbose:
        return

    print("=" * 76)
    print(f"WALK-FORWARD SUMMARY  |  {result.strategy_name} on {result.instrument}")
    print("=" * 76)

    # Per-window table
    hdr = f"{'Win':>3}  {'IS Period':>13}  {'OOS Period':>13}  "
    hdr += f"{'IS Sh':>6}  {'IS Ret%':>7}  "
    hdr += f"{'OOS Sh':>6}  {'OOS Ret%':>8}  {'Deg.':>5}  Trades(IS/OOS)"
    print(hdr)
    print("-" * 76)

    for w in result.windows:
        deg = f"{w.degradation_ratio:.2f}" if not np.isnan(w.degradation_ratio) else " N/A"
        flag = " ✓" if not np.isnan(w.degradation_ratio) and w.degradation_ratio > 0.6 else ""
        print(
            f"{w.window_idx+1:>3}  "
            f"{w.is_start}→{w.is_end}  "
            f"{w.oos_start}→{w.oos_end}  "
            f"{w.is_sharpe:>6.2f}  {w.is_return_pct:>+7.1f}%  "
            f"{w.oos_sharpe:>6.2f}  {w.oos_return_pct:>+8.1f}%  "
            f"{deg:>5}{flag}  "
            f"{w.is_trades}/{w.oos_trades}"
        )

    print("-" * 76)
    avg_str = f"{result.avg_degradation_ratio:.2f}" if not np.isnan(result.avg_degradation_ratio) else "N/A"
    print(f"Average degradation ratio: {avg_str}  (target > 0.6 per window, > 0.4 overall)")

    if result.overfitting_warning:
        print()
        print("⚠  WARNING: Strategy may be overfitted.  "
              "Average degradation ratio < 0.4.")
        print("   Consider simplifying parameters or using fewer free variables.")
    elif result.is_robust:
        print("✓  Parameters appear robust across walk-forward windows.")

    print()
    print(f"Recommended parameters:  {result.recommended_params}")
    print("=" * 76)

    # FVG-specific: max_candles_until_cancel analysis
    strat = result.strategy_name.lower().replace(" ", "_")
    if strat in ("fvg_retracement", "fvg"):
        _analyze_max_candles_effect(result.windows)


def _analyze_max_candles_effect(windows: List[WindowResult]) -> None:
    """
    For FVG strategy: group windows by the max_candles_until_cancel value
    chosen by IS optimisation, then show avg OOS Sharpe per group.

    Flags if candles > 12 shows meaningfully better OOS performance —
    suggesting that early London-session FVGs (which need more time to
    reach the NY-open entry window) carry real edge.
    """
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for w in windows:
        mc = w.best_params.get("max_candles_until_cancel")
        if mc is not None:
            groups[int(mc)].append(w.oos_sharpe)

    if not groups:
        return

    print()
    print("  max_candles_until_cancel — intra-session FVG analysis")
    print(f"  {'candles':>8}  {'windows':>7}  {'avg OOS Sharpe':>14}  {'pos. OOS':>8}")
    print(f"  {'-------':>8}  {'-------':>7}  {'---------------':>14}  {'--------':>8}")

    mc_avg: dict = {}
    for mc in sorted(groups.keys()):
        sharpes = groups[mc]
        avg     = float(np.mean(sharpes)) if sharpes else float("nan")
        n_pos   = sum(1 for s in sharpes if s > 0)
        mc_avg[mc] = avg
        print(f"  {mc:>8}  {len(sharpes):>7}  {avg:>14.3f}  {n_pos:>5}/{len(sharpes)}")

    # Compare short (≤12) vs long (>12) cancellation
    short_vals = [v for k, v in mc_avg.items() if k <= 12 and not np.isnan(v)]
    long_vals  = [v for k, v in mc_avg.items() if k >  12 and not np.isnan(v)]
    short_avg  = float(np.mean(short_vals)) if short_vals else float("nan")
    long_avg   = float(np.mean(long_vals))  if long_vals  else float("nan")

    print()
    if not np.isnan(long_avg) and not np.isnan(short_avg):
        diff = long_avg - short_avg
        if diff > 0.10:
            print(
                f"  ⚑  INTRA-SESSION BENEFIT: max_candles > 12 outperforms ≤12 by "
                f"{diff:+.3f} avg OOS Sharpe ({long_avg:.3f} vs {short_avg:.3f}).\n"
                f"     Early London-session FVGs (forming 03:00–05:30 ET) carry edge\n"
                f"     when allowed time to reach the 09:30 NY entry window.\n"
                f"     Recommended: keep max_candles_until_cancel >= 16."
            )
        else:
            print(
                f"  ✓  No clear intra-session benefit from max_candles > 12 "
                f"(diff = {diff:+.3f}).\n"
                f"     Tight cancellation (8–12 candles) performs comparably;\n"
                f"     consider favouring smaller values to avoid stale entries."
            )
    print()


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def save_result(result: WalkForwardResult, path: str | Path) -> Path:
    """Dump WalkForwardResult to JSON (windows list → dicts)."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    d = asdict(result)
    # Convert any float("nan") → null for JSON compatibility
    def _fix(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _fix(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_fix(v) for v in obj]
        return obj

    path.write_text(json.dumps(_fix(d), indent=2), encoding="utf-8")
    print(f"Walk-forward result saved: {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward optimisation for FTMO trading strategies."
    )
    parser.add_argument("--data",             required=True, type=Path,
                        help="Path to 15-min OHLCV parquet file")
    parser.add_argument("--strategy",         required=True,
                        choices=["london_breakout", "fvg_retracement"],
                        help="Strategy to optimise")
    parser.add_argument("--instrument",       default="EURUSD",
                        help="Instrument key from instruments.json")
    parser.add_argument("--output",           type=Path, default=None,
                        help="Save result as JSON to this path")
    parser.add_argument("--initial-balance",  type=float, default=10_000.0)
    parser.add_argument("--in-sample-months", type=int,   default=6)
    parser.add_argument("--oos-months",       type=int,   default=3)
    parser.add_argument("--step-months",      type=int,   default=3)
    parser.add_argument("--min-trades",       type=int,   default=20)
    parser.add_argument("--quiet",            action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    df = pd.read_parquet(args.data.expanduser())

    result = run_walk_forward(
        df=df,
        strategy_name=args.strategy,
        instrument=args.instrument,
        initial_balance=args.initial_balance,
        in_sample_months=args.in_sample_months,
        oos_months=args.oos_months,
        step_months=args.step_months,
        min_trades=args.min_trades,
        verbose=not args.quiet,
    )

    if args.output:
        save_result(result, args.output)


if __name__ == "__main__":
    _cli()
