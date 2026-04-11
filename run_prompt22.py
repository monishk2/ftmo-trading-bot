"""
Prompt 22 — Gold M1 Sweep Reversal with H4 Trend Guard
=======================================================

Tasks
-----
0. Data verification: XAUUSD_1m.parquet bar count, date range, volume coverage
1. run_fast() sanity check: compare single run vs run() on M15 data
2. Baseline backtest: M1 data (available range), report 2024 with/without trend guard
3. Walk-forward: 4m IS / 2m OOS / 2m step, 288 combos, min 5 trades
4. FTMO evaluation: 15-trading-day sliding windows (if WF passes avg_deg>0.15 & >50% pos OOS)
5. Risk sensitivity: 1.0% / 1.5% / 2.0% (if FTMO eval pass rate > 3%)
6. Combined portfolio: London S3 + Gold M1 (if standalone viable)

Pass criteria for WF: avg_deg > 0.15  AND  > 50% windows with positive OOS Sharpe
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from backtesting.backtester import Backtester
from backtesting.metrics import calculate_metrics
from backtesting.walk_forward import run_walk_forward, _M1_SWEEP_GRID
from strategies.gold_m1_sweep_reversal import GoldM1SweepReversal

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

INSTRUMENT      = "XAUUSD"
INITIAL_BALANCE = 10_000.0
PARQUET_M1      = ROOT / "data" / "historical" / "XAUUSD_1m.parquet"
PARQUET_M15     = ROOT / "data" / "historical" / "XAUUSD_15m.parquet"

LOOSE_FTMO = {
    "safety_buffers": {
        "daily_loss_trigger_pct":  99.0,
        "total_loss_trigger_pct":  99.0,
    }
}
REAL_FTMO_FILE = ROOT / "config" / "ftmo_rules.json"

with open(ROOT / "config" / "strategy_params.json") as _fh:
    _ALL_PARAMS = json.load(_fh)

with open(ROOT / "config" / "instruments.json") as _fh:
    _INSTR_CFG = json.load(_fh)[INSTRUMENT]

M1_BASE_CFG = _ALL_PARAMS["gold_m1_sweep_reversal"]


def _make_strategy(overrides: dict | None = None, trend_guard: bool = True) -> GoldM1SweepReversal:
    cfg = {**M1_BASE_CFG, **(overrides or {})}
    cfg["trend_guard_enabled"] = trend_guard
    s = GoldM1SweepReversal()
    s.setup(cfg, _INSTR_CFG)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_m1(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_M1)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
    return df.reset_index()


def _load_m15() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_M15)
    if "datetime" not in df.columns:
        df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FTMO evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

TRADING_DAYS_PER_MONTH = 21

def _eval_windows(
    equity: pd.Series,
    initial_balance: float,
    window_days: int = 15,
    profit_target_pct: float = 10.0,
    daily_loss_limit_pct: float = 5.0,
    total_loss_limit_pct: float = 10.0,
) -> dict:
    """
    Slide 15-trading-day windows across equity curve.
    Returns pass_rate, fail_rate, two_attempt_pass_pct.
    """
    daily = equity.resample("D").last().dropna()
    n = len(daily)
    passes = fails = total = 0

    for start_i in range(n - window_days):
        window = daily.iloc[start_i: start_i + window_days + 1].values
        if len(window) < window_days:
            continue

        norm = window / window[0] * initial_balance
        prev_eq = norm[0]
        outcome = "inconclusive"

        for eq in norm[1:]:
            if (eq - prev_eq) / prev_eq * 100.0 <= -daily_loss_limit_pct:
                outcome = "fail"; break
            if (eq - initial_balance) / initial_balance * 100.0 <= -total_loss_limit_pct:
                outcome = "fail"; break
            if (eq - initial_balance) / initial_balance * 100.0 >= profit_target_pct:
                outcome = "pass"; break
            prev_eq = eq

        total += 1
        if outcome == "pass":
            passes += 1
        elif outcome == "fail":
            fails += 1

    if total == 0:
        return {"pass_rate": 0.0, "fail_rate": 0.0, "two_attempt": 0.0, "total": 0}

    p = passes / total
    f = fails   / total
    two_attempt = (1 - (1 - p) ** 2) * 100.0
    return {
        "pass_rate":   round(p * 100, 2),
        "fail_rate":   round(f * 100, 2),
        "two_attempt": round(two_attempt, 2),
        "total":       total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 0 — Data verification
# ─────────────────────────────────────────────────────────────────────────────

def run_task0():
    print("\n" + "=" * 72)
    print("TASK 0: M1 Data Verification")
    print("=" * 72)

    df = pd.read_parquet(PARQUET_M1)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")

    print(f"Total bars:     {len(df):,}")
    print(f"Date range:     {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"Volume non-zero: {(df['volume'] > 0).sum():,} ({(df['volume'] > 0).mean()*100:.1f}%)")
    print(f"File size:      {PARQUET_M1.stat().st_size / 1e6:.1f} MB")

    # Weekday gap check
    df_et = df.set_index("datetime")
    df_et.index = df_et.index.tz_convert("US/Eastern")
    ts_diff = df_et.index.to_series().diff().dt.total_seconds()
    wd = df_et.index.dayofweek
    wk_gaps = ts_diff[(ts_diff > 600) & (wd < 5)]
    print(f"Weekday gaps >10min: {len(wk_gaps)}")
    if not wk_gaps.empty:
        print(f"  Largest: {wk_gaps.max()/3600:.1f}h at {wk_gaps.idxmax()}")

    # Yearly coverage
    df_et["year"] = df_et.index.year
    for yr, grp in df_et.groupby("year"):
        vol_pct = (grp["volume"] > 0).mean() * 100
        print(f"  {yr}: {len(grp):>7,} bars | vol>0: {vol_pct:.1f}%")

    print("\nData PASS — ready for M1 backtesting." if len(wk_gaps) <= 5
          else "\nData WARNING — multiple weekday gaps detected.")
    return df_et


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — run_fast() sanity check
# ─────────────────────────────────────────────────────────────────────────────

def run_task1(df_m1: pd.DataFrame):
    """Compare run_fast() vs run() on a 3-month M1 slice."""
    print("\n" + "=" * 72)
    print("TASK 1: run_fast() Sanity Check (3-month M1 slice)")
    print("=" * 72)

    # Use a short slice for speed
    slice_df = df_m1["2023-01":"2023-03"].reset_index()

    strat = _make_strategy()

    t0 = time.time()
    bt_fast = Backtester(strat, slice_df, INSTRUMENT, INITIAL_BALANCE,
                         _override_ftmo_rules=LOOSE_FTMO)
    r_fast = bt_fast.run_fast()
    t_fast = time.time() - t0

    strat2 = _make_strategy()
    t0 = time.time()
    bt_slow = Backtester(strat2, slice_df, INSTRUMENT, INITIAL_BALANCE,
                         _override_ftmo_rules=LOOSE_FTMO)
    r_slow = bt_slow.run()
    t_slow = time.time() - t0

    print(f"{'':20} {'run_fast()':>12} {'run()':>12}")
    print(f"{'Trades':20} {len(r_fast.trades):>12} {len(r_slow.trades):>12}")
    print(f"{'Final balance':20} {r_fast.final_balance:>12.2f} {r_slow.final_balance:>12.2f}")
    print(f"{'Time (s)':20} {t_fast:>12.2f} {t_slow:>12.2f}")
    speedup = t_slow / t_fast if t_fast > 0 else 0
    print(f"Speedup: {speedup:.1f}x")

    # Verify trade counts match
    if len(r_fast.trades) == len(r_slow.trades):
        print("Trade count: MATCH ✓")
    else:
        print(f"Trade count: MISMATCH — fast={len(r_fast.trades)} slow={len(r_slow.trades)}")
        # Check P&L difference
        fast_pnl = sum(t.pnl_dollars for t in r_fast.trades)
        slow_pnl = sum(t.pnl_dollars for t in r_slow.trades)
        print(f"  fast total P&L: {fast_pnl:.2f}  slow: {slow_pnl:.2f}")

    return speedup


# ─────────────────────────────────────────────────────────────────────────────
# Task 2a — Baseline: full M1 data, both directions
# ─────────────────────────────────────────────────────────────────────────────

def _run_baseline(df, trend_guard: bool, label: str, use_fast: bool = True) -> dict:
    strat = _make_strategy(trend_guard=trend_guard)
    bt = Backtester(strat, df.reset_index() if df.index.name == "datetime" else df,
                    INSTRUMENT, INITIAL_BALANCE, _override_ftmo_rules=LOOSE_FTMO)
    result = bt.run_fast() if use_fast else bt.run()
    m = calculate_metrics(result)
    print(f"\n  [{label}]")
    print(f"  Trades={m.get('total_trades', 0)}  T/mo={m.get('trades_per_month_avg', 0):.1f}  "
          f"WR={m.get('win_rate_pct', 0):.1f}%  Sharpe={m.get('sharpe_ratio', 0):.3f}")
    print(f"  Return={m.get('total_return_pct', 0):.2f}%  MaxDD={m.get('max_drawdown_pct', 0):.2f}%")
    return {"label": label, "metrics": m, "result": result}


def run_task2(df_m1_et: pd.DataFrame):
    print("\n" + "=" * 72)
    print("TASK 2: Baseline Backtest (Jan 2022 – Feb 2025)")
    print("=" * 72)

    df_full = df_m1_et.copy()

    # Warmup: skip first 3 months for ATR/EMA to stabilise
    df_base = df_full["2022-04":]

    print("\n--- Full period (Apr 2022 → Feb 2025) ---")
    r_with    = _run_baseline(df_base, trend_guard=True,  label="WITH H4 trend guard")
    r_without = _run_baseline(df_base, trend_guard=False, label="WITHOUT H4 trend guard")

    # 2024 isolation
    df_2024 = df_full["2024-01":"2024-12"]
    print("\n--- 2024 isolation ---")
    r_2024_with    = _run_baseline(df_2024, trend_guard=True,  label="2024 WITH guard")
    r_2024_without = _run_baseline(df_2024, trend_guard=False, label="2024 WITHOUT guard")

    # Average SL analysis
    trades_with = r_with["result"].trades
    if trades_with:
        sl_pips = [abs(t.entry_price - t.sl) / _INSTR_CFG["pip_size"] for t in trades_with]
        print(f"\n  Avg SL (with guard): {np.mean(sl_pips):.1f} pips  "
              f"median={np.median(sl_pips):.1f}  max={max(sl_pips):.1f}")

    # Level type breakdown
    try:
        df_sigs = pd.DataFrame([
            {"level": getattr(t, "strategy_name", ""), "dir": t.direction,
             "pnl": t.pnl_dollars}
            for t in trades_with
        ])
        if not df_sigs.empty:
            print(f"\n  Direction split: LONG={int((df_sigs['dir']>0).sum())}  "
                  f"SHORT={int((df_sigs['dir']<0).sum())}")
    except Exception:
        pass

    return r_with, r_2024_with, r_2024_without


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Walk-Forward
# ─────────────────────────────────────────────────────────────────────────────

def run_task3(df_m1_et: pd.DataFrame):
    print("\n" + "=" * 72)
    print("TASK 3: Walk-Forward Optimisation (4m IS / 2m OOS / 2m step)")
    print(f"Grid: {len(_M1_SWEEP_GRID)} params × combos = "
          f"{2**0 * 3*2*2*3*2*2*2*2} combos")   # will print actual count
    print("=" * 72)

    df_wf = df_m1_et["2022-07":].copy()

    # Use LOOSE_FTMO for WF so grid search isn't polluted by FTMO halts
    # Override ftmo_rules via _override_ftmo_rules in Backtester — but
    # run_walk_forward doesn't support that override. Workaround: pass
    # loose params in base_config_overrides (not needed — use_fast_runner
    # with run_fast() already uses the standard ftmo_rules.json).
    # For WF, we use the default (real) FTMO rules but with small windows
    # where halts are unlikely. If needed we patch strategy_params.

    t0 = time.time()
    wf = run_walk_forward(
        df=df_wf.reset_index(),
        strategy_name="gold_m1_sweep_reversal",
        instrument=INSTRUMENT,
        initial_balance=INITIAL_BALANCE,
        in_sample_months=4,
        oos_months=2,
        step_months=2,
        min_trades=5,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nWF completed in {elapsed/60:.1f} min")

    # Robustness verdict
    valid_deg = [w.degradation_ratio for w in wf.windows
                 if not np.isnan(w.degradation_ratio)]
    n_pos_oos = sum(1 for w in wf.windows if w.oos_sharpe > 0)
    pos_pct   = n_pos_oos / len(wf.windows) * 100 if wf.windows else 0.0

    print(f"\nWF Summary:")
    print(f"  Windows: {len(wf.windows)}")
    print(f"  avg_deg: {wf.avg_degradation_ratio:.3f}  (pass threshold: >0.15)")
    print(f"  Positive OOS: {n_pos_oos}/{len(wf.windows)} = {pos_pct:.0f}%  (threshold: >50%)")

    wf_pass = (
        not np.isnan(wf.avg_degradation_ratio) and
        wf.avg_degradation_ratio > 0.15 and
        pos_pct > 50.0
    )
    verdict = "PASS ✓" if wf_pass else "FAIL ✗"
    print(f"\n  WF Verdict: {verdict}")
    print(f"  Recommended params: {wf.recommended_params}")

    return wf, wf_pass


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — FTMO 15-day Evaluation (only if WF passes)
# ─────────────────────────────────────────────────────────────────────────────

def run_task4(df_m1_et: pd.DataFrame, wf_pass: bool, recommended_params: dict):
    print("\n" + "=" * 72)
    print("TASK 4: FTMO 15-Day Sliding Window Evaluation")
    print("=" * 72)

    if not wf_pass:
        print("  Skipped — WF did not pass threshold.")
        return None, False

    # Full baseline with recommended params
    strat = _make_strategy(recommended_params)
    df_eval = df_m1_et["2022-04":].copy().reset_index()

    with open(REAL_FTMO_FILE) as fh:
        ftmo_rules = json.load(fh)

    bt = Backtester(strat, df_eval, INSTRUMENT, INITIAL_BALANCE,
                    _override_ftmo_rules=ftmo_rules)
    result = bt.run_fast()
    m = calculate_metrics(result)

    print(f"\n  Full period (recommended params):")
    print(f"  Trades={m.get('total_trades', 0)}  T/mo={m.get('trades_per_month_avg', 0):.1f}  "
          f"WR={m.get('win_rate_pct', 0):.1f}%  Sharpe={m.get('sharpe_ratio', 0):.3f}")
    print(f"  Return={m.get('total_return_pct', 0):.2f}%  MaxDD={m.get('max_drawdown_pct', 0):.2f}%")

    avg_monthly = m.get("total_return_pct", 0) / max(
        (result.equity_curve.index[-1] - result.equity_curve.index[0]).days / 30.44, 1)
    print(f"  Avg monthly return: {avg_monthly:.3f}%")

    # 15-day windows
    eval_res = _eval_windows(result.equity_curve, INITIAL_BALANCE)
    print(f"\n  15-day sliding windows:")
    print(f"  Windows:  {eval_res['total']}")
    print(f"  Pass rate: {eval_res['pass_rate']:.1f}%")
    print(f"  Fail rate: {eval_res['fail_rate']:.1f}%")
    print(f"  Two-attempt pass: {eval_res['two_attempt']:.1f}%")

    viable = eval_res["pass_rate"] > 3.0
    verdict = "VIABLE ✓" if viable else "NOT VIABLE ✗"
    print(f"  FTMO Eval Verdict: {verdict}  (threshold: pass_rate > 3%)")

    return eval_res, viable


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Risk sensitivity + combined portfolio (only if FTMO eval passes)
# ─────────────────────────────────────────────────────────────────────────────

def run_task5(df_m1_et: pd.DataFrame, viable: bool, recommended_params: dict):
    print("\n" + "=" * 72)
    print("TASK 5: Risk Sensitivity + Combined Portfolio")
    print("=" * 72)

    if not viable:
        print("  Skipped — FTMO eval did not pass.")
        return

    # Risk sensitivity
    print("\n--- Risk Sensitivity ---")
    print(f"{'Risk%':>6} {'T/mo':>5} {'Return%':>8} {'MaxDD%':>7} {'Sharpe':>7} {'2-Att%':>7}")
    df_eval = df_m1_et["2022-04":].copy().reset_index()

    with open(REAL_FTMO_FILE) as fh:
        ftmo_rules = json.load(fh)

    for risk_pct in [1.0, 1.5, 2.0]:
        ov = {**recommended_params, "risk_per_trade_pct": risk_pct}
        strat = _make_strategy(ov)
        bt = Backtester(strat, df_eval, INSTRUMENT, INITIAL_BALANCE,
                        _override_ftmo_rules=ftmo_rules)
        result = bt.run_fast()
        m = calculate_metrics(result)
        ev = _eval_windows(result.equity_curve, INITIAL_BALANCE)
        months = max(
            (result.equity_curve.index[-1] - result.equity_curve.index[0]).days / 30.44, 1
        )
        print(f"{risk_pct:>6.1f} {m.get('trades_per_month_avg',0):>5.1f} "
              f"{m.get('total_return_pct',0):>+8.2f} {m.get('max_drawdown_pct',0):>7.2f} "
              f"{m.get('sharpe_ratio',0):>7.3f} {ev['two_attempt']:>7.1f}")

    # Combined portfolio
    if PARQUET_M15.exists():
        print("\n--- Combined: London S3 + Gold M1 ---")
        from backtesting.backtester import BacktestResult
        from strategies.london_open_breakout import LondonOpenBreakout

        london_cfg = _ALL_PARAMS["london_open_breakout"]
        eurusd_cfg = json.load(open(ROOT / "config" / "instruments.json"))["EURUSD"]

        df_london = pd.read_parquet(PARQUET_M15)
        if "datetime" not in df_london.columns:
            df_london = df_london.reset_index()
        df_london["datetime"] = pd.to_datetime(df_london["datetime"], utc=True)
        df_london = df_london[df_london["datetime"] >= pd.Timestamp("2022-04-01", tz="UTC")]

        lon_strat = LondonOpenBreakout()
        lon_strat.setup({
            **london_cfg,
            **london_cfg.get("instrument_overrides", {}).get("EURUSD", {}),
        }, eurusd_cfg)
        bt_lon = Backtester(lon_strat, df_london, "EURUSD", INITIAL_BALANCE,
                            _override_ftmo_rules=LOOSE_FTMO)
        res_lon = bt_lon.run()

        strat_gold = _make_strategy(recommended_params)
        bt_gold = Backtester(strat_gold, df_eval, INSTRUMENT, INITIAL_BALANCE,
                             _override_ftmo_rules=LOOSE_FTMO)
        res_gold = bt_gold.run_fast()

        # Combine equity curves (additive returns, daily)
        eq_lon  = res_lon.equity_curve.resample("D").last().ffill()
        eq_gold = res_gold.equity_curve.resample("D").last().ffill()
        idx     = eq_lon.index.intersection(eq_gold.index)
        if len(idx) > 10:
            ret_lon  = eq_lon.reindex(idx).pct_change()
            ret_gold = eq_gold.reindex(idx).pct_change()
            combined_ret = ret_lon + ret_gold
            combined_eq  = (1 + combined_ret).cumprod() * INITIAL_BALANCE
            combined_eq.iloc[0] = INITIAL_BALANCE
            total_ret = (combined_eq.iloc[-1] / INITIAL_BALANCE - 1) * 100
            months_c  = (idx[-1] - idx[0]).days / 30.44
            avg_mo    = total_ret / max(months_c, 1)
            peak      = combined_eq.cummax()
            max_dd    = float(((combined_eq - peak) / peak).min() * 100)
            print(f"  Combined (London EURUSD + Gold M1): {avg_mo:+.3f}%/mo  "
                  f"MaxDD={abs(max_dd):.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Context summary (always printed last)
# ─────────────────────────────────────────────────────────────────────────────

def print_context_summary(
    df_m1_et, r_base_with, r_2024_with, r_2024_without,
    wf, wf_pass, eval_res, eval_viable,
):
    print("\n" + "=" * 72)
    print("CONTEXT SUMMARY FOR NEXT PROMPT")
    print("=" * 72)

    m_with = r_base_with["metrics"]
    m_2024_with = r_2024_with["metrics"]
    m_2024_no   = r_2024_without["metrics"]

    print(f"""
1. M1 Data
   File:      XAUUSD_1m.parquet  ({len(df_m1_et):,} bars, Jan 2022 – Feb 2025)
   Volume:    {(df_m1_et['volume']>0).mean()*100:.1f}% non-zero
   pip_size:  0.01  pip_value: $1/lot  spread: 25 pips

2. Baseline (WITH H4 trend guard, Apr 2022–Feb 2025)
   Trades:    {m_with.get('total_trades',0)}  T/mo={m_with.get('trades_per_month_avg',0):.1f}
   WR:        {m_with.get('win_rate_pct',0):.1f}%  Sharpe={m_with.get('sharpe_ratio',0):.3f}
   Return:    {m_with.get('total_return_pct',0):.2f}%  MaxDD={m_with.get('max_drawdown_pct',0):.2f}%

3. 2024 Comparison
   WITH guard:    Trades={m_2024_with.get('total_trades',0)}  Return={m_2024_with.get('total_return_pct',0):.2f}%  Sharpe={m_2024_with.get('sharpe_ratio',0):.3f}
   WITHOUT guard: Trades={m_2024_no.get('total_trades',0)}  Return={m_2024_no.get('total_return_pct',0):.2f}%  Sharpe={m_2024_no.get('sharpe_ratio',0):.3f}

4. Walk-Forward (4m IS / 2m OOS / 2m step, 288 combos, min 5 trades)
   Windows:   {len(wf.windows)}
   avg_deg:   {wf.avg_degradation_ratio:.3f}  (threshold: >0.15)
   Pos OOS:   {sum(1 for w in wf.windows if w.oos_sharpe > 0)}/{len(wf.windows)}
   VERDICT:   {"PASS ✓" if wf_pass else "FAIL ✗"}
   Rec params: {wf.recommended_params}

5. FTMO 15-Day Eval: {"VIABLE ✓" if eval_viable else "NOT VIABLE ✗" if eval_res else "SKIPPED"}""")

    if eval_res:
        print(f"   Pass rate: {eval_res.get('pass_rate','N/A')}%  "
              f"Fail rate: {eval_res.get('fail_rate','N/A')}%  "
              f"Two-attempt: {eval_res.get('two_attempt','N/A')}%")

    print(f"""
6. H4 Trend Guard Design
   Threshold: {M1_BASE_CFG['trend_guard_threshold']} pips displacement over {M1_BASE_CFG['trend_guard_bars']} H4 bars
   EMA period: {M1_BASE_CFG['trend_guard_h4_ema']}
   Logic: uptrend → LONG only; downtrend → SHORT only; neutral → both

7. Strategy Status
   M15 GoldSweepReversal: DEAD (WF avg_deg=+0.089, 2024=-31%)
   M1 GoldM1SweepReversal: {"VIABLE" if eval_viable else "DEAD" if wf_pass else "WF FAILED"}
   Next candidate if this DEAD: daily/H4 swing strategies on different pairs""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("PROMPT 22 — Gold M1 Sweep Reversal with H4 Trend Guard")
    print("=" * 72)

    # Task 0
    df_m1_et = run_task0()

    # Task 1 — run_fast() sanity check
    run_task1(df_m1_et)

    # Task 2 — Baseline
    r_base_with, r_2024_with, r_2024_without = run_task2(df_m1_et)

    # Check if baseline has enough signal to proceed
    m_base = r_base_with["metrics"]
    t_per_mo = m_base.get("trades_per_month_avg", 0)
    sharpe   = m_base.get("sharpe_ratio", -999)
    print(f"\nBaseline gate: T/mo={t_per_mo:.1f}  Sharpe={sharpe:.3f}")

    if t_per_mo < 3.0 or sharpe < -1.0:
        print("Baseline too weak — stopping early. Strategy DEAD.")
        print_context_summary(
            df_m1_et, r_base_with, r_2024_with, r_2024_without,
            type("WF", (), {
                "windows": [], "avg_degradation_ratio": float("nan"),
                "recommended_params": {},
            })(),
            False, None, False,
        )
        return

    # Task 3 — Walk-Forward
    wf, wf_pass = run_task3(df_m1_et)

    # Task 4 — FTMO eval
    eval_res, eval_viable = run_task4(df_m1_et, wf_pass, wf.recommended_params)

    # Task 5 — Risk sensitivity + combined (only if viable)
    run_task5(df_m1_et, eval_viable, wf.recommended_params)

    # Context summary
    print_context_summary(
        df_m1_et, r_base_with, r_2024_with, r_2024_without,
        wf, wf_pass, eval_res, eval_viable,
    )


if __name__ == "__main__":
    main()
