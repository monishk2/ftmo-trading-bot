"""
Prompt 23 — Gold M1 Micro-Swing Scalper
========================================

Tasks
-----
0. Spread verification: re-run M15 GoldSweepReversal with 12-pip spread
   vs Prompt 21 result (+0.655 Sharpe at 25 pips). Report before/after.
1. [already done in previous session] Built strategies/gold_m1_scalper.py
2. Baseline: full M1 range (2022-2025). Report T/day, WR, avg SL pips, Sharpe, MaxDD.
3. Walk-forward: 3m IS / 2m OOS / 2m step, 243 combos, min 20 trades
   Pass: avg_deg > 0.1 AND > 50% OOS windows positive.
4. If WF passes: 15-day eval windows, risk sensitivity (0.5/0.8/1.0/1.5%)
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
from backtesting.walk_forward import run_walk_forward, _M1_SCALPER_GRID

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

INSTRUMENT      = "XAUUSD"
INITIAL_BALANCE = 10_000.0
PARQUET_M1      = ROOT / "data" / "historical" / "XAUUSD_1m.parquet"
PARQUET_M15     = ROOT / "data" / "historical" / "XAUUSD_15m.parquet"

# Reported Prompt 21 baseline (25-pip spread era)
P21_BASELINE_SHARPE = 0.655

LOOSE_FTMO = {
    "safety_buffers": {
        "daily_loss_trigger_pct": 99.0,
        "total_loss_trigger_pct": 99.0,
    }
}

with open(ROOT / "config" / "strategy_params.json") as _fh:
    _ALL_PARAMS = json.load(_fh)

with open(ROOT / "config" / "instruments.json") as _fh:
    _INSTR_CFG = json.load(_fh)

XAUUSD_CFG = _INSTR_CFG[INSTRUMENT]
M1_SCALPER_BASE = _ALL_PARAMS["gold_m1_scalper"]
SWEEP_BASE      = _ALL_PARAMS["gold_sweep_reversal"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_m15() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_M15)
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.columns = [c.lower() for c in df.columns]
    return df


def _load_m1(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_M1)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
    return df.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(strategy, df: pd.DataFrame, instrument: str, override_ftmo=None):
    kwargs = dict(
        strategy=strategy,
        df=df,
        instrument=instrument,
        initial_balance=INITIAL_BALANCE,
    )
    if override_ftmo:
        kwargs["_override_ftmo_rules"] = override_ftmo
    return Backtester(**kwargs).run()


def _run_fast(strategy, df: pd.DataFrame, instrument: str, override_ftmo=None):
    kwargs = dict(
        strategy=strategy,
        df=df,
        instrument=instrument,
        initial_balance=INITIAL_BALANCE,
    )
    if override_ftmo:
        kwargs["_override_ftmo_rules"] = override_ftmo
    return Backtester(**kwargs).run_fast()


def _print_metrics(m: dict, label: str = "") -> None:
    if "error" in m:
        print(f"  {label}: ERROR — {m['error']}")
        return
    trades = m["total_trades"]
    t_mo   = m["trades_per_month_avg"]
    wr     = m["win_rate_pct"]
    ret    = m["total_return_pct"]
    sharpe = m["sharpe_ratio"]
    maxdd  = m["max_drawdown_pct"]
    final  = m["final_balance"]
    print(f"  {label}")
    print(f"    Trades={trades}  T/mo={t_mo:.1f}  WR={wr:.1f}%")
    print(f"    Return={ret:.2f}%  Sharpe={sharpe:.3f}  MaxDD={maxdd:.2f}%")
    print(f"    Final balance: ${final:,.0f}")


def _avg_sl_pips(result) -> float:
    if not result.trades:
        return 0.0
    sl_pips = []
    for t in result.trades:
        sl_dist = abs(t.entry_price - t.sl)
        pips = sl_dist / XAUUSD_CFG["pip_size"]
        sl_pips.append(pips)
    return float(np.mean(sl_pips)) if sl_pips else 0.0


def _trades_per_day(result) -> float:
    if not result.trades:
        return 0.0
    dates = set(t.entry_time.date() for t in result.trades)
    trading_days = len(dates)
    return len(result.trades) / trading_days if trading_days > 0 else 0.0


def _eval_windows_15day(
    equity: pd.Series,
    initial_balance: float,
    window_days: int = 15,
    profit_target_pct: float = 10.0,
    daily_loss_limit_pct: float = 5.0,
    total_loss_limit_pct: float = 10.0,
) -> dict:
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
            if prev_eq > 0 and (eq - prev_eq) / prev_eq * 100.0 <= -daily_loss_limit_pct:
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


def _yearly_breakdown(result, label: str = "") -> None:
    if not result.trades:
        print(f"  {label}: no trades")
        return
    by_year: dict = {}
    for t in result.trades:
        y = t.entry_time.year
        by_year.setdefault(y, []).append(t.pnl_dollars)
    print(f"  {label} — yearly P&L:")
    for yr in sorted(by_year):
        pnls = by_year[yr]
        wins = sum(1 for p in pnls if p > 0)
        total = len(pnls)
        net   = sum(pnls)
        wr    = wins / total * 100 if total > 0 else 0
        print(f"    {yr}: {total:3d} trades  WR={wr:.0f}%  Net=${net:+,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 0 — Spread comparison: M15 GoldSweepReversal
# ─────────────────────────────────────────────────────────────────────────────

def run_task0():
    print("\n" + "=" * 72)
    print("TASK 0: Spread Comparison — M15 GoldSweepReversal")
    print("=" * 72)
    print(f"Prompt 21 baseline (25-pip spread): Sharpe = {P21_BASELINE_SHARPE:.3f}")
    print(f"Current instruments.json spread: {XAUUSD_CFG['typical_spread_pips']:.1f} pips")
    print()

    from strategies.gold_sweep_reversal import GoldSweepReversal

    df_m15 = _load_m15()
    # Match Prompt 21 start date (2020-07-01)
    df_m15_base = df_m15[df_m15.index >= pd.Timestamp("2020-07-01", tz="UTC")]

    cfg = dict(SWEEP_BASE)
    cfg["tp_mode"] = "A"
    s = GoldSweepReversal()
    s.setup(cfg, XAUUSD_CFG)

    t0 = time.perf_counter()
    result = _run(s, df_m15_base.reset_index(), INSTRUMENT, LOOSE_FTMO)
    elapsed = time.perf_counter() - t0

    m = calculate_metrics(result)
    new_sharpe = m.get("sharpe_ratio", 0.0)
    delta = new_sharpe - P21_BASELINE_SHARPE

    print(f"12-pip spread result  (TP mode A):")
    _print_metrics(m, f"GoldSweepReversal @12pip ({elapsed:.0f}s)")
    print()
    print(f"  Sharpe change: {P21_BASELINE_SHARPE:.3f} → {new_sharpe:.3f}  (Δ = {delta:+.3f})")
    if delta > 0.05:
        print(f"  ✓ Spread fix materially improves Sharpe (+{delta:.3f})")
    else:
        print(f"  ~ Spread fix has minimal impact on Sharpe ({delta:+.3f})")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Baseline: GoldM1Scalper full range
# ─────────────────────────────────────────────────────────────────────────────

def run_task2():
    print("\n" + "=" * 72)
    print("TASK 2: Baseline — GoldM1Scalper (Full M1 Range)")
    print("=" * 72)
    print(f"  M1 parquet: {PARQUET_M1}")
    print(f"  Instruments.json spread: {XAUUSD_CFG['typical_spread_pips']:.1f} pips")
    print()

    from strategies.gold_m1_scalper import GoldM1Scalper

    df_m1 = _load_m1()
    print(f"  Loaded {len(df_m1):,} M1 bars: {df_m1['datetime'].min()} → {df_m1['datetime'].max()}")

    s = GoldM1Scalper()
    s.setup(M1_SCALPER_BASE, XAUUSD_CFG)

    print("  Running baseline (run_fast)...")
    t0 = time.perf_counter()
    result = _run_fast(s, df_m1, INSTRUMENT, LOOSE_FTMO)
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.0f}s")

    m = calculate_metrics(result)
    if "error" in m:
        print(f"  ERROR: {m['error']}")
        return result, m

    t_day  = _trades_per_day(result)
    sl_avg = _avg_sl_pips(result)

    print()
    print(f"  Trades total:    {m['total_trades']}")
    print(f"  T/day avg:       {t_day:.2f}  (target: 4–6)")
    print(f"  T/month avg:     {m['trades_per_month_avg']:.1f}")
    print(f"  Win rate:        {m['win_rate_pct']:.1f}%  (break-even for 1.5:1 RR = 40.0%)")
    print(f"  Avg SL pips:     {sl_avg:.1f}")
    print(f"  Total return:    {m['total_return_pct']:.2f}%")
    print(f"  Sharpe:          {m['sharpe_ratio']:.3f}")
    print(f"  Sortino:         {m['sortino_ratio']:.3f}")
    print(f"  MaxDD:           {m['max_drawdown_pct']:.2f}%")
    print(f"  MaxDailyDD:      {m['max_daily_drawdown_pct']:.2f}%")
    print(f"  Profit factor:   {m['profit_factor']:.3f}")
    print(f"  Expectancy R:    {m['expectancy_r']:.4f}")
    print(f"  Avg win $:       ${m['avg_win_dollars']:.2f}")
    print(f"  Avg loss $:      ${m['avg_loss_dollars']:.2f}")
    print()

    # Yearly breakdown
    _yearly_breakdown(result, "GoldM1Scalper")

    # Exit reason breakdown
    exit_reasons: dict = {}
    for t in result.trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    print("\n  Exit reasons:")
    for reason, cnt in sorted(exit_reasons.items()):
        pct = cnt / len(result.trades) * 100 if result.trades else 0
        print(f"    {reason:15s}: {cnt:4d} ({pct:.1f}%)")

    # Level type breakdown
    level_types: dict = {}
    for t in result.trades:
        lt = getattr(t, "strategy_name", "?")
    # Check via df signal column if available
    print()

    if m["sharpe_ratio"] < -1.0:
        print("  GATE FAIL: Sharpe < -1.0 — strategy not viable, skip WF")
    elif m["win_rate_pct"] < 35.0:
        print(f"  GATE FAIL: WR={m['win_rate_pct']:.1f}% far below 40% break-even — likely not viable")
    else:
        print(f"  GATE PASS: WR={m['win_rate_pct']:.1f}% >= 35% threshold, proceed to WF")

    return result, m


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_task3(baseline_m: dict) -> bool:
    print("\n" + "=" * 72)
    print("TASK 3: Walk-Forward — GoldM1Scalper")
    print("=" * 72)

    # Hard gate: skip WF if baseline Sharpe < -1.0
    if baseline_m.get("sharpe_ratio", -999) < -1.0:
        print("  SKIPPED: Baseline Sharpe < -1.0 (walk-forward gate not passed)")
        return False

    df_m1 = _load_m1()

    print(f"  Grid: {len(list(__import__('itertools').product(*_M1_SCALPER_GRID.values())))} combos")
    print(f"  Windows: IS=3m  OOS=2m  step=2m  | min_trades=20")
    print()

    t0 = time.perf_counter()
    wf = run_walk_forward(
        df=df_m1,
        strategy_name="gold_m1_scalper",
        instrument=INSTRUMENT,
        initial_balance=INITIAL_BALANCE,
        in_sample_months=3,
        oos_months=2,
        step_months=2,
        min_trades=20,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  WF completed in {elapsed/60:.1f} minutes")

    # Summary
    wf_windows = wf.windows
    n_windows  = len(wf_windows)
    if n_windows == 0:
        print("  ERROR: No valid WF windows produced")
        return False

    avg_deg       = float(np.mean([w.degradation_ratio for w in wf_windows if w.oos_sharpe is not None]))
    oos_positive  = sum(1 for w in wf_windows if (w.oos_sharpe or 0) > 0)
    pct_positive  = oos_positive / n_windows * 100

    print(f"\n  WF Summary:")
    print(f"    Windows:         {n_windows}")
    print(f"    Avg degradation: {avg_deg:.4f}  (target: > 0.1)")
    print(f"    OOS positive:    {oos_positive}/{n_windows} = {pct_positive:.1f}%  (target: > 50%)")
    print()

    # Per-window table
    print(f"  {'Window':<8} {'IS Sharpe':>10} {'OOS Sharpe':>10} {'Deg':>8} {'OOS Trades':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for w in wf_windows:
        is_sh  = f"{w.is_sharpe:.3f}"  if w.is_sharpe  is not None else "  n/a"
        oos_sh = f"{w.oos_sharpe:.3f}" if w.oos_sharpe is not None else "  n/a"
        deg    = f"{w.degradation_ratio:.3f}" if w.degradation_ratio is not None else "  n/a"
        print(f"  {w.label:<8} {is_sh:>10} {oos_sh:>10} {deg:>8} {w.oos_trades:>10}")

    # Verdict
    pass_wf = avg_deg > 0.1 and pct_positive > 50.0
    print()
    if pass_wf:
        print(f"  WF PASS: avg_deg={avg_deg:.3f} > 0.1  AND  {pct_positive:.1f}% > 50% positive OOS")
    else:
        reasons = []
        if avg_deg <= 0.1:
            reasons.append(f"avg_deg={avg_deg:.3f} <= 0.1")
        if pct_positive <= 50.0:
            reasons.append(f"OOS positive={pct_positive:.1f}% <= 50%")
        print(f"  WF FAIL: {' | '.join(reasons)}")
        print("  VERDICT: GoldM1Scalper NOT viable — strategy is DEAD")

    return pass_wf


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — FTMO Evaluation (only if WF passes)
# ─────────────────────────────────────────────────────────────────────────────

def run_task4():
    print("\n" + "=" * 72)
    print("TASK 4: FTMO Evaluation + Risk Sensitivity")
    print("=" * 72)

    from strategies.gold_m1_scalper import GoldM1Scalper

    df_m1 = _load_m1()

    # 15-day window eval at baseline risk
    s = GoldM1Scalper()
    s.setup(M1_SCALPER_BASE, XAUUSD_CFG)
    result_base = _run_fast(s, df_m1, INSTRUMENT, LOOSE_FTMO)
    eq_base = result_base.equity_curve

    ev = _eval_windows_15day(eq_base, INITIAL_BALANCE)
    print(f"\n  15-day FTMO challenge eval (baseline risk={M1_SCALPER_BASE['risk_per_trade_pct']}%):")
    print(f"    Pass rate:      {ev['pass_rate']:.1f}%")
    print(f"    Fail rate:      {ev['fail_rate']:.1f}%")
    print(f"    2-attempt pass: {ev['two_attempt']:.1f}%")
    print(f"    Windows:        {ev['total']}")

    # Risk sensitivity
    risk_levels = [0.5, 0.8, 1.0, 1.5]
    print(f"\n  Risk sensitivity ({', '.join(f'{r}%' for r in risk_levels)}):")
    print(f"  {'Risk%':>6} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} {'WR%':>7} {'FTMO pass%':>11}")
    print(f"  {'-'*6} {'-'*8} {'-'*9} {'-'*8} {'-'*7} {'-'*11}")

    best_risk = None
    best_sharpe = -999
    for risk_pct in risk_levels:
        cfg = {**M1_SCALPER_BASE, "risk_per_trade_pct": risk_pct}
        s2 = GoldM1Scalper()
        s2.setup(cfg, XAUUSD_CFG)
        r2 = _run_fast(s2, df_m1, INSTRUMENT, LOOSE_FTMO)
        m2 = calculate_metrics(r2)
        if "error" in m2:
            print(f"  {risk_pct:6.1f} {'n/a':>8} {'n/a':>9} {'n/a':>8} {'n/a':>7} {'n/a':>11}")
            continue
        ev2 = _eval_windows_15day(r2.equity_curve, INITIAL_BALANCE)
        sh = m2["sharpe_ratio"]
        rt = m2["total_return_pct"]
        dd = m2["max_drawdown_pct"]
        wr = m2["win_rate_pct"]
        pr = ev2["pass_rate"]
        print(f"  {risk_pct:6.1f} {sh:8.3f} {rt:9.2f} {dd:8.2f} {wr:7.1f} {pr:11.1f}")
        if sh > best_sharpe:
            best_sharpe = sh
            best_risk   = risk_pct

    print(f"\n  Optimal risk by Sharpe: {best_risk}% (Sharpe={best_sharpe:.3f})")

    # Combined portfolio: London S3 + Gold M1 Scalper
    print("\n  Combined portfolio analysis (London S3 + Gold M1 Scalper):")
    _run_combined_portfolio(df_m1)


def _run_combined_portfolio(df_m1: pd.DataFrame) -> None:
    """Quick estimate of London S3 + Gold M1 combined equity curve."""
    from strategies.gold_m1_scalper import GoldM1Scalper
    from strategies.london_open_breakout import LondonOpenBreakout

    sp = _ALL_PARAMS
    ic = _INSTR_CFG

    london_pairs = [
        ("EURUSD", 1.2), ("GBPUSD", 1.2), ("EURJPY", 1.0),
        ("GBPJPY", 0.5), ("AUDUSD", 0.3), ("USDJPY", 0.5),
    ]
    london_overrides = {
        "EURJPY": {"min_asian_range_pips": 40, "max_asian_range_pips": 120, "entry_buffer_pips": 7},
        "GBPJPY": {"min_asian_range_pips": 30, "max_asian_range_pips": 120, "entry_buffer_pips": 7},
        "AUDUSD": {"min_asian_range_pips": 15, "max_asian_range_pips":  80, "entry_buffer_pips": 2},
        "USDJPY": {"min_asian_range_pips": 15, "max_asian_range_pips":  80, "entry_buffer_pips": 5},
    }

    equities = []

    # London S3 (load from parquet files if available)
    london_start = "2022-01-01"  # match M1 data start
    for pair, risk in london_pairs:
        pq = ROOT / "data" / "historical" / f"{pair}_15m.parquet"
        if not pq.exists():
            continue
        df_p = pd.read_parquet(pq)
        if "datetime" in df_p.columns:
            df_p = df_p.set_index("datetime")
        if df_p.index.tz is None:
            df_p.index = df_p.index.tz_localize("UTC")
        df_p.columns = [c.lower() for c in df_p.columns]
        df_p = df_p[df_p.index >= pd.Timestamp(london_start, tz="UTC")]

        cfg = dict(sp["london_open_breakout"])
        cfg.pop("instrument_overrides", None)
        cfg.update(london_overrides.get(pair, {}))
        cfg["risk_per_trade_pct"] = risk

        strat = LondonOpenBreakout()
        strat.setup(cfg, ic[pair])

        try:
            r = Backtester(
                strategy=strat, df=df_p.reset_index(), instrument=pair,
                initial_balance=INITIAL_BALANCE, _override_ftmo_rules=LOOSE_FTMO,
            ).run()
            equities.append(r.equity_curve.resample("1h").last().ffill())
        except Exception as e:
            print(f"    London {pair} failed: {e}")

    # Gold M1 Scalper
    s = GoldM1Scalper()
    s.setup(M1_SCALPER_BASE, XAUUSD_CFG)
    r_gold = _run_fast(s, df_m1, INSTRUMENT, LOOSE_FTMO)
    equities.append(r_gold.equity_curve.resample("1h").last().ffill())

    if not equities:
        print("    No results for combined portfolio")
        return

    # Combine: sum P&L, add to initial balance
    combined_pnl = pd.concat(equities, axis=1).fillna(0.0)
    combined_pnl = combined_pnl.diff().fillna(0.0).sum(axis=1)
    combined_eq  = INITIAL_BALANCE + combined_pnl.cumsum()

    # Stats
    daily_eq = combined_eq.resample("D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    peak   = np.maximum.accumulate(combined_eq.values)
    max_dd = float(abs(((combined_eq.values - peak) / np.where(peak == 0, 1, peak)).min())) * 100

    initial = combined_eq.iloc[0]
    final   = combined_eq.iloc[-1]
    total_r = (final - initial) / initial * 100
    years   = (combined_eq.index[-1] - combined_eq.index[0]).days / 365.25
    cagr    = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

    monthly = combined_eq.resample("ME").last()
    m_rets  = monthly.pct_change().dropna() * 100
    avg_mo  = float(m_rets.mean()) if len(m_rets) else 0

    print(f"    London S3 pairs loaded: {len(equities) - 1}")
    print(f"    Combined range: {combined_eq.index[0].date()} → {combined_eq.index[-1].date()}")
    print(f"    Avg monthly return: {avg_mo:.2f}%")
    print(f"    CAGR:               {cagr:.2f}%")
    print(f"    Total return:       {total_r:.2f}%")
    print(f"    Sharpe:             {sharpe:.3f}")
    print(f"    MaxDD:              {max_dd:.2f}%")

    ev = _eval_windows_15day(combined_eq, INITIAL_BALANCE)
    print(f"    15-day pass rate:   {ev['pass_rate']:.1f}%")
    print(f"    15-day 2-attempt:   {ev['two_attempt']:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("=" * 72)
    print("Prompt 23: Gold M1 Micro-Swing Scalper")
    print(f"venv Python {sys.version.split()[0]} | pandas {pd.__version__}")
    print(f"instruments.json XAUUSD spread: {XAUUSD_CFG['typical_spread_pips']:.1f} pips")
    print("=" * 72)

    # Task 0: spread comparison
    run_task0()

    # Task 2: baseline
    result_base, metrics_base = run_task2()

    # Task 3: walk-forward
    wf_pass = run_task3(metrics_base)

    # Task 4: FTMO eval (only if WF passes)
    if wf_pass:
        run_task4()
    else:
        print("\n" + "=" * 72)
        print("TASKS 4 SKIPPED: WF did not pass")
        print("=" * 72)

    print("\n" + "=" * 72)
    print("Prompt 23 COMPLETE")
    print("=" * 72)
