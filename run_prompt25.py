#!/usr/bin/env python3
"""
run_prompt25.py — News Event Straddle Strategy
===============================================
Task 0 : Build & verify news calendar
Task 1 : Verify straddle/fade code, event-bar sanity
Task 2 : Baseline backtest 2022-Apr2025 — all events, per event type, per instrument
Task 3 : Walk-forward 2022-2025 (6m IS / 3m OOS / 3m step, 243 combos)
Task 4 : Combined portfolio (London S3 + best news variant) + FTMO eval sim
Task 5 : Correlation analysis vs London S3
"""
from __future__ import annotations

import os, sys, json, pickle, warnings, itertools
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from data.news_calendar import build_calendar
from strategies.news_straddle import (
    INST_CFG, NORMAL_SPREAD_PIPS,
    generate_straddle_setups, generate_fade_setups, NewsSetup,
)

ET = pytz.timezone("US/Eastern")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Instrument data paths ─────────────────────────────────────────────────────
DATA_DIR = Path("data/historical")

INST_DATA = {
    "EURUSD": DATA_DIR / "EURUSD_15m.parquet",
    "GBPUSD": DATA_DIR / "GBPUSD_15m.parquet",
    "XAUUSD": DATA_DIR / "XAUUSD_15m.parquet",
}

# London S3 monthly P&L pickle (from Prompt 18 combined result)
LONDON_EQUITY_PKL = RESULTS_DIR / "combined_11strat_equity.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_m15(instrument: str) -> pd.DataFrame:
    path = INST_DATA[instrument]
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    # datetime is a column in these parquets
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        df = df.set_index("datetime")
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df.sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# Custom simulation engine
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NewsTradeResult:
    setup:       NewsSetup
    exit_bar:    int
    exit_price:  float
    exit_reason: str       # 'tp' | 'sl' | 'time_stop' | 'end_of_data'
    pnl_pips:    float
    pnl_dollars: float
    lot_size:    float
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp


def simulate_news(
    setups: List[NewsSetup],
    df: pd.DataFrame,
    initial_balance: float = 10_000.0,
    risk_pct: float = 1.0,
) -> Tuple[List[NewsTradeResult], pd.Series, float]:
    """
    Bar-by-bar simulation of news setups.

    Returns (trades, equity_series, final_balance).
    """
    balance = initial_balance
    equity_series: Dict[pd.Timestamp, float] = {}
    trades: List[NewsTradeResult] = []

    # Sort setups by event bar index (they should already be sorted by time)
    setups_sorted = sorted(setups, key=lambda s: df.index[s.event_bar_idx] if s.event_bar_idx < len(df) else pd.Timestamp.max)

    last_exit_idx = -1  # prevent overlapping positions

    for setup in setups_sorted:
        entry_i = setup.event_bar_idx
        if entry_i >= len(df) or entry_i <= last_exit_idx:
            continue

        pip_size  = setup.pip_size
        pip_value = setup.pip_value
        direction = setup.direction
        entry     = setup.entry_price
        sl        = setup.sl_price
        tp        = setup.tp_price
        ts        = setup.time_stop

        risk_pips  = abs(entry - sl) / pip_size
        if risk_pips < 0.1:
            continue
        risk_dollars = balance * risk_pct / 100.0
        lot_size     = risk_dollars / (risk_pips * pip_value)
        lot_size     = min(lot_size, 50.0)   # safety cap

        entry_time = df.index[entry_i]
        exit_i     = entry_i
        exit_price = entry
        exit_reason= "end_of_data"

        # Simulate bar by bar from entry_i onward
        for j in range(entry_i, len(df)):
            bar = df.iloc[j]
            bar_ts = df.index[j]

            # Check time stop first
            if bar_ts >= ts:
                exit_price  = float(bar["open"])
                exit_reason = "time_stop"
                exit_i      = j
                break

            # Check SL/TP using bar high/low
            if direction == 1:  # long
                if bar["low"] <= sl:
                    exit_price  = sl
                    exit_reason = "sl"
                    exit_i      = j
                    break
                if bar["high"] >= tp:
                    exit_price  = tp
                    exit_reason = "tp"
                    exit_i      = j
                    break
            else:               # short
                if bar["high"] >= sl:
                    exit_price  = sl
                    exit_reason = "sl"
                    exit_i      = j
                    break
                if bar["low"] <= tp:
                    exit_price  = tp
                    exit_reason = "tp"
                    exit_i      = j
                    break
        else:
            exit_i      = len(df) - 1
            exit_price  = float(df.iloc[exit_i]["close"])
            exit_reason = "end_of_data"

        pnl_pips    = direction * (exit_price - entry) / pip_size
        pnl_dollars = pnl_pips * pip_value * lot_size
        balance    += pnl_dollars
        last_exit_idx = exit_i

        t = NewsTradeResult(
            setup       = setup,
            exit_bar    = exit_i,
            exit_price  = exit_price,
            exit_reason = exit_reason,
            pnl_pips    = pnl_pips,
            pnl_dollars = pnl_dollars,
            lot_size    = lot_size,
            entry_time  = entry_time,
            exit_time   = df.index[exit_i],
        )
        trades.append(t)
        equity_series[entry_time] = balance

    eq = pd.Series(equity_series).sort_index()
    return trades, eq, balance


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe(trades: List[NewsTradeResult], initial_balance: float) -> float:
    if len(trades) < 3:
        return float("nan")
    rets = [t.pnl_dollars / initial_balance for t in trades]
    m, s = np.mean(rets), np.std(rets, ddof=1)
    if s == 0:
        return float("nan")
    return float(m / s * np.sqrt(252))


def _max_dd(trades: List[NewsTradeResult], initial_balance: float) -> float:
    """Max drawdown % from peak."""
    if not trades:
        return 0.0
    bal = initial_balance
    peak = bal
    max_dd = 0.0
    for t in trades:
        bal += t.pnl_dollars
        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _metrics(trades: List[NewsTradeResult], initial_balance: float) -> dict:
    if not trades:
        return dict(n=0, wr=0, ret_pct=0, sharpe=float("nan"), maxdd=0,
                    avg_pnl=0, t_per_mo=0)
    wins = sum(1 for t in trades if t.pnl_dollars > 0)
    wr   = wins / len(trades) * 100
    total_pnl = sum(t.pnl_dollars for t in trades)
    ret_pct   = total_pnl / initial_balance * 100
    # trades per month
    if len(trades) > 1:
        span_days = (trades[-1].exit_time - trades[0].entry_time).days + 1
        t_per_mo  = len(trades) / max(span_days / 30.44, 1.0)
    else:
        t_per_mo = len(trades)
    return dict(
        n       = len(trades),
        wr      = wr,
        ret_pct = ret_pct,
        sharpe  = _sharpe(trades, initial_balance),
        maxdd   = _max_dd(trades, initial_balance),
        avg_pnl = total_pnl / len(trades),
        t_per_mo= t_per_mo,
    )


def print_metrics(label: str, m: dict) -> None:
    if m["n"] == 0:
        print(f"  {label}: no trades")
        return
    print(f"  {label}: T={m['n']} WR={m['wr']:.1f}% Ret={m['ret_pct']:.2f}% "
          f"Sharpe={m['sharpe']:.3f} MaxDD={m['maxdd']:.1f}% T/mo={m['t_per_mo']:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 0: Build & verify news calendar
# ══════════════════════════════════════════════════════════════════════════════

def task0_calendar():
    print("\n" + "=" * 70)
    print("TASK 0: News Calendar")
    print("=" * 70)

    cal = build_calendar()
    out = Path("data/news_calendar.parquet")
    cal.to_parquet(out)
    print(f"Saved {len(cal)} events → {out}")

    print("\nEvents per type:")
    counts = cal.groupby("event_type").size().sort_values(ascending=False)
    for etype, cnt in counts.items():
        print(f"  {etype:15s}: {cnt:3d}")

    print(f"\nDate range: {cal['datetime'].min()} → {cal['datetime'].max()}")

    # Sample: show Jan 2022 events
    jan22 = cal[cal["datetime"].dt.year == 2022]
    jan22 = jan22[jan22["datetime"].dt.month == 1]
    print(f"\nJan 2022 events ({len(jan22)} total):")
    for _, r in jan22.iterrows():
        print(f"  {r['datetime'].strftime('%Y-%m-%d %H:%M %Z'):30s}  {r['event_type']:15s}  {r['impact']}")

    return cal


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1: Code verification & event-bar sanity
# ══════════════════════════════════════════════════════════════════════════════

def task1_verify(cal: pd.DataFrame):
    print("\n" + "=" * 70)
    print("TASK 1: Strategy Verification")
    print("=" * 70)

    # Check data availability
    for inst in ("EURUSD", "GBPUSD", "XAUUSD"):
        path = INST_DATA[inst]
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        df = load_m15(inst)
        print(f"  {inst}: {len(df)} bars  {df.index[0]} → {df.index[-1]}")

    # Verify NFP event bars exist in EURUSD data
    print("\nNFP event bar check (EURUSD, first 5):")
    df_eu = load_m15("EURUSD")
    nfp = cal[cal["event_type"] == "NFP"].head(5)
    for _, row in nfp.iterrows():
        evt_ts = row["datetime"]
        pos = df_eu.index.searchsorted(evt_ts)
        if pos < len(df_eu):
            bar = df_eu.iloc[pos]
            bar_ts = df_eu.index[pos]
            rng = (bar["high"] - bar["low"]) / INST_CFG["EURUSD"]["pip_size"]
            print(f"  {evt_ts.strftime('%Y-%m-%d %H:%M')} → bar at {bar_ts.strftime('%H:%M')} "
                  f"range={rng:.0f} pips")
        else:
            print(f"  {evt_ts.strftime('%Y-%m-%d')} → NOT FOUND")

    # Quick straddle setup count
    print("\nStraddle setup count (default params, EURUSD, all events):")
    df_eu = load_m15("EURUSD")
    setups = generate_straddle_setups(df_eu, cal, "EURUSD",
                                      bracket_atr_mult=0.5, rr_ratio=2.0,
                                      time_stop_hours=4.0, news_spread_mult=2.0)
    print(f"  Total straddle setups: {len(setups)}")
    by_type = {}
    for s in setups:
        by_type[s.event_type] = by_type.get(s.event_type, 0) + 1
    for et, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {et:15s}: {cnt:3d}")

    print("\nFade setup count (default params, EURUSD, all events):")
    fades = generate_fade_setups(df_eu, cal, "EURUSD",
                                 min_spike_pips=15.0, sl_buffer_pips=10.0,
                                 rr_ratio=2.0, time_stop_hours=4.0, news_spread_mult=2.0)
    print(f"  Total fade setups: {len(fades)}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2: Baseline backtest
# ══════════════════════════════════════════════════════════════════════════════

STRADDLE_PARAMS = dict(bracket_atr_mult=0.5, rr_ratio=2.0, time_stop_hours=4.0, news_spread_mult=2.0)
FADE_PARAMS     = dict(min_spike_pips=20.0, sl_buffer_pips=10.0, rr_ratio=2.0, time_stop_hours=4.0, news_spread_mult=2.0)
RISK_PCT        = 1.0
INITIAL_BAL     = 10_000.0

VERY_HIGH_EVENTS = ["NFP", "FOMC", "CPI", "GDP_ADV"]
HIGH_EVENTS      = ["PPI", "RETAIL_SALES", "PCE", "ISM_PMI"]


def task2_baseline(cal: pd.DataFrame):
    print("\n" + "=" * 70)
    print("TASK 2: Baseline Backtest (Jan 2022 – Apr 2025)")
    print("=" * 70)

    instruments = [inst for inst in ("EURUSD", "GBPUSD", "XAUUSD") if INST_DATA[inst].exists()]
    dfs = {inst: load_m15(inst) for inst in instruments}

    # ── Per instrument, per variant ────────────────────────────────────────
    all_straddle: List[NewsTradeResult] = []
    all_fade:     List[NewsTradeResult] = []

    print("\n--- All Events, All Instruments ---")
    for inst in instruments:
        df = dfs[inst]
        # Straddle
        setups_s = generate_straddle_setups(df, cal, inst, **STRADDLE_PARAMS)
        trades_s, _, _ = simulate_news(setups_s, df, INITIAL_BAL, RISK_PCT)
        all_straddle.extend(trades_s)
        m = _metrics(trades_s, INITIAL_BAL)
        print_metrics(f"Straddle {inst}", m)

        # Fade
        setups_f = generate_fade_setups(df, cal, inst, **FADE_PARAMS)
        trades_f, _, _ = simulate_news(setups_f, df, INITIAL_BAL, RISK_PCT)
        all_fade.extend(trades_f)
        m = _metrics(trades_f, INITIAL_BAL)
        print_metrics(f"Fade    {inst}", m)

    # ── Per event type (EURUSD only) ───────────────────────────────────────
    print("\n--- Per Event Type (EURUSD Straddle) ---")
    df_eu = dfs["EURUSD"]
    event_types = cal["event_type"].unique().tolist()
    event_results = {}
    for et in sorted(event_types):
        setups = generate_straddle_setups(df_eu, cal, "EURUSD",
                                          filter_event_types=[et], **STRADDLE_PARAMS)
        trades, _, _ = simulate_news(setups, df_eu, INITIAL_BAL, RISK_PCT)
        m = _metrics(trades, INITIAL_BAL)
        event_results[et] = m
        print_metrics(et, m)

    # ── Very-high-impact only (best filter) ───────────────────────────────
    print("\n--- VERY_HIGH Events Only (EURUSD Straddle) ---")
    setups_vh = generate_straddle_setups(df_eu, cal, "EURUSD",
                                          filter_event_types=VERY_HIGH_EVENTS,
                                          **STRADDLE_PARAMS)
    trades_vh, _, _ = simulate_news(setups_vh, df_eu, INITIAL_BAL, RISK_PCT)
    m_vh = _metrics(trades_vh, INITIAL_BAL)
    print_metrics("VERY_HIGH events straddle", m_vh)

    print("\n--- VERY_HIGH Events Only (EURUSD Fade) ---")
    setups_vhf = generate_fade_setups(df_eu, cal, "EURUSD",
                                       filter_event_types=VERY_HIGH_EVENTS,
                                       **FADE_PARAMS)
    trades_vhf, _, _ = simulate_news(setups_vhf, df_eu, INITIAL_BAL, RISK_PCT)
    m_vhf = _metrics(trades_vhf, INITIAL_BAL)
    print_metrics("VERY_HIGH events fade", m_vhf)

    # ── NFP + CPI + FOMC separately ───────────────────────────────────────
    print("\n--- NFP / CPI / FOMC individual (EURUSD Straddle) ---")
    for et in ["NFP", "CPI", "FOMC", "GDP_ADV"]:
        setups = generate_straddle_setups(df_eu, cal, "EURUSD",
                                          filter_event_types=[et], **STRADDLE_PARAMS)
        trades, _, _ = simulate_news(setups, df_eu, INITIAL_BAL, RISK_PCT)
        m = _metrics(trades, INITIAL_BAL)
        print_metrics(f"{et:12s}", m)

    # ── Root-cause analysis: straddle win-rate forensics ──────────────────
    print("\n--- Root-Cause: Straddle Exit Reasons (EURUSD, ALL events) ---")
    all_setups_s = generate_straddle_setups(df_eu, cal, "EURUSD", **STRADDLE_PARAMS)
    all_trades_s, _, _ = simulate_news(all_setups_s, df_eu, INITIAL_BAL, RISK_PCT)
    reasons = {}
    for t in all_trades_s:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r:15s}: {cnt} ({cnt/len(all_trades_s)*100:.1f}%)")

    # Breakdown: does WR vary with bracket size?
    print("\n--- Straddle WR vs bracket_atr_mult (EURUSD, ALL events) ---")
    for bam in [0.3, 0.5, 0.8, 1.2, 2.0]:
        setups_b = generate_straddle_setups(df_eu, cal, "EURUSD",
                                             bracket_atr_mult=bam, rr_ratio=2.0,
                                             time_stop_hours=4.0, news_spread_mult=2.0)
        trades_b, _, _ = simulate_news(setups_b, df_eu, INITIAL_BAL, RISK_PCT)
        if trades_b:
            wr = sum(1 for t in trades_b if t.pnl_dollars > 0) / len(trades_b) * 100
            avg_sl = np.mean([abs(s.entry_price - s.sl_price) / s.pip_size for s in setups_b])
            print(f"  bracket={bam:.1f}x: n={len(trades_b)} WR={wr:.1f}% avg_SL={avg_sl:.1f}pips")
        else:
            print(f"  bracket={bam:.1f}x: no setups")

    # Breakdown: fade WR — is it close to breakeven?
    print("\n--- Fade WR vs min_spike_pips (EURUSD, ALL events) ---")
    for msp in [10.0, 20.0, 30.0, 40.0, 50.0]:
        setups_f2 = generate_fade_setups(df_eu, cal, "EURUSD",
                                          min_spike_pips=msp, sl_buffer_pips=10.0,
                                          rr_ratio=2.0, time_stop_hours=4.0, news_spread_mult=2.0)
        trades_f2, _, _ = simulate_news(setups_f2, df_eu, INITIAL_BAL, RISK_PCT)
        if trades_f2:
            wr  = sum(1 for t in trades_f2 if t.pnl_dollars > 0) / len(trades_f2) * 100
            ret = sum(t.pnl_dollars for t in trades_f2) / INITIAL_BAL * 100
            sh  = _sharpe(trades_f2, INITIAL_BAL)
            sh_str = f"{sh:.3f}" if not np.isnan(sh) else "n/a"
            print(f"  min_spike={msp:.0f}p: n={len(trades_f2)} WR={wr:.1f}% Ret={ret:.1f}% Sharpe={sh_str}")
        else:
            print(f"  min_spike={msp:.0f}p: no setups")

    # Determine best baseline for WF
    best_label = "VERY_HIGH straddle EURUSD"
    best_sharpe = m_vh.get("sharpe", float("nan"))
    print(f"\nBest baseline: {best_label}  Sharpe={best_sharpe:.3f}")
    print(f"WF gate check: Sharpe > -1.0? {'YES - proceed to WF' if best_sharpe > -1.0 else 'NO - skip WF'}")

    return dict(
        dfs             = dfs,
        best_sharpe     = best_sharpe,
        proceed_wf      = best_sharpe > -1.0,
        event_results   = event_results,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3: Walk-Forward (6m IS / 3m OOS / 3m step)
# ══════════════════════════════════════════════════════════════════════════════

# 243 combos = 3^5
WF_GRID = list(itertools.product(
    [0.3, 0.5, 0.7],      # bracket_atr_mult
    [1.5, 2.0, 2.5],      # rr_ratio
    [0.5, 1.0, 1.5],      # risk_pct
    [1.5, 2.0, 2.5],      # news_spread_mult
    [2.0, 4.0, 6.0],      # time_stop_hours
))
assert len(WF_GRID) == 243

MIN_IS_TRADES = 8


def _run_straddle_window(
    df: pd.DataFrame,
    cal_window: pd.DataFrame,
    instrument: str,
    bracket_atr_mult: float,
    rr_ratio: float,
    risk_pct: float,
    news_spread_mult: float,
    time_stop_hours: float,
) -> float:
    """Run straddle + return Sharpe for a WF window."""
    setups = generate_straddle_setups(
        df, cal_window, instrument,
        bracket_atr_mult   = bracket_atr_mult,
        rr_ratio           = rr_ratio,
        time_stop_hours    = time_stop_hours,
        news_spread_mult   = news_spread_mult,
        filter_event_types = VERY_HIGH_EVENTS,
    )
    if len(setups) < MIN_IS_TRADES:
        return float("nan")
    trades, _, _ = simulate_news(setups, df, INITIAL_BAL, risk_pct)
    if len(trades) < MIN_IS_TRADES:
        return float("nan")
    return _sharpe(trades, INITIAL_BAL)


def _wf_windows(start_date: str, end_date: str, is_months: int = 6, oos_months: int = 3) -> list:
    """Generate (is_start, is_end, oos_start, oos_end) tuples."""
    s = pd.Timestamp(start_date, tz="US/Eastern")
    e = pd.Timestamp(end_date,   tz="US/Eastern")
    windows = []
    cur = s
    while True:
        is_end   = cur + pd.DateOffset(months=is_months)
        oos_end  = is_end + pd.DateOffset(months=oos_months)
        if oos_end > e:
            break
        windows.append((cur, is_end, is_end, oos_end))
        cur = cur + pd.DateOffset(months=oos_months)   # 3m step
    return windows


def task3_walkforward(task2_result: dict, cal: pd.DataFrame):
    print("\n" + "=" * 70)
    print("TASK 3: Walk-Forward (6m IS / 3m OOS / 3m step, 243 combos)")
    print("=" * 70)

    if not task2_result["proceed_wf"]:
        print("SKIP: baseline Sharpe < -1.0 — strategy is not viable")
        return None

    dfs  = task2_result["dfs"]
    inst = "EURUSD"
    df   = dfs[inst]

    windows = _wf_windows("2022-01-01", "2025-04-30")
    print(f"WF windows: {len(windows)}")
    for w in windows:
        print(f"  IS: {w[0].date()} – {w[1].date()}  OOS: {w[2].date()} – {w[3].date()}")

    oos_trades_all: List[NewsTradeResult] = []
    deg_scores: List[float] = []

    for w_i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        # Filter calendar for IS / OOS
        cal_is  = cal[(cal["datetime"] >= is_s)  & (cal["datetime"] < is_e)]
        cal_oos = cal[(cal["datetime"] >= oos_s) & (cal["datetime"] < oos_e)]

        # Filter dataframe slice
        df_is  = df[(df.index >= is_s)  & (df.index < is_e)]
        df_oos = df[(df.index >= oos_s) & (df.index < oos_e)]

        # Grid search on IS
        best_sh  = -np.inf
        best_params = WF_GRID[0]

        for params in WF_GRID:
            bam, rr, rp, nsm, tsh = params
            sh = _run_straddle_window(df_is, cal_is, inst, bam, rr, rp, nsm, tsh)
            if not np.isnan(sh) and sh > best_sh:
                best_sh    = sh
                best_params = params

        # OOS evaluation with best IS params
        bam, rr, rp, nsm, tsh = best_params
        setups_oos = generate_straddle_setups(
            df_oos, cal_oos, inst,
            bracket_atr_mult  = bam,
            rr_ratio          = rr,
            time_stop_hours   = tsh,
            news_spread_mult  = nsm,
            filter_event_types= VERY_HIGH_EVENTS,
        )
        trades_oos, _, _ = simulate_news(setups_oos, df_oos, INITIAL_BAL, rp)
        sh_oos  = _sharpe(trades_oos, INITIAL_BAL) if len(trades_oos) >= 3 else float("nan")
        deg     = sh_oos / best_sh if (not np.isnan(sh_oos) and best_sh > 0) else float("nan")
        oos_trades_all.extend(trades_oos)

        oos_sh_str  = f"{sh_oos:.3f}" if not np.isnan(sh_oos) else "n/a"
        deg_str     = f"{deg:.2f}"    if not np.isnan(deg)    else "n/a"
        print(f"\n  Window {w_i+1}: IS_Sharpe={best_sh:.3f} | OOS_Sharpe={oos_sh_str} | "
              f"deg={deg_str} | n_oos={len(trades_oos)}")
        print(f"    Best params: bracket={bam} rr={rr} risk={rp}% spread_mult={nsm} t_stop={tsh}h")

        if not np.isnan(deg):
            deg_scores.append(deg)

    # WF summary
    avg_deg = float(np.mean(deg_scores)) if deg_scores else float("nan")
    m_oos   = _metrics(oos_trades_all, INITIAL_BAL)
    print(f"\nWF SUMMARY:")
    print(f"  avg_degradation = {avg_deg:.3f}  (>0 = OOS tracks IS, <0 = degraded)")
    print_metrics("  Combined OOS", m_oos)

    proceed_eval = m_oos.get("sharpe", float("nan")) > 0.3
    print(f"\nEval sim gate: OOS Sharpe > 0.3? {'YES' if proceed_eval else 'NO'}")

    return dict(
        oos_trades   = oos_trades_all,
        avg_deg      = avg_deg,
        oos_metrics  = m_oos,
        proceed_eval = proceed_eval,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4: Combined portfolio + FTMO eval sim
# ══════════════════════════════════════════════════════════════════════════════

LONDON_MONTHLY_RETURN = 0.00896  # +0.896%/month from Prompt 18


def task4_portfolio(task3_result, cal: pd.DataFrame, dfs: dict):
    print("\n" + "=" * 70)
    print("TASK 4: Combined Portfolio + FTMO Eval Sim")
    print("=" * 70)

    # --- Always show standalone London S3 reference ---
    print("\nLondon S3 standalone (reference):")
    months_to_target = int(np.ceil(np.log(1.10) / np.log(1.0 + LONDON_MONTHLY_RETURN)))
    print(f"  Monthly return: +{LONDON_MONTHLY_RETURN*100:.3f}%")
    print(f"  Months to +10% FTMO target: {months_to_target}")
    print(f"  CAGR: {((1 + LONDON_MONTHLY_RETURN)**12 - 1)*100:.2f}%")

    if task3_result is None or not task3_result.get("proceed_eval", False):
        print("\nNews straddle eval SKIPPED (WF OOS Sharpe did not meet threshold)")
        print("Combined portfolio: London S3 only (unchanged)")
        return

    oos_trades = task3_result["oos_trades"]
    oos_m      = task3_result["oos_metrics"]

    # Load London equity if available
    news_monthly_return = oos_m["ret_pct"] / 100.0 / max((
        (oos_trades[-1].exit_time - oos_trades[0].entry_time).days / 30.44
        if oos_trades else 1.0
    ), 1.0)

    combined_monthly = LONDON_MONTHLY_RETURN + news_monthly_return
    combined_cagr    = (1 + combined_monthly)**12 - 1
    months_combined  = int(np.ceil(np.log(1.10) / np.log(1.0 + combined_monthly)))

    print(f"\nNews straddle OOS:")
    print_metrics("  OOS", oos_m)
    print(f"\nCombined portfolio:")
    print(f"  Monthly return: +{combined_monthly*100:.3f}%  ({LONDON_MONTHLY_RETURN*100:.3f}% London + {news_monthly_return*100:.3f}% News)")
    print(f"  CAGR:  {combined_cagr*100:.2f}%")
    print(f"  Months to +10% FTMO target: {months_combined}")

    # ── FTMO eval sim: 1000 simulated 60-day challenges ───────────────────
    print("\n--- FTMO Challenge Eval Sim (1000 trials, 60 trading days) ---")
    rng = np.random.default_rng(42)
    if len(oos_trades) >= 5:
        pnls = np.array([t.pnl_dollars / INITIAL_BAL for t in oos_trades])
    else:
        pnls = np.array([0.0])
    # London S3 trade-level pnls approximation (monthly +0.896% / 22 trades)
    london_daily_mu = LONDON_MONTHLY_RETURN / 22.0

    n_trials = 1000
    challenge_days = 60
    passed = 0
    account = 10_000.0
    daily_loss_limit = 0.04    # 4%
    total_dd_limit   = 0.09    # 9%
    target_profit    = 0.10    # 10%

    for _ in range(n_trials):
        bal     = account
        peak    = account
        halted  = False
        day_bal = account
        for day in range(challenge_days):
            if halted:
                break
            # London daily contribution
            london_d = rng.normal(london_daily_mu, london_daily_mu * 2)
            # News: ~1 event every 3 trading days
            news_d = 0.0
            if rng.random() < 1.0 / 3.0 and len(pnls) > 1:
                news_d = float(rng.choice(pnls)) * account
            daily_ret = london_d * account + news_d
            bal      += daily_ret

            # Daily loss check
            if (bal - day_bal) / day_bal < -daily_loss_limit:
                halted = True; break
            # Total DD check
            if bal > peak:
                peak = bal
            if (peak - bal) / account > total_dd_limit:
                halted = True; break
            day_bal = bal  # reset daily reference

        if not halted and (bal - account) / account >= target_profit:
            passed += 1

    pass_rate = passed / n_trials * 100
    print(f"  Pass rate: {pass_rate:.1f}%  (target ≥ 20% = viable)")
    print(f"  Expected months to pass FTMO at {pass_rate:.1f}% per attempt: "
          f"{100/max(pass_rate,1):.1f} (= {100/max(pass_rate,1)*2:.0f} months of 60-day windows)")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5: Correlation analysis
# ══════════════════════════════════════════════════════════════════════════════

def task5_correlation(cal: pd.DataFrame, dfs: dict):
    print("\n" + "=" * 70)
    print("TASK 5: Correlation Analysis — London S3 vs News Straddle")
    print("=" * 70)

    # Build news straddle daily P&L (EURUSD, VERY_HIGH only)
    df_eu = dfs.get("EURUSD")
    if df_eu is None:
        print("EURUSD data unavailable — skipping correlation")
        return

    setups = generate_straddle_setups(
        df_eu, cal, "EURUSD",
        filter_event_types=VERY_HIGH_EVENTS, **STRADDLE_PARAMS
    )
    trades, _, _ = simulate_news(setups, df_eu, INITIAL_BAL, RISK_PCT)

    # News straddle monthly returns
    if not trades:
        print("No news straddle trades — cannot compute correlation")
        return

    news_daily = {}
    for t in trades:
        d = t.entry_time.date()
        news_daily[d] = news_daily.get(d, 0.0) + t.pnl_dollars / INITIAL_BAL

    news_daily_s = pd.Series(news_daily).sort_index()
    news_daily_s.index = pd.to_datetime(news_daily_s.index)
    news_monthly  = news_daily_s.resample("ME").sum()

    print(f"\nNews straddle monthly stats ({len(news_monthly)} months):")
    print(f"  Mean/mo:  {news_monthly.mean()*100:.3f}%")
    print(f"  Std/mo:   {news_monthly.std()*100:.3f}%")
    print(f"  Min/mo:   {news_monthly.min()*100:.3f}%")
    print(f"  Max/mo:   {news_monthly.max()*100:.3f}%")

    # London S3: approximate as fixed monthly return with noise (no trade-level data from this session)
    if LONDON_EQUITY_PKL.exists():
        try:
            with open(LONDON_EQUITY_PKL, "rb") as fh:
                london_equity = pickle.load(fh)
            if isinstance(london_equity, pd.Series):
                london_monthly = london_equity.pct_change().dropna()
                london_monthly.index = pd.to_datetime(london_monthly.index)
                london_monthly = london_monthly.resample("ME").sum()
                common = news_monthly.index.intersection(london_monthly.index)
                if len(common) >= 3:
                    corr = news_monthly.loc[common].corr(london_monthly.loc[common])
                    print(f"\nLondon S3 vs News Straddle correlation: {corr:.3f}")
                    if corr < 0.2:
                        print("  → LOW correlation — good diversifier")
                    elif corr < 0.5:
                        print("  → MODERATE correlation")
                    else:
                        print("  → HIGH correlation — limited diversification benefit")
        except Exception as e:
            print(f"  Could not load London equity: {e}")
    else:
        print("\n  London S3 equity pkl not available — showing news straddle monthly distribution only")
        print("  London S3 is systematic session-breakout; News is event-driven spike")
        print("  → EXPECTED correlation: low (<0.2) — different triggers, different days")

    # Monthly P&L distribution
    print(f"\nNews straddle monthly P&L distribution:")
    positive = (news_monthly > 0).sum()
    negative = (news_monthly <= 0).sum()
    print(f"  Positive months: {positive}/{len(news_monthly)} ({positive/len(news_monthly)*100:.0f}%)")
    print(f"  Negative months: {negative}/{len(news_monthly)} ({negative/len(news_monthly)*100:.0f}%)")

    # Trade-day overlap with London (London trades 03:00-12:00 ET; news events at 8:30 or 14:00 ET)
    news_dates = set(t.entry_time.date() for t in trades)
    print(f"\nNews event trade days: {len(news_dates)} total")
    print(f"  Events at 08:30 ET (CPI/NFP/PPI/etc): can overlap with London session close")
    print(f"  Events at 14:00 ET (FOMC): no overlap with London session (03:00-12:00 ET)")
    fomc_trades = [t for t in trades if t.setup.event_type == "FOMC"]
    non_fomc    = [t for t in trades if t.setup.event_type != "FOMC"]
    print(f"  FOMC trades (14:00, no overlap): {len(fomc_trades)}")
    print(f"  Other trades (08:30, potential overlap): {len(non_fomc)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PROMPT 25 — News Event Straddle Strategy")
    print("=" * 70)

    # Task 0
    cal = task0_calendar()

    # Task 1
    task1_verify(cal)

    # Task 2
    t2 = task2_baseline(cal)
    dfs = t2["dfs"]

    # Task 3
    t3 = task3_walkforward(t2, cal)

    # Task 4
    task4_portfolio(t3, cal, dfs)

    # Task 5
    task5_correlation(cal, dfs)

    print("\n" + "=" * 70)
    print("PROMPT 25 COMPLETE")
    print("=" * 70)
