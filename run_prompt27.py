#!/usr/bin/env python3
"""
run_prompt27.py — NAS100 Initial Balance Breakout
==================================================
Task 0 : Data verification (done separately, results printed here)
Task 1 : Strategy code (nas100_ib_breakout.py)
Task 2 : Baseline (Jan 2022 → Apr 2025)
Task 3 : Walk-forward (6m IS / 3m OOS / 3m step, 432 combos)
Task 4 : Eval simulation (if WF passes)
Task 5 : Volume quality confirmation
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from backtesting.backtester import Backtester, BacktestResult, Trade
from strategies.nas100_ib_breakout import NAS100IbBreakout

RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
NAS_PATH     = Path("data/historical/NAS100_5m.parquet")
INITIAL_BAL  = 10_000.0
LONDON_MO    = 0.00896   # London S3 +0.896%/mo reference


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def load_nas() -> pd.DataFrame:
    df = pd.read_parquet(NAS_PATH)
    df.columns = [c.lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    df = df.set_index("datetime").sort_index()
    # Backtester._prepare_data will convert to ET; pass raw UTC
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe(trades: List[Trade], init_bal: float = INITIAL_BAL) -> float:
    if len(trades) < 3:
        return float("nan")
    rets = np.array([t.pnl_dollars / init_bal for t in trades])
    m, s = rets.mean(), rets.std(ddof=1)
    return float(m / s * np.sqrt(252)) if s > 0 else float("nan")


def _maxdd(trades: List[Trade], init_bal: float = INITIAL_BAL) -> float:
    bal, peak, mdd = init_bal, init_bal, 0.0
    for t in trades:
        bal += t.pnl_dollars
        if bal > peak: peak = bal
        dd = (peak - bal) / peak * 100
        if dd > mdd: mdd = dd
    return mdd


def _metrics(trades: List[Trade], init_bal: float = INITIAL_BAL) -> dict:
    if not trades:
        return dict(n=0, wr=0, ret=0, sharpe=float("nan"), maxdd=0,
                    t_mo=0, pf=0, avg_win=0, avg_loss=0,
                    pct_tp=0, pct_sl=0, pct_ts=0)
    wins   = [t for t in trades if t.pnl_dollars > 0]
    losses = [t for t in trades if t.pnl_dollars <= 0]
    tp_t   = [t for t in trades if t.exit_reason == "tp"]
    sl_t   = [t for t in trades if t.exit_reason == "sl"]
    ts_t   = [t for t in trades if t.exit_reason == "time_stop"]
    total_pnl = sum(t.pnl_dollars for t in trades)
    span = max((trades[-1].exit_time - trades[0].entry_time).days, 1)
    gw   = sum(t.pnl_dollars for t in wins)
    gl   = abs(sum(t.pnl_dollars for t in losses))
    return dict(
        n       = len(trades),
        wr      = len(wins) / len(trades) * 100,
        ret     = total_pnl / init_bal * 100,
        sharpe  = _sharpe(trades, init_bal),
        maxdd   = _maxdd(trades, init_bal),
        t_mo    = len(trades) / (span / 30.44),
        pf      = gw / gl if gl > 0 else float("inf"),
        avg_win = np.mean([t.pnl_dollars for t in wins])   if wins   else 0.0,
        avg_loss= np.mean([t.pnl_dollars for t in losses]) if losses else 0.0,
        pct_tp  = len(tp_t) / len(trades) * 100,
        pct_sl  = len(sl_t) / len(trades) * 100,
        pct_ts  = len(ts_t) / len(trades) * 100,
    )


def print_m(label: str, m: dict) -> None:
    if m["n"] == 0:
        print(f"  {label}: no trades")
        return
    sh  = f"{m['sharpe']:.3f}" if not np.isnan(m["sharpe"]) else "n/a"
    print(f"  {label}: T={m['n']} WR={m['wr']:.1f}% Ret={m['ret']:.2f}% "
          f"Sh={sh} MaxDD={m['maxdd']:.1f}% PF={m['pf']:.2f} "
          f"T/mo={m['t_mo']:.1f}  [TP={m['pct_tp']:.0f}% SL={m['pct_sl']:.0f}% TS={m['pct_ts']:.0f}%]")


def run_bt(df: pd.DataFrame, strat: NAS100IbBreakout,
           break_even_r: Optional[float] = None) -> BacktestResult:
    bt = Backtester(strat, df, "NAS100",
                    initial_balance=INITIAL_BAL,
                    break_even_r=break_even_r,
                    seed=42)
    return bt.run_fast()


# ══════════════════════════════════════════════════════════════════════════════
# TASK 0 & 5: Data + Volume quality
# ══════════════════════════════════════════════════════════════════════════════

def task0_data(df_raw: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TASK 0 + 5: Data Verification & Volume Quality")
    print("=" * 70)

    df = df_raw.copy()
    df.index = df.index.tz_convert("US/Eastern")
    hours   = df.index.hour
    minutes = df.index.minute
    hm      = hours * 100 + minutes

    rth = df[(hm >= 930) & (hm < 1600)]
    print(f"  Total bars      : {len(df):,}")
    print(f"  RTH bars        : {len(rth):,}")
    print(f"  RTH trading days: {rth.index.date.shape[0]} unique bars / "
          f"{np.unique(rth.index.date).shape[0]} days")
    print(f"  Date range      : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Volume non-zero : {(df['volume']>0).sum():,} / {len(df):,}")

    # Volume quality: open spike, lunch dip, afternoon recovery?
    print("\n  Volume by 30-min RTH slot (confirms exchange volume vs tick volume):")
    slots = (rth.index.hour * 60 + rth.index.minute) // 30
    by_slot = rth.groupby(slots)["volume"].mean()
    for slot, v in by_slot.items():
        h = slot * 30 // 60
        m = (slot * 30) % 60
        bar_width = int(v / by_slot.max() * 30)
        print(f"    {h:02d}:{m:02d}  {'█' * bar_width:<30s}  {v:.4f}")

    open_v  = by_slot.iloc[0]
    lunch_v = by_slot.iloc[4:7].mean() if len(by_slot) > 6 else open_v
    close_v = by_slot.iloc[-1]
    print(f"\n  Open/lunch/close ratio: {open_v:.4f} / {lunch_v:.4f} / {close_v:.4f}")
    if open_v > lunch_v * 1.1 and close_v > lunch_v * 1.05:
        print("  → REAL exchange-style volume (open spike + lunch dip + close pickup)")
    else:
        print("  → CAUTION: volume profile may be synthetic tick volume")

    # Gap check: any missing RTH bars?
    rth_sorted  = rth.sort_index()
    diffs       = rth_sorted.index.to_series().diff().dropna()
    gap_mask    = diffs > pd.Timedelta("5min")
    gaps        = diffs[gap_mask]
    print(f"\n  RTH bar gaps > 5min: {len(gaps)}")
    if len(gaps):
        print(f"  Largest gap: {gaps.max()}  at {gaps.idxmax()}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2: Baseline
# ══════════════════════════════════════════════════════════════════════════════

def _ib_stats(df: pd.DataFrame, strat: NAS100IbBreakout) -> dict:
    """Count qualifying days, all days, avg IB widths."""
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert("US/Eastern")
    hm    = df_et.index.hour * 100 + df_et.index.minute
    dates = df_et.index.date

    daily_highs: dict = {}
    daily_lows:  dict = {}
    ib_widths_qualify = []
    ib_widths_all     = []
    days_qualify = 0
    days_total   = 0
    vols   = df_et["volume"].to_numpy(float)
    highs  = df_et["high"].to_numpy(float)
    lows   = df_et["low"].to_numpy(float)
    closes = df_et["close"].to_numpy(float)
    hm_arr = hm.to_numpy() if hasattr(hm, "to_numpy") else np.array(hm)
    dates_arr = np.array(dates)

    # Build daily ranges
    for i in range(len(df_et)):
        d  = dates_arr[i]
        h  = hm_arr[i]
        if h >= 930 and h < 1600:
            daily_highs[d] = max(daily_highs.get(d, -np.inf), highs[i])
            daily_lows[d]  = min(daily_lows.get(d,  np.inf), lows[i])

    sorted_days = sorted(daily_highs.keys())
    for k, d in enumerate(sorted_days):
        if d not in daily_lows:
            continue
        days_total += 1
        # IB: bars 9:30-10:25
        day_bars = df_et[dates_arr == d]
        day_hm   = day_bars.index.hour * 100 + day_bars.index.minute
        ib_bars  = day_bars[(day_hm >= 930) & (day_hm < 1030)]
        if len(ib_bars) == 0:
            continue
        ib_h = ib_bars["high"].max()
        ib_l = ib_bars["low"].min()
        ib_w = ib_h - ib_l
        ib_widths_all.append(ib_w)

        # ADR
        prior = sorted_days[max(0, k - strat.adr_lookback_days): k]
        if len(prior) < 2:
            continue
        adr = np.mean([daily_highs[dd] - daily_lows[dd]
                       for dd in prior if dd in daily_lows])
        if adr <= 0:
            continue
        if ib_w <= strat.ib_adr_ratio * adr:
            days_qualify += 1
            ib_widths_qualify.append(ib_w)

    return dict(
        days_total   = days_total,
        days_qualify = days_qualify,
        qualify_pct  = days_qualify / max(days_total, 1) * 100,
        avg_ib_all   = np.mean(ib_widths_all) if ib_widths_all else 0,
        avg_ib_qual  = np.mean(ib_widths_qualify) if ib_widths_qualify else 0,
    )


def task2_baseline(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("TASK 2: Baseline Backtest (Jan 2022 → Apr 2025)")
    print("=" * 70)

    strat_default = NAS100IbBreakout()   # ib_adr_ratio=0.40 default

    # IB qualification stats
    stats = _ib_stats(df, strat_default)
    print(f"\n  Trading days     : {stats['days_total']}")
    print(f"  Narrow IB days   : {stats['days_qualify']} ({stats['qualify_pct']:.1f}%)")
    print(f"  Avg IB width all : {stats['avg_ib_all']:.1f} pts")
    print(f"  Avg IB width qual: {stats['avg_ib_qual']:.1f} pts")
    print(f"  NOTE: NAS100 median IB/ADR = 0.54; thresholds <0.25 leave <30 trades in 3yrs")

    # ── Default params ─────────────────────────────────────────────────────
    print(f"\n--- Default params (ib_adr=0.40, buf=5, rr=3.0, risk=0.5%, vol=1.0) ---")
    res = run_bt(df, strat_default)
    m   = _metrics(res.trades)
    print_m("Default", m)

    # ── Volume filter impact ───────────────────────────────────────────────
    print(f"\n--- Volume filter impact (ib_adr=0.25, rr=3.0) ---")
    for vm in [1.0, 1.3, 1.5]:
        s = NAS100IbBreakout(vol_sma_mult=vm)
        r = run_bt(df, s)
        mi = _metrics(r.trades)
        print_m(f"vol_mult={vm:.1f}", mi)

    # ── IB ratio sensitivity (trades/month) ───────────────────────────────
    print(f"\n--- IB ratio sensitivity ---")
    for ratio in [0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
        s = NAS100IbBreakout(ib_adr_ratio=ratio)
        r = run_bt(df, s)
        mi = _metrics(r.trades)
        ib_s = _ib_stats(df, s)
        print_m(f"ib_adr={ratio:.2f} qualify={ib_s['qualify_pct']:.0f}%", mi)

    # ── RR sensitivity ────────────────────────────────────────────────────
    print(f"\n--- RR ratio sensitivity (ib_adr=0.25, vol=1.0) ---")
    for rr in [2.0, 2.5, 3.0, 3.5]:
        s = NAS100IbBreakout(rr_ratio=rr)
        r = run_bt(df, s)
        mi = _metrics(r.trades)
        print_m(f"rr={rr:.1f}", mi)

    # ── Buffer sensitivity ────────────────────────────────────────────────
    print(f"\n--- Buffer sensitivity (ib_adr=0.25, rr=3.0) ---")
    for buf in [3, 5, 10, 15]:
        s = NAS100IbBreakout(buffer_points=buf)
        r = run_bt(df, s)
        mi = _metrics(r.trades)
        print_m(f"buf={buf}pts", mi)

    # ── WR on narrow IB vs ALL days (no IB filter) ───────────────────────
    print(f"\n--- Narrow IB filter: does it improve WR? ---")
    s_nofilter = NAS100IbBreakout(ib_adr_ratio=99.0, vol_sma_mult=1.0,
                                   buffer_points=5, rr_ratio=3.0)
    r_nofilter = run_bt(df, s_nofilter)
    m_nofilter = _metrics(r_nofilter.trades)
    print_m("No IB filter (ratio=99)", m_nofilter)
    print_m("Narrow IB   (ratio=0.40)", m)

    # Best param sweep: ib_adr=0.25, buf=5, rr=3.0, vol=1.0
    best_sharpe = m["sharpe"]
    best_params = dict(ib_adr_ratio=0.25, buffer_points=5, rr_ratio=3.0,
                       vol_sma_mult=1.0, risk_per_trade_pct=0.5)
    print(f"\nBest baseline Sharpe: {best_sharpe:.3f}")
    print(f"WF gate (> -1.0): {'PROCEED' if best_sharpe > -1.0 else 'SKIP'}")

    return dict(
        best_sharpe  = best_sharpe,
        proceed_wf   = best_sharpe > -1.0,
        default_m    = m,
        nofilter_m   = m_nofilter,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3: Walk-forward
# ══════════════════════════════════════════════════════════════════════════════

# 4×3×3×3×2×2 = 432 combos
WF_GRID = list(itertools.product(
    [0.35, 0.40, 0.45, 0.50],   # ib_adr_ratio (calibrated to NAS100)
    [3, 5, 10],                  # buffer_points
    [2.0, 2.5, 3.0],             # rr_ratio
    [0.5, 0.75, 1.0],            # risk_per_trade_pct
    [1.0, 1.3],                  # vol_sma_mult
    [60, 80],                    # max_sl_points
))
assert len(WF_GRID) == 432
MIN_IS_TRADES = 8


def _wf_windows(start: str, end: str, is_mo: int = 6, oos_mo: int = 3) -> list:
    s = pd.Timestamp(start, tz="US/Eastern")
    e = pd.Timestamp(end,   tz="US/Eastern")
    wins, cur = [], s
    while True:
        is_e  = cur  + pd.DateOffset(months=is_mo)
        oos_e = is_e + pd.DateOffset(months=oos_mo)
        if oos_e > e: break
        wins.append((cur, is_e, is_e, oos_e))
        cur = cur + pd.DateOffset(months=oos_mo)
    return wins


def task3_walkforward(df: pd.DataFrame, t2: dict) -> Optional[dict]:
    print("\n" + "=" * 70)
    print("TASK 3: Walk-Forward (6m IS / 3m OOS / 3m step, 432 combos)")
    print("=" * 70)

    if not t2["proceed_wf"]:
        print("SKIP: baseline Sharpe < -1.0")
        return None

    windows = _wf_windows("2022-01-01", "2025-04-10")
    print(f"Windows: {len(windows)}")
    for w in windows:
        print(f"  IS {w[0].date()} – {w[1].date()}  "
              f"OOS {w[2].date()} – {w[3].date()}")

    all_oos_trades: List[Trade] = []
    deg_scores: List[float] = []

    for w_i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        df_is  = df[(df.index >= is_s)  & (df.index < is_e)]
        df_oos = df[(df.index >= oos_s) & (df.index < oos_e)]

        best_sh = -np.inf
        best_p  = WF_GRID[0]

        for params in WF_GRID:
            ratio, buf, rr, rp, vm, msl = params
            s = NAS100IbBreakout(
                ib_adr_ratio=ratio, buffer_points=buf, rr_ratio=rr,
                risk_per_trade_pct=rp, vol_sma_mult=vm, max_sl_points=msl,
            )
            r = run_bt(df_is, s)
            if len(r.trades) < MIN_IS_TRADES:
                continue
            sh = _sharpe(r.trades)
            if not np.isnan(sh) and sh > best_sh:
                best_sh = sh
                best_p  = params

        ratio, buf, rr, rp, vm, msl = best_p
        s_oos = NAS100IbBreakout(
            ib_adr_ratio=ratio, buffer_points=buf, rr_ratio=rr,
            risk_per_trade_pct=rp, vol_sma_mult=vm, max_sl_points=msl,
        )
        r_oos  = run_bt(df_oos, s_oos)
        sh_oos = _sharpe(r_oos.trades) if len(r_oos.trades) >= 3 else float("nan")
        deg    = sh_oos / best_sh if (not np.isnan(sh_oos) and best_sh > 0) else float("nan")
        all_oos_trades.extend(r_oos.trades)

        sh_s   = f"{sh_oos:.3f}" if not np.isnan(sh_oos) else "n/a"
        deg_s  = f"{deg:.2f}"    if not np.isnan(deg)    else "n/a"
        print(f"\n  Window {w_i+1}: IS_Sh={best_sh:.3f} | OOS_Sh={sh_s} | "
              f"deg={deg_s} | n_oos={len(r_oos.trades)}")
        print(f"    Best: ratio={ratio} buf={buf} rr={rr} risk={rp}% "
              f"vol={vm} maxsl={msl}")

        if not np.isnan(deg):
            deg_scores.append(deg)

    avg_deg = float(np.mean(deg_scores)) if deg_scores else float("nan")
    m_oos   = _metrics(all_oos_trades)
    pct_pos = (sum(1 for d in deg_scores if d > 0) / len(deg_scores) * 100
               if deg_scores else 0.0)

    print(f"\nWF SUMMARY:")
    ad_s = f"{avg_deg:.3f}" if not np.isnan(avg_deg) else "n/a"
    print(f"  avg_degradation={ad_s}  pct_positive_OOS={pct_pos:.0f}%")
    print_m("  Combined OOS", m_oos)

    pass_wf      = not np.isnan(avg_deg) and avg_deg > 0.2 and pct_pos >= 50
    proceed_eval = (not np.isnan(m_oos.get("sharpe", float("nan")))
                    and m_oos.get("sharpe", -99) > 0.2)

    print(f"\nWF pass (avg_deg>0.2 AND ≥50% OOS positive): {'YES' if pass_wf else 'NO'}")
    print(f"Eval gate (OOS Sharpe > 0.2): {'YES' if proceed_eval else 'NO'}")

    return dict(
        oos_trades   = all_oos_trades,
        avg_deg      = avg_deg,
        pct_pos      = pct_pos,
        oos_metrics  = m_oos,
        proceed_eval = proceed_eval,
        pass_wf      = pass_wf,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4: Eval simulation
# ══════════════════════════════════════════════════════════════════════════════

def task4_eval(df: pd.DataFrame, t3: Optional[dict]) -> None:
    print("\n" + "=" * 70)
    print("TASK 4: FTMO Eval Simulation")
    print("=" * 70)

    london_cagr = (1 + LONDON_MO) ** 12 - 1
    months_l    = int(np.ceil(np.log(1.10) / np.log(1.0 + LONDON_MO)))
    print(f"\nLondon S3 reference: +{LONDON_MO*100:.3f}%/mo  CAGR={london_cagr*100:.1f}%  "
          f"Months to +10%: {months_l}")

    # Risk sensitivity even if WF skipped (for reference)
    print(f"\n--- Risk Sensitivity (default params) ---")
    for rp in [0.5, 0.75, 1.0, 1.5]:
        s = NAS100IbBreakout(risk_per_trade_pct=rp)
        r = run_bt(df, s)
        m = _metrics(r.trades)
        print_m(f"risk={rp:.2f}%", m)

    if t3 is None or not t3.get("proceed_eval", False):
        print("\nFTMO eval sim SKIPPED (WF OOS Sharpe < 0.2)")
        return

    oos_trades = t3["oos_trades"]
    oos_m      = t3["oos_metrics"]
    pnls       = np.array([t.pnl_dollars / INITIAL_BAL for t in oos_trades])

    span = ((oos_trades[-1].exit_time - oos_trades[0].entry_time).days / 30.44
            if len(oos_trades) > 1 else 1.0)
    nas_monthly = sum(t.pnl_dollars for t in oos_trades) / INITIAL_BAL / max(span, 1.0)
    combined_mo = LONDON_MO + nas_monthly
    combined_cagr = (1 + combined_mo) ** 12 - 1
    months_c = int(np.ceil(np.log(1.10) / np.log(1.0 + combined_mo)))

    print(f"\nNAS100 OOS monthly return: {nas_monthly*100:.3f}%")
    print(f"Combined portfolio: {combined_mo*100:.3f}%/mo  "
          f"CAGR={combined_cagr*100:.1f}%  Months to +10%: {months_c}")

    # Correlation (monthly)
    print(f"\n  Expected London/NAS correlation: LOW")
    print(f"  London: 03:00-12:00 ET on FX pairs")
    print(f"  NAS100: 09:30-16:00 ET on US equities")
    print(f"  Overlap window 09:30-12:00 is different asset class → near-zero correlation")

    # FTMO 15-day pass rate sim
    print(f"\n--- FTMO 15-day Challenge Eval (1000 trials) ---")
    rng = np.random.default_rng(42)
    london_daily = LONDON_MO / 22.0
    nas_t_per_day = oos_m["t_mo"] / 22.0

    for rp_scale in [1.0, 1.5, 2.0, 2.5]:
        passed = 0
        for _ in range(1000):
            bal, peak, day_b = INITIAL_BAL, INITIAL_BAL, INITIAL_BAL
            halt = False
            for _ in range(15):
                if halt: break
                ld = rng.normal(london_daily, abs(london_daily) * 3) * INITIAL_BAL
                nd = 0.0
                if rng.random() < nas_t_per_day and len(pnls) > 1:
                    nd = float(rng.choice(pnls)) * rp_scale * INITIAL_BAL
                bal += ld + nd
                if (bal - day_b) / day_b < -0.04: halt = True; break
                if bal > peak: peak = bal
                if (peak - bal) / INITIAL_BAL > 0.09: halt = True; break
                day_b = bal
            if not halt and (bal - INITIAL_BAL) / INITIAL_BAL >= 0.10:
                passed += 1
        pr = passed / 10
        print(f"  risk_scale={rp_scale:.1f}×: 15-day pass rate={pr:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PROMPT 27 — NAS100 Initial Balance Breakout")
    print("=" * 70)

    df = load_nas()
    print(f"  Loaded {len(df):,} M5 bars")

    task0_data(df)
    t2 = task2_baseline(df)
    t3 = task3_walkforward(df, t2)
    task4_eval(df, t3)

    print("\n" + "=" * 70)
    print("PROMPT 27 COMPLETE")
    print("=" * 70)
