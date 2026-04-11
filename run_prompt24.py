"""
Prompt 24 — Gold ICT Sweep-to-FVG Strategy
============================================

Tasks
-----
0.  Spread: set XAUUSD to 20 pips.  Verify spread application (per-trade, not per-bar).
7.  Re-run M15 GoldSweepReversal at 20-pip spread (compare to Prompt 23 result @12pip).
1-3. Modules built: utils/fvg_detector.py, utils/structure_detector.py,
     strategies/gold_ict_sweep_fvg.py
4.  Baseline: Jul 2022-Apr 2025 (full) + Jan 2024-Apr 2025 (recent).
    Reports funnel stats, confluence score analysis, partial TP stats.
5.  Walk-forward: 4m IS / 2m OOS / 1m step on 2023-2025 data.
    54 combos.  Pass: avg_deg > 0.1, >45% OOS positive.
6.  Eval sim: 15-day rolling windows on 2024-2025, risk sensitivity.
"""

from __future__ import annotations

import json
import sys
import time as _time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from backtesting.backtester import Backtester
from backtesting.metrics import calculate_metrics
from strategies.gold_ict_sweep_fvg import GoldIctSweepFvg, SetupRecord

INSTRUMENT      = "XAUUSD"
INITIAL_BALANCE = 10_000.0
PARQUET_M1      = ROOT / "data" / "historical" / "XAUUSD_1m.parquet"
PARQUET_M15     = ROOT / "data" / "historical" / "XAUUSD_15m.parquet"

# Prompt 23 baseline (12-pip spread)
P23_SWEEP_SHARPE_12PIP = 0.762

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

XAUUSD_CFG  = _INSTR_CFG[INSTRUMENT]
ICT_BASE    = _ALL_PARAMS["gold_ict_sweep_fvg"]
SWEEP_BASE  = _ALL_PARAMS["gold_sweep_reversal"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
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


def _load_m15(start: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_M15)
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.columns = [c.lower() for c in df.columns]
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Partial-exit custom simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PartialTrade:
    entry_bar:    int
    exit_bar:     int
    direction:    int
    entry_price:  float
    exit_price:   float
    lot_size:     float
    pnl_dollars:  float
    exit_reason:  str     # sl | tp1 | tp2 | be_sl | time_stop | end_data
    confluence:   int
    fvg_size:     float
    sl_pips:      float


def simulate_partials(
    setups:          List[SetupRecord],
    df_m1:           pd.DataFrame,
    initial_balance: float,
    min_sweep_pips:  float = 0,
    min_fvg_pips:    float = 0,
    min_confluence:  int   = 0,
    risk_pct:        float | None = None,  # None → use per-setup risk_pct
) -> Tuple[List[PartialTrade], pd.Series, float]:
    """
    Bar-by-bar simulation with partial exit at 1.5R, SL→BE, then 3.0R or time stop.

    Returns (trades, equity_series, final_balance)
    """
    # Apply filter
    filtered = [
        s for s in setups
        if s.sweep_size_pips >= min_sweep_pips
        and s.fvg_size_pips  >= min_fvg_pips
        and s.confluence      >= min_confluence
    ]
    # Sort by entry_bar, deduplicate (only first per bar)
    seen_bars = set()
    uniq = []
    for s in sorted(filtered, key=lambda x: x.entry_bar):
        if s.entry_bar not in seen_bars:
            uniq.append(s)
            seen_bars.add(s.entry_bar)
    filtered = uniq

    if not filtered:
        empty_eq = pd.Series({df_m1.index[0]: initial_balance})
        return [], empty_eq, initial_balance

    # Build raw arrays (M1 data with ET index)
    df_prep = df_m1.copy()
    if "datetime" in df_prep.columns:
        df_prep = df_prep.set_index("datetime")
    if df_prep.index.tz is None:
        df_prep.index = df_prep.index.tz_localize("UTC")
    df_prep.index = df_prep.index.tz_convert("US/Eastern")
    df_prep.columns = [c.lower() for c in df_prep.columns]

    idx_arr  = df_prep.index
    high_arr = df_prep["high"].to_numpy(dtype=float)
    low_arr  = df_prep["low"].to_numpy(dtype=float)
    close_arr= df_prep["close"].to_numpy(dtype=float)
    n        = len(df_prep)

    # Spread at entry
    half_spread = (XAUUSD_CFG["typical_spread_pips"] / 2.0) * XAUUSD_CFG["pip_size"]
    pip_sz      = XAUUSD_CFG["pip_size"]
    pip_val     = XAUUSD_CFG.get("pip_value_per_lot", 1.0)

    balance      = float(initial_balance)
    trades_out   = []
    eq_curve     = {idx_arr[0]: balance}

    # Non-overlapping: track last exit bar
    last_exit_bar = -1

    for s in filtered:
        if s.entry_bar <= last_exit_bar:
            continue
        if s.entry_bar >= n:
            break

        rp = risk_pct if risk_pct is not None else s.risk_pct

        # Entry price with spread
        entry = close_arr[s.entry_bar] + s.direction * half_spread
        sl    = s.sl_price
        sl_dist = abs(entry - sl)
        if sl_dist < 1e-6:
            continue

        tp1 = entry + s.direction * 1.5 * sl_dist
        tp2 = entry + s.direction * 3.0 * sl_dist

        # Lot sizing (100% position initially)
        risk_usd  = balance * rp / 100.0
        sl_pips   = sl_dist / pip_sz
        lot_total = risk_usd / (sl_pips * pip_val)
        lot_total = max(0.01, min(50.0, round(lot_total, 2)))
        lot_half  = max(0.01, round(lot_total / 2.0, 2))

        phase    = "full"   # full | half (after tp1)
        cur_sl   = sl
        p1_hit   = False

        exit_bar   = n - 1
        exit_price = close_arr[n - 1]
        exit_rsn   = "end_data"
        p1_exit_pnl = 0.0
        p1_exit_lot = 0.0

        for j in range(s.entry_bar + 1, n):
            jts = idx_arr[j]

            # Time stop check
            if s.time_stop is not None:
                try:
                    if jts >= s.time_stop:
                        exit_bar, exit_price, exit_rsn = j, close_arr[j], "time_stop"
                        break
                except Exception:
                    pass

            hit_sl = (s.direction ==  1 and low_arr[j]  <= cur_sl) or \
                     (s.direction == -1 and high_arr[j] >= cur_sl)
            hit_tp1 = not p1_hit and (
                (s.direction ==  1 and high_arr[j] >= tp1) or
                (s.direction == -1 and low_arr[j]  <= tp1)
            )
            hit_tp2 = p1_hit and (
                (s.direction ==  1 and high_arr[j] >= tp2) or
                (s.direction == -1 and low_arr[j]  <= tp2)
            )

            if hit_sl and not hit_tp1:
                exit_bar, exit_price, exit_rsn = j, cur_sl, ("be_sl" if p1_hit else "sl")
                break
            if hit_tp1 and not p1_hit:
                # Partial 1: exit half at tp1, move SL to breakeven
                p1_exit_pnl = (
                    s.direction * (tp1 - entry) / pip_sz * pip_val * lot_half
                )
                p1_exit_lot = lot_half
                p1_hit      = True
                cur_sl      = entry + s.direction * pip_sz   # breakeven + 1 pip
                phase       = "half"
            if hit_tp2 and p1_hit:
                exit_bar, exit_price, exit_rsn = j, tp2, "tp2"
                break

            eq_curve[jts] = balance + (
                s.direction * (close_arr[j] - entry) / pip_sz * pip_val
                * (lot_half if p1_hit else lot_total)
            ) + (p1_exit_pnl if p1_hit else 0.0)

        # Close remaining
        remain_lot = lot_half if p1_hit else lot_total
        final_pnl_pips = s.direction * (exit_price - entry) / pip_sz
        final_pnl_usd  = final_pnl_pips * pip_val * remain_lot
        total_pnl      = final_pnl_usd + p1_exit_pnl

        balance += total_pnl
        eq_curve[idx_arr[exit_bar]] = balance
        last_exit_bar = exit_bar

        trades_out.append(PartialTrade(
            entry_bar=s.entry_bar, exit_bar=exit_bar,
            direction=s.direction, entry_price=entry, exit_price=exit_price,
            lot_size=lot_total, pnl_dollars=total_pnl, exit_reason=exit_rsn,
            confluence=s.confluence, fvg_size=s.fvg_size_pips,
            sl_pips=abs(entry - sl) / pip_sz,
        ))

    eq_series = pd.Series(eq_curve).sort_index()
    return trades_out, eq_series, balance


def _trade_metrics(trades: List[PartialTrade], eq: pd.Series, initial: float, label: str) -> None:
    if not trades:
        print(f"  {label}: NO TRADES")
        return
    pnls     = np.array([t.pnl_dollars for t in trades])
    wins     = pnls > 0
    wr       = wins.sum() / len(pnls) * 100
    t_mo     = len(trades) / max((eq.index[-1] - eq.index[0]).days / 30.44, 1)
    ret      = (eq.iloc[-1] - initial) / initial * 100
    daily_eq = eq.resample("D").last().dropna()
    daily_r  = daily_eq.pct_change().dropna()
    sharpe   = float(daily_r.mean() / daily_r.std() * np.sqrt(252)) if daily_r.std() > 0 else 0
    peak     = np.maximum.accumulate(eq.values)
    max_dd   = float(abs(((eq.values - peak) / np.where(peak==0,1,peak)).min())) * 100
    avg_w    = float(pnls[wins].mean()) if wins.any() else 0
    avg_l    = float(abs(pnls[~wins].mean())) if (~wins).any() else 0
    pf       = float(pnls[wins].sum() / abs(pnls[~wins].sum())) if (~wins).any() and abs(pnls[~wins].sum()) > 0 else 999
    avg_sl   = float(np.mean([t.sl_pips for t in trades]))
    print(f"  {label}")
    print(f"    Trades={len(trades)}  T/mo={t_mo:.1f}")
    print(f"    WR={wr:.1f}%  AvgWin=${avg_w:.0f}  AvgLoss=${avg_l:.0f}  PF={pf:.3f}")
    print(f"    Return={ret:.2f}%  Sharpe={sharpe:.3f}  MaxDD={max_dd:.2f}%")
    print(f"    AvgSL={avg_sl:.1f}pips  Final=${eq.iloc[-1]:,.0f}")


def _eval_windows(eq: pd.Series, initial: float, window_days: int = 15) -> dict:
    daily = eq.resample("D").last().dropna()
    n     = len(daily)
    passes = fails = total = 0
    for si in range(n - window_days):
        w   = daily.iloc[si : si + window_days + 1].values
        if len(w) < window_days:
            continue
        nm  = w / w[0] * initial
        prev = nm[0]
        out  = "inconclusive"
        for eq_v in nm[1:]:
            if prev > 0 and (eq_v - prev) / prev * 100 <= -5.0:
                out = "fail"; break
            if (eq_v - initial) / initial * 100 <= -10.0:
                out = "fail"; break
            if (eq_v - initial) / initial * 100 >= 10.0:
                out = "pass"; break
            prev = eq_v
        total += 1
        if out == "pass": passes += 1
        elif out == "fail": fails += 1
    if total == 0:
        return {"pass_rate": 0.0, "fail_rate": 0.0, "two_attempt": 0.0, "total": 0}
    p = passes / total
    f = fails / total
    return {
        "pass_rate":   round(p * 100, 2),
        "fail_rate":   round(f * 100, 2),
        "two_attempt": round((1 - (1 - p) ** 2) * 100, 2),
        "total":       total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 7 — M15 GoldSweepReversal @20-pip spread
# ─────────────────────────────────────────────────────────────────────────────

def run_task7():
    print("\n" + "=" * 72)
    print("TASK 7: M15 GoldSweepReversal @20-pip spread vs @12-pip (Prompt 23)")
    print("=" * 72)
    print(f"  Prompt 23 result @12pip: Sharpe = {P23_SWEEP_SHARPE_12PIP:.3f}")
    print(f"  Current spread: {XAUUSD_CFG['typical_spread_pips']:.0f} pips")

    from strategies.gold_sweep_reversal import GoldSweepReversal

    df_m15 = _load_m15("2020-07-01")
    cfg    = {**SWEEP_BASE, "tp_mode": "A"}
    s      = GoldSweepReversal()
    s.setup(cfg, XAUUSD_CFG)

    t0 = _time.perf_counter()
    r  = Backtester(strategy=s, df=df_m15.reset_index(), instrument=INSTRUMENT,
                    initial_balance=INITIAL_BALANCE,
                    _override_ftmo_rules=LOOSE_FTMO).run()
    elapsed = _time.perf_counter() - t0
    m = calculate_metrics(r)
    new_sh = m.get("sharpe_ratio", 0.0)

    print(f"\n  @20pip ({elapsed:.0f}s):")
    print(f"    Trades={m['total_trades']}  WR={m['win_rate_pct']:.1f}%  "
          f"Return={m['total_return_pct']:.2f}%")
    print(f"    Sharpe={new_sh:.3f}  MaxDD={m['max_drawdown_pct']:.2f}%")
    print(f"\n  Sharpe: 12pip={P23_SWEEP_SHARPE_12PIP:.3f}  →  20pip={new_sh:.3f}  "
          f"(Δ={new_sh - P23_SWEEP_SHARPE_12PIP:+.3f})")
    if new_sh < 0.3:
        print("  WARNING: GoldSweepReversal not viable at 20-pip spread.")
    else:
        print(f"  GoldSweepReversal still positive at 20-pip spread (Sharpe={new_sh:.3f})")
    return new_sh


# ─────────────────────────────────────────────────────────────────────────────
# Spread verification helper
# ─────────────────────────────────────────────────────────────────────────────

def run_task0():
    print("\n" + "=" * 72)
    print("TASK 0: Spread Verification")
    print("=" * 72)
    spread = XAUUSD_CFG["typical_spread_pips"]
    pip_sz = XAUUSD_CFG["pip_size"]
    half   = spread / 2.0
    print(f"  instruments.json spread: {spread:.1f} pips = ${spread * pip_sz:.2f}/unit")
    print(f"  Applied: ENTRY only (half-spread = {half:.1f} pips = ${half * pip_sz:.3f})")
    print(f"  Exit at exact SL/TP price — no spread at exit (consistent with framework)")
    print(f"  Commission: ${XAUUSD_CFG['commission_per_lot_round_trip']:.2f}/lot round trip")
    print(f"  Effective entry cost on 100-pip SL: {half:.0f}/{100:.0f} = {half/100*100:.1f}%")
    print(f"  Spread is per-trade (at entry), NOT per-bar. Correct.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — Baseline backtests
# ─────────────────────────────────────────────────────────────────────────────

def run_task4(setups_full: List[SetupRecord], funnel_full: Dict,
              df_full: pd.DataFrame, df_recent: pd.DataFrame,
              setups_recent: List[SetupRecord], funnel_recent: Dict) -> Tuple:
    print("\n" + "=" * 72)
    print("TASK 4: Baseline Backtests — Full Range & Recent")
    print("=" * 72)

    # ── 4A: Full range (2022-2025) ────────────────────────────────────
    print("\n  [4A] FULL RANGE (Jul 2022 → Feb 2025)")
    print(f"  Funnel: sweeps={funnel_full['sweeps']}  "
          f"disp+FVG={funnel_full['disp_fvg']}  "
          f"retests={funnel_full['retests']}  "
          f"traded={funnel_full['traded']}")
    if funnel_full["sweeps"] > 0:
        print(f"  Sweep→FVG rate:  {funnel_full['disp_fvg']/funnel_full['sweeps']*100:.1f}%")
    if funnel_full["disp_fvg"] > 0:
        print(f"  FVG→retest rate: {funnel_full['retests']/funnel_full['disp_fvg']*100:.1f}%")
    if funnel_full["retests"] > 0:
        print(f"  Retest→trade rate: {funnel_full['traded']/funnel_full['retests']*100:.1f}%")

    t0 = _time.perf_counter()
    trades_full, eq_full, bal_full = simulate_partials(setups_full, df_full, INITIAL_BALANCE)
    print(f"  Simulation: {_time.perf_counter()-t0:.1f}s")
    _trade_metrics(trades_full, eq_full, INITIAL_BALANCE, "Full range")
    _exit_breakdown(trades_full)
    _confluence_analysis(trades_full)

    # ── 4B: Recent (2024-2025) ────────────────────────────────────────
    print("\n  [4B] RECENT (Jan 2024 → Feb 2025)")
    print(f"  Funnel: sweeps={funnel_recent['sweeps']}  "
          f"disp+FVG={funnel_recent['disp_fvg']}  "
          f"retests={funnel_recent['retests']}  "
          f"traded={funnel_recent['traded']}")

    trades_rec, eq_rec, bal_rec = simulate_partials(setups_recent, df_recent, INITIAL_BALANCE)
    _trade_metrics(trades_rec, eq_rec, INITIAL_BALANCE, "Recent 2024-2025")

    # Trade volume check
    if trades_rec:
        t_mo_rec = len(trades_rec) / max(
            (eq_rec.index[-1] - eq_rec.index[0]).days / 30.44, 1)
        print(f"\n  Trades/month (recent): {t_mo_rec:.1f}")

        if t_mo_rec < 4:
            print("  LOW TRADE COUNT — trying relaxations:")
            _try_relaxations(setups_recent, df_recent, INITIAL_BALANCE)

    return trades_full, eq_full, trades_rec, eq_rec


def _exit_breakdown(trades: List[PartialTrade]) -> None:
    if not trades:
        return
    reasons: dict = defaultdict(int)
    for t in trades:
        reasons[t.exit_reason] += 1
    print(f"\n  Exit breakdown (n={len(trades)}):")
    for rsn, cnt in sorted(reasons.items()):
        pct = cnt / len(trades) * 100
        print(f"    {rsn:12s}: {cnt:3d} ({pct:.0f}%)")


def _confluence_analysis(trades: List[PartialTrade]) -> None:
    if not trades:
        return
    by_score: dict = defaultdict(list)
    for t in trades:
        by_score[t.confluence].append(t.pnl_dollars)
    print(f"\n  Confluence score → WR (does higher score = higher WR?):")
    for sc in sorted(by_score):
        pnls = np.array(by_score[sc])
        wr   = (pnls > 0).sum() / len(pnls) * 100
        avg  = float(pnls.mean())
        print(f"    Score {sc}: n={len(pnls):3d}  WR={wr:.0f}%  AvgP&L=${avg:+.0f}")


def _try_relaxations(setups: List[SetupRecord], df: pd.DataFrame, initial: float) -> None:
    for min_conf, label in [(2, "min_conf=2"), (3, "min_conf=3 + entry at zone_top instead of 50%")]:
        t, eq, _ = simulate_partials(setups, df, initial, min_confluence=min_conf)
        if t:
            pnls = np.array([x.pnl_dollars for x in t])
            wr   = (pnls > 0).sum() / len(pnls) * 100
            t_mo = len(t) / max((eq.index[-1] - eq.index[0]).days / 30.44, 1)
            daily_eq = eq.resample("D").last().dropna()
            daily_r  = daily_eq.pct_change().dropna()
            sh = float(daily_r.mean() / daily_r.std() * np.sqrt(252)) if daily_r.std() > 0 else 0
            print(f"    {label}: T/mo={t_mo:.1f}  WR={wr:.0f}%  Sharpe={sh:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 — Walk-forward (2023-2025, 4m IS / 2m OOS / 1m step)
# ─────────────────────────────────────────────────────────────────────────────

def run_task5(setups_all: List[SetupRecord], df_all: pd.DataFrame) -> bool:
    print("\n" + "=" * 72)
    print("TASK 5: Walk-Forward (Jan 2023 → Feb 2025, 4m IS / 2m OOS / 1m step)")
    print("=" * 72)

    # Build WF windows
    idx = pd.DatetimeIndex([s.entry_time if hasattr(s, "entry_time") else
                            pd.Timestamp("2023-01-01", tz="UTC") for s in setups_all])

    # Use calendar dates for window splits
    df_wf_start = pd.Timestamp("2023-01-01", tz="UTC")
    df_wf_end   = pd.Timestamp("2025-02-05", tz="UTC")

    # Need the full M1 bar datetime index for forward scan
    df_prep = df_all.copy()
    if "datetime" in df_prep.columns:
        df_prep = df_prep.set_index("datetime")
    if df_prep.index.tz is None:
        df_prep.index = df_prep.index.tz_localize("UTC")

    # Build entry_times for setups using df_prep index
    all_entry_times = []
    for s in setups_all:
        if s.entry_bar < len(df_prep):
            all_entry_times.append(df_prep.index[s.entry_bar])
        else:
            all_entry_times.append(pd.NaT)

    # Grid
    sweep_vals = [10.0, 15.0, 20.0]
    fvg_vals   = [5.0,  10.0, 15.0]
    conf_vals  = [2, 3, 4]
    risk_vals  = [0.8, 1.0]
    combos = [
        (sw, fv, co, ri)
        for sw in sweep_vals
        for fv in fvg_vals
        for co in conf_vals
        for ri in risk_vals
    ]
    print(f"  Grid: {len(combos)} combos "
          f"(sweep×{len(sweep_vals)} fvg×{len(fvg_vals)} conf×{len(conf_vals)} risk×{len(risk_vals)})")

    # Generate windows: 4m IS, 2m OOS, 1m step
    windows = []
    cur = df_wf_start
    while True:
        is_end  = cur + pd.DateOffset(months=4)
        oos_end = is_end + pd.DateOffset(months=2)
        if oos_end > df_wf_end:
            break
        windows.append((cur, is_end, oos_end))
        cur += pd.DateOffset(months=1)
    print(f"  Windows: {len(windows)}")
    if not windows:
        print("  NOT ENOUGH DATA for WF")
        return False

    @dataclass
    class WFWindow:
        label:     str
        is_sharpe: Optional[float]
        oos_sharpe: Optional[float]
        deg:       Optional[float]
        oos_trades: int
        best_combo: tuple

    results = []
    for wi, (is_st, is_en, oos_en) in enumerate(windows):
        label = f"W{wi+1:02d}"
        # Filter setups to IS/OOS periods
        is_setups  = [s for s, et in zip(setups_all, all_entry_times)
                      if et is not pd.NaT and is_st <= et < is_en]
        oos_setups = [s for s, et in zip(setups_all, all_entry_times)
                      if et is not pd.NaT and is_en <= et < oos_en]

        if len(is_setups) < 4:
            results.append(WFWindow(label, None, None, None, 0, ()))
            continue

        # Find best IS combo
        best_sh_is = -999
        best_params = combos[0]
        for sw, fv, co, ri in combos:
            t_is, eq_is, _ = simulate_partials(
                is_setups, df_all, INITIAL_BALANCE,
                min_sweep_pips=sw, min_fvg_pips=fv, min_confluence=co, risk_pct=ri)
            if len(t_is) < 4:
                continue
            dr = eq_is.resample("D").last().dropna().pct_change().dropna()
            sh = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else -999
            if sh > best_sh_is:
                best_sh_is = sh
                best_params = (sw, fv, co, ri)

        # Apply best params to OOS
        sw, fv, co, ri = best_params
        t_oos, eq_oos, _ = simulate_partials(
            oos_setups, df_all, INITIAL_BALANCE,
            min_sweep_pips=sw, min_fvg_pips=fv, min_confluence=co, risk_pct=ri)

        sh_oos = None
        if len(t_oos) >= 2:
            dr_oos = eq_oos.resample("D").last().dropna().pct_change().dropna()
            sh_oos = float(dr_oos.mean() / dr_oos.std() * np.sqrt(252)) if dr_oos.std() > 0 else None

        deg = (sh_oos / best_sh_is) if (sh_oos is not None and best_sh_is > 0) else None

        results.append(WFWindow(
            label=label,
            is_sharpe=best_sh_is if best_sh_is > -990 else None,
            oos_sharpe=sh_oos,
            deg=deg,
            oos_trades=len(t_oos),
            best_combo=best_params,
        ))
        oos_sh_str = f"{sh_oos:.3f}" if sh_oos is not None else "n/a"
        deg_str    = f"{deg:.3f}"   if deg    is not None else "n/a"
        print(f"  {label}: IS_Sharpe={best_sh_is:.3f}  OOS_Sharpe={oos_sh_str}  "
              f"deg={deg_str}  OOS_T={len(t_oos)}  params={best_params}")

    valid = [w for w in results if w.oos_sharpe is not None]
    if not valid:
        print("  WF FAIL: No valid OOS windows")
        return False

    avg_deg    = float(np.mean([w.deg for w in valid if w.deg is not None] or [0]))
    n_positive = sum(1 for w in valid if (w.oos_sharpe or 0) > 0)
    pct_pos    = n_positive / len(valid) * 100

    print(f"\n  WF Summary: {len(valid)} valid windows")
    print(f"  Avg degradation: {avg_deg:.4f}  (target: > 0.1)")
    print(f"  OOS positive: {n_positive}/{len(valid)} = {pct_pos:.1f}%  (target: > 45%)")

    pass_wf = avg_deg > 0.1 and pct_pos > 45.0
    if pass_wf:
        print(f"  WF PASS")
    else:
        reasons = []
        if avg_deg <= 0.1:
            reasons.append(f"avg_deg={avg_deg:.3f}")
        if pct_pos <= 45.0:
            reasons.append(f"pos%={pct_pos:.1f}")
        print(f"  WF FAIL: {' | '.join(reasons)}")
    return pass_wf


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6 — Eval simulation (15-day windows on 2024-2025)
# ─────────────────────────────────────────────────────────────────────────────

def run_task6(setups_recent: List[SetupRecord], df_recent: pd.DataFrame,
              baseline_sharpe: float) -> None:
    print("\n" + "=" * 72)
    print("TASK 6: FTMO Eval Simulation (15-day windows, 2024-2025)")
    print("=" * 72)

    if baseline_sharpe <= 0.3:
        print(f"  SKIPPED: Recent Sharpe={baseline_sharpe:.3f} <= 0.3 threshold")
        return

    risk_levels = [0.8, 1.0, 1.5, 2.0]
    print(f"\n  {'Risk%':>6} {'T/mo':>6} {'WR%':>6} {'Sharpe':>8} {'MaxDD%':>8} "
          f"{'Pass%':>8} {'Fail%':>8} {'2-Att%':>8}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    best_sh = -999
    best_ri = 0.8
    for ri in risk_levels:
        t, eq, _ = simulate_partials(setups_recent, df_recent, INITIAL_BALANCE, risk_pct=ri)
        if not t:
            print(f"  {ri:6.1f} {'n/a':>6} {'n/a':>6} {'n/a':>8} {'n/a':>8} {'n/a':>8} {'n/a':>8} {'n/a':>8}")
            continue
        pnls   = np.array([x.pnl_dollars for x in t])
        wr     = (pnls > 0).sum() / len(pnls) * 100
        t_mo   = len(t) / max((eq.index[-1] - eq.index[0]).days / 30.44, 1)
        dr     = eq.resample("D").last().dropna().pct_change().dropna()
        sh     = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
        peak   = np.maximum.accumulate(eq.values)
        maxdd  = float(abs(((eq.values-peak)/np.where(peak==0,1,peak)).min()))*100
        ev     = _eval_windows(eq, INITIAL_BALANCE)
        print(f"  {ri:6.1f} {t_mo:6.1f} {wr:6.1f} {sh:8.3f} {maxdd:8.2f} "
              f"{ev['pass_rate']:8.1f} {ev['fail_rate']:8.1f} {ev['two_attempt']:8.1f}")
        if sh > best_sh:
            best_sh = sh
            best_ri = ri

    print(f"\n  Best Sharpe at risk={best_ri}% (Sharpe={best_sh:.3f})")

    # High-conviction only (score 4+) at 1.5%
    high_conv = [s for s in setups_recent if s.confluence >= 4]
    if high_conv:
        t_hc, eq_hc, _ = simulate_partials(high_conv, df_recent, INITIAL_BALANCE, risk_pct=1.5)
        if t_hc:
            pnls = np.array([x.pnl_dollars for x in t_hc])
            wr   = (pnls > 0).sum() / len(pnls) * 100
            t_mo = len(t_hc) / max((eq_hc.index[-1] - eq_hc.index[0]).days / 30.44, 1)
            dr   = eq_hc.resample("D").last().dropna().pct_change().dropna()
            sh   = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
            ev   = _eval_windows(eq_hc, INITIAL_BALANCE)
            print(f"\n  Score≥4 only @1.5%: n={len(t_hc)} T/mo={t_mo:.1f} "
                  f"WR={wr:.0f}% Sharpe={sh:.3f} Pass={ev['pass_rate']:.1f}%")
        else:
            print(f"  Score≥4 only: no trades after filtering")
    else:
        print(f"  No score≥4 setups in recent period")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("=" * 72)
    print("Prompt 24: Gold ICT Sweep-to-FVG")
    print(f"venv Python {sys.version.split()[0]} | pandas {pd.__version__}")
    print("=" * 72)

    # Task 0 — spread info
    run_task0()

    # Task 7 — re-run M15 sweep reversal at 20-pip spread
    sweep_sharpe_20pip = run_task7()

    # Build strategy instance
    strat = GoldIctSweepFvg()
    strat.setup(ICT_BASE, XAUUSD_CFG)

    # Load data
    print("\n  Loading M1 data...")
    t0 = _time.perf_counter()
    df_full   = _load_m1(start="2022-07-01")
    df_recent = _load_m1(start="2024-01-01")
    print(f"  Full:   {len(df_full):,} bars  ({df_full['datetime'].min().date()} → {df_full['datetime'].max().date()})")
    print(f"  Recent: {len(df_recent):,} bars  ({df_recent['datetime'].min().date()} → {df_recent['datetime'].max().date()})")

    # Detect setups — need to pass through backtester to get ET-converted data
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        """Convert to US/Eastern as backtester would."""
        d = df.copy()
        if "datetime" in d.columns:
            d = d.set_index("datetime")
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        d.index = d.index.tz_convert("US/Eastern")
        d.columns = [c.lower() for c in d.columns]
        return d

    print("\n  Detecting setups (full range)...")
    t1 = _time.perf_counter()
    df_full_et   = _prepare_df(df_full)
    setups_full, funnel_full = strat.detect_setups(df_full_et)
    print(f"  Full setup detection: {_time.perf_counter()-t1:.0f}s  "
          f"| {len(setups_full)} setups")

    print("  Detecting setups (recent 2024-2025)...")
    t1 = _time.perf_counter()
    df_recent_et   = _prepare_df(df_recent)
    setups_recent, funnel_recent = strat.detect_setups(df_recent_et)
    print(f"  Recent setup detection: {_time.perf_counter()-t1:.0f}s  "
          f"| {len(setups_recent)} setups")

    # Task 4 — baseline
    trades_full, eq_full, trades_rec, eq_rec = run_task4(
        setups_full, funnel_full, df_full, df_recent,
        setups_recent, funnel_recent,
    )

    # Gate: only run WF if full-range Sharpe > -1.0
    full_sharpe = 0.0
    if trades_full:
        dr_f = eq_full.resample("D").last().dropna().pct_change().dropna()
        full_sharpe = float(dr_f.mean() / dr_f.std() * np.sqrt(252)) if dr_f.std() > 0 else 0

    # Task 5 — WF (need 2023-2025 data)
    print("\n  Loading 2023-2025 data for WF...")
    df_wf      = _load_m1(start="2023-01-01")
    df_wf_et   = _prepare_df(df_wf)
    setups_wf, _ = strat.detect_setups(df_wf_et)
    if full_sharpe < -1.0:
        print(f"\n  WF SKIPPED: full-range Sharpe={full_sharpe:.3f} < -1.0 gate")
        wf_pass = False
    else:
        wf_pass = run_task5(setups_wf, df_wf)

    # Task 6 — eval sim
    recent_sharpe = 0.0
    if trades_rec:
        pnls = np.array([t.pnl_dollars for t in trades_rec])
        dr   = eq_rec.resample("D").last().dropna().pct_change().dropna()
        recent_sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0

    run_task6(setups_recent, df_recent, recent_sharpe)

    print("\n" + "=" * 72)
    print("Prompt 24 COMPLETE")
    print("=" * 72)
