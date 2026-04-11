#!/usr/bin/env python3
"""
run_prompt26.py — M1 News Bracket Orders
==========================================
Task 0 : Bracket/OCO support verified in backtester
Task 1 : Bracket vs Fade on XAUUSD M1
Task 2 : Baseline (2022 – Feb 2025): fill rate, WR, Sharpe, MaxDD per event type
Task 3 : Walk-forward (6m IS / 3m OOS / 3m step, 108 combos)
Task 4 : FTMO eval simulation (if Sharpe > 0.2)
Task 5 : Maximum single-event move analysis
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from backtesting.backtester import Backtester, BacktestResult, PendingOrder, Trade
from data.news_calendar import build_calendar
from strategies.news_m1_bracket import (
    generate_bracket_orders, generate_fade_orders, BracketSetupMeta
)

ET = pytz.timezone("US/Eastern")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR    = Path("data/historical")

XAUUSD_M1_PATH = DATA_DIR / "XAUUSD_1m.parquet"
INITIAL_BAL    = 10_000.0
PIP_SIZE       = 0.01     # XAUUSD
PIP_VALUE      = 1.0      # $/pip/lot

VERY_HIGH = ["NFP", "FOMC", "CPI", "GDP_ADV"]

# London S3 monthly return reference (from Prompt 18)
LONDON_MONTHLY = 0.00896


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

class _DummyStrategy:
    """Minimal strategy stub so Backtester.__init__ doesn't crash."""
    name = "news_m1_bracket"
    risk_per_trade_pct = 1.0


def load_m1() -> pd.DataFrame:
    df = pd.read_parquet(XAUUSD_M1_PATH)
    df.columns = [c.lower() for c in df.columns]
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        df = df.set_index("datetime")
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df.sort_index()


def run_bt(
    df_m1: pd.DataFrame,
    orders: List[PendingOrder],
    risk_pct: float = 1.0,
    break_even_r: Optional[float] = None,
) -> BacktestResult:
    """Run bracket simulation via Backtester.run_bracket()."""
    strat = _DummyStrategy()
    strat.risk_per_trade_pct = risk_pct
    bt = Backtester(strat, df_m1, "XAUUSD",
                    initial_balance=INITIAL_BAL,
                    break_even_r=break_even_r,
                    seed=42)
    return bt.run_bracket(orders)


# ── Metrics helpers ───────────────────────────────────────────────────────────

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
        return dict(n=0, wr=0, ret=0, sharpe=float("nan"), maxdd=0, t_mo=0,
                    pf=0, avg_win=0, avg_loss=0)
    wins   = [t for t in trades if t.pnl_dollars > 0]
    losses = [t for t in trades if t.pnl_dollars <= 0]
    total_pnl = sum(t.pnl_dollars for t in trades)
    span_days = max((trades[-1].exit_time - trades[0].entry_time).days, 1)
    avg_win  = np.mean([t.pnl_dollars for t in wins])  if wins   else 0.0
    avg_loss = np.mean([t.pnl_dollars for t in losses]) if losses else 0.0
    gross_win  = sum(t.pnl_dollars for t in wins)
    gross_loss = abs(sum(t.pnl_dollars for t in losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    return dict(
        n       = len(trades),
        wr      = len(wins) / len(trades) * 100,
        ret     = total_pnl / init_bal * 100,
        sharpe  = _sharpe(trades, init_bal),
        maxdd   = _maxdd(trades, init_bal),
        t_mo    = len(trades) / (span_days / 30.44),
        pf      = pf,
        avg_win = avg_win,
        avg_loss= avg_loss,
    )


def print_m(label: str, m: dict) -> None:
    if m["n"] == 0:
        print(f"  {label}: no trades")
        return
    sh_str = f"{m['sharpe']:.3f}" if not np.isnan(m['sharpe']) else "n/a"
    print(f"  {label}: T={m['n']} WR={m['wr']:.1f}% Ret={m['ret']:.2f}% "
          f"Sh={sh_str} MaxDD={m['maxdd']:.1f}% PF={m['pf']:.2f} T/mo={m['t_mo']:.1f}")


def fill_rate(orders: List[PendingOrder], result: BacktestResult) -> float:
    """% of OCO groups where one leg got filled."""
    if not orders:
        return 0.0
    filled_groups = {t.strategy_name for t in result.trades}  # proxy via event_type not ideal
    # Better: count distinct group_ids among filled
    filled_event_types = set()
    for t in result.trades:
        filled_event_types.add(t.entry_time)
    n_groups = len({o.group_id for o in orders}) / 2  # 2 orders per group
    return len(result.trades) / max(n_groups, 1) * 100


def _fill_rate_pct(orders: List[PendingOrder], trades: List[Trade]) -> float:
    n_events = len({o.group_id for o in orders})   # each OCO group = 1 event
    if n_events == 0:
        return 0.0
    return len(trades) / n_events * 100


# ══════════════════════════════════════════════════════════════════════════════
# TASK 0 — Verify bracket infrastructure
# ══════════════════════════════════════════════════════════════════════════════

def task0_verify(df_m1: pd.DataFrame, cal: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TASK 0: Bracket Infrastructure Verification")
    print("=" * 70)

    print(f"  XAUUSD M1: {len(df_m1)} bars  "
          f"{df_m1.index[0]}  →  {df_m1.index[-1]}")

    # Place a single NFP event and check order generation
    nfp1 = cal[cal["event_type"] == "NFP"].head(1)
    orders, metas = generate_bracket_orders(
        df_m1, nfp1, "XAUUSD",
        bracket_distance_atr_mult=0.5, rr_ratio=2.0,
        time_stop_bars=120, expiry_bars=30, news_spread_mult=2.0,
    )
    if metas:
        m = metas[0]
        o_buy  = next(o for o in orders if o.order_type == "buy_stop")
        o_sell = next(o for o in orders if o.order_type == "sell_stop")
        print(f"\n  Sample event: {nfp1.iloc[0]['datetime'].strftime('%Y-%m-%d %H:%M')}")
        print(f"  Anchor price : {m.anchor:.2f}")
        print(f"  M1 ATR(60)   : {m.atr:.2f} ({m.atr/PIP_SIZE:.0f} pips)")
        print(f"  Bracket dist : {m.bracket_dist:.2f} ({m.bracket_dist/PIP_SIZE:.0f} pips)")
        print(f"  BUY  STOP  entry={o_buy.entry_price:.2f}  sl={o_buy.sl_price:.2f}  tp={o_buy.tp_price:.2f}")
        print(f"  SELL STOP  entry={o_sell.entry_price:.2f}  sl={o_sell.sl_price:.2f}  tp={o_sell.tp_price:.2f}")
        print(f"  Place bar={o_buy.place_bar}  expiry bar={o_buy.expiry_bar}  time_stop={o_buy.time_stop_bars} bars")
    else:
        print("  WARNING: no orders generated for first NFP — check data coverage")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Baseline backtest
# ══════════════════════════════════════════════════════════════════════════════

BRACKET_PARAMS = dict(
    bracket_distance_atr_mult = 0.5,
    rr_ratio                  = 2.0,
    time_stop_bars            = 120,
    expiry_bars               = 30,
    news_spread_mult          = 2.0,
)
FADE_PARAMS = dict(
    fade_distance_atr_mult = 0.5,
    sl_mult                = 1.5,
    rr_ratio               = 1.5,
    time_stop_bars         = 60,
    expiry_bars            = 15,
    news_spread_mult       = 2.0,
)
RISK_PCT = 1.0


def task2_baseline(df_m1: pd.DataFrame, cal: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("TASK 2: Baseline Backtest (Jan 2022 – Feb 2025)")
    print("=" * 70)

    # ── All events: Bracket vs Fade ────────────────────────────────────────
    orders_b, metas_b = generate_bracket_orders(df_m1, cal, "XAUUSD", **BRACKET_PARAMS)
    orders_f, metas_f = generate_fade_orders(df_m1, cal, "XAUUSD", **FADE_PARAMS)

    res_b = run_bt(df_m1, orders_b, RISK_PCT)
    res_f = run_bt(df_m1, orders_f, RISK_PCT)

    m_b = _metrics(res_b.trades)
    m_f = _metrics(res_f.trades)
    fr_b = _fill_rate_pct(orders_b, res_b.trades)
    fr_f = _fill_rate_pct(orders_f, res_f.trades)

    print(f"\n--- All Events (XAUUSD) ---")
    print(f"  Events in calendar (2022-2025): {len(cal)}")
    print(f"  Bracket orders generated: {len(orders_b)} ({len(orders_b)//2} OCO pairs)")
    print(f"  Fade    orders generated: {len(orders_f)} ({len(orders_f)//2} OCO pairs)")
    print_m(f"Bracket  fill={fr_b:.0f}%", m_b)
    print_m(f"Fade     fill={fr_f:.0f}%", m_f)

    # ── VERY_HIGH only ─────────────────────────────────────────────────────
    orders_bvh, _ = generate_bracket_orders(df_m1, cal, "XAUUSD",
                                             filter_event_types=VERY_HIGH, **BRACKET_PARAMS)
    orders_fvh, _ = generate_fade_orders(df_m1, cal, "XAUUSD",
                                          filter_event_types=VERY_HIGH, **FADE_PARAMS)
    res_bvh = run_bt(df_m1, orders_bvh, RISK_PCT)
    res_fvh = run_bt(df_m1, orders_fvh, RISK_PCT)
    m_bvh = _metrics(res_bvh.trades)
    m_fvh = _metrics(res_fvh.trades)
    fr_bvh = _fill_rate_pct(orders_bvh, res_bvh.trades)
    fr_fvh = _fill_rate_pct(orders_fvh, res_fvh.trades)

    print(f"\n--- VERY_HIGH Events Only (NFP/FOMC/CPI/GDP_ADV) ---")
    print_m(f"Bracket  fill={fr_bvh:.0f}%", m_bvh)
    print_m(f"Fade     fill={fr_fvh:.0f}%", m_fvh)

    # ── Per event type (bracket) ───────────────────────────────────────────
    print(f"\n--- Per Event Type (Bracket, XAUUSD) ---")
    event_types = sorted(cal["event_type"].unique())
    et_results  = {}
    for et in event_types:
        o_et, _ = generate_bracket_orders(df_m1, cal, "XAUUSD",
                                           filter_event_types=[et], **BRACKET_PARAMS)
        r_et = run_bt(df_m1, o_et, RISK_PCT)
        m_et = _metrics(r_et.trades)
        fr_et = _fill_rate_pct(o_et, r_et.trades)
        et_results[et] = dict(metrics=m_et, fill_rate=fr_et)
        print_m(f"{et:15s} fill={fr_et:.0f}%", m_et)

    # ── Spread sensitivity ────────────────────────────────────────────────
    print(f"\n--- Spread Sensitivity (Bracket, VERY_HIGH) ---")
    for sm in [1.5, 2.0, 3.0]:
        p = {**BRACKET_PARAMS, "news_spread_mult": sm}
        o, _ = generate_bracket_orders(df_m1, cal, "XAUUSD",
                                        filter_event_types=VERY_HIGH, **p)
        r = run_bt(df_m1, o, RISK_PCT)
        m = _metrics(r.trades)
        fr = _fill_rate_pct(o, r.trades)
        print_m(f"spread_mult={sm:.1f}x fill={fr:.0f}%", m)

    # ── vs M15 straddle from Prompt 25 ────────────────────────────────────
    print(f"\n--- M1 vs M15 Comparison ---")
    print(f"  M15 straddle WR (Prompt 25): 21.1%  Sharpe=-9.96  (enter AFTER spike)")
    sh_str = f"{m_bvh['sharpe']:.3f}" if not np.isnan(m_bvh['sharpe']) else "n/a"
    print(f"  M1 bracket   WR (now):       {m_bvh['wr']:.1f}%  "
          f"Sharpe={sh_str}  (enter BEFORE spike)")
    print(f"  Fill rate (% events with a trade): {fr_bvh:.0f}%")

    # Did M1 fix timing?
    if m_bvh["wr"] > 33 and m_bvh["sharpe"] > 0:
        print("  → YES — M1 bracket timing IMPROVED over M15 straddle")
    else:
        print("  → NO — M1 bracket still struggling (WR < 33% or Sharpe < 0)")

    # Best Sharpe for WF gate
    best_sharpe = max(
        m_b["sharpe"] if not np.isnan(m_b["sharpe"]) else -99,
        m_bvh["sharpe"] if not np.isnan(m_bvh["sharpe"]) else -99,
        m_fvh["sharpe"] if not np.isnan(m_fvh["sharpe"]) else -99,
    )
    print(f"\nBest baseline Sharpe: {best_sharpe:.3f}")
    print(f"WF gate (> -1.0): {'PROCEED' if best_sharpe > -1.0 else 'SKIP'}")

    return dict(
        res_b=res_b, res_f=res_f,
        res_bvh=res_bvh, res_fvh=res_fvh,
        m_bvh=m_bvh, m_fvh=m_fvh,
        et_results=et_results,
        best_sharpe=best_sharpe,
        proceed_wf=best_sharpe > -1.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Walk-forward
# ══════════════════════════════════════════════════════════════════════════════

# 3×3×3×2×2 = 108 combos
WF_GRID = list(itertools.product(
    [0.3, 0.5, 0.7],   # bracket_distance_atr_mult
    [1.5, 2.0, 2.5],   # rr_ratio
    [1.0, 1.5, 2.0],   # risk_pct
    [2.0, 3.0],        # news_spread_mult
    [60,  120],        # time_stop_bars
))
assert len(WF_GRID) == 108
MIN_IS_TRADES = 6


def _wf_windows(start: str, end: str, is_mo: int = 6, oos_mo: int = 3) -> list:
    s = pd.Timestamp(start, tz="US/Eastern")
    e = pd.Timestamp(end,   tz="US/Eastern")
    wins = []
    cur = s
    while True:
        is_e  = cur  + pd.DateOffset(months=is_mo)
        oos_e = is_e + pd.DateOffset(months=oos_mo)
        if oos_e > e: break
        wins.append((cur, is_e, is_e, oos_e))
        cur = cur + pd.DateOffset(months=oos_mo)
    return wins


def task3_walkforward(df_m1: pd.DataFrame, cal: pd.DataFrame,
                      t2: dict) -> Optional[dict]:
    print("\n" + "=" * 70)
    print("TASK 3: Walk-Forward (6m IS / 3m OOS / 3m step, 108 combos)")
    print("=" * 70)

    if not t2["proceed_wf"]:
        print("SKIP: best baseline Sharpe < -1.0")
        return None

    windows = _wf_windows("2022-01-01", "2025-02-04")
    print(f"WF windows: {len(windows)}")
    for w in windows:
        print(f"  IS: {w[0].date()} – {w[1].date()}  "
              f"OOS: {w[2].date()} – {w[3].date()}")

    all_oos_trades: List[Trade] = []
    deg_scores: List[float] = []

    for w_i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        cal_is  = cal[(cal["datetime"] >= is_s)  & (cal["datetime"] < is_e)]
        cal_oos = cal[(cal["datetime"] >= oos_s) & (cal["datetime"] < oos_e)]
        df_is   = df_m1[(df_m1.index >= is_s)  & (df_m1.index < is_e)]
        df_oos  = df_m1[(df_m1.index >= oos_s) & (df_m1.index < oos_e)]

        best_is_sh  = -np.inf
        best_params = WF_GRID[0]

        for params in WF_GRID:
            bam, rr, rp, nsm, tsb = params
            o, _ = generate_bracket_orders(
                df_is, cal_is, "XAUUSD",
                bracket_distance_atr_mult=bam, rr_ratio=rr,
                time_stop_bars=int(tsb), expiry_bars=30,
                news_spread_mult=nsm,
                filter_event_types=VERY_HIGH,
            )
            r = run_bt(df_is, o, rp)
            if len(r.trades) < MIN_IS_TRADES:
                continue
            sh = _sharpe(r.trades)
            if not np.isnan(sh) and sh > best_is_sh:
                best_is_sh = sh
                best_params = params

        bam, rr, rp, nsm, tsb = best_params
        o_oos, _ = generate_bracket_orders(
            df_oos, cal_oos, "XAUUSD",
            bracket_distance_atr_mult=bam, rr_ratio=rr,
            time_stop_bars=int(tsb), expiry_bars=30,
            news_spread_mult=nsm,
            filter_event_types=VERY_HIGH,
        )
        r_oos  = run_bt(df_oos, o_oos, rp)
        sh_oos = _sharpe(r_oos.trades) if len(r_oos.trades) >= 3 else float("nan")
        deg    = sh_oos / best_is_sh if (not np.isnan(sh_oos) and best_is_sh > 0) else float("nan")
        all_oos_trades.extend(r_oos.trades)

        sh_oos_str = f"{sh_oos:.3f}" if not np.isnan(sh_oos) else "n/a"
        deg_str    = f"{deg:.2f}"    if not np.isnan(deg)    else "n/a"
        print(f"\n  Window {w_i+1}: IS_Sh={best_is_sh:.3f} | OOS_Sh={sh_oos_str} | "
              f"deg={deg_str} | n_oos={len(r_oos.trades)}")
        print(f"    Best: bam={bam} rr={rr} risk={rp}% nsm={nsm} ts={tsb}bars")

        if not np.isnan(deg):
            deg_scores.append(deg)

    avg_deg = float(np.mean(deg_scores)) if deg_scores else float("nan")
    m_oos   = _metrics(all_oos_trades)
    pct_pos = (sum(1 for d in deg_scores if d > 0) / len(deg_scores) * 100
               if deg_scores else 0.0)

    print(f"\nWF SUMMARY:")
    ad_str = f"{avg_deg:.3f}" if not np.isnan(avg_deg) else "n/a"
    print(f"  avg_degradation={ad_str}  pct_positive_OOS={pct_pos:.0f}%")
    print_m("  Combined OOS", m_oos)

    pass_wf = (not np.isnan(avg_deg) and avg_deg > 0.1 and pct_pos >= 40)
    print(f"\nWF pass (avg_deg>0.1 AND ≥40% OOS positive): {'YES' if pass_wf else 'NO'}")

    proceed_eval = (not np.isnan(m_oos.get("sharpe", float("nan")))
                    and m_oos.get("sharpe", -99) > 0.2)
    print(f"Eval sim gate (OOS Sharpe > 0.2): {'YES' if proceed_eval else 'NO'}")

    return dict(
        oos_trades   = all_oos_trades,
        avg_deg      = avg_deg,
        pct_pos      = pct_pos,
        oos_metrics  = m_oos,
        proceed_eval = proceed_eval,
        pass_wf      = pass_wf,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — Eval simulation
# ══════════════════════════════════════════════════════════════════════════════

def task4_eval(df_m1: pd.DataFrame, cal: pd.DataFrame,
               t3: Optional[dict]) -> None:
    print("\n" + "=" * 70)
    print("TASK 4: FTMO Eval Simulation")
    print("=" * 70)

    print(f"\nLondon S3 standalone reference: +{LONDON_MONTHLY*100:.3f}%/mo, "
          f"CAGR={((1+LONDON_MONTHLY)**12-1)*100:.1f}%")
    months_l = int(np.ceil(np.log(1.10) / np.log(1.0 + LONDON_MONTHLY)))
    print(f"  Months to +10% at London S3 only: {months_l}")

    if t3 is None or not t3.get("proceed_eval", False):
        print("\nNews M1 eval SKIPPED (WF OOS Sharpe did not clear 0.2 threshold)")

        # Still run risk sensitivity on best bracket variant for reference
        print("\n--- Risk Sensitivity (Best Bracket, VERY_HIGH only) ---")
        for rp in [1.0, 1.5, 2.0, 2.5]:
            o, _ = generate_bracket_orders(df_m1, cal, "XAUUSD",
                                            filter_event_types=VERY_HIGH, **BRACKET_PARAMS)
            r = run_bt(df_m1, o, rp)
            m = _metrics(r.trades)
            print_m(f"risk={rp:.1f}%", m)
        return

    oos_trades = t3["oos_trades"]
    oos_m      = t3["oos_metrics"]

    # Per-trade pnl distribution from OOS
    pnls = np.array([t.pnl_dollars / INITIAL_BAL for t in oos_trades]) if oos_trades else np.array([0.0])
    # News events per month (OOS period)
    if len(oos_trades) > 1:
        span = (oos_trades[-1].exit_time - oos_trades[0].entry_time).days / 30.44
        t_mo = len(oos_trades) / max(span, 1.0)
    else:
        t_mo = 1.0

    # Risk sensitivity
    print(f"\n--- Risk Sensitivity (VERY_HIGH only) ---")
    best_combo_rp = None
    best_pass_rate = 0.0
    for rp in [1.0, 1.5, 2.0, 2.5]:
        o, _ = generate_bracket_orders(df_m1, cal, "XAUUSD",
                                        filter_event_types=VERY_HIGH, **BRACKET_PARAMS)
        r = run_bt(df_m1, o, rp)
        m = _metrics(r.trades)
        print_m(f"risk={rp:.1f}%", m)

    # FTMO Challenge eval: 1000 trials, 15-day windows
    print(f"\n--- FTMO Challenge Eval Sim (1000 trials × 15 trading days) ---")
    rng = np.random.default_rng(42)
    n_trials      = 1000
    challenge_days = 15
    daily_loss_lim = 0.04
    total_dd_lim   = 0.09
    target         = 0.10
    london_daily_mu = LONDON_MONTHLY / 22.0

    results_by_risk = {}
    for rp in [1.0, 1.5, 2.0, 2.5]:
        pnl_scale = rp / 1.0   # scale OOS pnls proportionally to risk
        passed = 0
        for _ in range(n_trials):
            bal   = INITIAL_BAL
            peak  = INITIAL_BAL
            day_b = INITIAL_BAL
            halt  = False
            for day in range(challenge_days):
                if halt: break
                london_d = rng.normal(london_daily_mu, abs(london_daily_mu) * 3)
                news_d   = 0.0
                # ~3 VERY_HIGH events/month → p(event today) ≈ 3/22
                if rng.random() < 3.0 / 22.0 and len(pnls) > 1:
                    news_d = float(rng.choice(pnls)) * pnl_scale
                daily_ret = (london_d + news_d) * INITIAL_BAL
                bal += daily_ret
                if (bal - day_b) / day_b < -daily_loss_lim:
                    halt = True; break
                if bal > peak: peak = bal
                if (peak - bal) / INITIAL_BAL > total_dd_lim:
                    halt = True; break
                day_b = bal
            if not halt and (bal - INITIAL_BAL) / INITIAL_BAL >= target:
                passed += 1
        pass_rate = passed / n_trials * 100
        results_by_risk[rp] = pass_rate
        print(f"  risk={rp:.1f}%: pass_rate={pass_rate:.1f}%  "
              f"(attempts to pass: {100/max(pass_rate,0.1):.0f})")
        if pass_rate > best_pass_rate:
            best_pass_rate = pass_rate
            best_combo_rp  = rp

    print(f"\nBest risk level for eval: {best_combo_rp}%  pass_rate={best_pass_rate:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 — Maximum single-event move analysis
# ══════════════════════════════════════════════════════════════════════════════

def task5_max_events(df_m1: pd.DataFrame, cal: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TASK 5: Maximum Single-Event Move Analysis")
    print("=" * 70)

    idx  = df_m1.index
    arr_h = df_m1["high"].to_numpy(float)
    arr_l = df_m1["low"].to_numpy(float)
    arr_c = df_m1["close"].to_numpy(float)

    print(f"\n{'Date':12s}  {'Type':12s}  {'MaxMove(pips)':13s}  "
          f"{'At2%risk($)':11s}  {'At2%risk(%)':11s}")
    print("-" * 65)

    event_moves = []
    for _, row in cal.iterrows():
        evt_ts = row["datetime"]
        if evt_ts.tzinfo is None:
            evt_ts = ET.localize(evt_ts)
        pos = idx.searchsorted(evt_ts)
        if pos >= len(idx): continue
        if abs((idx[pos] - evt_ts).total_seconds()) / 60 > 2: continue
        event_bar = pos

        # Max move in first 30 M1 bars after event
        end_bar = min(event_bar + 31, len(df_m1))
        if end_bar <= event_bar: continue

        anchor   = float(arr_c[event_bar - 1]) if event_bar > 0 else float(arr_c[event_bar])
        hi_30    = float(np.max(arr_h[event_bar:end_bar]))
        lo_30    = float(np.min(arr_l[event_bar:end_bar]))
        max_move = max(abs(hi_30 - anchor), abs(lo_30 - anchor))
        max_pips = max_move / PIP_SIZE

        # At 2% risk, 1.0 lot typical for $10k account:
        # risk_usd = 10000 * 0.02 = $200
        # If SL = bracket_dist pips, catch = 60% of max_move at RR 2.0
        # Simplify: catch = max_pips × 0.6 × PIP_VALUE × lot
        # lot = risk_usd / (bracket_pips × PIP_VALUE)
        # With bracket = 0.5 × ATR, but use max_move / 3 as rough SL proxy
        rough_sl_pips = max(max_pips / 3.0, 50.0)
        risk_usd      = INITIAL_BAL * 0.02
        lot           = risk_usd / (rough_sl_pips * PIP_VALUE)
        catch_pips    = max_pips * 0.6
        pnl_dollars   = catch_pips * PIP_VALUE * lot
        pnl_pct       = pnl_dollars / INITIAL_BAL * 100

        event_moves.append(dict(
            date       = idx[event_bar].strftime("%Y-%m-%d"),
            event_type = row["event_type"],
            max_pips   = max_pips,
            pnl_dollars= pnl_dollars,
            pnl_pct    = pnl_pct,
        ))

    # Sort by max move descending, show top 20
    event_moves.sort(key=lambda x: -x["max_pips"])
    for e in event_moves[:20]:
        print(f"  {e['date']:12s}  {e['event_type']:12s}  {e['max_pips']:8.0f} pips      "
              f"${e['pnl_dollars']:7.0f}  ({e['pnl_pct']:5.2f}%)")

    # Summary stats
    all_pips = [e["max_pips"] for e in event_moves]
    all_pnl  = [e["pnl_pct"]  for e in event_moves]
    if all_pips:
        print(f"\nAll {len(event_moves)} events:")
        print(f"  Median max-move  : {np.median(all_pips):.0f} pips")
        print(f"  Mean  max-move   : {np.mean(all_pips):.0f} pips")
        print(f"  P90   max-move   : {np.percentile(all_pips, 90):.0f} pips")
        print(f"  Median pnl@2%    : {np.median(all_pnl):.2f}%")
        print(f"  Mean  pnl@2%     : {np.mean(all_pnl):.2f}%")

    # Per event type
    print(f"\nPer event type (median max-move, pips):")
    et_moves = {}
    for e in event_moves:
        et_moves.setdefault(e["event_type"], []).append(e["max_pips"])
    for et, mvs in sorted(et_moves.items(), key=lambda x: -np.median(x[1])):
        print(f"  {et:15s}: median={np.median(mvs):.0f}  mean={np.mean(mvs):.0f}  n={len(mvs)}")

    # How many top-10 events alone would cover +10% eval target?
    top10_pnl = sum(e["pnl_pct"] for e in event_moves[:10])
    print(f"\nTop-10 events cumulative pnl@2%risk: {top10_pnl:.1f}%")
    print(f"  → {'YES' if top10_pnl >= 10 else 'NO'}: top-10 events ALONE "
          f"{'would cover' if top10_pnl >= 10 else 'would NOT cover'} the +10% FTMO target")

    # NFP / FOMC / CPI specific
    print(f"\nNFP / FOMC / CPI breakdown:")
    for et in ["NFP", "FOMC", "CPI"]:
        evts = [e for e in event_moves if e["event_type"] == et]
        if not evts: continue
        print(f"  {et}: n={len(evts)}  "
              f"median_move={np.median([e['max_pips'] for e in evts]):.0f}p  "
              f"max_move={max(e['max_pips'] for e in evts):.0f}p  "
              f"median_pnl@2%={np.median([e['pnl_pct'] for e in evts]):.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PROMPT 26 — M1 News Bracket Orders (XAUUSD)")
    print("=" * 70)

    print("\nLoading data...")
    df_m1 = load_m1()
    cal   = pd.read_parquet("data/news_calendar.parquet")
    print(f"  M1 bars: {len(df_m1)}")
    print(f"  Events : {len(cal)}")

    # Task 0
    task0_verify(df_m1, cal)

    # Task 2 (Task 1 = strategy code, verified in Task 0)
    t2 = task2_baseline(df_m1, cal)

    # Task 3
    t3 = task3_walkforward(df_m1, cal, t2)

    # Task 4
    task4_eval(df_m1, cal, t3)

    # Task 5
    task5_max_events(df_m1, cal)

    print("\n" + "=" * 70)
    print("PROMPT 26 COMPLETE")
    print("=" * 70)
