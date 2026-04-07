#!/usr/bin/env python3
"""
Prompt 18: M15 Diagnosis & Redesign  +  London Risk Uplift
===========================================================
Run: venv/bin/python3 run_prompt18.py

Part A  — M15 variants V1-V4, MFE analysis, pullback audit, A3 WF
Part B  — London 7-strat risk scenarios S1/S2/S3 through FTMO evaluator
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
import time as time_module
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ta

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from backtesting.backtester import Backtester
from backtesting.ftmo_evaluator import FTMOEvaluator
from backtesting.metrics import calculate_metrics, _risk_adjusted
from strategies.london_open_breakout import LondonOpenBreakout
from strategies.m15_momentum_scalping import M15MomentumScalping
from strategies.ny_session_breakout import NYSessionBreakout

RESULTS = ROOT / "results"
DATA    = ROOT / "data" / "historical"
CFG_DIR = ROOT / "config"
INIT    = 10_000.0

# ── FTMO guardian disabled (see raw equity curve) ────────────────────────────
LOOSE_FTMO = {
    "challenge":    {"profit_target_pct": 10.0, "min_trading_days": 4,
                     "max_daily_loss_pct": 5.0,  "max_total_drawdown_pct": 10.0},
    "verification": {"profit_target_pct":  5.0,  "min_trading_days": 4,
                     "max_daily_loss_pct": 5.0,  "max_total_drawdown_pct": 10.0},
    "funded":       {"profit_target_pct": None,  "min_trading_days": 4,
                     "max_daily_loss_pct": 5.0,  "max_total_drawdown_pct": 10.0},
    "safety_buffers": {"daily_loss_trigger_pct": 99.0, "total_loss_trigger_pct": 99.0},
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_cfg():
    with open(CFG_DIR / "strategy_params.json") as f: sp = json.load(f)
    with open(CFG_DIR / "instruments.json")     as f: ic = json.load(f)
    return sp, ic


def _load_m15(pair: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"{pair}_15m.parquet")
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.columns = [c.lower() for c in df.columns]
    return df


def _sec(title: str):
    W = 72
    print(f"\n{'='*W}\n  {title}\n{'='*W}")


def _run_bt(strategy, df, pair, be_r=None, loose=True, seed=42):
    return Backtester(
        strategy=strategy, df=df, instrument=pair,
        initial_balance=INIT, phase="challenge", seed=seed,
        break_even_r=be_r,
        _override_ftmo_rules=LOOSE_FTMO if loose else None,
    ).run()


def _metrics_line(label, result, w=34):
    m = calculate_metrics(result)
    if "error" in m:
        print(f"  {label:<{w}}  (no trades)"); return
    print(
        f"  {label:<{w}}  T={m['total_trades']:>4}  WR={m['win_rate_pct']:4.1f}%"
        f"  Ret={m['total_return_pct']:>+7.2f}%  Sh={m['sharpe_ratio']:>+6.3f}"
        f"  DD={m['max_drawdown_pct']:>5.2f}%  PF={m['profit_factor']:>5.3f}"
        f"  T/mo={m['trades_per_month_avg']:.1f}"
        + (f"  AvgW=${m['avg_win_dollars']:.2f} AvgL=${m['avg_loss_dollars']:.2f}" if "avg_win_dollars" in m else "")
    )


def _combine_eq(results: list, init=INIT) -> pd.Series:
    eqs = [r.equity_curve for r in results if r is not None]
    if not eqs:
        return pd.Series(dtype=float)
    df = pd.concat(eqs, axis=1)
    df.columns = [f"s{i}" for i in range(len(eqs))]
    df = df.ffill().bfill()
    return init + df.diff().fillna(0).sum(axis=1).cumsum()


def _portfolio_stats(combined_eq: pd.Series) -> dict:
    cd = combined_eq.resample("D").last().dropna()
    if len(cd) < 2:
        return {}
    ret  = (cd.iloc[-1] - INIT) / INIT * 100
    yrs  = (cd.index[-1] - cd.index[0]).days / 365.25
    cagr = ((cd.iloc[-1] / INIT) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    sh, _ = _risk_adjusted(combined_eq)
    pk    = cd.expanding().max()
    dd    = float(abs(((cd - pk) / pk * 100).min()))
    mo    = cd.pct_change().resample("ME").apply(lambda x: (1+x).prod()-1).dropna() * 100
    # worst intraday daily loss (FTMO metric)
    worst_day = 0.0
    for date, grp in combined_eq.groupby(combined_eq.index.date):
        if grp.empty: continue
        sod = float(grp.iloc[0]); lo = float(grp.min())
        if sod > 0:
            worst_day = min(worst_day, (lo - sod) / sod * 100)
    # days to +10%
    above = cd[cd >= INIT * 1.10]
    d10 = (above.index[0] - cd.index[0]).days if len(above) else None
    return {
        "total_ret": ret, "cagr": cagr, "sharpe": sh, "max_dd": dd,
        "avg_mo": float(mo.mean()), "pos_mo": int((mo > 0).sum()),
        "neg_mo": int((mo <= 0).sum()), "worst_day": worst_day,
        "days_to_10": d10, "mo_series": mo,
    }


# ─────────────────────────────────────────────────────────────────────────────
# H4 slope helper (for V3 / V4 / A3)
# ─────────────────────────────────────────────────────────────────────────────

def _h4_ema200_slope(df_m15: pd.DataFrame, slope_lookback: int = 5) -> pd.Series:
    """
    Resample M15 to 4H, compute EMA(200), compute slope over `slope_lookback`
    H4 bars. Shift by 1 to avoid look-ahead, forward-fill onto M15 timestamps.
    Returns a Series aligned to df_m15.index (positive = uptrend, negative = down).
    """
    df_utc = df_m15.copy()
    if df_utc.index.tz is not None:
        df_utc.index = df_utc.index.tz_convert("UTC")

    h4_close = df_utc["close"].resample("4h").last().dropna()
    h4_ema   = ta.trend.EMAIndicator(
        close=h4_close, window=200, fillna=False
    ).ema_indicator()
    h4_slope = h4_ema - h4_ema.shift(slope_lookback)

    # Shift by 1 H4 bar — only use signal from the PREVIOUS completed H4 bar
    h4_slope_shifted = h4_slope.shift(1)

    # Forward-fill onto original M15 timestamps
    slope_m15 = h4_slope_shifted.reindex(df_utc.index).ffill()
    return slope_m15


def _apply_h4_filter(signals_df: pd.DataFrame, h4_slope: pd.Series) -> pd.DataFrame:
    """
    Zero out signals that go against the H4 EMA200 trend.
    Longs require slope > 0; shorts require slope < 0.
    """
    df = signals_df.copy()
    slope = h4_slope.reindex(df.index).ffill().fillna(0.0)

    bad_long  = (df["signal"] == 1)  & (slope <= 0)
    bad_short = (df["signal"] == -1) & (slope >= 0)
    mask = bad_long | bad_short

    df.loc[mask, "signal"]   = 0
    df.loc[mask, "sl_price"] = np.nan
    df.loc[mask, "tp_price"] = np.nan
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MFE / duration / exit-type helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mfe_analysis(trades, df_m15: pd.DataFrame):
    """
    For each trade compute Maximum Favorable Excursion in R multiples.
    Works on the OHLCV DataFrame (any tz-aware index).
    Returns list of MFE_R values, one per trade.
    """
    df = df_m15.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    mfe_list = []
    for t in trades:
        entry  = t.entry_price
        sl     = t.sl
        risk_r = abs(entry - sl)
        if risk_r < 1e-10:
            mfe_list.append(0.0)
            continue
        # Get bars between entry and exit (inclusive)
        mask    = (df.index >= t.entry_time) & (df.index <= t.exit_time)
        sub     = df.loc[mask]
        if sub.empty:
            mfe_list.append(0.0)
            continue
        if t.direction == 1:   # long: max high
            mfe_price = float(sub["high"].max())
            mfe_r     = (mfe_price - entry) / risk_r
        else:                  # short: min low
            mfe_price = float(sub["low"].min())
            mfe_r     = (entry - mfe_price) / risk_r
        mfe_list.append(max(0.0, mfe_r))
    return mfe_list


def _duration_bars(trades, df_m15: pd.DataFrame) -> list:
    """Count M15 bars between entry and exit for each trade."""
    idx = df_m15.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    durations = []
    for t in trades:
        n = int(((idx >= t.entry_time) & (idx <= t.exit_time)).sum())
        durations.append(n)
    return durations


def _exit_breakdown(trades):
    counts: dict = defaultdict(int)
    for t in trades:
        counts[t.exit_reason] += 1
    return dict(counts)


def _big_move_cooldown_check(trades, df_m15: pd.DataFrame, atr_mult_threshold=2.0):
    """
    For each LOSING trade, check whether any of the 3 bars immediately
    preceding entry had a move > atr_mult_threshold × ATR(14).
    Returns fraction of losing trades that had such a preceding big move.
    """
    df = df_m15.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")

    # Pre-compute ATR(14)
    atr = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14, fillna=False
    ).average_true_range()

    losers = [t for t in trades if t.pnl_dollars <= 0]
    triggered = 0
    for t in losers:
        entry_et = t.entry_time.tz_convert("US/Eastern")
        # Find position of entry bar
        try:
            entry_pos = df.index.get_indexer([entry_et], method="nearest")[0]
        except Exception:
            continue
        # Look at 3 bars before entry
        for offset in range(1, 4):
            i = entry_pos - offset
            if i < 0:
                break
            bar_range = float(df.iloc[i]["high"] - df.iloc[i]["low"])
            atr_val   = float(atr.iloc[i]) if not np.isnan(float(atr.iloc[i])) else 0
            if atr_val > 0 and bar_range > atr_mult_threshold * atr_val:
                triggered += 1
                break  # count this loser once

    return triggered / len(losers) if losers else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PART A — M15 Diagnostic Variants
# ─────────────────────────────────────────────────────────────────────────────

def run_part_a(sp, ic):
    m15_cfg  = sp["m15_momentum_scalping"]
    eu_instr = ic["EURUSD"]
    df_full  = _load_m15("EURUSD")

    # Pre-compute H4 slope (V3/V4 filter)
    print("  Pre-computing H4 EMA200 slope…", flush=True)
    h4_slope = _h4_ema200_slope(df_full)

    # ── Helper: build strategy and run ──────────────────────────────────────
    def _variant(cfg_override: dict, df, be_r=None, post_h4=False) -> tuple:
        cfg = {**m15_cfg, **cfg_override}
        cfg.setdefault("rsi_short_min", 100.0 - cfg.get("rsi_long_max", 65))
        cfg.setdefault("rsi_short_max", 100.0 - cfg.get("rsi_long_min", 45))
        s = M15MomentumScalping()
        s.setup(cfg, eu_instr)
        t0 = time_module.time()

        if post_h4:
            # H4 filter: generate signals then post-filter
            df_et  = Backtester(strategy=s, df=df, instrument="EURUSD",
                                initial_balance=INIT, _override_ftmo_rules=LOOSE_FTMO).strategy.generate_signals(
                                    df.copy().assign(**{"close": df["close"]})   # dummy — we need prepared df
                                )
            # Proper approach: prepare data then generate
            bt_tmp = Backtester.__new__(Backtester)
            bt_tmp.strategy = s
            bt_tmp.instrument = "EURUSD"
            bt_tmp.initial_balance = INIT
            bt_tmp.phase = "challenge"
            bt_tmp.rng = np.random.default_rng(42)
            bt_tmp.break_even_r = be_r
            bt_tmp.trail_atr_mult = None
            bt_tmp._load_configs(None, LOOSE_FTMO)
            df_prep = bt_tmp._prepare_data(df)
            sig_df  = s.generate_signals(df_prep.copy())
            sig_df  = _apply_h4_filter(sig_df, h4_slope)
            # Patch strategy to return pre-filtered signals
            import types
            def _patched_signals(self2, df2): return sig_df.copy()
            s.generate_signals = types.MethodType(_patched_signals, s)

        result = _run_bt(s, df, "EURUSD", be_r=be_r, loose=True)
        elapsed = time_module.time() - t0
        return result, elapsed

    # ── V1: BE disabled, full period ────────────────────────────────────────
    print("\n  Running V1…", flush=True)
    r1, e1 = _variant({}, df_full, be_r=None)

    # ── V2: BE disabled, start Jul 2020 ─────────────────────────────────────
    print("  Running V2…", flush=True)
    df_v2 = df_full[df_full.index >= pd.Timestamp("2020-07-01", tz="UTC")]
    r2, e2 = _variant({}, df_v2, be_r=None)

    # ── V3: V1 + H4 trend filter ─────────────────────────────────────────────
    print("  Running V3…", flush=True)
    r3, e3 = _variant({}, df_full, be_r=None, post_h4=True)

    # ── V4: V3 + session end 16:00 ───────────────────────────────────────────
    print("  Running V4…", flush=True)
    r4, e4 = _variant({"ny_session_end": "16:00"}, df_full, be_r=None, post_h4=True)

    # ── Print comparison table ───────────────────────────────────────────────
    _sec("PART A1 — M15 Diagnostic Variants (EURUSD)")
    print(f"\n  {'Variant':<38}  {'T':>4}  {'WR%':>5}  {'Ret%':>7}  {'Sh':>6}  "
          f"{'DD%':>5}  {'PF':>5}  {'T/mo':>5}  {'AvgW$':>6}  {'AvgL$':>6}")
    print(f"  {'-'*38}  {'-'*4}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}")
    for label, res in [
        ("V1: No BE, full 2020-2025",              r1),
        ("V2: No BE, start Jul 2020",              r2),
        ("V3: V1 + H4 EMA200 slope filter",        r3),
        ("V4: V3 + session 07:00-16:00 ET",        r4),
    ]:
        m = calculate_metrics(res)
        if "error" in m:
            print(f"  {label:<38}  (no trades)"); continue
        print(
            f"  {label:<38}  {m['total_trades']:>4}  {m['win_rate_pct']:>5.1f}"
            f"  {m['total_return_pct']:>+7.2f}%  {m['sharpe_ratio']:>+6.3f}"
            f"  {m['max_drawdown_pct']:>5.2f}%  {m['profit_factor']:>5.3f}"
            f"  {m['trades_per_month_avg']:>5.1f}"
            f"  ${m.get('avg_win_dollars', 0):>5.2f}  ${m.get('avg_loss_dollars', 0):>5.2f}"
        )

    # FTMO breach check for each variant
    ev = FTMOEvaluator(window_months=3)

    class _FR:
        def __init__(self, r):
            self.equity_curve=r.equity_curve; self.trades=r.trades
            self.initial_balance=INIT; self.final_balance=float(r.equity_curve.iloc[-1])
            self.ftmo_halt_reason=None; self.daily_pnl=pd.Series(dtype=float)

    print(f"\n  FTMO breach count (3-month windows, official 5%/10% limits):")
    for label, res in [("V1","r1"), ("V2","r2"), ("V3","r3"), ("V4","r4")]:
        r = locals()[res.strip('"')]
        er = ev.evaluate_single(_FR(r), label=label)
        print(f"    {label}: {er.daily_loss_breach_count+er.max_dd_breach_count} breaches  "
              f"({len(er.windows)} windows)")

    # Monthly return table for best variant (V1 as baseline)
    m1 = calculate_metrics(r1)
    if "error" not in m1:
        print(f"\n  V1 Monthly returns (%):")
        mo_d = r1.equity_curve.resample("D").last().dropna()
        mo_r = mo_d.pct_change().resample("ME").apply(lambda x:(1+x).prod()-1).dropna()*100
        mr = mo_r.to_frame("r"); mr["yr"]=mr.index.year; mr["mo"]=mr.index.month
        pv = mr.pivot_table(values="r",index="yr",columns="mo",aggfunc="first")
        MN = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        print("  " + "".join(f"  {m:>5}" for m in (["Year"]+MN+["Ann"])))
        for yr in pv.index:
            row=pv.loc[yr]; cells=""; ann=1.0
            for mo in range(1,13):
                v=row.get(mo,float("nan"))
                if not np.isnan(v): cells+=f"  {v:>+5.1f}"; ann*=(1+v/100)
                else: cells+=f"  {'—':>5}"
            print(f"  {yr}{cells}  {(ann-1)*100:>+6.1f}%")

    # ── MFE + Duration + Exit breakdown (V1) ────────────────────────────────
    _sec("PART A1 — V1 Trade Analytics (MFE / Duration / Exit types)")

    df_m15_et = df_full.copy()
    df_m15_et.index = df_m15_et.index.tz_convert("US/Eastern")

    if r1.trades:
        mfe_list  = _mfe_analysis(r1.trades, df_full)
        dur_list  = _duration_bars(r1.trades, df_full)
        exits     = _exit_breakdown(r1.trades)
        n         = len(r1.trades)

        mfe_arr = np.array(mfe_list)
        dur_arr = np.array(dur_list)

        print(f"\n  Exit breakdown (V1, n={n}):")
        for reason, cnt in sorted(exits.items(), key=lambda x: -x[1]):
            print(f"    {reason:<12}: {cnt:>4}  ({100*cnt/n:.0f}%)")

        print(f"\n  MFE analysis (Maximum Favorable Excursion in R):")
        print(f"    Median MFE:   {np.median(mfe_arr):.2f}R")
        print(f"    Mean MFE:     {np.mean(mfe_arr):.2f}R")
        print(f"    MFE < 0.5R:   {(mfe_arr < 0.5).sum():>4}  ({100*(mfe_arr<0.5).mean():.0f}%)")
        print(f"    0.5R≤MFE<1R:  {((mfe_arr>=0.5)&(mfe_arr<1)).sum():>4}  ({100*((mfe_arr>=0.5)&(mfe_arr<1)).mean():.0f}%)")
        print(f"    1R≤MFE<2R:    {((mfe_arr>=1)&(mfe_arr<2)).sum():>4}  ({100*((mfe_arr>=1)&(mfe_arr<2)).mean():.0f}%)")
        print(f"    MFE ≥ 2R:     {(mfe_arr>=2).sum():>4}  ({100*(mfe_arr>=2).mean():.0f}%)")
        if np.median(mfe_arr) > 1.2:
            print(f"\n  → Median MFE {np.median(mfe_arr):.2f}R > 1.2R: SIGNALS HAVE EDGE, exit logic is wrong")
        elif np.median(mfe_arr) < 0.8:
            print(f"\n  → Median MFE {np.median(mfe_arr):.2f}R < 0.8R: SIGNAL ITSELF IS WEAK")
        else:
            print(f"\n  → Median MFE {np.median(mfe_arr):.2f}R is borderline (0.8–1.2R)")

        print(f"\n  Trade duration (bars) distribution:")
        print(f"    Median: {np.median(dur_arr):.0f} bars  ({np.median(dur_arr)*15:.0f} min)")
        print(f"    Mean:   {np.mean(dur_arr):.1f} bars  ({np.mean(dur_arr)*15:.0f} min)")
        pcts = [25, 50, 75, 90]
        for p in pcts:
            v = np.percentile(dur_arr, p)
            print(f"    p{p:>2}: {v:.0f} bars ({v*15:.0f} min)")

        # Big-move cooldown check
        pct_losers_big = _big_move_cooldown_check(r1.trades, df_full)
        print(f"\n  Big-move cooldown check:")
        print(f"    % of losing trades entered within 3 bars of >2×ATR bar: "
              f"{pct_losers_big*100:.1f}%")
        if pct_losers_big > 0.60:
            print(f"    → >60% threshold triggered: ADD cooldown filter (skip if prior 3 bars moved >2×ATR)")
        else:
            print(f"    → <60%: cooldown filter NOT strongly indicated")

    # ── Pullback audit: last 20 entries from 2024 ────────────────────────────
    _sec("PART A2 — Pullback Logic Audit (last 20 EURUSD signals from 2024)")

    df_2024 = df_full[df_full.index.year == 2024].copy()
    if len(df_2024) > 0:
        s_audit = M15MomentumScalping()
        s_audit.setup(m15_cfg, eu_instr)
        bt_audit = Backtester.__new__(Backtester)
        bt_audit.strategy = s_audit; bt_audit.instrument = "EURUSD"
        bt_audit.initial_balance = INIT; bt_audit.phase = "challenge"
        bt_audit.rng = np.random.default_rng(42)
        bt_audit.break_even_r = None; bt_audit.trail_atr_mult = None
        bt_audit._load_configs(None, LOOSE_FTMO)
        df_prep_audit = bt_audit._prepare_data(df_2024)
        sig_df_audit  = s_audit.generate_signals(df_prep_audit.copy())

        signal_rows = sig_df_audit[sig_df_audit["signal"] != 0].copy()
        last_20 = signal_rows.tail(20)

        # Find outcomes from the actual V1 trades
        def _find_outcome(entry_ts, v1_trades):
            for t in v1_trades:
                if abs((t.entry_time - entry_ts).total_seconds()) < 3600:
                    return t.exit_reason, t.pnl_dollars
            return "unknown", None

        print(f"\n  {'#':<3}  {'Datetime (ET)':<22}  {'Dir':<5}  "
              f"{'Entry':>9}  {'EMA9':>8}  {'EMA21':>8}  {'EMA50':>8}  "
              f"{'RSI':>5}  {'Dist21p':>7}  {'ATR':>8}  {'Outcome':<15}")
        print("  " + "-"*130)

        eu_pip = eu_instr["pip_size"]
        for i, (ts, row) in enumerate(last_20.iterrows()):
            direction = "LONG" if int(row["signal"]) == 1 else "SHORT"
            dist_to_ema21_pips = (
                abs(row.get("close", 0) - row.get("ema_slow_val", 0)) / eu_pip
                if not np.isnan(row.get("ema_slow_val", float("nan"))) else float("nan")
            )
            ts_et = ts.tz_convert("US/Eastern")
            outcome, pnl = _find_outcome(ts_et, r1.trades)
            print(
                f"  {i+1:<3}  {str(ts_et)[:22]:<22}  {direction:<5}"
                f"  {row.get('close', 0):>9.5f}"
                f"  {row.get('ema_fast_val', float('nan')):>8.5f}"
                f"  {row.get('ema_slow_val', float('nan')):>8.5f}"
                f"  {row.get('ema_trend_val', float('nan')):>8.5f}"
                f"  {row.get('rsi_val', float('nan')):>5.1f}"
                f"  {dist_to_ema21_pips:>7.1f}"
                f"  {row.get('atr_val', float('nan')):>8.5f}"
                f"  {outcome:<15}"
                + (f"  ${pnl:.2f}" if pnl is not None else "")
            )
    else:
        print("  No 2024 data available.")

    # ── A3: WF on best variant ───────────────────────────────────────────────
    # Determine best variant: Sharpe > 0.2 AND trades/month >= 10
    variants = {"V1": r1, "V2": r2, "V3": r3, "V4": r4}
    best_v = None
    best_sh = -np.inf
    for vname, vres in variants.items():
        vm = calculate_metrics(vres)
        if "error" in vm: continue
        if vm["sharpe_ratio"] > 0.2 and vm["trades_per_month_avg"] >= 10:
            if vm["sharpe_ratio"] > best_sh:
                best_sh = vm["sharpe_ratio"]
                best_v  = (vname, vres, vm)

    _sec("PART A3 — Walk-Forward Decision")
    if best_v is None:
        best_variants_info = []
        for vname, vres in variants.items():
            vm = calculate_metrics(vres)
            if "error" not in vm:
                best_variants_info.append((vname, vm["sharpe_ratio"], vm["trades_per_month_avg"]))
        print(f"\n  No variant qualifies (needs Sharpe > 0.2 AND trades/month >= 10).")
        print(f"  Variant summary:")
        for vn, sh, tpm in best_variants_info:
            qual_sh = "✓" if sh > 0.2 else "✗"
            qual_tpm = "✓" if tpm >= 10 else "✗"
            print(f"    {vn}: Sharpe={sh:+.3f} {qual_sh}  T/mo={tpm:.1f} {qual_tpm}")
        print(f"\n  → M15 Momentum Scalping is DEAD in current form.")
        print(f"     Do not run WF. Required fixes before next attempt:")
        print(f"     1. Add H4 EMA200 trend filter (directional alignment)")
        print(f"     2. Wider session window 07:00–16:00 ET")
        print(f"     3. Cooldown after >2×ATR momentum bars")
        print(f"     4. Consider entry on first H1 pullback (not M15 granularity)")
        return {v: r for v, r in variants.items()}
    else:
        vname, vres, vm = best_v
        print(f"\n  Best variant: {vname}  Sharpe={vm['sharpe_ratio']:.3f}  T/mo={vm['trades_per_month_avg']:.1f}")
        print(f"  → Proceeding to walk-forward on {vname}…")
        _run_a3_wf(vname, vres, sp, ic, df_full, h4_slope)
        return {v: r for v, r in variants.items()}


def _run_a3_wf(vname, vres, sp, ic, df_full, h4_slope):
    """WF on best A1 variant (triggered only if Sharpe>0.2 and T/mo>=10)."""
    from backtesting.walk_forward import (
        _generate_windows, _expand_grid, _risk_adjusted as _ra,
        WalkForwardResult, WindowResult, _recommend_params, _print_summary, save_result
    )

    A3_GRID = {
        "ema_fast":           [9, 12],
        "ema_slow":           [21, 26],
        "rsi_long_min":       [40, 45],
        "rsi_long_max":       [60, 65, 70],
        "atr_tp_mult":        [1.5, 2.0, 2.5],
        "atr_sl_mult":        [1.0, 1.5],
        "risk_per_trade_pct": [0.5, 0.7],
    }
    n_combos = 1
    for v in A3_GRID.values():
        n_combos *= len(v)

    use_h4 = vname in ("V3", "V4")
    wide_session = vname == "V4"

    base_cfg = {**sp["m15_momentum_scalping"]}
    if wide_session:
        base_cfg["ny_session_end"] = "16:00"

    eu_instr  = ic["EURUSD"]
    df_norm   = df_full.copy()
    param_combos = list(_expand_grid(A3_GRID))
    windows   = list(_generate_windows(df_norm, 6, 3, 3))

    print(f"\n  A3 Walk-Forward: {n_combos} combos × {len(windows)} windows  "
          f"(H4 filter: {'ON' if use_h4 else 'OFF'})  …\n")

    window_results = []
    for w_idx, (is_df, oos_df, is_start, is_end, oos_start, oos_end) in enumerate(windows):
        print(f"  Win {w_idx+1}/{len(windows)} "
              f"IS:{is_start:%Y-%m}→{is_end:%Y-%m} "
              f"OOS:{oos_start:%Y-%m}→{oos_end:%Y-%m}", end=" ", flush=True)

        # Pre-compute H4 slope for IS slice
        h4_is  = _h4_ema200_slope(is_df)  if use_h4 else None
        h4_oos = _h4_ema200_slope(oos_df) if use_h4 else None

        import types

        best_params = None
        best_sharpe = -np.inf

        for combo_idx, params in enumerate(param_combos):
            cfg = {**base_cfg, **params}
            cfg["rsi_short_min"] = 100.0 - cfg.get("rsi_long_max", 65)
            cfg["rsi_short_max"] = 100.0 - cfg.get("rsi_long_min", 45)
            s = M15MomentumScalping()
            s.setup(cfg, eu_instr)

            if use_h4:
                # Prepare IS data, generate signals, apply H4 filter
                bt_tmp = Backtester.__new__(Backtester)
                bt_tmp.strategy = s; bt_tmp.instrument = "EURUSD"
                bt_tmp.initial_balance = INIT; bt_tmp.phase = "challenge"
                bt_tmp.rng = np.random.default_rng(combo_idx)
                bt_tmp.break_even_r = None; bt_tmp.trail_atr_mult = None
                bt_tmp._load_configs(None, None)
                df_prep = bt_tmp._prepare_data(is_df)
                sig_df  = s.generate_signals(df_prep.copy())
                sig_df  = _apply_h4_filter(sig_df, h4_is)
                _cached  = sig_df.copy()
                def _patched(self2, df2, _c=_cached): return _c
                s.generate_signals = types.MethodType(_patched, s)

            try:
                r = Backtester(strategy=s, df=is_df, instrument="EURUSD",
                               initial_balance=INIT, phase="challenge",
                               seed=combo_idx).run()
                if len(r.trades) < 8:
                    continue
                sh, _ = _risk_adjusted(r.equity_curve)
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_params = params
            except Exception:
                continue

        if best_params is None:
            print(f"  ⚠ no valid IS params"); continue

        # IS re-run
        cfg = {**base_cfg, **best_params}
        cfg["rsi_short_min"] = 100.0 - cfg.get("rsi_long_max", 65)
        cfg["rsi_short_max"] = 100.0 - cfg.get("rsi_long_min", 45)
        s_is = M15MomentumScalping(); s_is.setup(cfg, eu_instr)
        if use_h4:
            bt_is = Backtester.__new__(Backtester)
            bt_is.strategy = s_is; bt_is.instrument = "EURUSD"
            bt_is.initial_balance = INIT; bt_is.phase = "challenge"
            bt_is.rng = np.random.default_rng(0); bt_is.break_even_r = None; bt_is.trail_atr_mult = None
            bt_is._load_configs(None, None)
            df_prep = bt_is._prepare_data(is_df)
            sig_df  = s_is.generate_signals(df_prep.copy())
            sig_df  = _apply_h4_filter(sig_df, h4_is)
            _c = sig_df.copy()
            s_is.generate_signals = types.MethodType(lambda self2, df2: _c, s_is)
        r_is = Backtester(strategy=s_is, df=is_df, instrument="EURUSD",
                          initial_balance=INIT, phase="challenge", seed=0).run()
        is_ret = (r_is.final_balance - INIT) / INIT * 100

        # OOS run
        s_oos = M15MomentumScalping(); s_oos.setup(cfg, eu_instr)
        if use_h4:
            bt_oos = Backtester.__new__(Backtester)
            bt_oos.strategy = s_oos; bt_oos.instrument = "EURUSD"
            bt_oos.initial_balance = INIT; bt_oos.phase = "challenge"
            bt_oos.rng = np.random.default_rng(0); bt_oos.break_even_r = None; bt_oos.trail_atr_mult = None
            bt_oos._load_configs(None, None)
            df_prep_oos = bt_oos._prepare_data(oos_df)
            sig_oos = s_oos.generate_signals(df_prep_oos.copy())
            sig_oos = _apply_h4_filter(sig_oos, h4_oos)
            _co = sig_oos.copy()
            s_oos.generate_signals = types.MethodType(lambda self2, df2: _co, s_oos)
        r_oos = Backtester(strategy=s_oos, df=oos_df, instrument="EURUSD",
                           initial_balance=INIT, phase="challenge", seed=0).run()
        oos_sh, _ = _risk_adjusted(r_oos.equity_curve)
        oos_ret   = (r_oos.final_balance - INIT) / INIT * 100

        deg = oos_sh / best_sharpe if best_sharpe != 0 else float("nan")
        wr = WindowResult(
            window_idx=w_idx, is_start=is_start.strftime("%Y-%m"), is_end=is_end.strftime("%Y-%m"),
            oos_start=oos_start.strftime("%Y-%m"), oos_end=oos_end.strftime("%Y-%m"),
            best_params=best_params, is_sharpe=round(best_sharpe, 4),
            is_return_pct=round(is_ret, 4), is_trades=len(r_is.trades),
            oos_sharpe=round(oos_sh, 4), oos_return_pct=round(oos_ret, 4),
            oos_trades=len(r_oos.trades),
            degradation_ratio=round(deg, 4) if not np.isnan(deg) else float("nan"),
        )
        window_results.append(wr)
        deg_str = f"{deg:.2f}" if not np.isnan(deg) else "N/A"
        print(f"IS Sh={best_sharpe:.2f} OOS Sh={oos_sh:.2f} Deg={deg_str}")

    if not window_results:
        print("\n  WF produced no valid windows.")
        return

    valid_deg = [w.degradation_ratio for w in window_results if not np.isnan(w.degradation_ratio)]
    avg_deg   = float(np.mean(valid_deg)) if valid_deg else float("nan")
    pos_oos   = sum(1 for w in window_results if w.oos_sharpe > 0)
    pct_pos   = 100 * pos_oos / len(window_results) if window_results else 0
    is_robust = (not np.isnan(avg_deg)) and avg_deg > 0.3 and pct_pos >= 40

    wf = WalkForwardResult(
        strategy_name=f"M15_{vname}", instrument="EURUSD",
        windows=window_results, avg_degradation_ratio=avg_deg,
        recommended_params=_recommend_params(window_results),
        is_robust=is_robust, overfitting_warning=not is_robust,
        grid_used=A3_GRID, initial_balance=INIT,
    )
    _print_summary(wf, verbose=True)
    save_result(wf, RESULTS / f"walkforward_m15_{vname}_EURUSD.json")

    if is_robust:
        print(f"\n  → WF PASSES for {vname} (avg_deg={avg_deg:.3f}, pos_oos={pct_pos:.0f}%)")
        print("  → Expanding to 4 instruments with recommended params…")
        _expand_m15_4pairs(vname, wf.recommended_params, sp, ic)
    else:
        print(f"\n  → WF FAILS for {vname} (avg_deg={avg_deg:.3f}, pos_oos={pct_pos:.0f}%)")
        print(f"     M15 Momentum Scalping is DEAD.")


def _expand_m15_4pairs(vname, params, sp, ic):
    """Expand M15 to 4 instruments after WF pass."""
    pairs = ["EURUSD", "GBPUSD", "EURJPY", "USDJPY"]
    base  = sp["m15_momentum_scalping"]
    cfg   = {**base, **params}
    if vname == "V4":
        cfg["ny_session_end"] = "16:00"
    cfg["rsi_short_min"] = 100.0 - cfg.get("rsi_long_max", 65)
    cfg["rsi_short_max"] = 100.0 - cfg.get("rsi_long_min", 45)

    results = {}
    print(f"\n  {'Pair':<8}  {'T':>4}  {'WR%':>5}  {'Ret%':>7}  {'Sh':>6}  {'DD%':>5}")
    for pair in pairs:
        df = _load_m15(pair)
        s  = M15MomentumScalping()
        s.setup(cfg, ic[pair])
        r  = _run_bt(s, df, pair, be_r=None, loose=True)
        results[pair] = r
        with open(RESULTS / f"m15_{vname}_{pair}_backtest.pkl", "wb") as f:
            pickle.dump(r, f)
        m = calculate_metrics(r)
        if "error" not in m:
            print(f"  {pair:<8}  {m['total_trades']:>4}  {m['win_rate_pct']:>5.1f}"
                  f"  {m['total_return_pct']:>+7.2f}%  {m['sharpe_ratio']:>+6.3f}"
                  f"  {m['max_drawdown_pct']:>5.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# PART B — London Risk Uplift
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "S1 (baseline)": {
        "EURUSD": 0.7, "GBPUSD": 0.7, "EURJPY": 0.7,
        "GBPJPY": 0.7, "AUDUSD": 0.5, "USDJPY": 0.7,
        "NY_EURUSD": 0.7,
    },
    "S2 (robust uplift)": {
        "EURUSD": 1.0, "GBPUSD": 1.0, "EURJPY": 1.0,
        "GBPJPY": 0.5, "AUDUSD": 0.5, "USDJPY": 0.5,
        "NY_EURUSD": 0.5,
    },
    "S3 (aggressive)": {
        "EURUSD": 1.2, "GBPUSD": 1.2, "EURJPY": 1.0,
        "GBPJPY": 0.5, "AUDUSD": 0.3, "USDJPY": 0.5,
        "NY_EURUSD": 0.5,
    },
}

# London pair → (strategy_class, config_key, csv_or_break_even)
LONDON_PAIRS = {
    "EURUSD": ("london", "london_open_breakout", {}),
    "GBPUSD": ("london", "london_open_breakout", {}),
    "EURJPY": ("london", "london_open_breakout", {"min_asian_range_pips":40,"max_asian_range_pips":120,"entry_buffer_pips":7}),
    "GBPJPY": ("london", "london_open_breakout", {"min_asian_range_pips":30,"max_asian_range_pips":120,"entry_buffer_pips":7}),
    "AUDUSD": ("london", "london_open_breakout", {"min_asian_range_pips":15,"max_asian_range_pips":80,"entry_buffer_pips":2}),
    "USDJPY": ("london", "london_open_breakout", {"min_asian_range_pips":15,"max_asian_range_pips":80,"entry_buffer_pips":5}),
    "NY_EURUSD": ("ny", "ny_session_breakout", {}),
}


def _build_london_cfg(sp, pair_key, risk_pct, pair_overrides):
    base = sp["london_open_breakout"].copy()
    # Remove instrument_overrides block — we apply manually
    base.pop("instrument_overrides", None)
    cfg  = {**base, **pair_overrides, "risk_per_trade_pct": risk_pct}
    return cfg


def _build_ny_cfg(sp, risk_pct):
    cfg = {**sp["ny_session_breakout"], "risk_per_trade_pct": risk_pct}
    return cfg


def run_part_b(sp, ic):
    _sec("PART B — London Risk Uplift Scenarios")

    # Cache data
    dfs = {}
    for pair in ["EURUSD", "GBPUSD", "EURJPY", "GBPJPY", "AUDUSD", "USDJPY"]:
        dfs[pair] = _load_m15(pair)

    scenario_results = {}

    for sname, risks in SCENARIOS.items():
        print(f"\n  Running {sname}…", flush=True)
        t0 = time_module.time()
        pair_results = {}

        for pair_key, (stype, cfg_key, overrides) in LONDON_PAIRS.items():
            actual_pair = pair_key.replace("NY_", "")
            risk_pct    = risks[pair_key]

            if stype == "london":
                cfg = _build_london_cfg(sp, pair_key, risk_pct, overrides)
                s   = LondonOpenBreakout()
            else:
                cfg = _build_ny_cfg(sp, risk_pct)
                s   = NYSessionBreakout()

            s.setup(cfg, ic[actual_pair])
            r = _run_bt(s, dfs[actual_pair], actual_pair, be_r=1.0, loose=True)
            pair_results[pair_key] = r

        # Combine
        all_res    = list(pair_results.values())
        combined   = _combine_eq(all_res, INIT)
        stats      = _portfolio_stats(combined)
        all_trades = [t for r in all_res for t in r.trades]

        # FTMO official evaluation
        class _FR:
            def __init__(self):
                self.equity_curve=combined; self.trades=all_trades
                self.initial_balance=INIT; self.final_balance=float(combined.iloc[-1])
                self.ftmo_halt_reason=None; self.daily_pnl=pd.Series(dtype=float)

        ev     = FTMOEvaluator(window_months=3)
        ev_res = ev.evaluate_single(_FR(), label=sname)

        scenario_results[sname] = {"results": pair_results, "stats": stats,
                                   "ftmo": ev_res, "elapsed": time_module.time()-t0}

    # ── Comparison table ─────────────────────────────────────────────────────
    print(f"\n  {'Metric':<30}", end="")
    for sname in SCENARIOS:
        print(f"  {sname:<22}", end="")
    print()
    print("  " + "-"*30, end="")
    for _ in SCENARIOS:
        print("  " + "-"*22, end="")
    print()

    metrics_to_show = [
        ("Avg monthly return",  "avg_mo",      "+.3f", "%"),
        ("CAGR",                "cagr",         "+.2f", "%"),
        ("Total return",        "total_ret",    "+.2f", "%"),
        ("Sharpe ratio",        "sharpe",        "+.3f", ""),
        ("Max intraday DD",     "max_dd",         ".2f", "%"),
        ("Worst daily loss",    "worst_day",      ".3f", "%"),
        ("Days to +10%",        "days_to_10",       "s",  ""),
        ("Positive months",     "pos_mo",           "d",  ""),
    ]

    for label, key, fmt, unit in metrics_to_show:
        print(f"  {label:<30}", end="")
        for sname, sr in scenario_results.items():
            v = sr["stats"].get(key)
            if v is None:
                vstr = "—"
            elif isinstance(v, int) and fmt == "d":
                vstr = f"{v}{unit}"
            elif fmt == "s":
                vstr = str(v) + unit if v is not None else "not reached"
            else:
                vstr = f"{v:{fmt.lstrip('+')}}".replace("f", "") + unit
                if "+" in fmt and isinstance(v, (int, float)) and v >= 0:
                    vstr = "+" + vstr
            print(f"  {vstr:<22}", end="")
        print()

    print(f"\n  FTMO breach counts (24 rolling 3-month windows):")
    for sname, sr in scenario_results.items():
        ev_r   = sr["ftmo"]
        total_b = ev_r.daily_loss_breach_count + ev_r.max_dd_breach_count
        print(f"    {sname:<24}: {total_b} total  "
              f"(daily={ev_r.daily_loss_breach_count}  maxDD={ev_r.max_dd_breach_count})  "
              f"pass rate={ev_r.pass_rate_pct:.1f}%")

    # ── Pick best scenario ────────────────────────────────────────────────────
    print(f"\n  Selection criteria: 0 FTMO breaches AND worst_daily_loss > -3.5%")
    best_scenario = None
    best_avg_mo   = -np.inf
    for sname, sr in scenario_results.items():
        ev_r    = sr["ftmo"]
        total_b = ev_r.daily_loss_breach_count + ev_r.max_dd_breach_count
        wd      = sr["stats"].get("worst_day", -99)
        if total_b == 0 and wd > -3.5:
            if sr["stats"].get("avg_mo", -99) > best_avg_mo:
                best_avg_mo   = sr["stats"]["avg_mo"]
                best_scenario = sname

    if best_scenario:
        bs    = scenario_results[best_scenario]
        print(f"\n  ✓ BEST SCENARIO: {best_scenario}")
        print(f"    Avg monthly return: {bs['stats']['avg_mo']:+.3f}%")
        print(f"    CAGR:               {bs['stats']['cagr']:+.2f}%")
        print(f"    Sharpe:             {bs['stats']['sharpe']:+.3f}")
        print(f"    Max DD:             {bs['stats']['max_dd']:.2f}%")
        print(f"    Worst daily loss:   {bs['stats']['worst_day']:.3f}%")
        print(f"    Days to +10%:       {bs['stats']['days_to_10']}")
        # Print per-pair risks for best scenario
        print(f"\n    Optimal risk per trade:")
        for pair_key, risk in SCENARIOS[best_scenario].items():
            print(f"      {pair_key:<12}: {risk:.1f}%")

        # Save best scenario equity for reporting
        best_eq = _combine_eq(list(bs["results"].values()), INIT)
        with open(RESULTS / f"london_best_scenario_equity.pkl", "wb") as f:
            pickle.dump((best_scenario, best_eq, SCENARIOS[best_scenario]), f)
    else:
        print(f"\n  No scenario meets the criteria (0 breaches + worst_day > -3.5%).")
        print(f"  → Stick with S1 baseline risk (0.7% across board).")

    # Monthly return table for best scenario
    if best_scenario:
        eq = _combine_eq(list(scenario_results[best_scenario]["results"].values()), INIT)
        eq_d = eq.resample("D").last().dropna()
        mo_r = eq_d.pct_change().resample("ME").apply(lambda x:(1+x).prod()-1).dropna()*100
        print(f"\n  {best_scenario} Monthly returns (%):")
        mr = mo_r.to_frame("r"); mr["yr"]=mr.index.year; mr["mo"]=mr.index.month
        pv = mr.pivot_table(values="r",index="yr",columns="mo",aggfunc="first")
        MN = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        print("  " + "".join(f"  {m:>5}" for m in (["Year"]+MN+["Ann"])))
        for yr in pv.index:
            row=pv.loc[yr]; cells=""; ann=1.0
            for mo in range(1,13):
                v=row.get(mo,float("nan"))
                if not np.isnan(v): cells+=f"  {v:>+5.1f}"; ann*=(1+v/100)
                else: cells+=f"  {'—':>5}"
            print(f"  {yr}{cells}  {(ann-1)*100:>+6.1f}%")

    return scenario_results, best_scenario


# ─────────────────────────────────────────────────────────────────────────────
# Context Summary
# ─────────────────────────────────────────────────────────────────────────────

def context_summary(part_a_variants, best_a_variant, part_b_results, best_b_scenario):
    _sec("CONTEXT SUMMARY (Prompt 18 deliverables)")

    # 1. Best M15 variant
    best_v = best_a_variant
    if best_v:
        vname, vres, vm = best_v
        print(f"\n  1. Best M15 variant: {vname}  Sharpe={vm['sharpe_ratio']:+.3f}  T/mo={vm['trades_per_month_avg']:.1f}")
    else:
        # Find closest to qualifying
        closest = None
        closest_sh = -np.inf
        for vn, vr in part_a_variants.items():
            vm = calculate_metrics(vr)
            if "error" not in vm and vm["sharpe_ratio"] > closest_sh:
                closest_sh = vm["sharpe_ratio"]
                closest = (vn, vm)
        if closest:
            vn, vm = closest
            print(f"\n  1. Best M15 variant: {vn}  Sharpe={vm['sharpe_ratio']:+.3f}  T/mo={vm['trades_per_month_avg']:.1f}")
            print(f"     ✗ Does NOT qualify for WF (needs Sharpe>0.2 AND T/mo>=10)")
        else:
            print(f"\n  1. Best M15 variant: None qualified")

    # 2. MFE median (from V1)
    r1 = part_a_variants.get("V1")
    if r1 and r1.trades:
        df_eu = _load_m15("EURUSD")
        mfe_list = _mfe_analysis(r1.trades, df_eu)
        mfe_med = np.median(mfe_list)
        print(f"\n  2. MFE median (V1): {mfe_med:.2f}R", end="  ")
        if mfe_med > 1.2:
            print("→ Signals have edge, exit logic is wrong")
        elif mfe_med < 0.8:
            print("→ Signal itself is weak (entry quality problem)")
        else:
            print("→ Borderline (0.8–1.2R)")

    # 3. Pullback clustering
    if r1 and r1.trades:
        df_eu = _load_m15("EURUSD")
        pct_big = _big_move_cooldown_check(r1.trades, df_eu)
        print(f"\n  3. Pullback entries after extended moves: {pct_big*100:.1f}% of losers", end="  ")
        if pct_big > 0.60:
            print("→ COOLDOWN needed")
        else:
            print("→ NOT a dominant pattern")

    # 4. Best London risk scenario
    if best_b_scenario and part_b_results:
        bs = part_b_results[best_b_scenario]
        print(f"\n  4. Best London risk scenario: {best_b_scenario}")
        print(f"     Avg monthly return: {bs['stats']['avg_mo']:+.3f}%")
        print(f"     CAGR: {bs['stats']['cagr']:+.2f}%")
    else:
        print(f"\n  4. Best London risk scenario: S1 baseline (no uplift clears criteria)")

    # 5. M15 WF status
    print(f"\n  5. M15 walk-forward: {'DEAD — no variant qualifies' if not best_v else 'PROCEEDED — see A3 output'}")

    # 6. Combined best-case avg monthly return
    if best_b_scenario and part_b_results:
        avg_mo = part_b_results[best_b_scenario]["stats"]["avg_mo"]
        print(f"\n  6. Combined best-case avg monthly return (London {best_b_scenario}): {avg_mo:+.3f}%")
        print(f"     (+M15 if viable — currently M15 not viable)")
    else:
        s1 = part_b_results.get("S1 (baseline)")
        if s1:
            print(f"\n  6. Combined best-case avg monthly return (London S1): {s1['stats']['avg_mo']:+.3f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    RESULTS.mkdir(exist_ok=True)
    sp, ic = _load_cfg()
    t_start = time_module.time()

    print("Prompt 18: M15 Diagnosis + London Risk Uplift")
    print(f"venv Python {sys.version.split()[0]} | pandas {pd.__version__}")

    # Part A
    part_a_variants = run_part_a(sp, ic)

    # Determine best qualifying variant for context summary
    best_a_v = None
    best_sh = -np.inf
    for vn, vr in part_a_variants.items():
        vm = calculate_metrics(vr)
        if "error" not in vm and vm["sharpe_ratio"] > 0.2 and vm["trades_per_month_avg"] >= 10:
            if vm["sharpe_ratio"] > best_sh:
                best_sh = vm["sharpe_ratio"]
                best_a_v = (vn, vr, vm)

    # Part B
    part_b_results, best_b_scenario = run_part_b(sp, ic)

    # Summary
    context_summary(part_a_variants, best_a_v, part_b_results, best_b_scenario)

    print(f"\n  Total runtime: {(time_module.time()-t_start)/60:.1f} minutes")
    print("\nDone.")


if __name__ == "__main__":
    main()
