"""
Prompt 21: Gold Liquidity Sweep Reversal Strategy

Task 1: Strategy built in strategies/gold_sweep_reversal.py
Task 2: Baseline backtest (Jul 2020 → Apr 2025)
Task 3: Walk-forward validation
Task 4: Combined portfolio + standalone FTMO eval
Task 5: Sensitivity analysis (risk levels for eval)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

print(f"Prompt 21: Gold Liquidity Sweep Reversal")
print(f"venv Python {sys.version.split()[0]} | pandas {pd.__version__}")

DATA_DIR        = Path("data/historical")
INITIAL_BALANCE = 10_000.0
FULL_START      = "2020-01-01"
BASELINE_START  = "2020-07-01"

LOOSE_FTMO = {"safety_buffers": {"daily_loss_trigger_pct": 99.0, "total_loss_trigger_pct": 99.0}}

S3_RISK = {
    "EURUSD": 1.2, "GBPUSD": 1.2, "EURJPY": 1.0,
    "GBPJPY": 0.5, "AUDUSD": 0.3, "USDJPY": 0.5, "NY_EURUSD": 0.5,
}
LONDON_RANGE_OVERRIDES = {
    "EURJPY": {"min_asian_range_pips": 40, "max_asian_range_pips": 120, "entry_buffer_pips": 7},
    "GBPJPY": {"min_asian_range_pips": 30, "max_asian_range_pips": 120, "entry_buffer_pips": 7},
    "AUDUSD": {"min_asian_range_pips": 15, "max_asian_range_pips":  80, "entry_buffer_pips": 2},
    "USDJPY": {"min_asian_range_pips": 15, "max_asian_range_pips":  80, "entry_buffer_pips": 5},
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_m15(pair: str, start: Optional[str] = None) -> pd.DataFrame:
    path = DATA_DIR / f"{pair}_15m.parquet"
    df   = pd.read_parquet(path)
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.columns = [c.lower() for c in df.columns]
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    return df


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------

def _build_sweep(sp: Dict, ic: Dict, overrides: Optional[Dict] = None,
                 tp_mode: str = "A") -> object:
    from strategies.gold_sweep_reversal import GoldSweepReversal
    cfg = dict(sp["gold_sweep_reversal"])
    if overrides:
        cfg.update(overrides)
    cfg["tp_mode"] = tp_mode
    s = GoldSweepReversal()
    s.setup(cfg, ic["XAUUSD"])
    return s


def _build_london(pair: str, sp: Dict, ic: Dict, risk_pct: float) -> object:
    from strategies.london_open_breakout import LondonOpenBreakout
    cfg = dict(sp["london_open_breakout"])
    cfg.pop("instrument_overrides", None)
    cfg.update(LONDON_RANGE_OVERRIDES.get(pair, {}))
    cfg["risk_per_trade_pct"] = risk_pct
    s = LondonOpenBreakout()
    s.setup(cfg, ic[pair])
    return s


def _build_ny(sp: Dict, ic: Dict, risk_pct: float) -> object:
    from strategies.ny_session_breakout import NYSessionBreakout
    cfg = {**sp["ny_session_breakout"], "risk_per_trade_pct": risk_pct}
    s = NYSessionBreakout()
    s.setup(cfg, ic["EURUSD"])
    return s


# ---------------------------------------------------------------------------
# Backtest helper
# ---------------------------------------------------------------------------

def _run(strategy, df: pd.DataFrame, pair: str, override_ftmo=None):
    from backtesting.backtester import Backtester
    kwargs = dict(strategy=strategy, df=df, instrument=pair,
                  initial_balance=INITIAL_BALANCE)
    if override_ftmo:
        kwargs["_override_ftmo_rules"] = override_ftmo
    return Backtester(**kwargs).run()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _combine_equity(results, start: Optional[str] = None) -> pd.Series:
    pnl_frames = []
    for r in results:
        eq  = r.equity_curve.resample("15min").last().ffill()
        pnl = eq.diff().fillna(0.0)
        pnl_frames.append(pnl)
    combined_pnl = pd.concat(pnl_frames, axis=1).fillna(0.0).sum(axis=1)
    eq = INITIAL_BALANCE + combined_pnl.cumsum()
    if start:
        eq = eq[eq.index >= pd.Timestamp(start, tz="UTC")]
    return eq


def _portfolio_stats(equity: pd.Series, label: str = "") -> Dict:
    if len(equity) < 2:
        return {}
    monthly  = equity.resample("ME").last()
    m_rets   = monthly.pct_change().dropna() * 100.0
    avg_mo   = float(m_rets.mean()) if len(m_rets) else 0.0
    pos_mo   = int((m_rets > 0).sum())
    total_ret = (equity.iloc[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0
    years     = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr      = ((equity.iloc[-1] / INITIAL_BALANCE) ** (1 / years) - 1) * 100.0 if years > 0 else 0.0
    peak      = np.maximum.accumulate(equity.values)
    max_dd    = float(abs(((equity.values - peak) / np.where(peak == 0, 1, peak)).min())) * 100.0
    daily_eq  = equity.resample("D").last().dropna()
    worst_day = float(daily_eq.pct_change().dropna().min()) * 100.0
    daily_ret = daily_eq.pct_change().dropna()
    sharpe    = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    target    = INITIAL_BALANCE * 1.10
    above     = equity[equity >= target]
    days_to10 = int((above.index[0] - equity.index[0]).days) if len(above) else -1
    return dict(label=label, avg_monthly=avg_mo, cagr=cagr, total_ret=total_ret,
                sharpe=sharpe, max_dd=max_dd, worst_daily=worst_day,
                days_to10=days_to10, pos_months=pos_mo, n_months=len(m_rets))


def _count_ftmo_breaches(equity: pd.Series) -> Tuple[int, int]:
    class _Mock:
        equity_curve    = equity
        initial_balance = INITIAL_BALANCE
        trades          = []
    try:
        from backtesting.ftmo_evaluator import FTMOEvaluator
        ev  = FTMOEvaluator(window_months=3)
        res = ev.evaluate_single(_Mock(), label="")
        return len(res.windows), res.daily_loss_breach_count + res.max_dd_breach_count
    except Exception:
        return 0, 0


def _monthly_table(equity: pd.Series) -> str:
    monthly = equity.resample("ME").last()
    m_rets  = monthly.pct_change().dropna() * 100.0
    if m_rets.empty:
        return "  (no data)"
    df = pd.DataFrame({"year": m_rets.index.year, "month": m_rets.index.month, "r": m_rets.values})
    pivot = df.pivot_table(values="r", index="year", columns="month", aggfunc="first")
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    hdr = f"  {'Year':>5}" + "".join(f"  {names[m]:>5}" for m in sorted(pivot.columns)) + "     Ann%"
    lines = [hdr]
    for yr in sorted(pivot.index):
        row = f"  {yr:>5}"
        ann = 0.0
        for m in sorted(pivot.columns):
            v = pivot.loc[yr, m] if m in pivot.columns else float("nan")
            if not np.isnan(v):
                row += f"  {v:+5.1f}"; ann += v
            else:
                row += f"    {'—':>3}"
        lines.append(row + f"  {ann:+6.1f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TASK 2: Baseline backtest
# ---------------------------------------------------------------------------

def run_task2(sp: Dict, ic: Dict) -> Tuple[object, object, object]:
    df = _load_m15("XAUUSD", BASELINE_START)
    from backtesting.metrics import calculate_metrics
    from strategies.gold_sweep_reversal import GoldSweepReversal

    # Check volume data quality
    vol_arr = df["volume"].to_numpy()
    vol_pct = (vol_arr > 0).sum() / len(vol_arr) * 100.0
    print(f"  Volume data: {vol_pct:.1f}% non-zero bars", flush=True)
    if vol_pct < 10:
        print("  ⚠ Volume mostly zero — volume filter will be disabled automatically")

    print("  Running TP-A (fixed 2.5:1 RR)…", flush=True)
    s_a = _build_sweep(sp, ic, tp_mode="A")
    r_a = _run(s_a, df, "XAUUSD", LOOSE_FTMO)

    print("  Running TP-B (opposing level)…", flush=True)
    s_b = _build_sweep(sp, ic, tp_mode="B")
    r_b = _run(s_b, df, "XAUUSD", LOOSE_FTMO)

    def _report(r, label, strat=None):
        if not r.trades:
            print(f"\n  {label}: NO TRADES")
            return None, 0.0, 0.0
        m      = calculate_metrics(r)
        tpm    = m.get("trades_per_month_avg", 0.0)
        sh     = m.get("sharpe_ratio", 0.0)
        n_w, n_b = _count_ftmo_breaches(r.equity_curve)
        # Avg SL and trade duration from signal df
        avg_sl_pips = 0.0
        avg_dur_h   = 0.0
        if strat is not None:
            try:
                sig_df = strat.generate_signals(df.copy())
                slp = sig_df.loc[sig_df["signal"] != 0, "sl_pips"]
                if len(slp) > 0:
                    avg_sl_pips = float(slp.mean())
            except Exception:
                pass
        if r.trades:
            durs = []
            for t in r.trades:
                try:
                    d = (t.exit_time - t.entry_time).total_seconds() / 3600
                    durs.append(d)
                except Exception:
                    pass
            avg_dur_h = float(np.mean(durs)) if durs else 0.0

        print(f"\n  {label}:")
        print(f"    Trades={m['total_trades']}  T/mo={tpm:.1f}  WR={m['win_rate_pct']:.1f}%  PF={m['profit_factor']:.3f}")
        print(f"    AvgW=${m['avg_win_dollars']:.0f}  AvgL=${m['avg_loss_dollars']:.0f}"
              f"  Return={m['total_return_pct']:+.2f}%  Sharpe={sh:+.3f}  DD={m['max_drawdown_pct']:.2f}%")
        if avg_sl_pips > 0:
            print(f"    Avg SL: {avg_sl_pips:.0f} pips  Avg duration: {avg_dur_h:.1f}h")
        print(f"    FTMO breaches: {n_b} / {n_w} windows")
        return m, tpm, sh

    _report(r_a, "TP-A (fixed 2.5:1 RR)", s_a)
    _report(r_b, "TP-B (opposing level)", s_b)

    # ── Signal diagnostic: level type breakdown ──────────────────────
    print("\n  Level-type analysis (TP-A):")
    if r_a.trades:
        sig_df_a = s_a.generate_signals(df.copy())
        trade_bars = sig_df_a[sig_df_a["signal"] != 0]

        # Count by level type
        ltype_counts = trade_bars["level_type"].value_counts()
        # Win rate by level type (match with trade pnl using entry bar index)
        entry_times = {t.entry_time for t in r_a.trades}
        trade_results: Dict[str, List[bool]] = {}
        for t in r_a.trades:
            lt = ""
            # Find the signal bar near entry time
            near = sig_df_a.index[
                (sig_df_a.index >= t.entry_time - pd.Timedelta("15min")) &
                (sig_df_a.index <= t.entry_time + pd.Timedelta("15min")) &
                (sig_df_a["signal"] != 0)
            ]
            if len(near) > 0:
                lt = str(sig_df_a.loc[near[0], "level_type"])
            trade_results.setdefault(lt, []).append(t.pnl_dollars > 0)
        for ltype in ["PDH", "PDL", "ASH", "ASL", "LSH", "LSL", "NSH", "NSL"]:
            wins = trade_results.get(ltype, [])
            cnt  = ltype_counts.get(ltype, 0)
            if cnt > 0 or wins:
                wr = sum(wins) / len(wins) * 100 if wins else 0.0
                print(f"    {ltype}: signals={cnt}  WR={wr:.0f}%")

    # ── Sweep rejection stats ───────────────────────────────────────
    print("\n  Sweep detection analysis (TP-A):")
    sig_df_a2 = s_a.generate_signals(df.copy())
    n_signals  = int((sig_df_a2["signal"] != 0).sum())
    n_sweeps   = int(sig_df_a2["sweep_pips"].notna().sum())

    # Re-run without displacement filter to count displacement rejections
    s_nodis = _build_sweep(sp, ic, overrides={"candle_body_atr_min": 0.0})
    r_nodis = _run(s_nodis, df, "XAUUSD", LOOSE_FTMO)
    n_nodis = len(r_nodis.trades) if r_nodis.trades else 0

    # Re-run without volume filter
    s_novol = _build_sweep(sp, ic, overrides={"volume_mult": 1.0})
    r_novol = _run(s_novol, df, "XAUUSD", LOOSE_FTMO)
    n_novol = len(r_novol.trades) if r_novol.trades else 0

    print(f"    Accepted signals: {n_signals}")
    print(f"    Without displacement filter: {n_nodis} (+{n_nodis - n_signals} recovered)")
    print(f"    Without volume filter: {n_novol} (+{n_novol - n_signals} recovered)")

    # ── Viability check ─────────────────────────────────────────────
    ma = calculate_metrics(r_a) if r_a.trades else {}
    tpm_a = ma.get("trades_per_month_avg", 0.0)
    sh_a  = ma.get("sharpe_ratio", 0.0)
    viable = tpm_a >= 4 and sh_a > 0.0

    print(f"\n  Viability check (T/mo≥4, Sharpe>0):")
    print(f"    T/mo={tpm_a:.1f} {'✓' if tpm_a>=4 else '✗'}  Sharpe={sh_a:+.3f} {'✓' if sh_a>0 else '✗'}")

    # ── Relaxation tests if T/mo < 4 ────────────────────────────────
    if tpm_a < 4:
        print(f"\n  T/mo={tpm_a:.1f} < 4 — testing relaxations:")
        relaxations = [
            ("1. No displacement filter",   {"candle_body_atr_min": 0.0}),
            ("2. No volume filter",          {"volume_mult": 1.0}),
            ("3. Sweep threshold 10 pips",   {"sweep_min_pips": 10.0}),
            ("4. All three combined",        {"candle_body_atr_min": 0.0, "volume_mult": 1.0, "sweep_min_pips": 10.0}),
        ]
        for label, ovr in relaxations:
            s_r = _build_sweep(sp, ic, overrides=ovr)
            r_r = _run(s_r, df, "XAUUSD", LOOSE_FTMO)
            if r_r.trades:
                m_r   = calculate_metrics(r_r)
                tpm_r = m_r.get("trades_per_month_avg", 0.0)
                sh_r  = m_r.get("sharpe_ratio", 0.0)
                print(f"    {label}: T/mo={tpm_r:.1f}  Sharpe={sh_r:+.3f}  PF={m_r['profit_factor']:.3f}")
            else:
                print(f"    {label}: no trades")

    # ── Monthly returns ─────────────────────────────────────────────
    print(f"\n  TP-A Monthly returns:")
    print(_monthly_table(r_a.equity_curve))

    return r_a, r_b, s_a


# ---------------------------------------------------------------------------
# TASK 3: Walk-forward
# ---------------------------------------------------------------------------

def run_task3(sp: Dict, ic: Dict) -> Tuple[object, Optional[Dict]]:
    from backtesting.walk_forward import run_walk_forward

    df = _load_m15("XAUUSD", BASELINE_START)

    # Grid: 3×3×3×3×3×3×2×3 = 1458 — run full
    print("  WF: gold_sweep_reversal  |  256 combos × 6m/3m/3m", flush=True)
    print("  (min 5 trades per IS window)")

    try:
        wf = run_walk_forward(
            df=df,
            strategy_name="gold_sweep_reversal",
            instrument="XAUUSD",
            initial_balance=INITIAL_BALANCE,
            in_sample_months=6,
            oos_months=3,
            step_months=3,
            min_trades=5,
            verbose=True,
        )
    except RuntimeError as exc:
        print(f"  WF error: {exc}")
        return None, None

    n_w    = len(wf.windows)
    n_pos  = sum(1 for w in wf.windows if w.oos_sharpe > 0)
    avg_d  = wf.avg_degradation_ratio
    pos_fr = n_pos / n_w if n_w else 0.0
    passes = (not np.isnan(avg_d)) and avg_d > 0.2 and pos_fr > 0.40

    print(f"\n  WF summary: avg_deg={avg_d:.3f}  pos_OOS={n_pos}/{n_w} ({pos_fr:.0%})")
    print(f"  WF result: {'PASS ✓' if passes else 'FAIL ✗'}")
    if passes:
        print(f"  Recommended params: {wf.recommended_params}")

    return wf, wf.recommended_params if passes else None


# ---------------------------------------------------------------------------
# TASK 4: Combined + standalone FTMO eval
# ---------------------------------------------------------------------------

def run_task4(wf_best: Optional[Dict], sp: Dict, ic: Dict) -> Tuple[Dict, Dict]:
    """Returns (gold_stats, combined_stats) dicts."""
    from backtesting.metrics import calculate_metrics

    print("  Building London S3 + Gold sweep portfolio…", flush=True)

    london_pairs = ["EURUSD", "GBPUSD", "EURJPY", "GBPJPY", "AUDUSD", "USDJPY"]
    london_results = []
    for pair in london_pairs:
        df_p = _load_m15(pair, FULL_START)
        london_results.append(_run(_build_london(pair, sp, ic, S3_RISK[pair]), df_p, pair))

    df_ny = _load_m15("EURUSD", FULL_START)
    london_results.append(_run(_build_ny(sp, ic, S3_RISK["NY_EURUSD"]), df_ny, "EURUSD"))

    # Gold sweep (WF-best or default)
    df_g   = _load_m15("XAUUSD", BASELINE_START)
    s_gold = _build_sweep(sp, ic, overrides=wf_best)
    r_gold = _run(s_gold, df_g, "XAUUSD")

    # Gold standalone equity (normalised to INITIAL_BALANCE)
    gold_eq = r_gold.equity_curve
    gold_eq = INITIAL_BALANCE + (gold_eq - r_gold.initial_balance)

    # Combined
    all_results = london_results + [r_gold]
    combined_eq  = _combine_equity(all_results, start=BASELINE_START)

    # Stats
    gold_stats     = _portfolio_stats(gold_eq,    "Gold standalone")
    combined_stats = _portfolio_stats(combined_eq, "London S3 + Gold sweep")

    n_w_g, n_b_g = _count_ftmo_breaches(gold_eq)
    n_w_c, n_b_c = _count_ftmo_breaches(combined_eq)

    def _print_stats(s: Dict, n_w: int, n_b: int) -> None:
        print(f"    Avg monthly:    {s.get('avg_monthly', 0):+.3f}%")
        print(f"    CAGR:           {s.get('cagr', 0):+.2f}%")
        print(f"    Sharpe:         {s.get('sharpe', 0):+.3f}")
        print(f"    Max DD:         {s.get('max_dd', 0):.2f}%")
        print(f"    Worst daily:    {s.get('worst_daily', 0):.3f}%")
        d10 = s.get('days_to10', -1)
        print(f"    Days to +10%:   {d10 if d10 > 0 else 'not reached'}")
        print(f"    Pos months:     {s.get('pos_months',0)}/{s.get('n_months',0)}")
        print(f"    FTMO breaches:  {n_b}/{n_w} windows")

    print(f"\n  Combined (London S3 + Gold sweep):")
    _print_stats(combined_stats, n_w_c, n_b_c)

    print(f"\n  Gold standalone:")
    _print_stats(gold_stats, n_w_g, n_b_g)

    # ── 2-week performance windows ────────────────────────────────────
    print(f"\n  15-trading-day window analysis (gold standalone):")
    _eval_windows(gold_eq, label="Gold standalone")

    print(f"\n  15-trading-day window analysis (combined):")
    _eval_windows(combined_eq, label="Combined")

    # ── Monthly tables ─────────────────────────────────────────────
    print(f"\n  Gold Monthly Returns:")
    print(_monthly_table(gold_eq))

    print(f"\n  Combined Monthly Returns:")
    print(_monthly_table(combined_eq))

    return gold_stats, combined_stats, gold_eq, combined_eq


def _eval_windows(equity: pd.Series, label: str = "") -> None:
    """Slide 15-trading-day windows and compute eval statistics."""
    # Build daily equity
    daily = equity.resample("D").last().dropna()
    # Remove weekends
    daily = daily[daily.index.dayofweek < 5]

    n = len(daily)
    W = 15   # 15 trading days

    if n < W:
        print(f"    {label}: insufficient data ({n} trading days)")
        return

    target_pct    = 10.0
    daily_limit   = -5.0
    total_limit   = -10.0

    pass_count  = 0
    fail_count  = 0
    total_count = 0
    returns     = []

    for i in range(n - W + 1):
        window = daily.iloc[i : i + W]
        start_val = window.iloc[0]
        if start_val <= 0:
            continue

        # Compute returns relative to start of window (re-based)
        window_eq = INITIAL_BALANCE * (window / start_val)
        ret = (window.iloc[-1] - start_val) / start_val * 100.0
        returns.append(ret)
        total_count += 1

        # FTMO pass: final equity +10% above start
        passed = ret >= target_pct

        # FTMO fail: any day drops below -5% daily OR total -10%
        daily_rets = window.pct_change().dropna() * 100.0
        worst_d    = float(daily_rets.min()) if len(daily_rets) > 0 else 0.0
        max_dd_w   = float(((window / window.cummax()) - 1).min()) * 100.0
        failed     = (worst_d <= daily_limit) or (max_dd_w <= total_limit)

        if passed:
            pass_count += 1
        if failed:
            fail_count += 1

    if total_count == 0:
        print(f"    No valid windows")
        return

    returns_arr = np.array(returns)
    print(f"    Total 15-day windows: {total_count}")
    print(f"    Pass rate (≥+10%):    {pass_count}/{total_count} = {100*pass_count/total_count:.1f}%")
    print(f"    Fail rate (daily≤-5% or DD≤-10%): {fail_count}/{total_count} = {100*fail_count/total_count:.1f}%")
    print(f"    Median window return: {np.median(returns_arr):+.2f}%")
    print(f"    Best 15-day:  {np.max(returns_arr):+.2f}%")
    print(f"    Worst 15-day: {np.min(returns_arr):+.2f}%")


# ---------------------------------------------------------------------------
# TASK 4C: Focused eval simulation (if gold standalone > 3%/mo avg)
# ---------------------------------------------------------------------------

def run_task4c(gold_eq: pd.Series, gold_stats: Dict) -> None:
    avg_mo = gold_stats.get("avg_monthly", 0.0)
    if avg_mo <= 3.0:
        print(f"  Gold avg monthly = {avg_mo:+.2f}% < 3% — full eval sim skipped")
        print(f"  (Task 4C threshold not met)")
        return

    print(f"  Gold avg monthly = {avg_mo:+.2f}% > 3% — running full eval simulation…")
    _eval_windows(gold_eq, label="Gold standalone (full sim)")


# ---------------------------------------------------------------------------
# TASK 5: Sensitivity analysis
# ---------------------------------------------------------------------------

def run_task5(wf_best: Optional[Dict], sp: Dict, ic: Dict) -> None:
    from backtesting.metrics import calculate_metrics
    df = _load_m15("XAUUSD", BASELINE_START)

    risk_levels = [1.0, 1.5, 2.0]
    print(f"\n  Risk level sensitivity (gold standalone):")
    print(f"  {'Risk%':>6}  {'Avg Mo%':>8}  {'Sharpe':>7}  {'MaxDD%':>7}  {'WrstDay%':>9}  {'15d Pass%':>10}  {'15d Fail%':>10}")

    best_pass_rate = 0.0
    best_risk = 1.0

    for risk_pct in risk_levels:
        ovr = {**(wf_best or {}), "risk_per_trade_pct": risk_pct}
        s   = _build_sweep(sp, ic, overrides=ovr)
        r   = _run(s, df, "XAUUSD")

        if not r.trades:
            print(f"  {risk_pct:>6.1f}%  no trades")
            continue

        eq = r.equity_curve
        eq = INITIAL_BALANCE + (eq - r.initial_balance)
        st = _portfolio_stats(eq)
        m  = calculate_metrics(r)

        # 15-day pass/fail
        daily = eq.resample("D").last().dropna()
        daily = daily[daily.index.dayofweek < 5]
        n = len(daily); W = 15
        pass_c = 0; fail_c = 0; tot_c = 0
        for i in range(n - W + 1):
            win = daily.iloc[i:i+W]
            sv  = win.iloc[0]
            if sv <= 0: continue
            ret = (win.iloc[-1] - sv) / sv * 100.0
            dr  = win.pct_change().dropna() * 100.0
            wd  = float(dr.min()) if len(dr) > 0 else 0.0
            mdd = float(((win / win.cummax()) - 1).min()) * 100.0
            tot_c += 1
            if ret >= 10.0: pass_c += 1
            if wd <= -5.0 or mdd <= -10.0: fail_c += 1

        pass_rate = 100 * pass_c / tot_c if tot_c > 0 else 0.0
        fail_rate = 100 * fail_c / tot_c if tot_c > 0 else 0.0
        worst_d   = st.get("worst_daily", 0.0)

        print(f"  {risk_pct:>6.1f}%  {st['avg_monthly']:>+8.3f}%  "
              f"{st['sharpe']:>+7.3f}  {st['max_dd']:>7.2f}%  "
              f"{worst_d:>+9.3f}%  {pass_rate:>10.1f}%  {fail_rate:>10.1f}%")

        if pass_rate > best_pass_rate and worst_d > -3.5:
            best_pass_rate = pass_rate
            best_risk = risk_pct

    print(f"\n  Best risk level (max pass rate, worst_daily > -3.5%): {best_risk:.1f}%")


# ---------------------------------------------------------------------------
# CONTEXT SUMMARY
# ---------------------------------------------------------------------------

def context_summary(r_a, r_b, wf, wf_params, gold_stats, combined_stats,
                    gold_eq, sp: Dict) -> None:
    from backtesting.metrics import calculate_metrics

    print()

    if r_a and r_a.trades:
        m = calculate_metrics(r_a)
        print(f"  1. Trades/month: TP-A={m['trades_per_month_avg']:.1f}  Sharpe={m['sharpe_ratio']:+.3f}")
    else:
        print("  1. TP-A: NO TRADES")

    if r_b and r_b.trades:
        m = calculate_metrics(r_b)
        print(f"     TP-B: T/mo={m['trades_per_month_avg']:.1f}  Sharpe={m['sharpe_ratio']:+.3f}")

    if r_a and r_a.trades:
        # Level type win rates
        df_base = _load_m15("XAUUSD", BASELINE_START)
        from strategies.gold_sweep_reversal import GoldSweepReversal
        cfg = dict(sp["gold_sweep_reversal"])
        s = GoldSweepReversal(); s.setup(cfg, {"pip_size": 0.01})
        sig_df = s.generate_signals(df_base.copy())
        sig_bars = sig_df[sig_df["signal"] != 0]
        from collections import Counter
        lt_counts = Counter(sig_bars["level_type"])
        best_lt = lt_counts.most_common(1)[0][0] if lt_counts else "N/A"
        print(f"  2. Most frequent level swept: {best_lt}  distribution={dict(lt_counts)}")

    if r_a and r_a.trades:
        sig_df2 = s.generate_signals(df_base.copy())
        # volume filter impact: compare with/without
        df_base2 = _load_m15("XAUUSD", BASELINE_START)
        from strategies.gold_sweep_reversal import GoldSweepReversal as GSR
        cfg2 = dict(sp["gold_sweep_reversal"]); cfg2["volume_mult"] = 1.0
        s2 = GSR(); s2.setup(cfg2, {"pip_size": 0.01})
        sig_novol = s2.generate_signals(df_base2.copy())
        n_with = (sig_df2["signal"] != 0).sum()
        n_without = (sig_novol["signal"] != 0).sum()
        print(f"  3. Volume filter: {n_with} signals with vs {n_without} without → "
              f"removes {n_without - n_with} trades")

    if wf is None:
        print("  4. WF: NOT RUN")
    else:
        n_w = len(wf.windows)
        n_pos = sum(1 for w in wf.windows if w.oos_sharpe > 0)
        passed = wf_params is not None
        print(f"  4. WF: avg_deg={wf.avg_degradation_ratio:.3f}  {n_pos}/{n_w} pos OOS  "
              f"→ {'PASS ✓' if passed else 'FAIL ✗'}")

    if gold_stats:
        print(f"  5. Gold standalone avg monthly: {gold_stats.get('avg_monthly', 0):+.3f}%  "
              f"Sharpe={gold_stats.get('sharpe', 0):+.3f}")

    # 15-day eval windows from gold_eq
    if gold_eq is not None:
        daily = gold_eq.resample("D").last().dropna()
        daily = daily[daily.index.dayofweek < 5]
        n = len(daily); W = 15; pass_c = 0; fail_c = 0; tot_c = 0
        for i in range(n - W + 1):
            win = daily.iloc[i:i+W]
            sv  = win.iloc[0]
            if sv <= 0: continue
            ret = (win.iloc[-1] - sv) / sv * 100.0
            dr  = win.pct_change().dropna() * 100.0
            wd  = float(dr.min()) if len(dr) > 0 else 0.0
            mdd = float(((win / win.cummax()) - 1).min()) * 100.0
            tot_c += 1
            if ret >= 10.0: pass_c += 1
            if wd <= -5.0 or mdd <= -10.0: fail_c += 1
        pass_rate = 100 * pass_c / tot_c if tot_c > 0 else 0.0
        fail_rate = 100 * fail_c / tot_c if tot_c > 0 else 0.0
        print(f"  6. 15-day eval PASS rate (≥+10%): {pass_c}/{tot_c} = {pass_rate:.1f}%")
        print(f"  7. 15-day eval FAIL rate: {fail_c}/{tot_c} = {fail_rate:.1f}%")

    print(f"  8. Recommended risk: see Task 5 sensitivity table")

    # Verdict
    avg_mo = gold_stats.get("avg_monthly", 0.0) if gold_stats else 0.0
    pass_r = pass_rate if gold_eq is not None else 0.0
    ftmo_b = 0
    if gold_stats:
        try:
            class _Mock:
                equity_curve    = gold_eq
                initial_balance = INITIAL_BALANCE
                trades          = []
            from backtesting.ftmo_evaluator import FTMOEvaluator
            ev  = FTMOEvaluator(window_months=3)
            res = ev.evaluate_single(_Mock(), label="")
            ftmo_b = res.daily_loss_breach_count + res.max_dd_breach_count
        except Exception:
            pass

    print(f"\n  9. Verdict:")
    # Criteria for ">30% probability across 2 attempts":
    # Single attempt pass rate > ~17% → 2-attempt rate = 1 - (1-0.17)² ≈ 30%
    two_attempt_prob = (1 - (1 - pass_r / 100) ** 2) * 100 if pass_r > 0 else 0.0
    passes_verdict = avg_mo > 2.0 and pass_r > 5.0 and ftmo_b == 0
    print(f"    Gold avg monthly: {avg_mo:+.2f}%  FTMO breaches: {ftmo_b}")
    print(f"    Single-attempt eval pass rate: {pass_r:.1f}%")
    print(f"    Two-attempt probability: {two_attempt_prob:.1f}%")
    if passes_verdict:
        print(f"    → YES — system can likely pass FTMO in 2-3 weeks (>30% over 2 attempts)")
    else:
        print(f"    → NO — insufficient edge for reliable FTMO pass in 2-3 week timeframe")
        if avg_mo <= 2.0:
            print(f"      (avg monthly return too low: {avg_mo:+.2f}% — need >2%)")
        if pass_r <= 5.0:
            print(f"      (15-day pass rate too low: {pass_r:.1f}% — need >5% per attempt)")
        if ftmo_b > 0:
            print(f"      ({ftmo_b} FTMO breaches — strategy violates risk rules)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import time as _time
    t0 = _time.time()

    sp = json.loads(Path("config/strategy_params.json").read_text())
    ic = json.loads(Path("config/instruments.json").read_text())

    print("\n" + "=" * 72)
    print("  TASK 2 — Baseline Backtest (Jul 2020 → Apr 2025)")
    print("=" * 72)
    r_a, r_b, s_a = run_task2(sp, ic)

    print("\n" + "=" * 72)
    print("  TASK 3 — Walk-Forward Validation")
    print("=" * 72)
    wf, wf_params = run_task3(sp, ic)

    print("\n" + "=" * 72)
    print("  TASK 4 — Combined Portfolio + Standalone FTMO Eval")
    print("=" * 72)
    gold_stats, combined_stats, gold_eq, combined_eq = run_task4(wf_params, sp, ic)

    print("\n" + "=" * 72)
    print("  TASK 4C — Focused Eval Simulation")
    print("=" * 72)
    run_task4c(gold_eq, gold_stats)

    print("\n" + "=" * 72)
    print("  TASK 5 — Risk Sensitivity Analysis")
    print("=" * 72)
    run_task5(wf_params, sp, ic)

    print("\n" + "=" * 72)
    print("  CONTEXT SUMMARY")
    print("=" * 72)
    context_summary(r_a, r_b, wf, wf_params, gold_stats, combined_stats, gold_eq, sp)

    elapsed = _time.time() - t0
    print(f"\n  Total runtime: {elapsed / 60:.1f} minutes")
    print("\nDone.")


if __name__ == "__main__":
    main()
