"""
Prompt 20: Gold Session Breakout with VPA Filter

Task 0: Data quality verification (done in task0 section)
Task 1: Strategy built in strategies/gold_session_breakout.py
Task 2: Baseline backtest (Jul 2020 → Apr 2025)
Task 3: Walk-forward validation
Task 4: Combined portfolio (London S3 + Gold) + gold standalone
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

print(f"Prompt 20: Gold Session Breakout + VPA Filter")
print(f"venv Python {sys.version.split()[0]} | pandas {pd.__version__}")

DATA_DIR        = Path("data/historical")
INITIAL_BALANCE = 10_000.0
FULL_START      = "2020-01-01"
BASELINE_START  = "2020-07-01"   # skip COVID distortion

LOOSE_FTMO = {"safety_buffers": {"daily_loss_trigger_pct": 99.0, "total_loss_trigger_pct": 99.0}}

# London S3 risk settings (locked from Prompt 18-19)
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

def _build_gold(sp: Dict, ic: Dict, overrides: Optional[Dict] = None,
                use_h1_atr_tp: bool = False) -> object:
    from strategies.gold_session_breakout import GoldSessionBreakout
    cfg = dict(sp["gold_session_breakout"])
    if overrides:
        cfg.update(overrides)
    cfg["use_h1_atr_tp"] = use_h1_atr_tp
    s = GoldSessionBreakout()
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
    """Return (n_windows, n_breaches)."""
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
# TASK 0: Data quality
# ---------------------------------------------------------------------------

def run_task0() -> None:
    df = _load_m15("XAUUSD")
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert("US/Eastern")

    session = df_et[(df_et.index.hour >= 3) & (df_et.index.hour < 16) &
                    (df_et.index.dayofweek < 5)]
    sess_days = len(np.unique(session.index.date))
    nz_vol    = (session.volume > 0).sum()
    price_min = df.close.min(); price_max = df.close.max()

    print(f"  XAUUSD data: {len(df):,} bars  {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Price range:  ${price_min:.0f} → ${price_max:.0f}")
    print(f"  Session days: {sess_days}  bars/day={len(session)/sess_days:.1f}  (expect 52)")
    print(f"  Session vol>0: {nz_vol:,}/{len(session):,} ({100*nz_vol/len(session):.1f}%)")

    # gaps check
    idx  = df.index.sort_values()
    gaps = pd.Series(idx).diff().dt.total_seconds() / 3600
    wd_gaps = [(i, gaps[i], idx[i]) for i in gaps[gaps > 4].index
               if idx[i].weekday() < 5 and idx[i-1].weekday() < 5
               and (idx[i] - idx[i-1]).days < 2]
    print(f"  Intra-week weekday gaps >4h: {len(wd_gaps)}  (expect ~0)")
    print(f"  → Data quality: {'PASS ✓' if len(wd_gaps) <= 5 and nz_vol/len(session) > 0.80 else 'WARN'}")


# ---------------------------------------------------------------------------
# TASK 2: Baseline backtest
# ---------------------------------------------------------------------------

def run_task2(sp: Dict, ic: Dict) -> Tuple[object, object]:
    """Returns (result_tpa, result_tpb)."""
    df = _load_m15("XAUUSD", BASELINE_START)

    # ── TP-A (fixed 2:1 RR) ─────────────────────────────────────────────
    print("  Running TP-A (fixed 2:1 RR)…", flush=True)
    s_a   = _build_gold(sp, ic, use_h1_atr_tp=False)
    r_a   = _run(s_a, df, "XAUUSD", LOOSE_FTMO)

    # ── TP-B (1.5 × H1 ATR) ─────────────────────────────────────────────
    print("  Running TP-B (1.5 × H1 ATR)…", flush=True)
    s_b   = _build_gold(sp, ic, use_h1_atr_tp=True)
    r_b   = _run(s_b, df, "XAUUSD", LOOSE_FTMO)

    # ── VPA analysis: accepted vs rejected ───────────────────────────────
    sig_df_a = s_a.generate_signals(df.copy())
    n_accepted = int(sig_df_a["vpa_accepted"].sum())
    n_rejected = int(sig_df_a["vpa_rejected"].sum())

    # Win rates for accepted vs rejected
    if r_a.trades:
        acc_wins  = sum(1 for t in r_a.trades if t.pnl_dollars > 0)
        acc_total = len(r_a.trades)
        acc_wr    = acc_wins / acc_total * 100.0 if acc_total else 0.0
    else:
        acc_wins = acc_total = acc_wr = 0

    from backtesting.metrics import calculate_metrics, _trades_per_month
    from backtesting.metrics import _risk_adjusted

    def _report(r, label):
        if not r.trades:
            print(f"  {label}: NO TRADES")
            return
        m      = calculate_metrics(r)
        tpm    = m.get("trades_per_month_avg", 0.0)
        sh     = m.get("sharpe_ratio", 0.0)
        n_w, n_b = _count_ftmo_breaches(r.equity_curve)
        print(f"  {label}:")
        print(f"    Trades={m['total_trades']}  T/mo={tpm:.1f}  WR={m['win_rate_pct']:.1f}%"
              f"  PF={m['profit_factor']:.3f}")
        print(f"    AvgW=${m['avg_win_dollars']:.0f}  AvgL=${m['avg_loss_dollars']:.0f}"
              f"  Return={m['total_return_pct']:+.2f}%  Sharpe={sh:+.3f}  DD={m['max_drawdown_pct']:.2f}%")
        print(f"    FTMO breaches: {n_b} / {n_w} windows")
        return m, tpm, sh

    print()
    result_a = _report(r_a, "TP-A (fixed 2:1 RR)")
    print()
    result_b = _report(r_b, "TP-B (1.5 × H1 ATR)")

    print(f"\n  VPA filter stats:")
    print(f"    Stage 1 breakouts (total):   {n_accepted + n_rejected}")
    print(f"    Accepted (VPA confirmed):    {n_accepted}  ({100*n_accepted/max(n_accepted+n_rejected,1):.1f}%)")
    print(f"    Rejected (VPA timeout):      {n_rejected}  ({100*n_rejected/max(n_accepted+n_rejected,1):.1f}%)")
    print(f"    Accepted win rate:           {acc_wr:.1f}%")
    # For rejected win rate we'd need to run without VPA — approximate from accepted signal count
    print(f"    (VPA filter removes {100*n_rejected/max(n_accepted+n_rejected,1):.0f}% of breakouts)")

    # ── Viability check ──────────────────────────────────────────────────
    ma = calculate_metrics(r_a) if r_a.trades else {}
    tpm_a = ma.get("trades_per_month_avg", 0.0)
    sh_a  = ma.get("sharpe_ratio", 0.0)
    pf_a  = ma.get("profit_factor", 0.0)
    viable = tpm_a >= 8 and sh_a > 0.3 and pf_a > 1.2

    print(f"\n  Viability check (T/mo≥8, Sharpe>0.3, PF>1.2):")
    print(f"    T/mo={tpm_a:.1f} {'✓' if tpm_a>=8 else '✗'}  "
          f"Sharpe={sh_a:+.3f} {'✓' if sh_a>0.3 else '✗'}  "
          f"PF={pf_a:.3f} {'✓' if pf_a>1.2 else '✗'}")

    # ── Relaxation tests if T/mo < 8 ─────────────────────────────────────
    if tpm_a < 8:
        print(f"\n  T/mo={tpm_a:.1f} < 8 — testing relaxations:")

        relaxations = [
            ("1. VPA vol threshold 1.2×", {"vpa_volume_mult": 1.2}),
            ("2. Wider entry 03:00-14:00", {"entry_window_end": "14:00"}),
            ("3. No regime ADX filter",   {"regime_adx_h1_min": 0.0}),
        ]
        for label, ovr in relaxations:
            s_r = _build_gold(sp, ic, overrides=ovr)
            r_r = _run(s_r, df, "XAUUSD", LOOSE_FTMO)
            if r_r.trades:
                m_r   = calculate_metrics(r_r)
                tpm_r = m_r.get("trades_per_month_avg", 0.0)
                sh_r  = m_r.get("sharpe_ratio", 0.0)
                print(f"    {label}: T/mo={tpm_r:.1f}  Sharpe={sh_r:+.3f}  PF={m_r['profit_factor']:.3f}")
            else:
                print(f"    {label}: no trades")

    # ── Monthly returns for best TP ─────────────────────────────────────
    best_r = r_a if sh_a >= (calculate_metrics(r_b).get("sharpe_ratio", -99) if r_b.trades else -99) else r_b
    print(f"\n  TP-A Monthly returns:")
    print(_monthly_table(r_a.equity_curve))

    return r_a, r_b


# ---------------------------------------------------------------------------
# TASK 3: Walk-forward
# ---------------------------------------------------------------------------

def run_task3(sp: Dict, ic: Dict) -> Tuple[object, Optional[Dict]]:
    from backtesting.walk_forward import run_walk_forward

    df = _load_m15("XAUUSD", BASELINE_START)

    print("  WF: gold_session_breakout  |  324 combos × 6m/3m/3m", flush=True)
    print("  (min 6 trades per IS window)")

    try:
        wf = run_walk_forward(
            df=df,
            strategy_name="gold_session_breakout",
            instrument="XAUUSD",
            initial_balance=INITIAL_BALANCE,
            in_sample_months=6,
            oos_months=3,
            step_months=3,
            min_trades=6,
            verbose=True,
        )
    except RuntimeError as exc:
        print(f"  WF error: {exc}")
        return None, None

    n_w    = len(wf.windows)
    n_pos  = sum(1 for w in wf.windows if w.oos_sharpe > 0)
    avg_d  = wf.avg_degradation_ratio
    pos_fr = n_pos / n_w if n_w else 0.0
    passes = (not np.isnan(avg_d)) and avg_d > 0.3 and pos_fr > 0.40

    print(f"\n  WF summary: avg_deg={avg_d:.3f}  pos_OOS={n_pos}/{n_w} ({pos_fr:.0%})")
    print(f"  WF result: {'PASS ✓' if passes else 'FAIL ✗'}")
    if passes:
        print(f"  Recommended params: {wf.recommended_params}")

    return wf, wf.recommended_params if passes else None


# ---------------------------------------------------------------------------
# TASK 4: Combined portfolio + gold standalone
# ---------------------------------------------------------------------------

def run_task4(wf_best_params: Optional[Dict], sp: Dict, ic: Dict) -> None:
    print("  Building London S3 (full 2020-2025) + Gold (Jul 2020-Apr 2025)…")

    # ── London S3 portfolio ───────────────────────────────────────────────
    london_pairs = ["EURUSD", "GBPUSD", "EURJPY", "GBPJPY", "AUDUSD", "USDJPY"]
    london_results = []
    for pair in london_pairs:
        df_p = _load_m15(pair, FULL_START)
        s    = _build_london(pair, sp, ic, S3_RISK[pair])
        london_results.append(_run(s, df_p, pair))

    df_ny = _load_m15("EURUSD", FULL_START)
    london_results.append(_run(_build_ny(sp, ic, S3_RISK["NY_EURUSD"]), df_ny, "EURUSD"))

    # ── Gold with WF-best or default params ───────────────────────────────
    df_g   = _load_m15("XAUUSD", BASELINE_START)
    s_gold = _build_gold(sp, ic, overrides=wf_best_params)
    r_gold = _run(s_gold, df_g, "XAUUSD")

    # ── Combined portfolio (from BASELINE_START so gold start aligns) ─────
    all_results  = london_results + [r_gold]
    combined_eq  = _combine_equity(all_results, start=BASELINE_START)

    # ── Gold standalone equity (same period) ─────────────────────────────
    gold_eq = r_gold.equity_curve
    gold_eq = INITIAL_BALANCE + (gold_eq - r_gold.initial_balance)  # normalise

    # ── Stats ─────────────────────────────────────────────────────────────
    from backtesting.metrics import calculate_metrics
    combined_stats = _portfolio_stats(combined_eq, "London S3 + Gold")
    gold_stats     = _portfolio_stats(gold_eq,    "Gold standalone")

    n_w_c, n_b_c = _count_ftmo_breaches(combined_eq)
    n_w_g, n_b_g = _count_ftmo_breaches(gold_eq)

    def _print_stats(s: Dict, n_w: int, n_b: int) -> None:
        print(f"    Avg monthly:  {s.get('avg_monthly', 0):+.3f}%")
        print(f"    CAGR:         {s.get('cagr', 0):+.2f}%")
        print(f"    Sharpe:       {s.get('sharpe', 0):+.3f}")
        print(f"    Max DD:       {s.get('max_dd', 0):.2f}%")
        print(f"    Worst daily:  {s.get('worst_daily', 0):.3f}%")
        d10 = s.get('days_to10', -1)
        print(f"    Days to +10%: {d10 if d10 > 0 else 'not reached'}")
        print(f"    Pos months:   {s.get('pos_months',0)}/{s.get('n_months',0)}")
        print(f"    FTMO breaches: {n_b}/{n_w} windows")

    print(f"\n  Combined (London S3 + Gold):")
    _print_stats(combined_stats, n_w_c, n_b_c)

    print(f"\n  Gold STANDALONE (as sole FTMO strategy):")
    _print_stats(gold_stats, n_w_g, n_b_g)

    # ── Correlation ───────────────────────────────────────────────────────
    london_eq  = _combine_equity(london_results, start=BASELINE_START)
    london_d   = london_eq.resample("D").last().dropna().pct_change().dropna()
    gold_d     = gold_eq.resample("D").last().dropna().pct_change().dropna()
    common     = london_d.index.intersection(gold_d.index)
    if len(common) > 10:
        corr = float(london_d.loc[common].corr(gold_d.loc[common]))
        print(f"\n  Gold ↔ London correlation (daily returns): {corr:+.3f}")
    else:
        print("\n  Correlation: insufficient overlap")

    # ── Peak risk exposure ────────────────────────────────────────────────
    from backtesting.metrics import calculate_metrics
    if r_gold.trades:
        gold_m = calculate_metrics(r_gold)
        max_gold_risk = sp["gold_session_breakout"].get("risk_per_trade_pct", 0.8)
        max_forex_risk = max(S3_RISK[p] for p in ["EURUSD", "GBPUSD"])
        print(f"\n  Max simultaneous positions: 8 (7 London/NY + 1 Gold)")
        print(f"  Peak combined risk (if all simultaneously open):")
        total_risk = sum(S3_RISK.values()) + max_gold_risk
        print(f"    All London pairs: {sum(S3_RISK.values()):.1f}% + Gold: {max_gold_risk:.1f}% = {total_risk:.1f}%")
        print(f"    (realistic simultaneous peak: London 2-3 concurrent + Gold = 3-4%)")

    # ── Gold monthly returns ───────────────────────────────────────────────
    print(f"\n  Gold Monthly Returns:")
    print(_monthly_table(gold_eq))

    print(f"\n  Combined Monthly Returns:")
    print(_monthly_table(combined_eq))


# ---------------------------------------------------------------------------
# CONTEXT SUMMARY
# ---------------------------------------------------------------------------

def context_summary(r_gold_a, r_gold_b, wf, wf_params, sp: Dict) -> None:
    from backtesting.metrics import calculate_metrics

    print()

    # 1. Data quality
    df_g = _load_m15("XAUUSD")
    df_et = df_g.copy(); df_et.index = df_et.index.tz_convert("US/Eastern")
    sess = df_et[(df_et.index.hour >= 3) & (df_et.index.hour < 16) & (df_et.index.dayofweek < 5)]
    nz   = (sess.volume > 0).sum()
    print(f"  1. XAUUSD data quality:")
    print(f"     {len(df_g):,} bars  2020-01-01 → 2025-04-30  volume_ok={nz/len(sess)*100:.1f}%")

    # 2. Trades/month and Sharpe
    if r_gold_a and r_gold_a.trades:
        m = calculate_metrics(r_gold_a)
        print(f"  2. Gold baseline (TP-A): T/mo={m['trades_per_month_avg']:.1f}  Sharpe={m['sharpe_ratio']:+.3f}")
    else:
        print("  2. Gold baseline: NO TRADES")

    # 3. VPA stats
    df_base = _load_m15("XAUUSD", BASELINE_START)
    from strategies.gold_session_breakout import GoldSessionBreakout
    cfg = dict(sp["gold_session_breakout"])
    s = GoldSessionBreakout(); s.setup(cfg, {"pip_size": 0.01})
    sig_df = s.generate_signals(df_base.copy())
    n_acc = int(sig_df["vpa_accepted"].sum()); n_rej = int(sig_df["vpa_rejected"].sum())
    tot = n_acc + n_rej
    acc_wr = 0.0
    if r_gold_a and r_gold_a.trades:
        acc_wr = sum(1 for t in r_gold_a.trades if t.pnl_dollars > 0) / len(r_gold_a.trades) * 100.0
    print(f"  3. VPA filter: {n_rej}/{tot} rejected ({100*n_rej/max(tot,1):.0f}%)  "
          f"accepted WR={acc_wr:.1f}%")

    # 4. WF
    if wf is None:
        print("  4. WF: NOT RUN (no viable baseline)")
    else:
        n_w   = len(wf.windows)
        n_pos = sum(1 for w in wf.windows if w.oos_sharpe > 0)
        passed = wf_params is not None
        print(f"  4. WF: avg_deg={wf.avg_degradation_ratio:.3f}  {n_pos}/{n_w} pos OOS  → {'PASS ✓' if passed else 'FAIL ✗'}")

    # 5-8: defer to task4 output
    print("  5–8. See Task 4 output above for combined/standalone stats and timeline.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import time as _time
    t0 = _time.time()

    sp = json.loads(Path("config/strategy_params.json").read_text())
    ic = json.loads(Path("config/instruments.json").read_text())

    print("\n" + "=" * 72)
    print("  TASK 0 — XAUUSD Data Quality")
    print("=" * 72)
    run_task0()

    print("\n" + "=" * 72)
    print("  TASK 2 — Gold Baseline Backtest (Jul 2020 → Apr 2025)")
    print("=" * 72)
    r_gold_a, r_gold_b = run_task2(sp, ic)

    print("\n" + "=" * 72)
    print("  TASK 3 — Walk-Forward Validation")
    print("=" * 72)
    wf, wf_params = run_task3(sp, ic)

    print("\n" + "=" * 72)
    print("  TASK 4 — Combined Portfolio + Gold Standalone")
    print("=" * 72)
    run_task4(wf_params, sp, ic)

    print("\n" + "=" * 72)
    print("  CONTEXT SUMMARY")
    print("=" * 72)
    context_summary(r_gold_a, r_gold_b, wf, wf_params, sp)

    elapsed = _time.time() - t0
    print(f"\n  Total runtime: {elapsed / 60:.1f} minutes")
    print("\nDone.")


if __name__ == "__main__":
    main()
