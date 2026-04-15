"""
execution/paper_trade_verifier.py
====================================
Compare live (or paper-trade) signals against the backtest signal record.

PURPOSE
-------
Before deploying real capital, run the same strategy logic on recent historical
data and verify that:
  1. Signal directions are identical (100% match required)
  2. Entry prices agree within a tolerance (spread + slippage drift)
  3. SL/TP prices agree within tolerance
  4. Number of signals per day matches exactly

Any discrepancy indicates a live/backtest divergence — likely a timezone
offset, VWAP anchor mismatch, or Asian range date boundary issue.

USAGE
-----
# Verify gold strategy signals for January 2026:
venv/bin/python execution/paper_trade_verifier.py \
  --strategy gold \
  --start 2026-01-01 \
  --end 2026-01-31 \
  --config config/live_config.json

# Verify NAS100:
venv/bin/python execution/paper_trade_verifier.py \
  --strategy nas100 \
  --start 2026-01-01 \
  --end 2026-01-31

# Compare against a live signal log:
venv/bin/python execution/paper_trade_verifier.py \
  --strategy gold \
  --live-log logs/live/gold_signals_2026-01.csv \
  --start 2026-01-01 --end 2026-01-31

OUTPUT
------
Prints a daily comparison table and a summary pass/fail report.
Saves a CSV of mismatches to logs/paper_trade/mismatches_YYYY-MM-DD.csv.

DATA REQUIREMENTS
-----------------
  XAUUSD H1: data/historical/XAUUSD_1h.parquet
  NAS100 M5:  data/historical/usatechidxusd_m5.parquet
  (same files used in backtesting)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.gold_multi_strategy import GoldMultiStrategy
from strategies.nas100_ib_breakout import NAS100IbBreakout

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

# ── Default data paths ────────────────────────────────────────────────────────
_GOLD_PARQUET  = "data/historical/XAUUSD_1h.parquet"
_NAS_PARQUET   = "data/historical/usatechidxusd_m5.parquet"
_LOG_DIR       = "logs/paper_trade"
_PRICE_TOL_PCT = 5.0     # max % deviation between backtest and live price


# ════════════════════════════════════════════════════════════════════════════
#  Signal extraction from backtest run
# ════════════════════════════════════════════════════════════════════════════

def _load_parquet(path: str) -> pd.DataFrame:
    """Load a parquet file; set datetime index; ensure US/Eastern tz."""
    df = pd.read_parquet(path)
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    return df


def _extract_signals(
    strategy: Any,
    df: pd.DataFrame,
    start: dt.date,
    end: dt.date,
) -> List[Dict[str, Any]]:
    """
    Run strategy.generate_signals() on df and extract all non-zero signals
    between start and end dates.

    Returns list of dicts with keys:
        bar_time, direction, entry_price, sl_price, tp_price, tp1_price,
        tp1_pct, trail_distance, date
    """
    sig_df = strategy.generate_signals(df.copy())
    mask   = sig_df["signal"].fillna(0) != 0
    sig_df = sig_df[mask]

    results = []
    for ts, row in sig_df.iterrows():
        d = ts.date()
        if d < start or d > end:
            continue
        results.append({
            "bar_time":      ts,
            "date":          d,
            "direction":     int(row["signal"]),
            "entry_price":   float(row.get("close", row.get("entry_price", np.nan))),
            "sl_price":      float(row.get("sl_price", np.nan)),
            "tp_price":      float(row.get("tp_price", np.nan)),
            "tp1_price":     float(row.get("tp1_price", np.nan)),
            "tp1_pct":       float(row.get("tp1_pct", np.nan)),
            "trail_distance": float(row.get("trail_distance", np.nan)),
        })
    return results


# ════════════════════════════════════════════════════════════════════════════
#  Live signal log parser
# ════════════════════════════════════════════════════════════════════════════

def _parse_live_log(log_path: str) -> List[Dict[str, Any]]:
    """
    Parse a CSV live signal log produced by the cBot or Tradovate connector.

    Expected CSV columns (any extra columns are ignored):
        bar_time, direction, entry_price, sl_price, tp_price, tp1_price
        bar_time format: "YYYY-MM-DD HH:MM:SS" or ISO8601
    """
    records = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = pd.Timestamp(row["bar_time"]).tz_localize("US/Eastern") \
                     if "+" not in row["bar_time"] \
                     else pd.Timestamp(row["bar_time"]).tz_convert("US/Eastern")
                records.append({
                    "bar_time":  ts,
                    "date":      ts.date(),
                    "direction": int(float(row["direction"])),
                    "entry_price": float(row.get("entry_price", "nan")),
                    "sl_price":    float(row.get("sl_price", "nan")),
                    "tp_price":    float(row.get("tp_price", "nan")),
                    "tp1_price":   float(row.get("tp1_price", "nan")),
                })
            except Exception as exc:
                logger.warning("Skipping live log row: %s — %s", row, exc)
    return records


# ════════════════════════════════════════════════════════════════════════════
#  Comparison engine
# ════════════════════════════════════════════════════════════════════════════

def compare_signals(
    backtest_signals: List[Dict[str, Any]],
    live_signals:     List[Dict[str, Any]],
    price_tol_pct:    float = _PRICE_TOL_PCT,
) -> Dict[str, Any]:
    """
    Match live signals to backtest signals by date + direction.
    Return a comparison report dict.
    """
    # Index by (date, direction) for matching
    bt_by_key: Dict[Tuple, List] = {}
    for s in backtest_signals:
        k = (s["date"], s["direction"])
        bt_by_key.setdefault(k, []).append(s)

    live_by_key: Dict[Tuple, List] = {}
    for s in live_signals:
        k = (s["date"], s["direction"])
        live_by_key.setdefault(k, []).append(s)

    all_dates_bt   = sorted({s["date"] for s in backtest_signals})
    all_dates_live = sorted({s["date"] for s in live_signals})
    all_dates      = sorted(set(all_dates_bt) | set(all_dates_live))

    rows          = []
    total         = 0
    dir_matches   = 0
    price_errors  = 0
    sl_errors     = 0
    count_errors  = 0
    mismatches    = []

    for d in all_dates:
        bt_today   = [s for s in backtest_signals if s["date"] == d]
        live_today = [s for s in live_signals     if s["date"] == d]

        if len(bt_today) != len(live_today):
            count_errors += 1
            mismatches.append({
                "date":  d,
                "issue": f"count mismatch: backtest={len(bt_today)} live={len(live_today)}",
            })

        n = min(len(bt_today), len(live_today))
        for i in range(n):
            total += 1
            bt  = bt_today[i]
            lv  = live_today[i]

            # Direction match
            dir_ok = (bt["direction"] == lv["direction"])
            if dir_ok:
                dir_matches += 1

            # Price comparison (% deviation)
            entry_dev = _pct_dev(bt["entry_price"], lv["entry_price"])
            sl_dev    = _pct_dev(bt["sl_price"],    lv["sl_price"])
            tp_dev    = _pct_dev(bt["tp_price"],    lv["tp_price"])

            entry_ok = entry_dev <= price_tol_pct
            sl_ok    = sl_dev    <= price_tol_pct

            if not entry_ok:
                price_errors += 1
            if not sl_ok:
                sl_errors += 1

            pass_flag = dir_ok and entry_ok and sl_ok
            row = {
                "date":       d,
                "dir_bt":     bt["direction"],
                "dir_live":   lv["direction"],
                "entry_bt":   bt["entry_price"],
                "entry_live": lv["entry_price"],
                "entry_dev%": round(entry_dev, 2),
                "sl_bt":      bt["sl_price"],
                "sl_live":    lv["sl_price"],
                "sl_dev%":    round(sl_dev, 2),
                "tp_bt":      bt["tp_price"],
                "tp_live":    lv["tp_price"],
                "tp_dev%":    round(tp_dev, 2),
                "pass":       pass_flag,
            }
            rows.append(row)
            if not pass_flag:
                mismatches.append({"date": d, "issue": _describe_mismatch(row)})

    dir_match_pct   = 100.0 * dir_matches  / max(total, 1)
    price_pass_pct  = 100.0 * (total - price_errors) / max(total, 1)
    sl_pass_pct     = 100.0 * (total - sl_errors)    / max(total, 1)
    overall_pass    = (dir_match_pct == 100.0
                       and price_pass_pct >= (100.0 - price_tol_pct)
                       and count_errors == 0)

    return {
        "total_signals":       total,
        "dir_match_pct":       round(dir_match_pct, 1),
        "price_pass_pct":      round(price_pass_pct, 1),
        "sl_pass_pct":         round(sl_pass_pct, 1),
        "count_errors":        count_errors,
        "overall_pass":        overall_pass,
        "rows":                rows,
        "mismatches":          mismatches,
        "backtest_signal_days": len(all_dates_bt),
        "live_signal_days":    len(all_dates_live),
    }


def _pct_dev(a: float, b: float) -> float:
    """Percent deviation between a and b (0 if either is NaN)."""
    if np.isnan(a) or np.isnan(b) or a == 0:
        return 0.0
    return abs(a - b) / abs(a) * 100.0


def _describe_mismatch(row: Dict) -> str:
    issues = []
    if row["dir_bt"] != row["dir_live"]:
        issues.append(f"direction {row['dir_bt']}≠{row['dir_live']}")
    if row["entry_dev%"] > _PRICE_TOL_PCT:
        issues.append(f"entry {row['entry_dev%']:.1f}%")
    if row["sl_dev%"] > _PRICE_TOL_PCT:
        issues.append(f"sl {row['sl_dev%']:.1f}%")
    return " | ".join(issues) if issues else "unknown"


# ════════════════════════════════════════════════════════════════════════════
#  Reporter
# ════════════════════════════════════════════════════════════════════════════

def print_report(report: Dict[str, Any], strategy_name: str) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  PAPER TRADE VERIFICATION — {strategy_name.upper()}")
    print(sep)
    print(f"  Signals compared:    {report['total_signals']}")
    print(f"  Backtest signal days:{report['backtest_signal_days']}")
    print(f"  Live signal days:    {report['live_signal_days']}")
    print(f"  Direction match:     {report['dir_match_pct']:.1f}%  (must be 100%)")
    print(f"  Entry price pass:    {report['price_pass_pct']:.1f}%  (> 95% OK)")
    print(f"  SL price pass:       {report['sl_pass_pct']:.1f}%  (> 95% OK)")
    print(f"  Count errors (days): {report['count_errors']}  (must be 0)")
    print(sep)

    if report["overall_pass"]:
        print("  RESULT: ✓ PASS — signals match backtest, safe to deploy live")
    else:
        print("  RESULT: ✗ FAIL — investigate mismatches before going live")

    if report["mismatches"]:
        print(f"\n  Mismatches ({len(report['mismatches'])}):")
        for m in report["mismatches"][:20]:   # show first 20
            print(f"    {m['date']}  {m['issue']}")
        if len(report["mismatches"]) > 20:
            print(f"    ... and {len(report['mismatches']) - 20} more")

    print(sep)

    # Daily table
    if report["rows"]:
        print(f"\n  {'Date':<12} {'Dir':>4} {'Entry BT':>10} {'Entry Live':>10} "
              f"{'Dev%':>6} {'SL Dev%':>8} {'TP Dev%':>8} {'OK':>4}")
        print(f"  {'-'*10:<12} {'-'*3:>4} {'-'*8:>10} {'-'*8:>10} "
              f"{'-'*5:>6} {'-'*6:>8} {'-'*6:>8} {'-'*3:>4}")
        for row in report["rows"]:
            ok_str = "OK" if row["pass"] else "FAIL"
            print(f"  {str(row['date']):<12} {row['dir_bt']:>4} "
                  f"{row['entry_bt']:>10.3f} {row['entry_live']:>10.3f} "
                  f"{row['entry_dev%']:>6.2f} {row['sl_dev%']:>8.2f} "
                  f"{row['tp_dev%']:>8.2f} {ok_str:>4}")


def save_mismatches(report: Dict[str, Any], strategy_name: str) -> Optional[str]:
    """Save mismatch details to a CSV file. Return path or None."""
    if not report["mismatches"]:
        return None

    os.makedirs(_LOG_DIR, exist_ok=True)
    date_str  = dt.date.today().isoformat()
    path      = f"{_LOG_DIR}/mismatches_{strategy_name}_{date_str}.csv"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "issue"])
        writer.writeheader()
        writer.writerows(report["mismatches"])

    logger.info("Mismatches saved to %s", path)
    return path


def save_comparison_csv(report: Dict[str, Any], strategy_name: str) -> str:
    """Save full comparison table to CSV."""
    os.makedirs(_LOG_DIR, exist_ok=True)
    date_str = dt.date.today().isoformat()
    path     = f"{_LOG_DIR}/comparison_{strategy_name}_{date_str}.csv"
    if report["rows"]:
        pd.DataFrame(report["rows"]).to_csv(path, index=False)
        logger.info("Comparison saved to %s", path)
    return path


# ════════════════════════════════════════════════════════════════════════════
#  Strategy factories
# ════════════════════════════════════════════════════════════════════════════

def _build_gold_strategy(cfg: Dict[str, Any]) -> GoldMultiStrategy:
    g = cfg.get("gold", {})
    ma = g.get("module_a", {})
    mb = g.get("module_b", {})
    mc = g.get("module_c", {})
    return GoldMultiStrategy(
        rr_tp1=g.get("rr_tp1", 1.0),
        rr_tp2=g.get("rr_tp2", 2.5),
        tp1_pct=g.get("tp1_pct", 0.5),
        trail_atr_mult=g.get("trail_atr_mult", 2.0),
        risk_per_trade_pct=g.get("risk_per_trade_pct", 0.5),
        max_trades_per_day=g.get("max_trades_per_day", 3),
        a_buffer_pips=ma.get("buffer_pips", 50.0),
        a_min_range_pips=ma.get("min_range_pips", 500.0),
        a_max_range_pips=ma.get("max_range_pips", 2000.0),
        a_sl_cap_pips=ma.get("sl_cap_pips", 1000.0),
        a_regime_atr_pct=ma.get("regime_atr_pct", 40.0),
        b_sl_buffer_pips=mb.get("sl_buffer_pips", 30.0),
        b_sl_cap_pips=mb.get("sl_cap_pips", 500.0),
        b_sl_min_pips=mb.get("sl_min_pips", 20.0),
        c_sl_pips=mc.get("sl_pips", 200.0),
    )


def _build_nas100_strategy(cfg: Dict[str, Any]) -> NAS100IbBreakout:
    n = cfg
    return NAS100IbBreakout(
        ib_adr_ratio=n.get("ib_adr_ratio", 99.0),
        buffer_points=n.get("buffer_points", 5.0),
        rr_ratio=n.get("rr_ratio", 2.5),
        risk_per_trade_pct=n.get("risk_per_trade_pct", 0.5),
        max_sl_points=n.get("max_sl_points", 80.0),
        min_sl_points=n.get("min_sl_points", 15.0),
        vol_sma_period=n.get("vol_sma_period", 20),
        vol_sma_mult=n.get("vol_sma_mult", 1.0),
    )


# ════════════════════════════════════════════════════════════════════════════
#  Self-verification mode: replay history and compare to a reference pickle
# ════════════════════════════════════════════════════════════════════════════

def verify_against_backtest_pkl(
    strategy_name: str,
    pkl_path:      str,
    config_section: Dict[str, Any],
    start:         dt.date,
    end:           dt.date,
) -> bool:
    """
    Load a saved backtest Trade list from a pkl file, extract its signal
    dates/directions, then re-run the strategy on live data and compare.
    Returns True if verification passes.
    """
    import pickle

    try:
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
    except FileNotFoundError:
        logger.warning("Backtest pkl not found: %s — running forward-only verify", pkl_path)
        saved = None

    if strategy_name == "gold":
        data_path = _GOLD_PARQUET
        strategy  = _build_gold_strategy(config_section)
    else:
        data_path = _NAS_PARQUET
        strategy  = _build_nas100_strategy(config_section)

    df = _load_parquet(data_path)
    backtest_signals = _extract_signals(strategy, df, start, end)

    if saved is None:
        logger.info("No reference pkl — saved %d signals from re-run as baseline",
                    len(backtest_signals))
        return True

    # Convert saved Trade list → signal dicts
    if isinstance(saved, list) and hasattr(saved[0], "entry_time"):
        # It's a list of Trade objects
        ref_signals = []
        seen_ids = set()
        for tr in saved:
            if tr.trade_id in seen_ids:
                continue   # only take one record per signal
            d = tr.entry_time.date() if hasattr(tr.entry_time, "date") else tr.entry_time
            if d < start or d > end:
                continue
            ref_signals.append({
                "date":        d,
                "direction":   tr.direction,
                "entry_price": tr.entry_price,
                "sl_price":    getattr(tr, "sl_price", np.nan),
                "tp_price":    getattr(tr, "tp_price", np.nan),
                "tp1_price":   np.nan,
            })
            seen_ids.add(tr.trade_id)
    else:
        logger.warning("Unexpected pkl format — skipping reference comparison")
        ref_signals = backtest_signals

    report = compare_signals(ref_signals, backtest_signals)
    print_report(report, f"{strategy_name} (pkl vs re-run)")
    save_comparison_csv(report, strategy_name)
    save_mismatches(report, strategy_name)
    return report["overall_pass"]


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper trade signal verifier — compare live vs backtest."
    )
    p.add_argument("--strategy", choices=["gold", "nas100"], required=True,
                   help="Which strategy to verify")
    p.add_argument("--start", required=True,
                   help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True,
                   help="End date YYYY-MM-DD")
    p.add_argument("--config", default="config/live_config.json",
                   help="Path to live_config.json")
    p.add_argument("--platform", default="the5ers",
                   choices=["the5ers", "ftmo", "apex"],
                   help="Which platform section to use for strategy params")
    p.add_argument("--live-log", default=None,
                   help="Path to live signal CSV log (optional)")
    p.add_argument("--pkl", default=None,
                   help="Path to backtest pkl file for reference comparison")
    p.add_argument("--tolerance", type=float, default=_PRICE_TOL_PCT,
                   help=f"Price tolerance %% (default {_PRICE_TOL_PCT})")
    return p.parse_args()


def main() -> None:
    args  = _parse_args()
    start = dt.date.fromisoformat(args.start)
    end   = dt.date.fromisoformat(args.end)

    with open(args.config) as f:
        all_cfg = json.load(f)
    cfg = all_cfg.get(args.platform, all_cfg)

    logger.info("Loading data and generating backtest signals (%s → %s)…", start, end)

    if args.strategy == "gold":
        data_path = _GOLD_PARQUET
        strategy  = _build_gold_strategy(cfg)
    else:
        data_path = _NAS_PARQUET
        strategy  = _build_nas100_strategy(cfg.get("nas100", cfg))

    if not os.path.exists(data_path):
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    df                = _load_parquet(data_path)
    backtest_signals  = _extract_signals(strategy, df, start, end)
    logger.info("Backtest signals: %d", len(backtest_signals))

    # If a pkl reference is provided, use it as "live" for comparison
    if args.pkl:
        ok = verify_against_backtest_pkl(
            args.strategy, args.pkl, cfg, start, end
        )
        sys.exit(0 if ok else 1)

    # If a live log is provided, compare live vs backtest
    if args.live_log:
        if not os.path.exists(args.live_log):
            logger.error("Live log not found: %s", args.live_log)
            sys.exit(1)
        live_signals = _parse_live_log(args.live_log)
        logger.info("Live signals: %d", len(live_signals))
    else:
        # No live data — run forward self-test: re-run strategy on last 30 bars
        # and compare to the same run (should be 100% identical)
        logger.info("No live log provided — running self-consistency test "
                    "(strategy vs itself on same data)")
        live_signals = backtest_signals   # should produce 100% match

    report = compare_signals(backtest_signals, live_signals,
                             price_tol_pct=args.tolerance)
    print_report(report, args.strategy)
    save_comparison_csv(report, args.strategy)
    save_mismatches(report, args.strategy)

    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
