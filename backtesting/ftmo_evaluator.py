"""
FTMO Evaluation Wrapper
=======================

Applies **official** FTMO evaluation criteria to BacktestResult objects —
using the real limits (5 % daily, 10 % max DD, +10 % profit target) rather
than the guardian's conservative safety buffers (4 % / 9 %).

Answers four questions
----------------------
1. Did any rolling window breach the daily loss limit (5 %)?
2. Did any rolling window breach max drawdown (10 %)?
3. In net-profitable windows, how many trading days to reach +10 %?
4. Worst intraday equity drawdown vs worst closed-trade drawdown
   (the "measurement gap").

Portfolio mode
--------------
Combine two BacktestResult objects (EURUSD + GBPUSD) into a single
equity curve representing one unified FTMO account (starting equity =
sum of both initial balances) and evaluate that combined account.

CLI
---
  python -m backtesting.ftmo_evaluator \\
      --eurusd  results/london_breakout_EURUSD_backtest.pkl \\
      --gbpusd  results/london_breakout_GBPUSD_backtest.pkl \\
      [--window-months 3]
"""

from __future__ import annotations

import argparse
import datetime
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ── FTMO official limits (challenge) ─────────────────────────────────────────
DAILY_LOSS_LIMIT_PCT  = 5.0    # equity must not drop > 5 % below start-of-day balance
MAX_DD_LIMIT_PCT      = 10.0   # equity must not drop > 10 % below initial balance
PROFIT_TARGET_PCT     = 10.0   # +10 % above initial = challenge passed
MIN_TRADING_DAYS      = 4      # minimum distinct calendar days with a closed trade


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class WindowEval:
    """FTMO evaluation for a single window (rolling period or full span)."""
    label:                  str              # e.g. "2022-01 → 2022-04"
    start_date:             str
    end_date:               str
    initial_equity:         float
    final_equity:           float
    return_pct:             float

    # FTMO pass / fail
    passed:                 bool             # no breach AND profit target hit
    breach_type:            Optional[str]    # "daily_loss" | "max_drawdown" | None
    breach_date:            Optional[str]    # ISO date string
    breach_equity:          Optional[float]

    # Profit target
    profit_target_hit:      bool
    profit_target_ts:       Optional[str]    # ISO timestamp
    days_to_target:         Optional[int]    # calendar days from window start

    # Activity
    trading_days:           int              # distinct days with a closed trade
    min_trading_days_met:   bool

    # Drawdown analysis
    max_intraday_dd_pct:    float            # from equity_curve (includes floating)
    worst_intraday_ts:      Optional[str]
    max_closed_dd_pct:      float            # from realised balance only
    dd_gap_pp:              float            # intraday_dd - closed_dd ≥ 0

    # Daily loss proximity
    worst_daily_loss_pct:   float            # worst single-day equity drop (negative = loss)
    worst_daily_loss_date:  Optional[str]
    closest_to_limit_pct:   float            # how close to 5 % limit (0 = never close)


@dataclass
class FTMOEvalResult:
    """Full evaluation result for one instrument or the combined portfolio."""
    label:                  str
    full_period:            WindowEval
    windows:                List[WindowEval]

    # Aggregate statistics across rolling windows
    pass_rate_pct:          float            # % of windows where FTMO criteria met
    avg_days_to_target:     float            # avg trading days in target-hit windows
    breach_rate_pct:        float            # % of windows with any breach
    daily_loss_breach_count: int
    max_dd_breach_count:    int

    # Measurement gap: extra risk hidden in open positions
    avg_dd_gap_pp:          float            # avg (intraday_dd - closed_dd)
    max_dd_gap_pp:          float            # worst single-window gap


# ── Core evaluator ────────────────────────────────────────────────────────────

class FTMOEvaluator:
    """
    Applies FTMO evaluation criteria to one or two BacktestResult objects.

    Parameters
    ----------
    window_months :
        Length of each rolling evaluation window in months.
        Default 3 = quarterly FTMO challenge periods.
    """

    def __init__(self, window_months: int = 3) -> None:
        self.window_months = window_months

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate_single(
        self,
        result,          # BacktestResult
        label: str = "",
    ) -> FTMOEvalResult:
        """Evaluate a single instrument BacktestResult."""
        eq   = result.equity_curve
        init = result.initial_balance
        trades = result.trades

        full = self._eval_window(
            eq, init, trades,
            label=label or "Full period",
        )
        windows = self._rolling_windows(eq, init, trades)
        return self._build_result(label, full, windows)

    def evaluate_portfolio(
        self,
        result_a,        # BacktestResult — EURUSD
        result_b,        # BacktestResult — GBPUSD
        label: str = "Portfolio (EURUSD + GBPUSD)",
    ) -> FTMOEvalResult:
        """
        Combine two BacktestResults into one FTMO account.

        The combined account starts at the sum of both initial balances.
        Both equity curves are aligned and summed bar-by-bar (forward-fill
        gaps where one instrument has no bar).
        """
        combined_eq, combined_init = self._combine_equity(
            result_a.equity_curve, result_a.initial_balance,
            result_b.equity_curve, result_b.initial_balance,
        )
        combined_trades = result_a.trades + result_b.trades

        full = self._eval_window(
            combined_eq, combined_init, combined_trades,
            label="Full period (portfolio)",
        )
        windows = self._rolling_windows(combined_eq, combined_init, combined_trades)
        return self._build_result(label, full, windows)

    # ── Window evaluation ─────────────────────────────────────────────────

    def _eval_window(
        self,
        equity: pd.Series,
        initial: float,
        all_trades: list,
        label: str = "",
    ) -> WindowEval:
        """Run all FTMO checks on an equity slice."""
        if len(equity) == 0:
            return self._empty_window(label)

        start_ts  = equity.index[0]
        end_ts    = equity.index[-1]
        final_eq  = float(equity.iloc[-1])
        ret_pct   = (final_eq - initial) / initial * 100.0

        # Filter trades that close within this window
        start_date = start_ts.date()
        end_date   = end_ts.date()
        win_trades = [
            t for t in all_trades
            if start_date <= t.exit_time.date() <= end_date
        ]

        # ── Max drawdown (intraday equity) ───────────────────────────────
        max_intra_dd, worst_intra_ts = self._max_drawdown(equity)

        # ── Max drawdown (closed-trade balance only) ─────────────────────
        max_closed_dd = self._closed_trade_dd(win_trades, initial)

        dd_gap = max_intra_dd - max_closed_dd   # intraday always >= closed

        # ── Daily loss check ─────────────────────────────────────────────
        worst_daily_loss, worst_daily_date = self._worst_daily_loss(equity)

        # Closest approach to the 5 % daily limit (0 = no risk; 5 = breached)
        closest_to_limit = min(abs(worst_daily_loss), DAILY_LOSS_LIMIT_PCT)

        # ── Max drawdown breach ──────────────────────────────────────────
        max_dd_pct = (equity.min() - initial) / initial * 100.0  # negative
        dd_breach       = max_dd_pct < -MAX_DD_LIMIT_PCT
        dd_breach_idx   = None
        if dd_breach:
            below = equity[equity < initial * (1 - MAX_DD_LIMIT_PCT / 100)]
            if not below.empty:
                dd_breach_idx = below.index[0]

        # ── Daily loss breach ────────────────────────────────────────────
        dl_breach, dl_breach_date = self._find_daily_loss_breach(equity)

        # ── Profit target ────────────────────────────────────────────────
        target_eq  = initial * (1 + PROFIT_TARGET_PCT / 100)
        target_hit = final_eq >= target_eq
        target_ts: Optional[pd.Timestamp] = None
        if target_hit or (equity >= target_eq).any():
            idx = (equity >= target_eq).idxmax()
            if equity[idx] >= target_eq:
                target_ts  = idx
                target_hit = True

        days_to_target: Optional[int] = None
        if target_ts is not None:
            days_to_target = (target_ts.date() - start_date).days

        # ── Trading days ─────────────────────────────────────────────────
        trade_days = len({t.exit_time.date() for t in win_trades})

        # ── Determine breach (daily loss takes priority over max DD) ─────
        breach_type: Optional[str] = None
        breach_date: Optional[str] = None
        breach_eq:   Optional[float] = None

        if dl_breach:
            breach_type = "daily_loss"
            breach_date = dl_breach_date.isoformat() if dl_breach_date else None
        elif dd_breach:
            breach_type = "max_drawdown"
            breach_date = dd_breach_idx.date().isoformat() if dd_breach_idx is not None else None
            breach_eq   = float(equity[dd_breach_idx]) if dd_breach_idx is not None else None

        passed = (breach_type is None) and target_hit and (trade_days >= MIN_TRADING_DAYS)

        return WindowEval(
            label=label,
            start_date=str(start_date),
            end_date=str(end_date),
            initial_equity=round(initial, 2),
            final_equity=round(final_eq, 2),
            return_pct=round(ret_pct, 3),
            passed=passed,
            breach_type=breach_type,
            breach_date=breach_date,
            breach_equity=round(breach_eq, 2) if breach_eq else None,
            profit_target_hit=target_hit,
            profit_target_ts=str(target_ts) if target_ts is not None else None,
            days_to_target=days_to_target,
            trading_days=trade_days,
            min_trading_days_met=trade_days >= MIN_TRADING_DAYS,
            max_intraday_dd_pct=round(max_intra_dd, 3),
            worst_intraday_ts=str(worst_intra_ts) if worst_intra_ts is not None else None,
            max_closed_dd_pct=round(max_closed_dd, 3),
            dd_gap_pp=round(dd_gap, 3),
            worst_daily_loss_pct=round(worst_daily_loss, 3),
            worst_daily_loss_date=str(worst_daily_date) if worst_daily_date else None,
            closest_to_limit_pct=round(closest_to_limit, 3),
        )

    def _rolling_windows(
        self,
        equity: pd.Series,
        initial: float,
        all_trades: list,
    ) -> List[WindowEval]:
        """
        Slice equity into rolling window_months windows, rebase each to
        $initial, and run a fresh FTMO evaluation on each slice.
        """
        windows: List[WindowEval] = []
        start = equity.index[0]
        data_end = equity.index[-1]
        cursor = pd.Timestamp(
            year=start.year, month=start.month, day=1, tz=equity.index.tz
        )

        win_idx = 0
        while True:
            win_end = cursor + pd.DateOffset(months=self.window_months)
            if win_end > data_end + pd.Timedelta(days=1):
                break

            slice_eq = equity[(equity.index >= cursor) & (equity.index < win_end)]
            if len(slice_eq) < 10:
                cursor += pd.DateOffset(months=self.window_months)
                continue

            # Rebase: treat the first equity value in this window as $initial
            win_initial = float(slice_eq.iloc[0])
            rebased     = slice_eq / win_initial * initial

            label = (
                f"{cursor.strftime('%Y-%m')} → "
                f"{(win_end - pd.Timedelta(days=1)).strftime('%Y-%m')}"
            )
            win_trades = [
                t for t in all_trades
                if cursor.date() <= t.exit_time.date() < win_end.date()
            ]
            weval = self._eval_window(rebased, initial, win_trades, label=label)
            windows.append(weval)

            cursor    += pd.DateOffset(months=self.window_months)
            win_idx   += 1

        return windows

    # ── Aggregate ─────────────────────────────────────────────────────────

    def _build_result(
        self,
        label: str,
        full: WindowEval,
        windows: List[WindowEval],
    ) -> FTMOEvalResult:
        if not windows:
            return FTMOEvalResult(
                label=label, full_period=full, windows=[],
                pass_rate_pct=0.0, avg_days_to_target=0.0,
                breach_rate_pct=0.0, daily_loss_breach_count=0,
                max_dd_breach_count=0, avg_dd_gap_pp=0.0, max_dd_gap_pp=0.0,
            )

        n = len(windows)
        pass_rate    = 100.0 * sum(1 for w in windows if w.passed) / n
        breach_rate  = 100.0 * sum(1 for w in windows if w.breach_type) / n
        dl_breaches  = sum(1 for w in windows if w.breach_type == "daily_loss")
        dd_breaches  = sum(1 for w in windows if w.breach_type == "max_drawdown")

        target_days  = [w.days_to_target for w in windows
                        if w.profit_target_hit and w.days_to_target is not None]
        avg_target   = float(np.mean(target_days)) if target_days else float("nan")

        gaps         = [w.dd_gap_pp for w in windows]
        avg_gap      = float(np.mean(gaps)) if gaps else 0.0
        max_gap      = float(max(gaps)) if gaps else 0.0

        return FTMOEvalResult(
            label=label,
            full_period=full,
            windows=windows,
            pass_rate_pct=round(pass_rate, 1),
            avg_days_to_target=round(avg_target, 1) if not np.isnan(avg_target) else float("nan"),
            breach_rate_pct=round(breach_rate, 1),
            daily_loss_breach_count=dl_breaches,
            max_dd_breach_count=dd_breaches,
            avg_dd_gap_pp=round(avg_gap, 3),
            max_dd_gap_pp=round(max_gap, 3),
        )

    # ── Indicator helpers ──────────────────────────────────────────────────

    @staticmethod
    def _combine_equity(
        eq_a: pd.Series, init_a: float,
        eq_b: pd.Series, init_b: float,
    ):
        """Sum two equity curves after aligning on their shared index."""
        combined_init = init_a + init_b
        df = pd.DataFrame({"a": eq_a, "b": eq_b})
        # Forward-fill gaps (one instrument may have fewer bars on some days)
        df = df.ffill().bfill()
        combined = df["a"] + df["b"]
        return combined, combined_init

    @staticmethod
    def _max_drawdown(equity: pd.Series):
        """
        Maximum drawdown as a positive percentage.
        Returns (max_dd_pct, timestamp_of_trough).
        """
        roll_max = equity.expanding().max()
        dd       = (equity - roll_max) / roll_max * 100.0
        worst    = float(dd.min())
        ts       = dd.idxmin() if not dd.empty else None
        return abs(worst), ts

    @staticmethod
    def _closed_trade_dd(trades: list, initial: float) -> float:
        """
        Max drawdown computed only from realised trade P&L — ignores
        floating / intraday exposure.  Returns positive percentage.
        """
        if not trades:
            return 0.0
        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        balance = initial
        peak    = initial
        worst   = 0.0
        for t in sorted_trades:
            balance += t.pnl_dollars
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100.0
            if dd > worst:
                worst = dd
        return worst

    @staticmethod
    def _worst_daily_loss(equity: pd.Series):
        """
        For each calendar day, compute (min_equity − first_equity) / first_equity.
        Returns (worst_loss_pct, date).  Loss is negative when equity falls.
        """
        worst_pct  = 0.0
        worst_date = None
        for date, grp in equity.groupby(equity.index.date):
            if grp.empty:
                continue
            sod     = float(grp.iloc[0])
            low_eq  = float(grp.min())
            if sod <= 0:
                continue
            loss_pct = (low_eq - sod) / sod * 100.0
            if loss_pct < worst_pct:
                worst_pct  = loss_pct
                worst_date = date
        return worst_pct, worst_date

    @staticmethod
    def _find_daily_loss_breach(equity: pd.Series):
        """
        Scan bar-by-bar for the first day where equity dropped > DAILY_LOSS_LIMIT_PCT
        below the start-of-day equity.  Returns (breached: bool, date | None).
        """
        for date, grp in equity.groupby(equity.index.date):
            if grp.empty:
                continue
            sod = float(grp.iloc[0])
            if sod <= 0:
                continue
            for ts, eq_val in grp.items():
                loss_pct = (float(eq_val) - sod) / sod * 100.0
                if loss_pct < -DAILY_LOSS_LIMIT_PCT:
                    return True, date
        return False, None

    @staticmethod
    def _empty_window(label: str) -> WindowEval:
        return WindowEval(
            label=label, start_date="", end_date="",
            initial_equity=0.0, final_equity=0.0, return_pct=0.0,
            passed=False, breach_type=None, breach_date=None, breach_equity=None,
            profit_target_hit=False, profit_target_ts=None, days_to_target=None,
            trading_days=0, min_trading_days_met=False,
            max_intraday_dd_pct=0.0, worst_intraday_ts=None,
            max_closed_dd_pct=0.0, dd_gap_pp=0.0,
            worst_daily_loss_pct=0.0, worst_daily_loss_date=None,
            closest_to_limit_pct=0.0,
        )


# ── Console report ────────────────────────────────────────────────────────────

def print_report(result: FTMOEvalResult, window_months: int = 3) -> None:
    """Formatted console output for an FTMOEvalResult."""
    W = 76

    def rule(c="─"):
        print(c * W)

    rule("═")
    print(f"  FTMO EVALUATION  |  {result.label}")
    rule("═")

    fp = result.full_period
    print(f"\n  Full period: {fp.start_date} → {fp.end_date}")
    print(f"  Initial equity: ${fp.initial_equity:,.2f}   "
          f"Final: ${fp.final_equity:,.2f}   "
          f"Return: {fp.return_pct:+.2f}%")
    print(f"  Trading days: {fp.trading_days}   "
          f"Min-days requirement (≥4): {'✓' if fp.min_trading_days_met else '✗'}")
    print()

    # Profit target
    if fp.profit_target_hit:
        print(f"  Profit target (+{PROFIT_TARGET_PCT:.0f}%): ✓ HIT  "
              f"on {fp.profit_target_ts}  ({fp.days_to_target} calendar days)")
    else:
        print(f"  Profit target (+{PROFIT_TARGET_PCT:.0f}%): ✗ NOT HIT  "
              f"(final return {fp.return_pct:+.2f}%)")

    # Drawdown
    print()
    print(f"  Max intraday equity DD:      {fp.max_intraday_dd_pct:.3f}%  "
          f"(FTMO limit: {MAX_DD_LIMIT_PCT:.0f}%)  "
          f"{'❌ BREACH' if fp.breach_type == 'max_drawdown' else '✓ OK'}")
    print(f"  Max closed-trade balance DD: {fp.max_closed_dd_pct:.3f}%")
    print(f"  ▸ Measurement gap (intraday − closed): +{fp.dd_gap_pp:.3f}pp")
    print(f"    → Closed-trade balance UNDERSTATES true risk by {fp.dd_gap_pp:.3f}pp")

    # Daily loss
    print()
    print(f"  Worst single-day equity loss: {fp.worst_daily_loss_pct:.3f}%  "
          f"on {fp.worst_daily_loss_date}  "
          f"(limit: -{DAILY_LOSS_LIMIT_PCT:.0f}%)  "
          f"{'❌ BREACH' if fp.breach_type == 'daily_loss' else '✓ OK'}")
    pct_of_limit = abs(fp.worst_daily_loss_pct) / DAILY_LOSS_LIMIT_PCT * 100
    print(f"    → Closest approach: {abs(fp.worst_daily_loss_pct):.3f}% = "
          f"{pct_of_limit:.0f}% of the 5% limit consumed")

    # Overall verdict
    print()
    if fp.passed:
        print(f"  FULL-PERIOD VERDICT: ✓ PASS")
    else:
        reasons = []
        if fp.breach_type:
            reasons.append(f"{fp.breach_type} breach on {fp.breach_date}")
        if not fp.profit_target_hit:
            reasons.append(f"profit target not reached ({fp.return_pct:+.2f}%)")
        if not fp.min_trading_days_met:
            reasons.append(f"too few trading days ({fp.trading_days} < 4)")
        print(f"  FULL-PERIOD VERDICT: ✗ FAIL  ({'; '.join(reasons)})")

    # Rolling windows table
    print()
    rule()
    if not result.windows:
        print("  No rolling windows generated.")
        rule("═")
        print()
        return
    print(f"  Rolling {result.windows[0].label.split('→')[0].strip()[:7]}"
          f"… windows  ({len(result.windows)} total, "
          f"{window_months}-month each)")
    print(f"  Pass rate: {result.pass_rate_pct:.1f}%   "
          f"Breach rate: {result.breach_rate_pct:.1f}%   "
          f"Daily-loss breaches: {result.daily_loss_breach_count}   "
          f"Max-DD breaches: {result.max_dd_breach_count}")
    print()

    hdr = (f"  {'Window':<22}  {'Ret%':>6}  {'IntraDD':>7}  {'CloseDD':>7}  "
           f"{'Gap':>5}  {'WrstDay':>8}  {'TDays':>5}  {'Status':<16}")
    print(hdr)
    rule("-")

    for w in result.windows:
        status = "✓ PASS" if w.passed else (
            f"✗ {w.breach_type}" if w.breach_type else
            ("no breach/tgt" if not w.profit_target_hit else "✗ min days")
        )
        print(
            f"  {w.label:<22}  {w.return_pct:>+6.2f}%  "
            f"{w.max_intraday_dd_pct:>6.2f}%  {w.max_closed_dd_pct:>6.2f}%  "
            f"{w.dd_gap_pp:>5.2f}  {w.worst_daily_loss_pct:>8.2f}%  "
            f"{w.trading_days:>5}  {status:<16}"
        )

    rule("-")

    # Profitable-window stats
    profit_wins = [w for w in result.windows if w.profit_target_hit]
    if profit_wins:
        avg_d = np.mean([w.days_to_target for w in profit_wins
                         if w.days_to_target is not None])
        print(f"\n  Windows hitting +10%: {len(profit_wins)}/{len(result.windows)} "
              f"({100*len(profit_wins)/len(result.windows):.0f}%)   "
              f"Avg calendar days to target: {avg_d:.0f}")
    else:
        print(f"\n  No rolling window hit the +10% profit target.")

    # DD gap summary
    print(f"\n  Avg intraday−closed DD gap: {result.avg_dd_gap_pp:.3f}pp   "
          f"Max gap: {result.max_dd_gap_pp:.3f}pp")
    print(f"  → The floating exposure from open positions adds at most "
          f"{result.max_dd_gap_pp:.2f}pp of hidden drawdown not visible in "
          f"closed-trade balance.")
    rule("═")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_pkl(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FTMO accuracy evaluation wrapper for backtest results."
    )
    parser.add_argument("--eurusd",  required=True, type=Path,
                        help="Path to EURUSD BacktestResult PKL file")
    parser.add_argument("--gbpusd",  required=True, type=Path,
                        help="Path to GBPUSD BacktestResult PKL file")
    parser.add_argument("--window-months", type=int, default=3,
                        help="Rolling window length in months (default 3)")
    args = parser.parse_args()

    eu = _load_pkl(args.eurusd)
    gu = _load_pkl(args.gbpusd)

    evaluator = FTMOEvaluator(window_months=args.window_months)

    print("\nEvaluating EURUSD...")
    eu_result = evaluator.evaluate_single(eu, label="London Breakout — EURUSD")

    print("Evaluating GBPUSD...")
    gu_result = evaluator.evaluate_single(gu, label="London Breakout — GBPUSD")

    print("Evaluating Portfolio...")
    port_result = evaluator.evaluate_portfolio(eu, gu)

    print_report(eu_result,   window_months=args.window_months)
    print_report(gu_result,   window_months=args.window_months)
    print_report(port_result, window_months=args.window_months)


if __name__ == "__main__":
    main()
