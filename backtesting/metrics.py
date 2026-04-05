"""
Performance metrics for BacktestResult objects.

All monetary values in USD unless stated otherwise.
Returns and rates use the 0-100 scale (i.e. 5.0 means 5%).

Public API
----------
calculate_metrics(result)          -> dict
simulate_ftmo_pass_rate(equity, …) -> float
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtesting.backtester import BacktestResult

_TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def calculate_metrics(result: "BacktestResult") -> Dict:
    """
    Compute comprehensive performance metrics from a BacktestResult.

    Returns
    -------
    dict with keys:

    Scalars
        total_return_pct, cagr_pct, total_trades, win_rate_pct,
        avg_win_dollars, avg_loss_dollars, avg_win_r, avg_loss_r,
        expectancy_r, profit_factor, gross_profit, gross_loss,
        max_drawdown_pct, max_drawdown_duration_days, max_daily_drawdown_pct,
        sharpe_ratio, sortino_ratio, trades_per_month_avg,
        max_consecutive_wins, max_consecutive_losses,
        ftmo_pass_rate_pct, initial_balance, final_balance

    DataFrames / Series
        monthly_returns   (DataFrame: index=year, columns=month 1..12)
        win_rate_by_dow   (Series indexed by weekday name)
        win_rate_by_hour  (Series indexed by hour 0-23)
    """
    if not result.trades:
        return {"error": "No trades to analyse"}

    trades  = result.trades
    equity  = result.equity_curve
    initial = result.initial_balance
    final   = result.final_balance

    pnl_usd   = np.array([t.pnl_dollars for t in trades], dtype=float)
    wins_mask = pnl_usd > 0
    loss_mask = ~wins_mask

    # ── Returns ──────────────────────────────────────────────────────────
    total_return_pct = (final - initial) / initial * 100.0
    years            = _years(equity)
    cagr_pct         = ((final / initial) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0

    # ── Win / loss stats ─────────────────────────────────────────────────
    win_rate_pct    = float(wins_mask.sum()) / len(pnl_usd) * 100.0
    avg_win_dollars = float(pnl_usd[wins_mask].mean()) if wins_mask.any() else 0.0
    avg_loss_dollars = float(abs(pnl_usd[loss_mask].mean())) if loss_mask.any() else 0.0
    gross_profit    = float(pnl_usd[wins_mask].sum()) if wins_mask.any() else 0.0
    gross_loss      = float(abs(pnl_usd[loss_mask].sum())) if loss_mask.any() else 0.0
    profit_factor   = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # R multiples — treat average loss as 1R
    avg_win_r    = avg_win_dollars / avg_loss_dollars if avg_loss_dollars > 0 else 0.0
    avg_loss_r   = 1.0
    win_frac     = win_rate_pct / 100.0
    expectancy_r = win_frac * avg_win_r - (1.0 - win_frac) * avg_loss_r

    # ── Drawdown ─────────────────────────────────────────────────────────
    max_dd_pct, max_dd_days = _max_drawdown(equity)
    max_daily_dd_pct        = _max_daily_drawdown(result.daily_pnl, initial)

    # ── Risk-adjusted returns ─────────────────────────────────────────────
    sharpe, sortino = _risk_adjusted(equity)

    # ── Time-based analysis ───────────────────────────────────────────────
    monthly_returns     = _monthly_returns(equity)
    trades_per_month    = _trades_per_month(trades)
    win_rate_by_dow     = _win_rate_by_dow(trades)
    win_rate_by_hour    = _win_rate_by_hour(trades)

    # ── Streaks ──────────────────────────────────────────────────────────
    max_cons_wins, max_cons_losses = _consecutive(pnl_usd)

    # ── FTMO simulation ──────────────────────────────────────────────────
    ftmo_pass_rate = simulate_ftmo_pass_rate(equity, initial)

    return {
        # Returns
        "total_return_pct":            round(total_return_pct, 4),
        "cagr_pct":                    round(cagr_pct, 4),
        # Trade stats
        "total_trades":                len(trades),
        "win_rate_pct":                round(win_rate_pct, 2),
        "avg_win_dollars":             round(avg_win_dollars, 2),
        "avg_loss_dollars":            round(avg_loss_dollars, 2),
        "avg_win_r":                   round(avg_win_r, 4),
        "avg_loss_r":                  avg_loss_r,
        "expectancy_r":                round(expectancy_r, 4),
        "profit_factor":               round(profit_factor, 4),
        "gross_profit":                round(gross_profit, 2),
        "gross_loss":                  round(gross_loss, 2),
        # Drawdown
        "max_drawdown_pct":            round(max_dd_pct, 4),
        "max_drawdown_duration_days":  max_dd_days,
        "max_daily_drawdown_pct":      round(max_daily_dd_pct, 4),
        # Risk-adjusted
        "sharpe_ratio":                round(sharpe, 4),
        "sortino_ratio":               round(sortino, 4),
        # Time-based
        "monthly_returns":             monthly_returns,
        "trades_per_month_avg":        round(trades_per_month, 2),
        "win_rate_by_dow":             win_rate_by_dow,
        "win_rate_by_hour":            win_rate_by_hour,
        # Streaks
        "max_consecutive_wins":        max_cons_wins,
        "max_consecutive_losses":      max_cons_losses,
        # FTMO
        "ftmo_pass_rate_pct":          round(ftmo_pass_rate, 2),
        # Summary
        "initial_balance":             initial,
        "final_balance":               final,
    }


# ---------------------------------------------------------------------------
# FTMO challenge Monte Carlo simulation
# ---------------------------------------------------------------------------

def simulate_ftmo_pass_rate(
    equity: pd.Series,
    initial_balance: float,
    n_simulations: int = 1_000,
    window_trading_days: int = 30,
    profit_target_pct: float = 10.0,
    daily_loss_limit_pct: float = 5.0,
    total_loss_limit_pct: float = 10.0,
    seed: int = 42,
) -> float:
    """
    Monte Carlo FTMO challenge pass-rate estimate.

    Draws ``n_simulations`` random windows of ``window_trading_days`` business
    days from ``equity``, normalises each window to start at ``initial_balance``,
    then replays FTMO rules bar-by-bar:

    Pass: equity reaches +profit_target_pct before any limit is breached.
    Fail: daily equity drops ≥ daily_loss_limit_pct vs previous-day close, OR
          cumulative equity drops ≥ total_loss_limit_pct vs window start.

    Returns the pass rate as a percentage (0–100).
    """
    rng = np.random.default_rng(seed)

    # One equity sample per business day
    daily = equity.resample("B").last().dropna()
    n_days = len(daily)

    if n_days <= window_trading_days:
        return 0.0

    passes = 0
    max_start = n_days - window_trading_days - 1  # ensure full window fits

    for _ in range(n_simulations):
        start = int(rng.integers(0, max_start + 1))
        window = daily.iloc[start: start + window_trading_days + 1].values

        # Normalise to initial_balance
        norm     = window / window[0] * initial_balance
        prev_eq  = norm[0]
        outcome  = "inconclusive"

        for eq in norm[1:]:
            # Daily loss: today vs previous day close
            if (eq - prev_eq) / prev_eq * 100.0 <= -daily_loss_limit_pct:
                outcome = "fail"
                break
            # Total drawdown: vs window start
            if (eq - initial_balance) / initial_balance * 100.0 <= -total_loss_limit_pct:
                outcome = "fail"
                break
            # Profit target
            if (eq - initial_balance) / initial_balance * 100.0 >= profit_target_pct:
                outcome = "pass"
                break
            prev_eq = eq

        if outcome == "pass":
            passes += 1

    return passes / n_simulations * 100.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _years(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return (equity.index[-1] - equity.index[0]).days / 365.25


def _max_drawdown(equity: pd.Series) -> Tuple[float, int]:
    """Return (max_drawdown_pct, max_drawdown_duration_days)."""
    arr  = equity.values.astype(float)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / np.where(peak == 0, 1.0, peak) * 100.0
    max_dd = float(abs(dd.min())) if len(dd) else 0.0

    # Longest contiguous period below the running peak
    max_days  = 0
    start_idx = None
    for i, d in enumerate(dd):
        if d < 0 and start_idx is None:
            start_idx = i
        elif d >= 0 and start_idx is not None:
            days      = (equity.index[i] - equity.index[start_idx]).days
            max_days  = max(max_days, days)
            start_idx = None
    if start_idx is not None:  # still in drawdown at end
        days     = (equity.index[-1] - equity.index[start_idx]).days
        max_days = max(max_days, days)

    return max_dd, max_days


def _max_daily_drawdown(daily_pnl: pd.Series, initial_balance: float) -> float:
    """Worst single-day loss as % of that day's opening balance."""
    if daily_pnl.empty:
        return 0.0
    running = initial_balance
    max_dd  = 0.0
    for date in sorted(daily_pnl.index):
        pnl    = float(daily_pnl[date])
        if running > 0:
            loss_pct = -min(0.0, pnl) / running * 100.0
            max_dd   = max(max_dd, loss_pct)
        running += pnl
    return max_dd


def _risk_adjusted(equity: pd.Series) -> Tuple[float, float]:
    """Return (sharpe, sortino), annualised from daily returns (0-mean assumed risk-free)."""
    daily = equity.resample("D").last().dropna()
    rets  = daily.pct_change().dropna()
    if len(rets) < 2:
        return 0.0, 0.0

    mu    = float(rets.mean())
    sigma = float(rets.std())
    sharpe = mu / sigma * np.sqrt(_TRADING_DAYS_PER_YEAR) if sigma > 0 else 0.0

    down       = rets[rets < 0]
    down_sigma = float(down.std()) if len(down) > 1 else 0.0
    sortino    = mu / down_sigma * np.sqrt(_TRADING_DAYS_PER_YEAR) if down_sigma > 0 else 0.0

    return sharpe, sortino


def _monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Pivot table of monthly returns. Rows = year, Columns = month (1–12)."""
    monthly = equity.resample("M").last()
    m_rets  = monthly.pct_change().dropna() * 100.0
    if m_rets.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        "year":   m_rets.index.year,
        "month":  m_rets.index.month,
        "return": m_rets.values,
    })
    return df.pivot_table(values="return", index="year", columns="month", aggfunc="first")


def _trades_per_month(trades) -> float:
    if not trades:
        return 0.0
    # Convert to UTC-naive before period conversion to avoid timezone warning
    times = pd.Series([t.entry_time.tz_convert("UTC").tz_localize(None) for t in trades])
    periods = times.dt.to_period("M")
    return float(periods.value_counts().mean())


def _win_rate_by_dow(trades) -> pd.Series:
    """Win rate (%) grouped by weekday name of entry."""
    df = pd.DataFrame([
        {"dow": t.entry_time.day_name(), "win": t.pnl_dollars > 0}
        for t in trades
    ])
    return (df.groupby("dow")["win"].mean() * 100.0).rename("win_rate_pct")


def _win_rate_by_hour(trades) -> pd.Series:
    """Win rate (%) grouped by entry hour (US/Eastern)."""
    df = pd.DataFrame([
        {"hour": t.entry_time.hour, "win": t.pnl_dollars > 0}
        for t in trades
    ])
    return (df.groupby("hour")["win"].mean() * 100.0).rename("win_rate_pct")


def _consecutive(pnl_arr: np.ndarray) -> Tuple[int, int]:
    """Return (max_consecutive_wins, max_consecutive_losses)."""
    max_wins = max_losses = 0
    cur_wins = cur_losses = 0
    for p in pnl_arr:
        if p > 0:
            cur_wins  += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins   = 0
        max_wins   = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses
