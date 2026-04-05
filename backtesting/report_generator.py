"""
Backtesting Report Generator
=============================

Creates a self-contained, interactive HTML report from a BacktestResult.

Charts (all Plotly, embedded via CDN)
--------------------------------------
 1. Equity curve                          — zoomable line chart
 2. Drawdown (underwater)                 — area chart below zero
 3. Monthly returns heatmap               — year × month, colour-coded
 4. Trade P&L distribution                — histogram with commission impact
 5. Win rate by day of week               — horizontal bar chart
 6. Win rate by hour (Eastern)            — horizontal bar chart
 7. Cumulative P&L by strategy            — overlaid lines
 8. Rolling 30-day Sharpe ratio           — line chart
 9. Regime distribution                   — pie chart

Summary table at top
---------------------
 All metrics from metrics.py PLUS:
  - Total commission paid
  - Average slippage per trade
  - Net R:R vs theoretical R:R
  - Best / worst month
  - Strategy-by-strategy breakdown (trades, win%, avg P&L)

Usage
-----
  python -m backtesting.report_generator \\
      --result backtest_result.pkl \\
      --output reports/backtest_report.html \\
      [--regime-log regime_log.json]   # optional — {date: regime} mapping

  # Or call directly from Python:
  from backtesting.report_generator import generate_report
  generate_report(result, output_path)
"""

from __future__ import annotations

import argparse
import json
import pickle
from calendar import month_abbr
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.backtester import BacktestResult, Trade, PIP_VALUE_PER_LOT
from backtesting.metrics import calculate_metrics

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(
    result: BacktestResult,
    output_path: str | Path,
    regime_log: Optional[Dict[str, str]] = None,
    title: str = "FTMO Backtest Report",
) -> Path:
    """
    Build a self-contained HTML file at ``output_path``.

    Parameters
    ----------
    result :
        BacktestResult from Backtester.run().
    output_path :
        Destination HTML file path.
    regime_log :
        Optional dict of {date_str: regime_str} from RegimeFilter so we can
        render the regime distribution pie.  If omitted the pie is skipped.
    title :
        Page <title> and <h1> text.

    Returns
    -------
    Path  — resolved output path.
    """
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = calculate_metrics(result)
    if "error" in metrics:
        raise ValueError(f"Cannot generate report: {metrics['error']}")

    # ── 1. Equity curve ───────────────────────────────────────────────────
    eq = result.equity_curve
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=eq.index, y=eq.values, mode="lines",
        name="Equity", line=dict(color="#2196F3", width=1.5),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig_equity.add_hline(y=result.initial_balance,
                         line_dash="dash", line_color="gray", line_width=1)
    fig_equity.update_layout(**_layout("Equity Curve", "Equity (USD)", height=340))

    # ── 2. Drawdown (underwater) ──────────────────────────────────────────
    arr  = eq.values.astype(float)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / np.where(peak == 0, 1.0, peak) * 100.0
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=eq.index, y=dd, mode="lines", fill="tozeroy",
        name="Drawdown",
        line=dict(color="#F44336", width=1),
        fillcolor="rgba(244,67,54,0.15)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}%<extra></extra>",
    ))
    fig_dd.update_layout(**_layout("Drawdown (Underwater)", "Drawdown %", height=260))

    # ── 3. Monthly returns heatmap ────────────────────────────────────────
    mr = metrics.get("monthly_returns", pd.DataFrame())
    if isinstance(mr, pd.DataFrame) and not mr.empty:
        z     = mr.values.tolist()
        years = [str(y) for y in mr.index.tolist()]
        months = [month_abbr[m] for m in mr.columns.tolist()]
        text  = [[f"{v:.1f}%" if not np.isnan(v) else ""
                  for v in row] for row in mr.values]
        fig_heatmap = go.Figure(go.Heatmap(
            z=z, x=months, y=years, text=text, texttemplate="%{text}",
            colorscale=[[0, "#F44336"], [0.5, "#FFFFFF"], [1, "#4CAF50"]],
            zmid=0, showscale=True,
            hovertemplate="Year=%{y}  Month=%{x}<br>Return=%{z:.2f}%<extra></extra>",
        ))
        fig_heatmap.update_layout(**_layout("Monthly Returns Heatmap", height=max(200, 40 + 30 * len(years))))
    else:
        fig_heatmap = _empty_fig("Monthly Returns — insufficient data")

    # ── 4. Trade P&L distribution ─────────────────────────────────────────
    pnl_list = [t.pnl_dollars for t in result.trades]
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=pnl_list, nbinsx=40, name="P&L",
        marker_color=["#4CAF50" if p >= 0 else "#F44336" for p in pnl_list],
        marker_line_width=0.5,
        hovertemplate="Bin centre: $%{x:.0f}<br>Count: %{y}<extra></extra>",
    ))
    fig_hist.update_layout(**_layout("Trade P&L Distribution", "P&L (USD)", height=300))
    fig_hist.add_vline(x=0, line_dash="dash", line_color="white", line_width=1)

    # ── 5. Win rate by day of week ────────────────────────────────────────
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    wr_dow = metrics.get("win_rate_by_dow", pd.Series(dtype=float))
    if isinstance(wr_dow, pd.Series) and not wr_dow.empty:
        wr_dow = wr_dow.reindex([d for d in dow_order if d in wr_dow.index])
        fig_dow = go.Figure(go.Bar(
            y=wr_dow.index.tolist(), x=wr_dow.values, orientation="h",
            marker_color="#2196F3",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig_dow.add_vline(x=50, line_dash="dash", line_color="gray", line_width=1)
        fig_dow.update_layout(**_layout("Win Rate by Day of Week", "Win Rate %", height=260))
    else:
        fig_dow = _empty_fig("Win Rate by DoW — no data")

    # ── 6. Win rate by hour ───────────────────────────────────────────────
    wr_hour = metrics.get("win_rate_by_hour", pd.Series(dtype=float))
    if isinstance(wr_hour, pd.Series) and not wr_hour.empty:
        fig_hour = go.Figure(go.Bar(
            x=wr_hour.index.tolist(), y=wr_hour.values,
            marker_color="#9C27B0",
            hovertemplate="Hour %{x}:00 ET<br>Win rate: %{y:.1f}%<extra></extra>",
        ))
        fig_hour.add_hline(y=50, line_dash="dash", line_color="gray", line_width=1)
        fig_hour.update_layout(**_layout("Win Rate by Hour (US/Eastern)", "Win Rate %", height=280))
    else:
        fig_hour = _empty_fig("Win Rate by Hour — no data")

    # ── 7. Cumulative P&L by strategy ─────────────────────────────────────
    fig_strat = _cumulative_pnl_by_strategy(result.trades)

    # ── 8. Rolling 30-day Sharpe ──────────────────────────────────────────
    fig_sharpe = _rolling_sharpe(eq)

    # ── 9. Regime pie ─────────────────────────────────────────────────────
    fig_regime = _regime_pie(regime_log)

    # ── Execution cost metrics (new) ──────────────────────────────────────
    exec_metrics = _execution_cost_metrics(result)

    # ── Strategy breakdown table ──────────────────────────────────────────
    strat_rows = _strategy_breakdown(result.trades, result.initial_balance)

    # ── Monthly best / worst ──────────────────────────────────────────────
    monthly_df = metrics.get("monthly_returns", pd.DataFrame())
    best_month, worst_month = _best_worst_month(monthly_df)

    # ── Assemble HTML ─────────────────────────────────────────────────────
    plots = [
        fig_equity, fig_dd, fig_heatmap, fig_hist,
        fig_dow, fig_hour, fig_strat, fig_sharpe, fig_regime,
    ]
    html = _build_html(
        title=title,
        metrics=metrics,
        exec_metrics=exec_metrics,
        strat_rows=strat_rows,
        best_month=best_month,
        worst_month=worst_month,
        plots=plots,
        halt_reason=result.ftmo_halt_reason,
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved: {output_path}  ({output_path.stat().st_size / 1024:.0f} KB)")
    return output_path


# ---------------------------------------------------------------------------
# Execution cost analysis
# ---------------------------------------------------------------------------

def _execution_cost_metrics(result: BacktestResult) -> Dict[str, Any]:
    """
    Returns:
      total_commission      — sum of (pnl_gross - pnl_net) approximation
      avg_slippage_pips     — average slippage per trade (entry vs theoretical)
      theoretical_rr        — average config R:R ratio from trade geometry
      net_rr                — actual realised R:R from trade P&L
    """
    trades = result.trades
    if not trades:
        return {}

    # Commission: stored per-trade as (gross_usd - pnl_dollars).
    # We cannot reconstruct gross directly here without the lot_size and
    # pnl_pips already on the Trade object, so compute gross from pips.
    config = result.config
    # pip_size from config if available, else default
    pip_size = 0.0001  # fallback
    inst_cfg = config.get("instrument_config", {})
    if inst_cfg:
        pip_size = float(inst_cfg.get("pip_size", 0.0001))

    total_commission = 0.0
    slippages = []

    for t in trades:
        gross_usd = t.pnl_pips * PIP_VALUE_PER_LOT * t.lot_size
        commission = gross_usd - t.pnl_dollars
        total_commission += commission

        # Slippage is the difference between theoretical entry (bar close
        # with only spread) and actual entry.  We don't have bar close stored
        # on Trade, so slippage is approximated from remaining cost vs zero
        # (spread = config spread).  We report the total execution cost.
        slippages.append(max(0.0, commission / max(t.lot_size, 0.01)))

    avg_commission_per_trade = total_commission / len(trades)

    # Theoretical R:R from SL/TP geometry
    rr_list = []
    for t in trades:
        sl_dist = abs(t.entry_price - t.sl) / pip_size if t.sl else None
        tp_dist = abs(t.entry_price - t.tp) / pip_size if t.tp else None
        if sl_dist and tp_dist and sl_dist > 0:
            rr_list.append(tp_dist / sl_dist)
    theoretical_rr = float(np.mean(rr_list)) if rr_list else 0.0

    # Net realised R:R — wins' average vs losses' average in pips
    pnl_pips = np.array([t.pnl_pips for t in trades])
    wins  = pnl_pips[pnl_pips > 0]
    losses = pnl_pips[pnl_pips < 0]
    avg_win_pips  = float(wins.mean())  if len(wins)   else 0.0
    avg_loss_pips = float(abs(losses.mean())) if len(losses) else 0.0
    net_rr = avg_win_pips / avg_loss_pips if avg_loss_pips > 0 else 0.0

    return {
        "total_commission":          round(total_commission, 2),
        "avg_commission_per_trade":  round(avg_commission_per_trade, 2),
        "theoretical_rr":            round(theoretical_rr, 2),
        "net_rr_after_costs":        round(net_rr, 2),
        "rr_degradation":            round(theoretical_rr - net_rr, 2),
    }


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _cumulative_pnl_by_strategy(trades: List[Trade]):
    import plotly.graph_objects as go

    strategies = sorted({t.strategy_name for t in trades})
    colours    = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    fig = go.Figure()

    for i, sname in enumerate(strategies):
        st_trades = [t for t in trades if t.strategy_name == sname]
        st_trades.sort(key=lambda t: t.exit_time)
        cum_pnl = np.cumsum([t.pnl_dollars for t in st_trades])
        times   = [t.exit_time for t in st_trades]
        fig.add_trace(go.Scatter(
            x=times, y=cum_pnl, mode="lines", name=sname,
            line=dict(color=colours[i % len(colours)], width=2),
            hovertemplate=f"{sname}<br>%{{x|%Y-%m-%d}}<br>$%{{y:,.2f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(**_layout("Cumulative P&L by Strategy", "Cumulative P&L (USD)", height=320))
    return fig


def _rolling_sharpe(equity: pd.Series, window_days: int = 30):
    import plotly.graph_objects as go

    daily  = equity.resample("D").last().dropna()
    rets   = daily.pct_change().dropna()
    factor = np.sqrt(252)

    roll_mean  = rets.rolling(window_days).mean()
    roll_std   = rets.rolling(window_days).std()
    roll_sharpe = (roll_mean / roll_std.replace(0, np.nan)) * factor

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe.values, mode="lines",
        name=f"Rolling {window_days}d Sharpe",
        line=dict(color="#FF9800", width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_hline(y=1, line_dash="dot",  line_color="#4CAF50", line_width=1)
    fig.update_layout(**_layout(f"Rolling {window_days}-Day Sharpe Ratio", "Sharpe", height=280))
    return fig


def _regime_pie(regime_log: Optional[Dict[str, str]]):
    import plotly.graph_objects as go

    if not regime_log:
        return _empty_fig("Regime Distribution — no regime log provided")

    from collections import Counter
    counts   = Counter(regime_log.values())
    labels   = list(counts.keys())
    values   = list(counts.values())
    colours  = {
        "high_vol": "#FF9800", "normal": "#4CAF50",
        "low_vol":  "#2196F3", "dead":   "#9E9E9E",
    }
    marker_colours = [colours.get(l, "#607D8B") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.35,
        marker_colors=marker_colours,
        hovertemplate="%{label}<br>%{value} days (%{percent})<extra></extra>",
    ))
    fig.update_layout(**_layout("Regime Distribution", height=300))
    return fig


def _empty_fig(msg: str):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="gray"))
    fig.update_layout(**_layout(msg, height=200))
    return fig


def _layout(title: str, yaxis_title: str = "", height: int = 300) -> Dict:
    return dict(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        margin=dict(l=50, r=20, t=45, b=40),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#2d3561", zerolinecolor="#2d3561"),
        yaxis=dict(title=yaxis_title, gridcolor="#2d3561", zerolinecolor="#2d3561"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _strategy_breakdown(trades: List[Trade], initial_balance: float) -> List[Dict]:
    from collections import defaultdict
    groups = defaultdict(list)
    for t in trades:
        groups[t.strategy_name].append(t)

    rows = []
    for sname, st in sorted(groups.items()):
        pnl = [t.pnl_dollars for t in st]
        wins = [p for p in pnl if p > 0]
        rows.append({
            "strategy":       sname,
            "trades":         len(pnl),
            "win_rate":       f"{len(wins)/len(pnl)*100:.1f}%",
            "total_pnl":      f"${sum(pnl):,.2f}",
            "avg_pnl":        f"${np.mean(pnl):,.2f}",
            "avg_win":        f"${np.mean(wins):,.2f}" if wins else "—",
            "avg_loss":       f"${abs(np.mean([p for p in pnl if p <= 0])):,.2f}"
                              if any(p <= 0 for p in pnl) else "—",
            "return_pct":     f"{sum(pnl)/initial_balance*100:.2f}%",
        })
    return rows


def _best_worst_month(monthly_df: pd.DataFrame):
    if isinstance(monthly_df, pd.DataFrame) and not monthly_df.empty:
        flat  = monthly_df.stack().dropna()
        best  = flat.idxmax()   # (year, month)
        worst = flat.idxmin()
        bv, wv = flat[best], flat[worst]
        return (
            f"{month_abbr[best[1]]} {best[0]}: +{bv:.1f}%",
            f"{month_abbr[worst[1]]} {worst[0]}: {wv:.1f}%",
        )
    return "N/A", "N/A"


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _build_html(
    title: str,
    metrics: Dict,
    exec_metrics: Dict,
    strat_rows: List[Dict],
    best_month: str,
    worst_month: str,
    plots,
    halt_reason: Optional[str],
) -> str:
    import plotly.io as pio

    # Render each figure to a <div> with embedded JSON (no JS file needed)
    plot_divs = [
        pio.to_html(fig, include_plotlyjs="cdn" if i == 0 else False,
                    full_html=False, config={"responsive": True})
        for i, fig in enumerate(plots)
    ]

    # Pair plots into a 2-column grid
    grid_items = []
    for i in range(0, len(plot_divs), 2):
        left  = plot_divs[i]
        right = plot_divs[i+1] if i+1 < len(plot_divs) else ""
        grid_items.append(f"""
        <div class="row">
          <div class="col">{left}</div>
          <div class="col">{right}</div>
        </div>""")

    summary_table  = _render_summary_table(metrics, exec_metrics, best_month, worst_month)
    strategy_table = _render_strategy_table(strat_rows)
    halt_banner    = (f'<div class="halt-banner">⚠ BACKTEST HALTED: {halt_reason}</div>'
                      if halt_reason else "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title}</title>
  <style>
    :root {{ --bg: #0f0f23; --surface: #1a1a2e; --border: #2d3561;
             --text: #e0e0e0; --muted: #8888aa; --green: #4CAF50;
             --red: #F44336; --blue: #2196F3; --orange: #FF9800; }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text);
            font-family: "Segoe UI", system-ui, sans-serif; font-size: 14px; }}
    h1 {{ padding: 24px 32px 8px; font-size: 22px; color: var(--blue); }}
    h2 {{ padding: 20px 32px 10px; font-size: 16px; color: var(--muted);
          text-transform: uppercase; letter-spacing: 1px; border-top: 1px solid var(--border); }}
    .halt-banner {{ margin: 0 32px 16px; padding: 12px 16px;
                    background: rgba(244,67,54,0.15); border: 1px solid var(--red);
                    border-radius: 4px; color: var(--red); font-size: 13px; }}
    /* Metrics grid */
    .metrics-grid {{ display: grid;
                     grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                     gap: 12px; padding: 0 32px 24px; }}
    .metric-card {{ background: var(--surface); border: 1px solid var(--border);
                    border-radius: 6px; padding: 14px 16px; }}
    .metric-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase;
                     letter-spacing: 0.8px; margin-bottom: 6px; }}
    .metric-value {{ font-size: 18px; font-weight: 600; }}
    .positive {{ color: var(--green); }}
    .negative {{ color: var(--red); }}
    .neutral  {{ color: var(--text); }}
    /* Tables */
    table {{ width: calc(100% - 64px); margin: 0 32px 24px;
             border-collapse: collapse; font-size: 13px; }}
    th {{ background: var(--surface); color: var(--muted); padding: 8px 12px;
          text-align: left; border-bottom: 1px solid var(--border);
          text-transform: uppercase; font-size: 11px; letter-spacing: 0.8px; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid rgba(45,53,97,0.5); }}
    tr:hover td {{ background: rgba(33,150,243,0.05); }}
    /* Charts */
    .row {{ display: flex; gap: 0; padding: 0 24px; }}
    .col {{ flex: 1; min-width: 0; }}
  </style>
</head>
<body>
  <h1>📊 {title}</h1>
  {halt_banner}
  <h2>Performance Summary</h2>
  {summary_table}
  <h2>Strategy Breakdown</h2>
  {strategy_table}
  <h2>Charts</h2>
  {"".join(grid_items)}
</body>
</html>"""


def _render_summary_table(
    m: Dict,
    ec: Dict,
    best_month: str,
    worst_month: str,
) -> str:
    def _pct(v, fmt=".2f"):
        cls = "positive" if v >= 0 else "negative"
        return f'<span class="{cls}">{v:{fmt}}%</span>'

    def _usd(v):
        cls = "positive" if v >= 0 else "negative"
        return f'<span class="{cls}">${v:,.2f}</span>'

    def _r(v):
        cls = "positive" if v >= 0 else "negative"
        return f'<span class="{cls}">{v:.3f}R</span>'

    def _n(v, fmt=".2f"):
        return f'<span class="neutral">{v:{fmt}}</span>'

    cards = [
        ("Total Return",        _pct(m["total_return_pct"])),
        ("CAGR",                _pct(m["cagr_pct"])),
        ("Sharpe Ratio",        _n(m["sharpe_ratio"])),
        ("Sortino Ratio",       _n(m["sortino_ratio"])),
        ("Max Drawdown",        f'<span class="negative">{m["max_drawdown_pct"]:.2f}%</span>'),
        ("Max DD Duration",     f'<span class="neutral">{m["max_drawdown_duration_days"]}d</span>'),
        ("Max Daily DD",        f'<span class="negative">{m["max_daily_drawdown_pct"]:.2f}%</span>'),
        ("Win Rate",            _n(m["win_rate_pct"], ".1f") + "%"),
        ("Total Trades",        f'<span class="neutral">{m["total_trades"]}</span>'),
        ("Trades / Month",      _n(m["trades_per_month_avg"])),
        ("Profit Factor",       _n(m["profit_factor"])),
        ("Expectancy",          _r(m["expectancy_r"])),
        ("Avg Win",             _usd(m["avg_win_dollars"])),
        ("Avg Loss",            _usd(-m["avg_loss_dollars"])),
        ("Gross Profit",        _usd(m["gross_profit"])),
        ("Gross Loss",          _usd(-m["gross_loss"])),
        ("Best Month",          f'<span class="positive">{best_month}</span>'),
        ("Worst Month",         f'<span class="negative">{worst_month}</span>'),
        ("Max Consec. Wins",    _n(m["max_consecutive_wins"], "d")),
        ("Max Consec. Losses",  _n(m["max_consecutive_losses"], "d")),
        ("FTMO Pass Rate",      _pct(m["ftmo_pass_rate_pct"])),
        ("Initial Balance",     f'<span class="neutral">${m["initial_balance"]:,.2f}</span>'),
        ("Final Balance",       _usd(m["final_balance"] - m["initial_balance"])),
        # Execution cost section
        ("Total Commission",    f'<span class="negative">${ec.get("total_commission", 0):,.2f}</span>'),
        ("Avg Commission/Trade",f'<span class="negative">${ec.get("avg_commission_per_trade", 0):,.2f}</span>'),
        ("Theoretical R:R",     f'<span class="neutral">{ec.get("theoretical_rr", 0):.2f}</span>'),
        ("Net R:R (After Costs)",f'<span class="neutral">{ec.get("net_rr_after_costs", 0):.2f}</span>'),
        ("R:R Degradation",     f'<span class="negative">-{ec.get("rr_degradation", 0):.2f}</span>'),
    ]

    items = "".join(f"""
      <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
      </div>""" for label, value in cards)

    return f'<div class="metrics-grid">{items}</div>'


def _render_strategy_table(rows: List[Dict]) -> str:
    if not rows:
        return "<p style='padding:0 32px 24px;color:var(--muted)'>No strategy data.</p>"

    headers = ["Strategy", "Trades", "Win Rate", "Total P&L", "Avg P&L", "Avg Win", "Avg Loss", "Return"]
    keys    = ["strategy", "trades", "win_rate", "total_pnl", "avg_pnl", "avg_win", "avg_loss", "return_pct"]

    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join(
        "<tr>" + "".join(f"<td>{row.get(k, '—')}</td>" for k in keys) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an HTML backtest report from a pickled BacktestResult."
    )
    parser.add_argument("--result", "-r", required=True, type=Path,
                        help="Path to the pickled BacktestResult (.pkl)")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="Output HTML file path")
    parser.add_argument("--regime-log", type=Path, default=None,
                        help="Optional JSON file: {date_str: regime_str}")
    parser.add_argument("--title", default="FTMO Backtest Report")
    args = parser.parse_args()

    with open(args.result.expanduser(), "rb") as fh:
        result = pickle.load(fh)

    regime_log = None
    if args.regime_log and args.regime_log.exists():
        with open(args.regime_log) as fh:
            regime_log = json.load(fh)

    generate_report(result, args.output, regime_log=regime_log, title=args.title)


if __name__ == "__main__":
    _cli()
