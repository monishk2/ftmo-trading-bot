"""
FTMO Trading Bot — Main Orchestrator
=====================================

Usage
-----
  Backtest (all strategies, full date range):
    python main.py --mode backtest --strategy all

  Backtest (single strategy, date-filtered):
    python main.py --mode backtest --strategy london_breakout \\
        --instrument EURUSD --start 2022-01-01 --end 2024-12-31

  Walk-forward optimisation:
    python main.py --mode walkforward --strategy london_breakout \\
        --instrument EURUSD

  Live trading via cTrader:
    python main.py --mode live --instrument EURUSD --phase challenge

  Paper trading (mock connector, no real orders):
    python main.py --mode live --simulate

  Generate HTML report from a saved backtest pickle:
    python main.py --mode report --result results/my_result.pkl

Architecture
------------
  backtest  → loads data → Backtester per strategy → RegimeFilter gates
              each trading day → HTML report saved to reports/

  walkforward → WalkForwardOptimizer → JSON result + console table

  live      → Guardian + OrderManager + (MockOrReal)Connector
              Schedule loop runs in main thread; graceful Ctrl+C shutdown.

  report    → loads pickle/parquet → ReportGenerator → HTML

Scheduler (live mode)
---------------------
  02:30 ET  Regime filter — determine active strategies for today
  02:45 ET  Pre-compute Asian range (London Breakout)
  03:00 ET  London open — monitor for breakout entries (until 05:30)
  06:00 ET  Scan for FVGs from London session
  09:30 ET  NY open — monitor FVG limit orders (until 12:00)
  12:00 ET  Cancel unfilled FVG orders; London time-stop fires
  16:00 ET  Close everything; generate daily summary
  Fri 16:00 Weekend shutdown + guardian state save

Guardian equity check runs every 60 seconds in the main loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import signal
import sys
import time as time_mod
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz

# ---------------------------------------------------------------------------
# Logging — file + console, set up BEFORE any module imports
# ---------------------------------------------------------------------------
_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_DIR / "main.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

_EASTERN = pytz.timezone("US/Eastern")
_CONFIG_DIR = Path("config")
_DATA_DIR   = Path("data") / "historical"
_RESULTS_DIR = Path("results")
_REPORTS_DIR = Path("reports")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="FTMO Prop Firm Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode", required=True,
        choices=["backtest", "walkforward", "live", "report"],
        help="Execution mode",
    )
    p.add_argument(
        "--strategy", default="all",
        choices=["all", "london_breakout", "fvg_retracement"],
        help="Strategy to run (backtest / walkforward / live)",
    )
    p.add_argument(
        "--instrument", default="EURUSD",
        choices=["EURUSD", "GBPUSD"],
        help="Trading instrument",
    )
    p.add_argument(
        "--phase", default="challenge",
        choices=["challenge", "verification", "funded"],
        help="FTMO account phase (affects risk rules)",
    )
    p.add_argument(
        "--start", default=None,
        help="Backtest start date YYYY-MM-DD (default: use all available data)",
    )
    p.add_argument(
        "--end", default=None,
        help="Backtest end date YYYY-MM-DD (default: use all available data)",
    )
    p.add_argument(
        "--initial-balance", type=float, default=10_000.0,
        help="Starting account balance in USD",
    )
    p.add_argument(
        "--result", type=Path, default=None,
        help="Path to saved backtest pickle (report mode only)",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output path override (report HTML or walk-forward JSON)",
    )
    p.add_argument(
        "--simulate", action="store_true",
        help="Live mode: use MockCTraderConnector (no real orders)",
    )
    p.add_argument(
        "--in-sample-months", type=int, default=6,
        help="Walk-forward IS window length in months",
    )
    p.add_argument(
        "--oos-months", type=int, default=3,
        help="Walk-forward OOS window length in months",
    )
    p.add_argument(
        "--step-months", type=int, default=3,
        help="Walk-forward step size in months",
    )
    p.add_argument(
        "--min-trades", type=int, default=20,
        help="Minimum trades per walk-forward IS window",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output (useful for scripting)",
    )
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(instrument: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Load 15-min OHLCV Parquet for instrument, optionally date-filtered."""
    candidates = [
        _DATA_DIR / f"{instrument}_15m.parquet",
        _DATA_DIR / f"{instrument.lower()}_15m.parquet",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        # Try CSV fallback
        csv_candidates = list(_DATA_DIR.glob(f"{instrument}*.csv"))
        if not csv_candidates:
            logger.error(
                "No data file found for %s in %s. "
                "Run: python data/download_data.py --help",
                instrument, _DATA_DIR,
            )
            sys.exit(1)
        path = csv_candidates[0]
        logger.info("Loading CSV data from %s", path)
        df = pd.read_csv(path)
        # Minimal normalisation — download_data.py stores clean Parquet
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.set_index("datetime")
        df.columns = [c.lower() for c in df.columns]
    else:
        logger.info("Loading Parquet data from %s", path)
        df = pd.read_parquet(path)
        if "datetime" in df.columns:
            df = df.set_index("datetime")

    # Ensure UTC-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.columns = [c.lower() for c in df.columns]

    # Date filter
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end + " 23:59:59", tz="UTC")]

    if df.empty:
        logger.error("DataFrame is empty after date filtering (%s to %s)", start, end)
        sys.exit(1)

    logger.info(
        "Loaded %d bars for %s  (%s → %s)",
        len(df), instrument,
        df.index[0].date(), df.index[-1].date(),
    )
    return df


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_strategy_params() -> Dict[str, Any]:
    return _load_json(_CONFIG_DIR / "strategy_params.json")


def _load_instruments() -> Dict[str, Any]:
    return _load_json(_CONFIG_DIR / "instruments.json")


def _load_ftmo_rules() -> Dict[str, Any]:
    return _load_json(_CONFIG_DIR / "ftmo_rules.json")


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def _build_strategy(
    name: str,
    params: Dict[str, Any],
    instrument_cfg: Dict[str, Any],
):
    if name == "london_breakout":
        from strategies.london_open_breakout import LondonOpenBreakout
        s = LondonOpenBreakout()
        s.setup(params["london_open_breakout"], instrument_cfg)
        return s
    if name == "fvg_retracement":
        from strategies.fvg_retracement import FVGRetracement
        s = FVGRetracement()
        s.setup(params["fvg_retracement"], instrument_cfg)
        return s
    raise ValueError(f"Unknown strategy: {name!r}")


def _strategy_names(strategy_arg: str) -> List[str]:
    if strategy_arg == "all":
        return ["london_breakout", "fvg_retracement"]
    return [strategy_arg]


# ---------------------------------------------------------------------------
# MODE: backtest
# ---------------------------------------------------------------------------

def _run_backtest(args: argparse.Namespace) -> None:
    from backtesting.backtester import Backtester
    from backtesting.metrics import calculate_metrics
    from backtesting.report_generator import generate_report
    from strategies.regime_filter import RegimeFilter

    _RESULTS_DIR.mkdir(exist_ok=True)
    _REPORTS_DIR.mkdir(exist_ok=True)

    params       = _load_strategy_params()
    instruments  = _load_instruments()
    instr_cfg    = instruments[args.instrument]

    df = _load_data(args.instrument, args.start, args.end)

    # Regime filter (shared across strategies, evaluated per day in backtester)
    rf = RegimeFilter()
    rf.setup(params["regime_filter"])

    strategy_names = _strategy_names(args.strategy)
    all_results = {}

    print(f"\n{'='*60}")
    print(f"  BACKTEST  |  {args.instrument}  |  phase={args.phase}")
    print(f"  Balance: ${args.initial_balance:,.2f}")
    if args.start or args.end:
        print(f"  Date range: {args.start or 'start'} → {args.end or 'end'}")
    print(f"{'='*60}\n")

    for strat_name in strategy_names:
        strategy = _build_strategy(strat_name, params, instr_cfg)

        print(f"Running {strat_name} on {args.instrument}...", flush=True)

        bt = Backtester(
            strategy=strategy,
            df=df,
            instrument=args.instrument,
            initial_balance=args.initial_balance,
            phase=args.phase,
        )
        result = bt.run()
        all_results[strat_name] = result

        m = calculate_metrics(result)

        # Print strategy summary
        n_trades = len(result.trades)
        wr       = m.get("win_rate_pct", 0)
        sharpe   = m.get("sharpe_ratio", float("nan"))
        ret_pct  = (result.final_balance - args.initial_balance) / args.initial_balance * 100
        max_dd   = m.get("max_drawdown_pct", float("nan"))

        print(f"  Trades:     {n_trades}")
        print(f"  Win rate:   {wr:.1f}%")
        print(f"  Return:     {ret_pct:+.2f}%")
        print(f"  Sharpe:     {sharpe:.3f}")
        print(f"  Max DD:     {max_dd:.2f}%")
        if result.ftmo_halt_reason:
            print(f"  FTMO HALT:  {result.ftmo_halt_reason}")
        print()

        # Save raw result
        result_path = _RESULTS_DIR / f"{strat_name}_{args.instrument}_backtest.pkl"
        with open(result_path, "wb") as fh:
            pickle.dump(result, fh)
        logger.info("Saved result to %s", result_path)

        # HTML report (skip if no trades)
        if not result.trades:
            print("  Report:     (skipped — no trades)")
            print()
        else:
            report_path = (
                args.output
                if (args.output and len(strategy_names) == 1)
                else _REPORTS_DIR / f"{strat_name}_{args.instrument}_report.html"
            )
            try:
                generate_report(result, report_path, title=f"{strat_name} — {args.instrument}")
                print(f"  Report:     {report_path}")
            except Exception as exc:
                print(f"  Report:     (error — {exc})")
            print()

    # Regime distribution summary
    _print_regime_summary(df, rf)


def _print_regime_summary(df: pd.DataFrame, rf: RegimeFilter) -> None:
    """Show regime breakdown for the backtest period."""
    try:
        counts: Dict[str, int] = {}
        trading_days = sorted({d.date() for d in df.index.to_pydatetime()})
        for day in trading_days[-90:]:   # last 90 days only to keep it fast
            regime = rf.get_regime(df, day)
            counts[regime] = counts.get(regime, 0) + 1

        total = sum(counts.values())
        if total:
            print("Regime distribution (last 90 trading days):")
            for regime, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"  {regime:<12} {cnt:>3}d  ({cnt/total*100:.0f}%)")
            print()
    except Exception:
        pass   # regime summary is best-effort, never crash the run


# ---------------------------------------------------------------------------
# MODE: walkforward
# ---------------------------------------------------------------------------

def _run_walkforward(args: argparse.Namespace) -> None:
    from backtesting.walk_forward import run_walk_forward, save_result

    _RESULTS_DIR.mkdir(exist_ok=True)

    df = _load_data(args.instrument, args.start, args.end)

    strategy_names = _strategy_names(args.strategy)
    if len(strategy_names) > 1:
        logger.error("Walk-forward requires exactly one strategy. Use --strategy.")
        sys.exit(1)

    strat_name = strategy_names[0]

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD  |  {strat_name}  |  {args.instrument}")
    print(f"  IS={args.in_sample_months}m  OOS={args.oos_months}m  step={args.step_months}m")
    print(f"{'='*60}\n")

    result = run_walk_forward(
        df=df,
        strategy_name=strat_name,
        instrument=args.instrument,
        initial_balance=args.initial_balance,
        in_sample_months=args.in_sample_months,
        oos_months=args.oos_months,
        step_months=args.step_months,
        min_trades=args.min_trades,
        phase=args.phase,
        verbose=not args.quiet,
    )

    out = (
        args.output
        or _RESULTS_DIR / f"walkforward_{strat_name}_{args.instrument}.json"
    )
    save_result(result, out)
    print(f"\nResult saved to: {out}")


# ---------------------------------------------------------------------------
# MODE: report
# ---------------------------------------------------------------------------

def _run_report(args: argparse.Namespace) -> None:
    from backtesting.report_generator import generate_report

    _REPORTS_DIR.mkdir(exist_ok=True)

    if args.result is None:
        logger.error("--result PATH required for report mode")
        sys.exit(1)
    if not args.result.exists():
        logger.error("Result file not found: %s", args.result)
        sys.exit(1)

    print(f"\nLoading result from {args.result}...")
    with open(args.result, "rb") as fh:
        result = pickle.load(fh)

    out = args.output or _REPORTS_DIR / (args.result.stem + "_report.html")

    print(f"Generating HTML report → {out}")
    generate_report(result, out)
    print(f"Done.  Open in browser: {out.resolve()}")


# ---------------------------------------------------------------------------
# MODE: live
# ---------------------------------------------------------------------------

def _run_live(args: argparse.Namespace) -> None:
    from backtesting.metrics import calculate_metrics
    from execution.ctrader_connector import MockCTraderConnector
    from execution.ftmo_guardian import FTMOGuardian, TradeRecord
    from execution.order_manager import OrderManager
    from strategies.regime_filter import RegimeFilter

    _RESULTS_DIR.mkdir(exist_ok=True)

    instruments  = _load_instruments()
    params       = _load_strategy_params()
    ftmo_rules   = _load_ftmo_rules()
    instr_cfg    = instruments[args.instrument]

    mode_label = "SIMULATION (Mock Connector)" if args.simulate else "LIVE (cTrader)"
    print(f"\n{'='*60}")
    print(f"  LIVE TRADING  |  {mode_label}")
    print(f"  Instrument:  {args.instrument}")
    print(f"  Phase:       {args.phase}")
    print(f"  Balance:     ${args.initial_balance:,.2f}")
    print(f"{'='*60}\n")

    # ── Connector ─────────────────────────────────────────────────────
    if args.simulate:
        connector = MockCTraderConnector(initial_balance=args.initial_balance)
        logger.info("Using MockCTraderConnector — no real orders will be placed")
    else:
        # Real cTrader connector would be instantiated here.
        # For now, we guard against accidental live execution without real auth.
        logger.error(
            "Real cTrader connector not yet configured. "
            "Pass --simulate for paper trading. "
            "To add live trading: implement CTraderConnector subclass "
            "using the ctrader-open-api SDK and wire it here."
        )
        sys.exit(1)

    if not connector.connect():
        logger.error("Failed to connect to broker")
        sys.exit(1)

    # ── Guardian ──────────────────────────────────────────────────────
    state_path = _RESULTS_DIR / "guardian_state.json"
    guardian = FTMOGuardian(
        initial_balance=args.initial_balance,
        config=ftmo_rules,
        mode=args.phase,
        state_path=state_path,
    )

    # ── Order manager ─────────────────────────────────────────────────
    order_mgr = OrderManager(
        guardian=guardian,
        connector=connector,
        instruments_config=instruments,
    )

    # ── Regime filter ─────────────────────────────────────────────────
    rf = RegimeFilter()
    rf.setup(params["regime_filter"])

    # ── State ─────────────────────────────────────────────────────────
    live_state = _LiveState(
        args=args,
        params=params,
        instr_cfg=instr_cfg,
        guardian=guardian,
        order_mgr=order_mgr,
        connector=connector,
        rf=rf,
    )

    # ── Graceful shutdown on Ctrl+C ───────────────────────────────────
    def _shutdown(sig, frame):
        print("\n\nCtrl+C received — shutting down gracefully...")
        live_state.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"Guardian state: {guardian.get_status()}")
    print("\nScheduler running. Press Ctrl+C to stop.\n")

    # ── Main event loop ───────────────────────────────────────────────
    live_state.run()


# ---------------------------------------------------------------------------
# Live trading state machine
# ---------------------------------------------------------------------------

class _LiveState:
    """
    Encapsulates all live-mode state and the scheduler loop.

    The scheduler ticks every second, fires time-based jobs when due,
    and runs the guardian equity check every 60 seconds.
    """

    # Scheduled events: (hour, minute) Eastern → method name
    _SCHEDULE = [
        (2,  30, "_job_regime_filter"),
        (2,  45, "_job_precompute_asian_range"),
        (3,   0, "_job_london_open"),
        (6,   0, "_job_fvg_scan"),
        (9,  30, "_job_ny_open"),
        (12,  0, "_job_midday_cutoff"),
        (16,  0, "_job_daily_close"),
    ]

    def __init__(
        self,
        args,
        params,
        instr_cfg,
        guardian,
        order_mgr,
        connector,
        rf,
    ):
        self._args       = args
        self._params     = params
        self._instr_cfg  = instr_cfg
        self._guardian   = guardian
        self._order_mgr  = order_mgr
        self._connector  = connector
        self._rf         = rf

        self._active_strategies: List[str] = ["london_breakout", "fvg_retracement"]
        self._fired_jobs_today: set         = set()
        self._last_equity_check: float      = 0.0
        self._last_midnight_reset: date     = date.min
        self._running: bool                 = True

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        while self._running:
            now_et = datetime.now(_EASTERN)
            today  = now_et.date()

            # Daily midnight reset
            if today > self._last_midnight_reset:
                self._do_midnight_reset(now_et)
                self._last_midnight_reset = today

            # Scheduled jobs
            self._tick_schedule(now_et)

            # Guardian equity check every 60 seconds
            if time_mod.monotonic() - self._last_equity_check >= 60:
                self._check_equity()
                self._last_equity_check = time_mod.monotonic()

            # Manage open positions (SL/TP/time-stop)
            try:
                prices = self._get_current_prices()
                self._order_mgr.manage_open_positions(
                    pd.Timestamp.now(tz="UTC"), prices
                )
            except Exception as exc:
                logger.error("Error managing positions: %s", exc)

            # Guardian permanent halt → exit
            if self._guardian._permanently_halted:
                logger.critical(
                    "PERMANENT HALT active: %s — shutting down",
                    self._guardian._halt_reason,
                )
                self.shutdown()
                break

            time_mod.sleep(1)

    # ------------------------------------------------------------------ #
    # Scheduler                                                            #
    # ------------------------------------------------------------------ #

    def _tick_schedule(self, now_et: datetime) -> None:
        for hour, minute, method in self._SCHEDULE:
            key = (now_et.date(), hour, minute)
            if key in self._fired_jobs_today:
                continue
            if now_et.hour == hour and now_et.minute >= minute:
                self._fired_jobs_today.add(key)
                logger.info("Firing scheduled job %s at %s ET", method, now_et.strftime("%H:%M"))
                try:
                    getattr(self, method)(now_et)
                except Exception as exc:
                    logger.error("Scheduled job %s failed: %s", method, exc)

    # ------------------------------------------------------------------ #
    # Scheduled jobs                                                       #
    # ------------------------------------------------------------------ #

    def _job_regime_filter(self, now_et: datetime) -> None:
        """02:30 ET — determine which strategies are active today."""
        try:
            # Load recent data for regime calculation
            df = _load_data(self._args.instrument, start=None, end=None)
            today = now_et.date()
            self._active_strategies = self._rf.get_active_strategies(df, today)
            logger.info("Regime filter: active strategies today = %s", self._active_strategies)
            print(f"[{now_et.strftime('%H:%M')} ET] Regime filter → active: {self._active_strategies}")
        except Exception as exc:
            logger.error("Regime filter failed — defaulting to all strategies: %s", exc)
            self._active_strategies = ["london_breakout", "fvg_retracement"]

    def _job_precompute_asian_range(self, now_et: datetime) -> None:
        """02:45 ET — pre-flight check / log for London Breakout."""
        if "london_breakout" not in self._active_strategies:
            logger.info("LondonOpenBreakout not active today — skipping Asian range pre-compute")
            return
        print(f"[{now_et.strftime('%H:%M')} ET] London Breakout active — monitoring for breakout entry from 03:00")

    def _job_london_open(self, now_et: datetime) -> None:
        """03:00 ET — London open; live signal generation handled by manage_open_positions."""
        if "london_breakout" not in self._active_strategies:
            print(f"[{now_et.strftime('%H:%M')} ET] London open — LondonOpenBreakout INACTIVE (regime filter)")
            return
        print(f"[{now_et.strftime('%H:%M')} ET] London open — monitoring breakout entries until 05:30 ET")
        self._try_generate_signals("london_breakout", now_et)

    def _job_fvg_scan(self, now_et: datetime) -> None:
        """06:00 ET — scan for FVGs from London session."""
        if "fvg_retracement" not in self._active_strategies:
            print(f"[{now_et.strftime('%H:%M')} ET] FVG scan — FVGRetracement INACTIVE (regime filter)")
            return
        print(f"[{now_et.strftime('%H:%M')} ET] FVG scan — detecting Fair Value Gaps from London session")
        self._try_generate_signals("fvg_retracement", now_et)

    def _job_ny_open(self, now_et: datetime) -> None:
        """09:30 ET — NY open; FVG entries monitored until 12:00."""
        if "fvg_retracement" not in self._active_strategies:
            return
        print(f"[{now_et.strftime('%H:%M')} ET] NY open — monitoring FVG retracement entries until 12:00 ET")

    def _job_midday_cutoff(self, now_et: datetime) -> None:
        """12:00 ET — cancel unfilled FVG orders; London time-stop fires."""
        print(f"[{now_et.strftime('%H:%M')} ET] Midday cutoff — London positions time-stopped; FVG entries cancelled")
        # manage_open_positions handles actual time-stops; this is the log marker

    def _job_daily_close(self, now_et: datetime) -> None:
        """16:00 ET — close everything, generate daily summary."""
        weekday = now_et.weekday()  # 4 = Friday
        if weekday == 4:
            print(f"[{now_et.strftime('%H:%M')} ET] Friday close — weekend shutdown")
            self._order_mgr.close_all_positions("friday_weekend_shutdown")
            self._print_daily_summary()
            self._guardian.save_state(_RESULTS_DIR / "guardian_state.json")
            self._running = False
            return

        print(f"[{now_et.strftime('%H:%M')} ET] Daily close — closing all positions")
        self._order_mgr.close_all_positions("daily_close_1600")
        self._print_daily_summary()

    # ------------------------------------------------------------------ #
    # Signal generation (live tick path)                                   #
    # ------------------------------------------------------------------ #

    def _try_generate_signals(self, strategy_name: str, now_et: datetime) -> None:
        """
        Generate signals from recent OHLCV data and submit any entries
        to order_manager.  In a production system this would be driven by
        a streaming price feed; here we generate from the latest stored data.
        """
        if self._guardian._daily_halted or self._guardian._permanently_halted:
            return
        try:
            df = _load_data(self._args.instrument, start=None, end=None)
            strategy = _build_strategy(
                strategy_name,
                self._params,
                self._instr_cfg,
            )
            signals_df = strategy.generate_signals(df)

            # Look for a signal on the most recent bar
            last = signals_df.iloc[-1]
            if last.get("signal", 0) != 0:
                price_info = self._get_current_prices().get(self._args.instrument, {})
                entry = float(price_info.get("ask", last["close"]))
                signal = {
                    "instrument":  self._args.instrument,
                    "direction":   int(last["signal"]),
                    "entry_price": entry,
                    "sl_price":    float(last["sl_price"]),
                    "tp_price":    float(last["tp_price"]),
                    "lot_size":    None,
                    "risk_pct":    float(
                        self._params[
                            "london_open_breakout"
                            if strategy_name == "london_breakout"
                            else "fvg_retracement"
                        ]["risk_per_trade_pct"]
                    ),
                    "time_stop":   last.get("time_stop"),
                    "timestamp":   pd.Timestamp.now(tz="UTC"),
                }
                placed = self._order_mgr.execute_trade(signal, strategy_name)
                if placed:
                    print(
                        f"  → Trade placed: {signal['instrument']} "
                        f"{'LONG' if signal['direction']==1 else 'SHORT'}"
                    )
        except Exception as exc:
            logger.error("Signal generation failed for %s: %s", strategy_name, exc)

    # ------------------------------------------------------------------ #
    # Equity monitoring                                                    #
    # ------------------------------------------------------------------ #

    def _check_equity(self) -> None:
        """Pull account info from connector and update guardian."""
        try:
            info = self._connector.get_account_info()
            equity    = float(info.get("equity", self._guardian._equity))
            open_pnl  = float(info.get("unrealised", 0.0))
            n_open    = int(info.get("open_positions", self._guardian._open_positions))

            self._guardian.update_equity(
                current_equity=equity,
                open_pnl=open_pnl,
                open_positions_count=n_open,
            )

            status = self._guardian.get_status()
            logger.debug(
                "Equity check: equity=%.2f  daily_dd=%.2f%%  total_dd=%.2f%%  "
                "open=%d  daily_halted=%s",
                equity,
                status["daily_drawdown_pct"],
                status["total_drawdown_pct"],
                n_open,
                status["daily_halted"],
            )

            if status["daily_halted"] and not status["permanently_halted"]:
                logger.warning(
                    "Daily halt active — closing all positions: %s",
                    status["halt_reason"],
                )
                self._order_mgr.close_all_positions("guardian_daily_halt")

        except Exception as exc:
            logger.error("Equity check failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Midnight reset                                                       #
    # ------------------------------------------------------------------ #

    def _do_midnight_reset(self, now_et: datetime) -> None:
        """Call guardian.daily_reset() at the start of each new trading day."""
        try:
            info    = self._connector.get_account_info()
            balance = float(info.get("balance", self._guardian._balance))
            equity  = float(info.get("equity",  self._guardian._equity))
            self._guardian.daily_reset(balance, equity)
            self._fired_jobs_today = set()   # clear job-fire records for new day
            logger.info(
                "Midnight reset: balance=%.2f equity=%.2f | %s",
                balance, equity, now_et.strftime("%Y-%m-%d"),
            )
            print(f"\n[MIDNIGHT RESET] New trading day: {now_et.date()} | "
                  f"Balance: ${balance:,.2f}")
        except Exception as exc:
            logger.error("Midnight reset failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Daily summary                                                        #
    # ------------------------------------------------------------------ #

    def _print_daily_summary(self) -> None:
        status = self._guardian.get_status()
        print(f"\n{'─'*50}")
        print("  DAILY SUMMARY")
        print(f"{'─'*50}")
        print(f"  Balance:        ${status['current_balance']:>10,.2f}")
        print(f"  Equity:         ${status['current_equity']:>10,.2f}")
        print(f"  Daily P&L:      ${status['daily_pnl']:>+10,.2f}")
        print(f"  Daily DD:       {status['daily_drawdown_pct']:>8.2f}%  "
              f"(limit {status['daily_drawdown_limit_pct']:.1f}%)")
        print(f"  Total DD:       {status['total_drawdown_pct']:>8.2f}%  "
              f"(limit {status['total_drawdown_limit_pct']:.1f}%)")
        print(f"  Active strats:  {self._active_strategies}")
        if status["daily_halted"]:
            print(f"  HALTED:         {status['halt_reason']}")
        print(f"{'─'*50}\n")

    # ------------------------------------------------------------------ #
    # Price feed                                                           #
    # ------------------------------------------------------------------ #

    def _get_current_prices(self) -> Dict[str, Dict[str, float]]:
        """
        Return current bid/ask prices.

        In simulation mode the mock connector holds prices from the last
        update_prices() call.  In live mode this would poll the cTrader
        streaming feed.
        """
        price = self._connector.get_symbol_price(self._args.instrument)
        if price["bid"] == 0.0 and price["ask"] == 0.0:
            # No price set yet — return a neutral placeholder so position
            # management doesn't crash on startup
            return {}
        return {self._args.instrument: price}

    # ------------------------------------------------------------------ #
    # Shutdown                                                             #
    # ------------------------------------------------------------------ #

    def shutdown(self) -> None:
        """Graceful shutdown: close positions, save guardian state."""
        self._running = False
        print("\nShutting down...")

        try:
            if self._order_mgr.open_position_count > 0:
                response = input(
                    f"  {self._order_mgr.open_position_count} open position(s) — "
                    "close them? [Y/n] "
                ).strip().lower()
                if response in ("", "y", "yes"):
                    self._order_mgr.close_all_positions("graceful_shutdown")
                    print("  All positions closed.")
                else:
                    print("  Positions left open.")
        except (EOFError, KeyboardInterrupt):
            # Non-interactive context (piped / CI) — close everything
            self._order_mgr.close_all_positions("forced_shutdown")

        self._guardian.save_state(_RESULTS_DIR / "guardian_state.json")
        print(f"  Guardian state saved to {_RESULTS_DIR / 'guardian_state.json'}")
        self._print_daily_summary()
        self._connector.disconnect()
        print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    logger.info(
        "FTMO Trading Bot starting | mode=%s strategy=%s instrument=%s phase=%s",
        args.mode, args.strategy, args.instrument, args.phase,
    )

    if args.mode == "backtest":
        _run_backtest(args)
    elif args.mode == "walkforward":
        _run_walkforward(args)
    elif args.mode == "live":
        _run_live(args)
    elif args.mode == "report":
        _run_report(args)
    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
