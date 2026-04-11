"""
Vectorized event-driven backtesting engine.

Architecture
------------
Signal generation is vectorised (strategy processes the full DataFrame at once).
Execution is simulated bar-by-bar to correctly model:
  - Position state (only one position at a time)
  - FTMO daily loss limit (intraday equity check at every bar)
  - FTMO total drawdown halt

Entry model
-----------
  Signal fires at bar close (t).
  entry_price = close ± spread/2 ± slippage
    Long:  buy at ask  → close + spread_adj + slippage_adj
    Short: sell at bid → close - spread_adj - slippage_adj
  Slippage is drawn from a session-aware uniform distribution:
    - Within session_open_window_minutes of 03:00 or 09:30 Eastern → session_open_pips max
    - Otherwise → normal_pips max
  SL/TP checked from bar t+1 onward using bar high/low.
  If both SL and TP hit in the same bar → exit at SL (worst case).

P&L model
---------
  pnl_pips    = direction × (exit_price − entry_price) / pip_size
  pnl_dollars = pnl_pips × PIP_VALUE_PER_LOT × lot_size − commission
  equity      = balance (realised) + floating (open position marked to close)

FTMO daily loss
---------------
  Reference balance = account BALANCE (realised only) at midnight US/Eastern.
  Check = (equity − midnight_balance) / midnight_balance × 100
  Trigger at -daily_loss_trigger_pct% (default 4%) → block trading + close position.

Strategy interface (duck-typed)
--------------------------------
  strategy.name                  : str
  strategy.risk_per_trade_pct    : float   (used for auto lot sizing)
  strategy.generate_signals(df)  : pd.DataFrame
      Adds columns to df:
        signal    int    1=long, -1=short, 0=no trade
        sl_price  float  stop-loss price
        tp_price  float  take-profit price
        lot_size  float  (optional) overrides auto-sizing
        time_stop pd.Timestamp  (optional) force-close at this bar
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.timezone import is_within_session_open_window

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent / "config"

# USD per pip per standard lot for USD-quoted pairs (EURUSD, GBPUSD).
# Based on 100,000 unit contract: pip_value = pip_size × 100,000 / quote ≈ $10.
PIP_VALUE_PER_LOT = 10.0


# ---------------------------------------------------------------------------
# Pending / bracket order support
# ---------------------------------------------------------------------------

@dataclass
class PendingOrder:
    """
    A stop or limit entry order placed before price reaches entry_price.

    Fields
    ------
    order_type      : "buy_stop" | "sell_stop" | "buy_limit" | "sell_limit"
    entry_price     : trigger price (order fills when bar H/L pierces this)
    sl_price        : stop-loss price (placed when order fills)
    tp_price        : take-profit price
    place_bar       : bar index at which this order becomes active
    expiry_bar      : cancel without fill after this bar index
    group_id        : OCO group — when any order in this group fills,
                      all others with the same group_id are cancelled
    time_stop_bars  : bars after fill before forced time-stop exit (0 = disabled)
    news_spread_mult: spread multiplier applied when filling within news_window_bars
                      of event_bar (models wide spreads at release time)
    event_bar       : bar index of the underlying news event (for spread calc)
    news_window_bars: how many bars either side of event_bar the multiplier applies
    event_type      : metadata string for reporting (e.g. 'NFP')
    """
    order_type:       str    # "buy_stop" | "sell_stop" | "buy_limit" | "sell_limit"
    entry_price:      float
    sl_price:         float
    tp_price:         float
    place_bar:        int
    expiry_bar:       int
    group_id:         int    = 0
    time_stop_bars:   int    = 0
    news_spread_mult: float  = 1.0
    event_bar:        int    = -1
    news_window_bars: int    = 5
    event_type:       str    = ""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time:    pd.Timestamp
    exit_time:     pd.Timestamp
    direction:     int          # 1 = long, -1 = short
    entry_price:   float
    exit_price:    float
    sl:            float
    tp:            float
    lot_size:      float
    pnl_pips:      float
    pnl_dollars:   float
    pnl_pct:       float        # relative to balance at entry
    strategy_name: str
    instrument:    str
    exit_reason:   str          # sl | tp | time_stop | ftmo_daily | ftmo_total | end_of_data


@dataclass
class BacktestResult:
    trades:          List[Trade]
    equity_curve:    pd.Series   # equity (balance + floating) at every bar close
    daily_pnl:       pd.Series   # keyed by date (US/Eastern)
    config:          Dict[str, Any]
    initial_balance: float
    final_balance:   float
    ftmo_halt_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Backtester:
    """
    Bar-by-bar backtesting engine with realistic cost simulation
    and FTMO rule enforcement.

    Parameters
    ----------
    strategy :
        Object implementing the strategy interface (see module docstring).
    df :
        OHLCV DataFrame. Index must be tz-aware UTC datetimes (or a 'datetime'
        column). Columns: open, high, low, close, volume.
        The engine converts the index to US/Eastern internally.
    instrument :
        Key in instruments.json (e.g. "EURUSD").
    initial_balance :
        Starting account balance in USD.
    phase :
        "challenge" | "verification" | "funded" — selects FTMO rule set.
    seed :
        RNG seed for reproducible slippage draws.
    _override_instrument_config :
        Inject a custom instrument config dict instead of loading from file.
        Intended for unit tests only.
    _override_ftmo_rules :
        Inject custom FTMO rules dict instead of loading from file.
        Intended for unit tests only.
    """

    def __init__(
        self,
        strategy,
        df: pd.DataFrame,
        instrument: str,
        initial_balance: float = 10_000.0,
        phase: str = "challenge",
        seed: int = 42,
        break_even_r: Optional[float] = None,
        trail_atr_mult: Optional[float] = None,
        _override_instrument_config: Optional[Dict] = None,
        _override_ftmo_rules: Optional[Dict] = None,
    ) -> None:
        self.strategy = strategy
        self.instrument = instrument
        self.initial_balance = initial_balance
        self.phase = phase
        self.rng = np.random.default_rng(seed)
        # Optional trade-management overlays (both default to None = disabled)
        self.break_even_r   = break_even_r    # move SL to entry+1pip once +N×R hit
        self.trail_atr_mult = trail_atr_mult  # trail at N×ATR(14) after +1.5R hit

        self._load_configs(_override_instrument_config, _override_ftmo_rules)
        self.df = self._prepare_data(df)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_configs(
        self,
        override_instrument: Optional[Dict],
        override_ftmo: Optional[Dict],
    ) -> None:
        if override_ftmo is not None:
            self.ftmo_rules = override_ftmo
        else:
            with open(_CONFIG_DIR / "ftmo_rules.json") as fh:
                self.ftmo_rules = json.load(fh)

        if override_instrument is not None:
            inst = override_instrument
        else:
            with open(_CONFIG_DIR / "instruments.json") as fh:
                inst = json.load(fh)[self.instrument]

        self.pip_size: float            = inst["pip_size"]
        self.pip_value_per_lot: float   = inst.get("pip_value_per_lot", PIP_VALUE_PER_LOT)
        self.spread_pips: float         = inst["typical_spread_pips"]
        self.slippage_model: Dict       = inst["slippage_model"]
        self.commission_per_lot: float  = inst["commission_per_lot_round_trip"]

        buffers = self.ftmo_rules["safety_buffers"]
        self.daily_loss_trigger_pct: float = buffers["daily_loss_trigger_pct"]
        self.total_loss_trigger_pct: float = buffers["total_loss_trigger_pct"]

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise to US/Eastern tz-aware DatetimeIndex with lower-case columns."""
        df = df.copy()
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("US/Eastern")
        df.columns = [c.lower() for c in df.columns]
        return df

    # ------------------------------------------------------------------
    # Cost modelling
    # ------------------------------------------------------------------

    def _slippage_pips(self, ts: pd.Timestamp) -> float:
        """Draw random slippage in pips, elevated near session opens."""
        ts_utc = ts.tz_convert("UTC")
        window = self.slippage_model["session_open_window_minutes"]
        if is_within_session_open_window(ts_utc, window):
            max_slip = self.slippage_model["session_open_pips"]
        else:
            max_slip = self.slippage_model["normal_pips"]
        return float(self.rng.uniform(0.0, max_slip))

    def _calc_entry_price(self, close: float, direction: int, ts: pd.Timestamp) -> float:
        """Apply half-spread + slippage to the signal-bar close price."""
        spread_adj = (self.spread_pips / 2.0) * self.pip_size
        slip_adj   = self._slippage_pips(ts) * self.pip_size
        if direction == 1:   # long: buy at ask
            return close + spread_adj + slip_adj
        return close - spread_adj - slip_adj  # short: sell at bid

    def _auto_lot_size(self, balance: float, entry: float, sl: float) -> float:
        """Size position so that an SL hit risks risk_per_trade_pct of balance."""
        risk_pct = getattr(self.strategy, "risk_per_trade_pct", 0.75)
        risk_usd = balance * risk_pct / 100.0
        sl_pips  = abs(entry - sl) / self.pip_size
        if sl_pips < 0.01:
            return 0.01
        lot = risk_usd / (sl_pips * self.pip_value_per_lot)
        lot = min(lot, 50.0)   # hard cap: never more than 50 lots on one trade
        return max(0.01, round(lot, 2))

    # ------------------------------------------------------------------
    # Trade lifecycle helpers
    # ------------------------------------------------------------------

    def _floating_pnl(self, pos: Dict, close: float) -> float:
        """Unrealised P&L in USD (commission excluded, charged only on close)."""
        pnl_pips = pos["direction"] * (close - pos["entry_price"]) / self.pip_size
        return pnl_pips * self.pip_value_per_lot * pos["lot_size"]

    # ------------------------------------------------------------------
    # Trade-management helpers (break-even / trailing stop)
    # ------------------------------------------------------------------

    def _compute_atr14(self, df: pd.DataFrame) -> pd.Series:
        """
        Wilder ATR(14) using the same ta-library call as regime_filter.py.
        Returns a Series aligned to df.index.
        """
        import ta  # lazy import — only needed when trail_atr_mult is set
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"],
            window=14, fillna=False,
        )
        return atr_indicator.average_true_range()

    @staticmethod
    def _atr_at(atr_series: pd.Series, ts: pd.Timestamp) -> float:
        """Return ATR at ts; fall back to most recent non-NaN value."""
        if ts in atr_series.index:
            val = float(atr_series[ts])
            if not np.isnan(val):
                return val
        candidates = atr_series.dropna()
        candidates = candidates[candidates.index <= ts]
        return float(candidates.iloc[-1]) if len(candidates) else float(atr_series.dropna().iloc[0])

    def _update_sl_for_management(
        self,
        pos: Dict,
        row,
        atr_series: Optional[pd.Series],
        ts: pd.Timestamp,
    ) -> None:
        """
        Mutates pos["sl"] in-place for break-even and trailing-stop logic.
        Called every bar while a position is open, *before* the SL/TP check.

        Break-even rule
        ---------------
        Once the bar's extreme touches break_even_r × original_sl_dist from
        entry, the SL is moved to entry + 1 pip (in trade direction).
        The SL can only move in the favourable direction — never worsened.

        Trailing-stop rule
        ------------------
        Activates when price first reaches +1.5R from entry.
        After activation, the SL trails trail_atr_mult × ATR(14) behind the
        running extreme (highest high for longs, lowest low for shorts).
        Again, the SL can only move in the favourable direction.
        """
        direction = pos["direction"]
        sl        = pos["sl"]
        orig_dist = pos["original_sl_dist"]   # |entry - original_sl|
        ep        = pos["entry_price"]

        # ── Break-even ────────────────────────────────────────────────────
        if self.break_even_r is not None and not pos["be_triggered"]:
            be_target = ep + direction * orig_dist * self.break_even_r
            hit = (
                (direction ==  1 and row["high"] >= be_target) or
                (direction == -1 and row["low"]  <= be_target)
            )
            if hit:
                be_sl = ep + direction * self.pip_size   # entry + 1 pip
                if (direction == 1 and be_sl > sl) or (direction == -1 and be_sl < sl):
                    pos["sl"] = be_sl
                    sl = be_sl
                pos["be_triggered"] = True

        # ── Trailing stop ─────────────────────────────────────────────────
        if self.trail_atr_mult is not None and atr_series is not None:
            trail_target = ep + direction * orig_dist * 1.5

            if not pos["trail_active"]:
                activated = (
                    (direction ==  1 and row["high"] >= trail_target) or
                    (direction == -1 and row["low"]  <= trail_target)
                )
                if activated:
                    pos["trail_active"]   = True
                    pos["trail_extreme"]  = row["high"] if direction == 1 else row["low"]

            if pos["trail_active"]:
                atr_val = self._atr_at(atr_series, ts)
                if direction == 1:
                    pos["trail_extreme"] = max(pos["trail_extreme"], row["high"])
                    new_sl = pos["trail_extreme"] - self.trail_atr_mult * atr_val
                    if new_sl > pos["sl"]:
                        pos["sl"] = new_sl
                else:
                    pos["trail_extreme"] = min(pos["trail_extreme"], row["low"])
                    new_sl = pos["trail_extreme"] + self.trail_atr_mult * atr_val
                    if new_sl < pos["sl"]:
                        pos["sl"] = new_sl

    def _close_position(
        self,
        pos: Dict,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str,
    ) -> Trade:
        """Build a closed Trade, deducting round-trip commission."""
        direction  = pos["direction"]
        pnl_pips   = direction * (exit_price - pos["entry_price"]) / self.pip_size
        gross_usd  = pnl_pips * self.pip_value_per_lot * pos["lot_size"]
        commission = self.commission_per_lot * pos["lot_size"]
        pnl_usd    = gross_usd - commission
        pnl_pct    = pnl_usd / pos["balance_at_entry"] * 100.0

        logger.debug(
            "%s %s exit=%s reason=%s pnl_pips=%.1f pnl_usd=%.2f commission=%.2f",
            self.instrument,
            "LONG" if direction == 1 else "SHORT",
            exit_time, exit_reason, pnl_pips, pnl_usd, commission,
        )

        return Trade(
            entry_time=pos["entry_time"],
            exit_time=exit_time,
            direction=direction,
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            sl=pos["sl"],
            tp=pos["tp"],
            lot_size=pos["lot_size"],
            pnl_pips=pnl_pips,
            pnl_dollars=pnl_usd,
            pnl_pct=pnl_pct,
            strategy_name=self.strategy.name,
            instrument=self.instrument,
            exit_reason=exit_reason,
        )

    # ------------------------------------------------------------------
    # Parsing helpers (handle NaN / NaT / missing columns gracefully)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_signal(row) -> int:
        val = row.get("signal", 0)
        return 0 if pd.isna(val) else int(val)

    @staticmethod
    def _get_float(row, key: str) -> float:
        val = row.get(key, np.nan)
        return np.nan if pd.isna(val) else float(val)

    @staticmethod
    def _get_timestamp(row, key: str) -> Optional[pd.Timestamp]:
        val = row.get(key, None)
        return None if (val is None or pd.isna(val)) else val

    # ------------------------------------------------------------------
    # Fast sparse loop (for high-frequency data like M1)
    # ------------------------------------------------------------------

    def run_fast(self) -> BacktestResult:
        """
        Sparse backtester — iterates only signal bars + bars while a position
        is open. Skips idle bars entirely.

        ~15-20x faster than run() for strategies with O(200) annual signals
        on M1 data (~700k bars). Designed for WF grid search.

        Limitations vs run():
        - Trailing stop (trail_atr_mult) not supported.
        - Break-even is supported.
        - Equity curve is sparse (only records bars with open position +
          first bar of each new day). Metrics remain accurate because
          metrics.py resamples to daily anyway.
        """
        df = self.strategy.generate_signals(self.df.copy())
        n  = len(df)

        if n == 0:
            empty_eq = pd.Series({df.index[0]: float(self.initial_balance)})
            return BacktestResult([], empty_eq, pd.Series(dtype=float), {},
                                  self.initial_balance, self.initial_balance)

        idx      = df.index
        dates    = idx.date                          # array of datetime.date
        highs    = df["high"].to_numpy(dtype=float)
        lows     = df["low"].to_numpy(dtype=float)
        closes   = df["close"].to_numpy(dtype=float)
        sig_arr  = df["signal"].fillna(0).to_numpy(dtype=float)
        sl_arr   = (df["sl_price"].to_numpy(dtype=float)
                    if "sl_price" in df.columns else np.full(n, np.nan))
        tp_arr   = (df["tp_price"].to_numpy(dtype=float)
                    if "tp_price" in df.columns else np.full(n, np.nan))
        has_ts   = "time_stop" in df.columns
        ts_col   = df["time_stop"].to_numpy() if has_ts else None

        # Signal positions (sorted ascending)
        sig_pos  = np.where(sig_arr != 0)[0]

        if len(sig_pos) == 0:
            eq = pd.Series({idx[0]: self.initial_balance, idx[-1]: self.initial_balance})
            return BacktestResult([], eq, pd.Series(dtype=float), {},
                                  self.initial_balance, self.initial_balance)

        loose_ftmo = (
            self.daily_loss_trigger_pct >= 90.0 and
            self.total_loss_trigger_pct >= 90.0
        )

        balance          = float(self.initial_balance)
        trades: List[Trade] = []
        eq_curve: Dict   = {}
        day_pnl: Dict    = {}
        midnight_balance = balance
        prev_date        = None
        halt_reason: Optional[str] = None
        blocked_dates: set = set()       # dates where daily limit was hit
        last_exit_idx    = -1            # skip signals while a position is open

        # Seed equity at first bar
        prev_date = dates[0]
        eq_curve[idx[0]] = balance

        for sig_idx in sig_pos:
            # Skip signals that fall inside a currently open trade
            if sig_idx <= last_exit_idx:
                continue
            d = dates[sig_idx]

            # Update day_pnl for any days elapsed since last trade
            if prev_date is not None and d != prev_date:
                day_pnl[prev_date] = balance - midnight_balance
                midnight_balance   = balance
                prev_date          = d

            if d in blocked_dates:
                continue

            sig = int(sig_arr[sig_idx])
            sl  = float(sl_arr[sig_idx])
            tp  = float(tp_arr[sig_idx])
            if np.isnan(sl) or np.isnan(tp):
                continue

            entry_ts    = idx[sig_idx]
            entry_price = self._calc_entry_price(closes[sig_idx], sig, entry_ts)

            # Guard: SL must be on correct side of entry
            if (sig == 1 and sl >= entry_price) or (sig == -1 and sl <= entry_price):
                continue

            lot_size         = self._auto_lot_size(balance, entry_price, sl)
            orig_dist        = abs(entry_price - sl)
            current_sl       = sl
            be_triggered     = False
            balance_at_entry = balance

            # Time-stop
            ts_stop = None
            if has_ts:
                raw = ts_col[sig_idx]
                if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
                    ts_stop = pd.Timestamp(raw)

            eq_curve[entry_ts] = balance   # entry bar: position just opened

            exit_idx    = n - 1
            exit_price  = closes[n - 1]
            exit_reason = "end_of_data"
            cur_date    = d

            for j in range(sig_idx + 1, n):
                jts   = idx[j]
                jdate = dates[j]

                # ── Day boundary ─────────────────────────────────────
                if jdate != cur_date:
                    floating = (sig * (closes[j - 1] - entry_price)
                                / self.pip_size * self.pip_value_per_lot * lot_size)
                    day_pnl[cur_date]  = (balance + floating) - midnight_balance
                    midnight_balance   = balance          # reset to realised balance
                    cur_date           = jdate
                    prev_date          = jdate

                    if not loose_ftmo:
                        equity  = balance + floating
                        dd_pct  = (equity - midnight_balance) / max(midnight_balance, 1.0) * 100.0
                        if dd_pct <= -self.daily_loss_trigger_pct:
                            blocked_dates.add(jdate)
                            exit_idx = j; exit_price = closes[j]; exit_reason = "ftmo_daily"
                            break
                        tot_pct = (equity - self.initial_balance) / self.initial_balance * 100.0
                        if tot_pct <= -self.total_loss_trigger_pct:
                            exit_idx = j; exit_price = closes[j]; exit_reason = "ftmo_total"
                            halt_reason = f"Total drawdown {tot_pct:.2f}% at {jts}"
                            break

                # ── Break-even ───────────────────────────────────────
                if self.break_even_r is not None and not be_triggered:
                    be_target = entry_price + sig * orig_dist * self.break_even_r
                    hit_be = (
                        (sig ==  1 and highs[j] >= be_target) or
                        (sig == -1 and lows[j]  <= be_target)
                    )
                    if hit_be:
                        new_sl = entry_price + sig * self.pip_size
                        if (sig == 1 and new_sl > current_sl) or (sig == -1 and new_sl < current_sl):
                            current_sl = new_sl
                        be_triggered = True

                # ── SL / TP check ────────────────────────────────────
                hit_sl = (sig ==  1 and lows[j]  <= current_sl) or (sig == -1 and highs[j] >= current_sl)
                hit_tp = (sig ==  1 and highs[j] >= tp)         or (sig == -1 and lows[j]  <= tp)

                if hit_sl or hit_tp:
                    exit_idx = j
                    if hit_sl:
                        exit_price, exit_reason = current_sl, "sl"   # SL wins on tie
                    else:
                        exit_price, exit_reason = tp, "tp"
                    break

                # ── Time stop ────────────────────────────────────────
                if ts_stop is not None and jts >= ts_stop:
                    exit_idx = j; exit_price = closes[j]; exit_reason = "time_stop"
                    break

                # ── Record equity (floating) ─────────────────────────
                floating = (sig * (closes[j] - entry_price)
                            / self.pip_size * self.pip_value_per_lot * lot_size)
                eq_curve[jts] = balance + floating

            # ── Close trade ──────────────────────────────────────────
            trade = self._close_position(
                {
                    "entry_time":       entry_ts,
                    "direction":        sig,
                    "entry_price":      entry_price,
                    "sl":               current_sl,
                    "tp":               tp,
                    "lot_size":         lot_size,
                    "balance_at_entry": balance_at_entry,
                },
                exit_price,
                idx[exit_idx],
                exit_reason,
            )
            trades.append(trade)
            balance      += trade.pnl_dollars
            eq_curve[idx[exit_idx]] = balance
            prev_date    = dates[exit_idx]
            last_exit_idx = exit_idx      # no new trade can open before this bar

            if halt_reason or exit_reason == "ftmo_total":
                break

            if not loose_ftmo:
                tot_pct = (balance - self.initial_balance) / self.initial_balance * 100.0
                if tot_pct <= -self.total_loss_trigger_pct:
                    halt_reason = f"Total drawdown {tot_pct:.2f}% at {idx[exit_idx]}"
                    break

        # Final day P&L bucket
        if prev_date is not None:
            day_pnl[prev_date] = balance - midnight_balance

        eq_curve[idx[-1]] = balance   # ensure series ends at final balance

        config = {
            "instrument":      self.instrument,
            "initial_balance": self.initial_balance,
            "phase":           self.phase,
            "strategy_name":   self.strategy.name,
            "ftmo_rules":      self.ftmo_rules,
        }

        return BacktestResult(
            trades=trades,
            equity_curve=pd.Series(eq_curve).sort_index(),
            daily_pnl=pd.Series(day_pnl),
            config=config,
            initial_balance=self.initial_balance,
            final_balance=balance,
            ftmo_halt_reason=halt_reason,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest. Returns a BacktestResult."""
        df = self.strategy.generate_signals(self.df.copy())

        # Pre-compute ATR(14) for trailing stop (None if feature not enabled)
        atr_series: Optional[pd.Series] = (
            self._compute_atr14(df) if self.trail_atr_mult is not None else None
        )

        balance   = self.initial_balance
        trades:   List[Trade] = []
        eq_curve: Dict[pd.Timestamp, float] = {}
        day_pnl:  Dict = {}

        open_pos:       Optional[Dict] = None
        prev_date                      = None
        midnight_balance               = balance
        daily_blocked                  = False
        halt_reason: Optional[str]     = None

        for ts, row in df.iterrows():
            current_date = ts.date()

            # ── Day boundary ──────────────────────────────────────────────
            if current_date != prev_date:
                if prev_date is not None:
                    day_pnl[prev_date] = balance - midnight_balance
                # Snapshot REALISED balance at midnight Eastern
                midnight_balance = balance
                daily_blocked    = False
                prev_date        = current_date

            # ── Equity = realised balance + floating unrealised ───────────
            equity = balance + (self._floating_pnl(open_pos, row["close"]) if open_pos else 0.0)

            # ── FTMO daily loss check ────────────────────────────────────
            daily_dd_pct = (equity - midnight_balance) / midnight_balance * 100.0

            if (not daily_blocked) and daily_dd_pct <= -self.daily_loss_trigger_pct:
                daily_blocked = True
                logger.warning(
                    "FTMO daily-loss triggered at %s | dd=%.2f%% | threshold=%.1f%%",
                    ts, daily_dd_pct, -self.daily_loss_trigger_pct,
                )
                if open_pos:
                    trade = self._close_position(open_pos, row["close"], ts, "ftmo_daily")
                    trades.append(trade)
                    balance  += trade.pnl_dollars
                    equity    = balance
                    open_pos  = None

            # ── FTMO total drawdown check ────────────────────────────────
            total_dd_pct = (equity - self.initial_balance) / self.initial_balance * 100.0

            if total_dd_pct <= -self.total_loss_trigger_pct:
                if open_pos:
                    trade = self._close_position(open_pos, row["close"], ts, "ftmo_total")
                    trades.append(trade)
                    balance  += trade.pnl_dollars
                    open_pos  = None
                halt_reason = (
                    f"Total drawdown {total_dd_pct:.2f}% reached safety buffer "
                    f"(-{self.total_loss_trigger_pct}%) at {ts}"
                )
                logger.error(halt_reason)
                eq_curve[ts] = balance
                break

            # ── Break-even / trailing: update SL before the exit check ──
            if open_pos and (self.break_even_r is not None or self.trail_atr_mult is not None):
                self._update_sl_for_management(open_pos, row, atr_series, ts)

            # ── Check open position for SL / TP / time-stop ─────────────
            if open_pos:
                direction = open_pos["direction"]
                sl, tp    = open_pos["sl"], open_pos["tp"]  # read after potential SL update

                exit_price:  Optional[float] = None
                exit_reason: Optional[str]   = None

                if direction == 1:   # long
                    if row["low"] <= sl:
                        exit_price, exit_reason = sl, "sl"
                    elif row["high"] >= tp:
                        exit_price, exit_reason = tp, "tp"
                else:                # short
                    if row["high"] >= sl:
                        exit_price, exit_reason = sl, "sl"
                    elif row["low"] <= tp:
                        exit_price, exit_reason = tp, "tp"

                # Time stop: force-close at bar close if timestamp reached
                if exit_price is None:
                    time_stop = open_pos.get("time_stop")
                    if time_stop is not None and ts >= time_stop:
                        exit_price, exit_reason = row["close"], "time_stop"

                if exit_price is not None:
                    trade = self._close_position(open_pos, exit_price, ts, exit_reason)
                    trades.append(trade)
                    balance  += trade.pnl_dollars
                    equity    = balance
                    open_pos  = None

            # ── Enter new position ───────────────────────────────────────
            signal = self._get_signal(row)

            if open_pos is None and not daily_blocked and signal != 0:
                sl_price = self._get_float(row, "sl_price")
                tp_price = self._get_float(row, "tp_price")

                if not (np.isnan(sl_price) or np.isnan(tp_price)):
                    entry_price = self._calc_entry_price(row["close"], signal, ts)

                    # Guard: SL must be on the correct side of entry.
                    # For LONG: sl < entry.  For SHORT: sl > entry.
                    # If violated (e.g. bar punched through a FVG zone and
                    # closed beyond the SL), skip — position sizing would
                    # explode and the trade has no valid risk-reward geometry.
                    sl_on_wrong_side = (
                        (signal == 1  and sl_price >= entry_price) or
                        (signal == -1 and sl_price <= entry_price)
                    )
                    if sl_on_wrong_side:
                        logger.debug(
                            "Skipping signal at %s: SL %.5f on wrong side of "
                            "entry %.5f (direction=%d)", ts, sl_price, entry_price, signal
                        )
                        continue

                    raw_lot  = self._get_float(row, "lot_size")
                    lot_size = (
                        self._auto_lot_size(balance, entry_price, sl_price)
                        if np.isnan(raw_lot)
                        else raw_lot
                    )

                    open_pos = {
                        "entry_time":       ts,
                        "direction":        signal,
                        "entry_price":      entry_price,
                        "sl":               sl_price,
                        "tp":               tp_price,
                        "lot_size":         lot_size,
                        "balance_at_entry": balance,
                        "time_stop":        self._get_timestamp(row, "time_stop"),
                        # trade-management state
                        "original_sl_dist": abs(entry_price - sl_price),
                        "be_triggered":     False,
                        "trail_active":     False,
                        "trail_extreme":    entry_price,
                    }
                    logger.debug(
                        "%s %s | entry=%.5f sl=%.5f tp=%.5f lots=%.2f @ %s",
                        self.instrument, "LONG" if signal == 1 else "SHORT",
                        entry_price, sl_price, tp_price, lot_size, ts,
                    )

            # Record equity (floating included)
            eq_curve[ts] = balance + (self._floating_pnl(open_pos, row["close"]) if open_pos else 0.0)

        # ── End-of-data: close any still-open position ────────────────────
        if open_pos:
            last_ts    = df.index[-1]
            last_close = df.iloc[-1]["close"]
            trade = self._close_position(open_pos, last_close, last_ts, "end_of_data")
            trades.append(trade)
            balance += trade.pnl_dollars

        # Final daily P&L bucket
        if prev_date is not None:
            day_pnl[prev_date] = balance - midnight_balance

        config = {
            "instrument":    self.instrument,
            "initial_balance": self.initial_balance,
            "phase":         self.phase,
            "strategy_name": self.strategy.name,
            "ftmo_rules":    self.ftmo_rules,
        }

        return BacktestResult(
            trades=trades,
            equity_curve=pd.Series(eq_curve),
            daily_pnl=pd.Series(day_pnl),
            config=config,
            initial_balance=self.initial_balance,
            final_balance=balance,
            ftmo_halt_reason=halt_reason,
        )

    # ------------------------------------------------------------------
    # Bracket / OCO order simulation
    # ------------------------------------------------------------------

    def run_bracket(self, pending_orders: List[PendingOrder]) -> BacktestResult:
        """
        Simulate bracket / OCO stop-or-limit orders.

        Instead of calling generate_signals(), takes a pre-built list of
        PendingOrder objects (usually produced by a strategy's
        generate_bracket_orders() method).

        Design
        ------
        - Sparse: only iterates bars inside active event windows
          (place_bar → place_bar + max_look_ahead per event group).
          ~100× faster than a full M1 bar loop on 1.6 M bars.
        - OCO: when one order in a group fills, all others in the same
          group are immediately cancelled.
        - One position at a time — a new event window is skipped if the
          previous trade is still open.
        - Spread at fill: news_spread_mult × normal_spread, applied only
          within news_window_bars of event_bar.
        - Break-even supported (uses self.break_even_r).
        - FTMO daily / total drawdown enforced between events.
        """
        from collections import defaultdict

        df  = self.df
        n   = len(df)
        idx = df.index
        highs  = df["high"].to_numpy(float)
        lows   = df["low"].to_numpy(float)
        closes = df["close"].to_numpy(float)
        dates  = idx.date

        # Group orders by group_id; sort groups by their earliest place_bar
        groups: Dict[int, List[PendingOrder]] = defaultdict(list)
        for o in pending_orders:
            groups[o.group_id].append(o)
        sorted_groups = sorted(groups.values(), key=lambda g: min(o.place_bar for o in g))

        balance          = float(self.initial_balance)
        trades: List[Trade] = []
        eq_curve: Dict   = {}
        day_pnl: Dict    = {}
        midnight_balance = balance
        prev_date        = dates[0]
        halt_reason: Optional[str] = None
        eq_curve[idx[0]] = balance
        last_exit_bar    = -1

        for grp_orders in sorted_groups:
            if halt_reason:
                break

            place_bar = min(o.place_bar for o in grp_orders)
            if place_bar >= n or place_bar <= last_exit_bar:
                continue

            # Maximum bar to scan: expiry_bar of latest order + max time_stop_bars
            max_expiry   = max(o.expiry_bar for o in grp_orders)
            max_ts_bars  = max((o.time_stop_bars for o in grp_orders), default=0)
            scan_end     = min(max_expiry + max_ts_bars + 1, n)

            active_orders: List[PendingOrder] = list(grp_orders)
            pos: Optional[Dict] = None

            for i in range(place_bar, scan_end):
                if i >= n:
                    break

                bar_ts   = idx[i]
                bar_date = dates[i]
                bar_h    = highs[i]
                bar_l    = lows[i]
                bar_c    = closes[i]

                # ── Day boundary ──────────────────────────────────────
                if bar_date != prev_date:
                    floating = 0.0
                    if pos is not None:
                        floating = (pos["direction"] * (bar_c - pos["entry_price"])
                                    / self.pip_size * self.pip_value_per_lot * pos["lot_size"])
                    day_pnl[prev_date] = (balance + floating) - midnight_balance
                    midnight_balance   = balance
                    prev_date          = bar_date

                # ── Manage open position ──────────────────────────────
                if pos is not None:
                    direction  = pos["direction"]
                    ep         = pos["entry_price"]
                    orig_dist  = pos["orig_dist"]
                    current_sl = pos["sl"]

                    # Break-even
                    if self.break_even_r is not None and not pos["be_triggered"]:
                        be_target = ep + direction * orig_dist * self.break_even_r
                        hit_be = (
                            (direction ==  1 and bar_h >= be_target) or
                            (direction == -1 and bar_l <= be_target)
                        )
                        if hit_be:
                            new_sl = ep + direction * self.pip_size
                            if (direction == 1 and new_sl > current_sl) or \
                               (direction == -1 and new_sl < current_sl):
                                pos["sl"] = new_sl
                                current_sl = pos["sl"]
                            pos["be_triggered"] = True

                    # SL / TP check
                    hit_sl = ((direction ==  1 and bar_l <= current_sl) or
                              (direction == -1 and bar_h >= current_sl))
                    hit_tp = ((direction ==  1 and bar_h >= pos["tp"]) or
                              (direction == -1 and bar_l <= pos["tp"]))
                    # Time stop
                    ts_bar = pos.get("time_stop_bar")
                    hit_ts = ts_bar is not None and i >= ts_bar

                    exit_price  = None
                    exit_reason = None
                    if hit_sl:
                        exit_price, exit_reason = current_sl, "sl"
                    elif hit_tp:
                        exit_price, exit_reason = pos["tp"], "tp"
                    elif hit_ts:
                        exit_price, exit_reason = bar_c, "time_stop"

                    if exit_price is not None:
                        trade = self._close_position(pos, exit_price, bar_ts, exit_reason)
                        trades.append(trade)
                        balance += trade.pnl_dollars
                        eq_curve[bar_ts] = balance
                        last_exit_bar    = i
                        pos = None
                        active_orders    = []  # OCO partners already gone

                        # FTMO checks
                        tot_pct = (balance - self.initial_balance) / self.initial_balance * 100.0
                        if tot_pct <= -self.total_loss_trigger_pct:
                            halt_reason = f"Total drawdown {tot_pct:.2f}% at {bar_ts}"
                            break
                        break  # event window done
                    else:
                        floating = (direction * (bar_c - ep)
                                    / self.pip_size * self.pip_value_per_lot * pos["lot_size"])
                        eq_curve[bar_ts] = balance + floating
                    continue

                # ── Check pending orders for fill ─────────────────────
                filled_order: Optional[PendingOrder] = None
                for o in list(active_orders):
                    if i > o.expiry_bar:
                        active_orders.remove(o)
                        continue

                    triggered = False
                    if   o.order_type == "buy_stop"   and bar_h >= o.entry_price: triggered = True
                    elif o.order_type == "sell_stop"  and bar_l <= o.entry_price: triggered = True
                    elif o.order_type == "buy_limit"  and bar_l <= o.entry_price: triggered = True
                    elif o.order_type == "sell_limit" and bar_h >= o.entry_price: triggered = True

                    if triggered:
                        filled_order = o
                        break

                if filled_order is None:
                    continue

                # ── Fill the order ─────────────────────────────────────
                o         = filled_order
                direction = 1 if o.order_type in ("buy_stop", "buy_limit") else -1

                # Spread: elevated near event bar
                news_nearby  = abs(i - o.event_bar) <= o.news_window_bars
                spread_mult  = o.news_spread_mult if news_nearby else 1.0
                eff_spread   = self.spread_pips * spread_mult
                slip         = self._slippage_pips(bar_ts)
                fill_price   = (o.entry_price
                                + direction * ((eff_spread / 2.0) * self.pip_size
                                               + slip * self.pip_size))

                # Revalidate SL is on correct side
                sl = o.sl_price
                if (direction == 1 and sl >= fill_price) or (direction == -1 and sl <= fill_price):
                    continue  # degenerate order — skip

                lot_size = self._auto_lot_size(balance, fill_price, sl)
                ts_bar   = (i + o.time_stop_bars) if o.time_stop_bars > 0 else None

                pos = {
                    "entry_time":      bar_ts,
                    "entry_price":     fill_price,
                    "direction":       direction,
                    "sl":              sl,
                    "tp":              o.tp_price,
                    "lot_size":        lot_size,
                    "balance_at_entry":balance,
                    "orig_dist":       abs(fill_price - sl),
                    "be_triggered":    False,
                    "time_stop_bar":   ts_bar,
                    "event_type":      o.event_type,
                }

                eq_curve[bar_ts] = balance
                # Cancel all other OCO partners
                active_orders = []

            # If position still open at end of scan window → force-close
            if pos is not None:
                close_bar = min(scan_end - 1, n - 1)
                trade = self._close_position(
                    pos, closes[close_bar], idx[close_bar], "end_of_data"
                )
                trades.append(trade)
                balance       += trade.pnl_dollars
                eq_curve[idx[close_bar]] = balance
                last_exit_bar = close_bar
                pos = None

        # Final day P&L
        if prev_date is not None:
            day_pnl[prev_date] = balance - midnight_balance

        eq_curve[idx[-1]] = balance

        # Use a dummy strategy name if strategy lacks a name
        strat_name = getattr(self.strategy, "name", "news_m1_bracket")

        config = {
            "instrument":      self.instrument,
            "initial_balance": self.initial_balance,
            "phase":           self.phase,
            "strategy_name":   strat_name,
        }
        return BacktestResult(
            trades=trades,
            equity_curve=pd.Series(eq_curve).sort_index(),
            daily_pnl=pd.Series(day_pnl),
            config=config,
            initial_balance=self.initial_balance,
            final_balance=balance,
            ftmo_halt_reason=halt_reason,
        )
