"""
ctrader_bots/gold_multi_strategy.py
=====================================
cTrader Automate Python cBot — Gold Multi-Strategy (Modules A + B + C)

TEMPLATE STRUCTURE
------------------
Uses the exact cTrader Python cBot template:
  import clr
  clr.AddReference("cAlgo.API")
  from cAlgo.API import *
  from robot_wrapper import *

  class GoldMultiStrategy(object):
      def on_start(self): ...
      def on_bar(self):   ...
      def on_stop(self):  ...

PARAMETERS IN PYTHON CBOTS
---------------------------
Python cBots do NOT display a Parameter panel in the cTrader UI the way
C# cBots do (this is expected — the robot_wrapper doesn't bridge the
Parameter attribute system). Edit the CONFIGURATION constants at the top
of the class, then click Build before each deployment.

MODULES
-------
  A — London H1 Breakout     (03:00–07:59 ET = 08:00–12:59 UTC)
  B — VWAP Pullback H1       (09:00–14:59 ET = 14:00–19:59 UTC, max 2/day)
  C — NY Pullback to Level   (09:00–14:59 ET = 14:00–19:59 UTC, 1/day)
      Module C uses NO partial profits — full exit at C_RR_TP (2R).
      EV: 0.560R (vs 0.103R with partials — 5.4× improvement).

TRAIL: V4 — Close TP1_FRACTION at TP1, lock SL at +25% of TP1 dist,
            trail at TRAIL_ATR_MULT × ATR(14) once price reaches 1.5R.
            (V4 applies to Modules A and B only; Module C uses full exit.)

TIMEZONE
--------
All session logic uses UTC hours derived from bar.OpenTime.ToUniversalTime().
This is unambiguous regardless of broker server time.
  ET winter (UTC-5):  03:00 ET = 08:00 UTC
  ET summer (UTC-4):  03:00 ET = 07:00 UTC
The UTC windows below use winter offsets; summer shifts windows 1h earlier
(no significant impact on H1 strategy — may miss/add 1 bar at window edges).

DEPLOY
------
  1. cTrader Automate → My Robots → New → Python cBot
  2. Paste this entire file → Build → attach to XAUUSD H1 chart
  3. Edit the CONFIGURATION constants below, then rebuild, before each launch
  4. Run in Cloud mode for 24/7 execution

cAlgo.API METHODS USED
-----------------------
  api.MarketData.GetBars(TimeFrame.Hour, symbolName)  → Bars series
  bars.Last(1)                                          → last COMPLETED bar
  bars.Last(0)                                          → current forming bar
  bars.Count                                            → total bar count
  bar.Open / .High / .Low / .Close / .OpenTime / .TickVolume
  bar.OpenTime.ToUniversalTime()                        → UTC DateTime
  api.Account.Balance / .Equity
  api.Positions                                        → iterable of Position
  pos.Id / .SymbolName / .TradeType / .VolumeInUnits
  pos.EntryPrice / .StopLoss / .TakeProfit / .NetProfit
  api.Symbol[symbolName].PipSize / .PipValue / .LotSize
  api.Symbol[symbolName].VolumeInUnitsMin
  api.Symbol[symbolName].NormalizeVolumeInUnits(v)
  api.ExecuteMarketOrder(TradeType.Buy, symbol, volume, label)  → TradeResult
  result.IsSuccessful / result.Position / result.Error
  api.ModifyPosition(position, stopLossPrice, takeProfitPrice)
  api.ClosePosition(position)                          → full close
  api.ClosePosition(position, volumeInUnits)           → partial close
  api.Print(message)                                   → cTrader log
  int(bar.OpenTime.ToUniversalTime().DayOfWeek)         → 0=Sun…5=Fri…6=Sat
"""

import clr
clr.AddReference("cAlgo.API")
from cAlgo.API import *
from robot_wrapper import *

from collections import defaultdict


class GoldMultiStrategy():

    # ═══════════════════════════════════════════════════════════════════
    #  CONFIGURATION — edit these constants, then rebuild before launch
    # ═══════════════════════════════════════════════════════════════════

    SYMBOL          = "XAUUSD"      # confirm symbol name with your broker
    RISK_PCT        = 1.50          # % of balance risked per trade
    MAX_TRADES_DAY  = 3             # combined cap across A + B + C
    DAILY_KILL_PCT  = 3.5           # close all gold api.Positions if intraday DD >= this %
                                    # (use 3.5% on The5ers to stay inside 5% static floor)

    # Partial-profit (V4 trail variant — Modules A and B only)
    RR_TP1          = 1.0           # reward:risk to TP1
    RR_TP2          = 2.5           # reward:risk to TP2 (final target for A and B)
    TP1_FRACTION    = 0.50          # fraction of position to close at TP1
    TRAIL_ATR_MULT  = 2.0           # ATR(14) multiplier for trailing SL after TP1

    # Module A — London H1 Breakout
    A_BUF_PIPS      = 50.0          # pips beyond Asian range to confirm breakout
    A_MIN_RANGE     = 500.0         # min valid Asian range (pips, = $5.00)
    A_MAX_RANGE     = 2000.0        # max valid Asian range (pips, = $20.00)
    A_SL_CAP        = 1000.0        # SL cap from entry (pips)
    A_REGIME_PCT    = 40.0          # today's ATR must exceed this pct of 60-day ATR history
    A_TIME_STOP_H   = 20            # hours after bar open → close position (≈ 22:00 ET)

    # Module B — VWAP Pullback
    B_SL_BUFFER     = 30.0          # pips below/above bar extreme for SL
    B_SL_CAP        = 500.0         # SL cap (pips)
    B_SL_MIN        = 20.0          # SL minimum (pips)
    B_EMA_PERIOD    = 50
    B_TIME_STOP_H   = 12            # hours after bar open → close (≈ 21:00 ET)

    # Module C — NY Session Pullback to London Level (NO partials — full exit only)
    C_SL_PIPS       = 200.0         # fixed SL distance from breakout level (pips)
    C_TIME_STOP_H   = 12
    C_RR_TP         = 2.0           # full TP at 2R (no TP1 partial; EV 0.560R vs 0.103R with partials)

    PIP             = 0.01          # XAUUSD pip size ($0.01)

    # UTC hour windows (all session logic runs in UTC for broker independence)
    # Asian range:  19:00–02:59 ET  →  00:00–07:59 UTC  (winter)
    # London open:  03:00–07:59 ET  →  08:00–12:59 UTC
    # NY session:   09:00–14:59 ET  →  14:00–19:59 UTC
    UTC_ASIAN_S  = 0
    UTC_ASIAN_E  = 7
    UTC_LONDON_S = 8
    UTC_LONDON_E = 12
    UTC_NY_S     = 14
    UTC_NY_E     = 19

    # ═══════════════════════════════════════════════════════════════════

    def on_start(self):
        # Persistent cross-day state
        self._pos_meta  = {}    # position Id (int) → metadata dict
        self._atr_hist  = []    # daily ATR proxy values for regime filter
        self._ema       = None  # EMA50 value
        self._ema_k     = 2.0 / (self.B_EMA_PERIOD + 1.0)

        # Daily state (reset each calendar day)
        self._day_date  = None
        self._asian_h   = []
        self._asian_l   = []
        self._range_h   = None  # finalised Asian high
        self._range_l   = None  # finalised Asian low
        self._london_d  = None  # +1 / -1 / None
        self._trades    = 0
        self._a_done    = False
        self._b_count   = 0
        self._c_done    = False
        self._vwap_n    = 0.0   # VWAP numerator (Σ close × vol)
        self._vwap_d    = 0.0   # VWAP denominator (Σ vol)
        self._vols      = []    # volume history for SMA filter
        self._day_bal   = float(api.Account.Balance)

        self._prewarm()
        api.Print("GoldMultiStrategy started | Balance={:.2f}  Risk={:.1f}%".format(
            api.Account.Balance, self.RISK_PCT))

    def on_bar(self):
        bars = api.MarketData.GetBars(TimeFrame.Hour, self.SYMBOL)
        if bars.Count < 60:
            return

        bar   = bars.Last(1)                       # last COMPLETED bar
        t_utc = bar.OpenTime.ToUniversalTime()
        uh    = t_utc.Hour                         # UTC hour of bar open
        ud    = t_utc.Date                         # UTC date (System.DateTime)
        dow   = int(t_utc.DayOfWeek)               # 0=Sun 1=Mon … 5=Fri 6=Sat

        # ── Daily reset ────────────────────────────────────────────
        if self._day_date is None or ud != self._day_date:
            self._reset_day(ud, bars)

        # ── Kill switch ────────────────────────────────────────────
        if self._kill_check():
            return

        # ── Manage open api.Positions ──────────────────────────────────
        self._manage(bar)

        # ── EMA update ────────────────────────────────────────────
        self._ema_update(bar.Close)

        # ── Collect Asian range bars (UTC 00–07) ───────────────────
        if self.UTC_ASIAN_S <= uh <= self.UTC_ASIAN_E:
            self._asian_h.append(bar.High)
            self._asian_l.append(bar.Low)

        # Finalise Asian range at first London bar (UTC 08:xx)
        if uh == self.UTC_LONDON_S and self._range_h is None and self._asian_h:
            self._range_h = max(self._asian_h)
            self._range_l = min(self._asian_l)

        # ── VWAP accumulator (reset at first NY bar, UTC 14:xx) ───
        if uh == self.UTC_NY_S:
            self._vwap_n = 0.0
            self._vwap_d = 0.0
        if uh >= self.UTC_NY_S:
            vol = max(bar.TickVolume, 1)
            self._vwap_n += bar.Close * vol
            self._vwap_d += vol
            self._vols.append(vol)

        # ── No new entries on Friday ───────────────────────────────
        if dow == 5:
            return

        # ── No entry if a gold position is already open ────────────
        if self._has_pos():
            return

        # ── Daily cap ──────────────────────────────────────────────
        if self._trades >= self.MAX_TRADES_DAY:
            return

        # ── Module A — London Breakout (UTC 08–12) ─────────────────
        if self.UTC_LONDON_S <= uh <= self.UTC_LONDON_E and not self._a_done:
            self._mod_a(bar)

        if self._has_pos():
            return

        # ── Module B — VWAP Pullback (UTC 14–19, max 2/day) ────────
        if self.UTC_NY_S <= uh <= self.UTC_NY_E and self._b_count < 2:
            vwap = self._vwap_n / self._vwap_d if self._vwap_d > 0 else bar.Close
            self._mod_b(bar, vwap)

        if self._has_pos():
            return

        # ── Module C — Session Pullback (UTC 14–19, 1/day) ─────────
        if self.UTC_NY_S <= uh <= self.UTC_NY_E and not self._c_done:
            # Determine London direction once window has passed
            if self._london_d is None and uh >= self.UTC_LONDON_E:
                self._detect_london_dir(ud, bars)
            if self._london_d is not None:
                self._mod_c(bar)

    def on_stop(self):
        for pos in api.Positions:
            if pos.SymbolName == self.SYMBOL:
                api.ClosePosition(pos)
        api.Print("GoldMultiStrategy stopped — all api.Positions closed.")

    # ═══════════════════════════════════════════════════════════════════
    #  Module A — London H1 Breakout
    # ═══════════════════════════════════════════════════════════════════

    def _mod_a(self, bar):
        if self._range_h is None:
            return

        ps  = self.PIP
        buf = self.A_BUF_PIPS * ps
        rng = self._range_h - self._range_l

        if not (self.A_MIN_RANGE * ps <= rng <= self.A_MAX_RANGE * ps):
            return
        if not self._regime_ok():
            return

        sig = 0
        if bar.Close > self._range_h + buf:
            sig = 1
        elif bar.Close < self._range_l - buf:
            sig = -1
        if sig == 0:
            return

        cap = self.A_SL_CAP * ps
        if sig == 1:
            sl_dist = min(bar.Close - (self._range_l - buf), cap)
        else:
            sl_dist = min((self._range_h + buf) - bar.Close, cap)

        if sl_dist <= 0:
            return

        if self._enter(sig, bar, sl_dist, "A", self.A_TIME_STOP_H):
            self._a_done = True

    # ═══════════════════════════════════════════════════════════════════
    #  Module B — VWAP Pullback
    # ═══════════════════════════════════════════════════════════════════

    def _mod_b(self, bar, vwap):
        if self._ema is None:
            return

        ps     = self.PIP
        buf    = self.B_SL_BUFFER * ps
        cap    = self.B_SL_CAP * ps
        mn     = self.B_SL_MIN * ps
        vol_sm = self._vol_sma()

        sig = 0
        if (bar.Close > self._ema
                and bar.Low <= vwap <= bar.Close
                and bar.TickVolume > vol_sm):
            sig = 1
        elif (bar.Close < self._ema
              and bar.High >= vwap >= bar.Close
              and bar.TickVolume > vol_sm):
            sig = -1
        if sig == 0:
            return

        sl_dist = (min(max(bar.Close - bar.Low + buf, mn), cap) if sig == 1
                   else min(max(bar.High - bar.Close + buf, mn), cap))

        if self._enter(sig, bar, sl_dist, "B", self.B_TIME_STOP_H):
            self._b_count += 1

    # ═══════════════════════════════════════════════════════════════════
    #  Module C — NY Session Pullback to London Level
    # ═══════════════════════════════════════════════════════════════════

    def _detect_london_dir(self, utc_date, bars):
        """Scan bars from London window to determine breakout direction."""
        if self._range_h is None:
            return
        ps  = self.PIP
        buf = self.A_BUF_PIPS * ps
        rng = self._range_h - self._range_l
        if not (self.A_MIN_RANGE * ps <= rng <= self.A_MAX_RANGE * ps):
            return

        # Walk recent bars backwards looking for London-window bars
        for i in range(bars.Count - 1, max(bars.Count - 20, -1), -1):
            b    = bars[i]
            b_ut = b.OpenTime.ToUniversalTime()
            if b_ut.Date != utc_date:
                continue
            if self.UTC_LONDON_S <= b_ut.Hour <= self.UTC_LONDON_E:
                if b.Close > self._range_h + buf:
                    self._london_d = 1
                    return
                elif b.Close < self._range_l - buf:
                    self._london_d = -1
                    return

    def _mod_c(self, bar):
        if self._range_h is None or self._london_d is None:
            return

        sig   = self._london_d
        ps    = self.PIP
        level = self._range_h if sig == 1 else self._range_l

        if sig == 1:
            touch = bar.Low <= level and bar.Close > level
        else:
            touch = bar.High >= level and bar.Close < level
        if not touch:
            return

        sl_dist = self.C_SL_PIPS * ps
        sl_price = (level - sl_dist) if sig == 1 else (level + sl_dist)

        # Guard: SL must be on wrong side of entry
        if (sig == 1 and sl_price >= bar.Close) or (sig == -1 and sl_price <= bar.Close):
            return

        actual_dist = abs(bar.Close - sl_price)

        if self._enter(sig, bar, actual_dist, "C", self.C_TIME_STOP_H):
            self._c_done = True

    # ═══════════════════════════════════════════════════════════════════
    #  Order entry
    # ═══════════════════════════════════════════════════════════════════

    def _enter(self, sig, bar, sl_dist, label, ts_hours):
        """
        Place a market order.
        ts_hours: hours after bar.OpenTime before time-stop triggers.
        Returns True if order filled.
        Module C uses full exit at C_RR_TP (no TP1 partial, no trail).
        """
        volume = self._size(sl_dist)
        if volume <= 0:
            api.Print("Gold{}: volume=0 (balance too low or SL too wide)".format(label))
            return False

        trade_type = TradeType.Buy if sig == 1 else TradeType.Sell
        result = api.ExecuteMarketOrder(trade_type, self.SYMBOL, volume, "Gold" + label)

        if not result.IsSuccessful:
            api.Print("Gold{}: order rejected — {}".format(label, result.Error))
            return False

        pos    = result.Position
        entry  = pos.EntryPrice          # actual fill price (includes spread)
        sl_prc = entry - sig * sl_dist

        # Module C: single full-exit TP at C_RR_TP (no partial, no trail)
        full_exit = (label == "C")
        tp_rr  = self.C_RR_TP if full_exit else self.RR_TP2
        tp1    = entry + sig * self.RR_TP1 * sl_dist   # unused for Module C
        tp2    = entry + sig * tp_rr * sl_dist

        # Attach SL and TP to the position (prices, not pips)
        api.ModifyPosition(pos, sl_prc, tp2)

        # Time stop = bar open time + ts_hours (server time, broker-agnostic)
        time_stop = bar.OpenTime.AddHours(ts_hours)

        self._pos_meta[pos.Id] = {
            "sig":       sig,
            "entry":     entry,
            "sl_dist":   sl_dist,
            "cur_sl":    sl_prc,
            "tp1":       tp1,
            "tp2":       tp2,
            "vol":       volume,
            "tp1_hit":   False,
            "trail_on":  False,
            "trail_ext": entry,
            "ts":        time_stop,   # System.DateTime (server tz)
            "label":     label,
            "full_exit": full_exit,   # True for Module C: broker manages TP/SL
        }
        self._trades += 1
        if full_exit:
            api.Print("ENTERED Gold{} {} {:.0f}u @ {:.2f}  SL={:.2f}  TP={:.2f} [full]".format(
                label, "BUY" if sig == 1 else "SELL", volume, entry, sl_prc, tp2))
        else:
            api.Print("ENTERED Gold{} {} {:.0f}u @ {:.2f}  SL={:.2f}  TP1={:.2f}  TP2={:.2f}".format(
                label, "BUY" if sig == 1 else "SELL", volume, entry, sl_prc, tp1, tp2))
        return True

    # ═══════════════════════════════════════════════════════════════════
    #  Position management (called every bar)
    # ═══════════════════════════════════════════════════════════════════

    def _manage(self, bar):
        """Walk all tracked gold api.Positions: time stop, TP1 partial, V4 trail."""
        to_del = []

        for pos in api.Positions:
            if pos.SymbolName != self.SYMBOL:
                continue
            pid = pos.Id
            if pid not in self._pos_meta:
                continue

            m   = self._pos_meta[pid]
            sig = m["sig"]

            # ── Time stop ─────────────────────────────────────────
            # Compare server-time bar open against stored DateTime
            if bar.OpenTime >= m["ts"]:
                api.ClosePosition(pos)
                to_del.append(pid)
                api.Print("TIME STOP Gold" + m["label"])
                continue

            # ── Module C: full exit — broker manages TP/SL; no partial or trail ──
            if m.get("full_exit"):
                continue

            # ── Phase 1: pre-TP1 ──────────────────────────────────
            if not m["tp1_hit"]:
                hit_tp1 = ((sig == 1 and bar.High >= m["tp1"]) or
                           (sig == -1 and bar.Low  <= m["tp1"]))
                if hit_tp1:
                    self._handle_tp1(pos, m)
                    continue

                # Detect if broker already closed position at SL
                still_open = any(p.Id == pid for p in api.Positions)
                if not still_open:
                    to_del.append(pid)
                    continue

            # ── Phase 2: post-TP1 (V4 trail) ──────────────────────
            else:
                self._handle_trail(pos, m, bar, sig)

        for pid in to_del:
            self._pos_meta.pop(pid, None)

    def _handle_tp1(self, pos, m):
        """
        Partial close at TP1; lock SL at 25% of TP1 distance (V4).
        The position object remains valid after partial close (same Id, reduced volume).
        """
        sig  = m["sig"]
        sym  = api.Symbol[self.SYMBOL]
        pvol = sym.NormalizeVolumeInUnits(m["vol"] * self.TP1_FRACTION)

        if pvol >= sym.VolumeInUnitsMin:
            api.ClosePosition(pos, pvol)   # partial close — pos.Id unchanged

        # V4: lock SL → entry + 25% of TP1 dist (never backward)
        lock_dist = abs(m["tp1"] - m["entry"]) * 0.25
        lock_sl   = m["entry"] + sig * lock_dist
        if (sig == 1 and lock_sl > m["cur_sl"]) or (sig == -1 and lock_sl < m["cur_sl"]):
            m["cur_sl"] = lock_sl
            api.ModifyPosition(pos, lock_sl, m["tp2"])

        m["tp1_hit"]  = True
        m["trail_on"] = False
        m["trail_ext"] = m["tp1"]
        api.Print("TP1 partial Gold{} | locked SL @ {:.2f}".format(m["label"], m["cur_sl"]))

    def _handle_trail(self, pos, m, bar, sig):
        """
        V4 trail variant: full ATR trail activates only after price reaches 1.5R.
        Before that, SL stays at the V4 lock level.
        """
        orig_dist = m["sl_dist"]
        trigger   = m["entry"] + sig * 1.5 * orig_dist

        if not m["trail_on"]:
            hit_trig = ((sig == 1 and bar.High >= trigger) or
                        (sig == -1 and bar.Low  <= trigger))
            if hit_trig:
                m["trail_on"]  = True
                m["trail_ext"] = bar.High if sig == 1 else bar.Low

        if not m["trail_on"]:
            return

        atr   = self._atr14()
        trail = self.TRAIL_ATR_MULT * atr
        if trail <= 0:
            return

        if sig == 1:
            m["trail_ext"] = max(m["trail_ext"], bar.High)
            new_sl = m["trail_ext"] - trail
        else:
            m["trail_ext"] = min(m["trail_ext"], bar.Low)
            new_sl = m["trail_ext"] + trail

        if (sig == 1 and new_sl > m["cur_sl"]) or (sig == -1 and new_sl < m["cur_sl"]):
            m["cur_sl"] = new_sl
            # pos.TakeProfit is a nullable double — pass as-is to keep current TP
            api.ModifyPosition(pos, new_sl, pos.TakeProfit)

    # ═══════════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════════

    def _prewarm(self):
        """
        Seed EMA50 and daily ATR history from the existing bar series
        so indicators are valid from the very first bar.
        """
        bars = api.MarketData.GetBars(TimeFrame.Hour, self.SYMBOL)
        n    = bars.Count
        if n < self.B_EMA_PERIOD + 20:
            api.Print("Insufficient history for pre-warm ({} bars)".format(n))
            return

        # EMA50 (SMA seed over first B_EMA_PERIOD bars, then EMA thereafter)
        closes = [bars[i].Close for i in range(n)]
        ema    = sum(closes[:self.B_EMA_PERIOD]) / float(self.B_EMA_PERIOD)
        for c in closes[self.B_EMA_PERIOD:]:
            ema = c * self._ema_k + ema * (1.0 - self._ema_k)
        self._ema = ema

        # Daily ATR proxy: sum of H-L across all H1 bars for each UTC calendar day
        daily_hl = defaultdict(list)
        for i in range(n):
            d = bars[i].OpenTime.ToUniversalTime().Date
            daily_hl[d].append(bars[i].High - bars[i].Low)
        for d in sorted(daily_hl.keys()):
            self._atr_hist.append(sum(daily_hl[d]))
        self._atr_hist = self._atr_hist[-120:]

        api.Print("Pre-warmed | EMA50={:.2f}  ATR-days={}".format(self._ema, len(self._atr_hist)))

    def _reset_day(self, utc_date, bars):
        """New UTC calendar day: append ATR proxy, reset all daily counters."""
        atr = self._atr14(bars)
        if atr > 0:
            self._atr_hist.append(atr * 24.0)   # H1 ATR scaled → daily proxy
            self._atr_hist = self._atr_hist[-120:]

        self._day_date  = utc_date
        self._asian_h   = []
        self._asian_l   = []
        self._range_h   = None
        self._range_l   = None
        self._london_d  = None
        self._trades    = 0
        self._a_done    = False
        self._b_count   = 0
        self._c_done    = False
        self._vwap_n    = 0.0
        self._vwap_d    = 0.0
        self._vols      = []
        self._day_bal   = float(api.Account.Balance)

    def _ema_update(self, close):
        if self._ema is None:
            self._ema = close
        else:
            self._ema = close * self._ema_k + self._ema * (1.0 - self._ema_k)

    def _vol_sma(self):
        v = self._vols[-20:] if len(self._vols) >= 20 else self._vols
        return sum(v) / max(len(v), 1)

    def _atr14(self, bars=None):
        """ATR(14) from the last 15 H1 bars using True Range."""
        if bars is None:
            bars = api.MarketData.GetBars(TimeFrame.Hour, self.SYMBOL)
        n = bars.Count
        if n < 15:
            return 3.70   # fallback: historical avg H1 ATR for XAUUSD
        trs = []
        for i in range(n - 15, n - 1):
            b0 = bars[i]
            b1 = bars[i + 1]
            trs.append(max(
                b1.High - b1.Low,
                abs(b1.High - b0.Close),
                abs(b1.Low  - b0.Close),
            ))
        return sum(trs[-14:]) / 14.0

    def _regime_ok(self):
        """
        Regime filter for Module A: today's H-L range proxy must exceed
        the A_REGIME_PCT percentile of the last 60 days' range history.
        Returns True (allow trade) if insufficient history.
        """
        if len(self._atr_hist) < 10:
            return True
        recent    = sorted(self._atr_hist[-60:])
        idx       = max(0, int(len(recent) * self.A_REGIME_PCT / 100.0) - 1)
        threshold = recent[idx]

        bars = api.MarketData.GetBars(TimeFrame.Hour, self.SYMBOL)
        n    = bars.Count
        if n < 24:
            return True
        today_range = sum(bars[n - 1 - i].High - bars[n - 1 - i].Low for i in range(24))
        return today_range > threshold

    def _size(self, sl_dist):
        """
        Compute position volume in units.
        risk_usd = balance × RISK_PCT / 100
        lots     = risk_usd / (sl_pips × pip_value_per_lot)
        units    = lots × lot_size, normalised to broker minimum step.
        """
        if sl_dist <= 0:
            return 0
        sym      = api.Symbol[self.SYMBOL]
        risk_usd = float(api.Account.Balance) * self.RISK_PCT / 100.0
        sl_pips  = sl_dist / self.PIP
        pip_val  = sym.PipValue        # USD per pip per 1 standard lot
        if pip_val <= 0:
            return 0
        lots  = risk_usd / (sl_pips * pip_val)
        units = sym.NormalizeVolumeInUnits(lots * sym.LotSize)
        return max(units, sym.VolumeInUnitsMin)

    def _has_pos(self):
        """True if any XAUUSD position is currently open."""
        for pos in api.Positions:
            if pos.SymbolName == self.SYMBOL:
                return True
        return False

    def _kill_check(self):
        """
        Daily kill switch: if intraday equity drawdown >= DAILY_KILL_PCT,
        close all gold api.Positions and block new entries for the rest of the day.
        Reference: account balance at start of UTC calendar day.
        """
        eq  = float(api.Account.Equity)
        dd  = (self._day_bal - eq) / max(self._day_bal, 1.0) * 100.0
        if dd >= self.DAILY_KILL_PCT:
            for pos in api.Positions:
                if pos.SymbolName == self.SYMBOL:
                    api.ClosePosition(pos)
            self._trades = self.MAX_TRADES_DAY   # block further entries today
            api.Print("KILL SWITCH | DD={:.2f}% | all gold api.Positions closed".format(dd))
            return True
        return False
