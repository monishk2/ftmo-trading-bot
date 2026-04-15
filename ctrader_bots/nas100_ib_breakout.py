"""
ctrader_bots/nas100_ib_breakout.py
====================================
cTrader Python cBot — NAS100 Initial Balance Breakout (M5)

STRATEGY SUMMARY
----------------
  1. During 09:30-10:30 ET, track Initial Balance: IB_high, IB_low
  2. After IB is finalised, watch for M5 close beyond IB boundary + buffer
     with volume > N-bar SMA filter
  3. Enter at market on the signal bar (M5 close ≈ next open)
  4. SL = opposite IB wall ± buffer, capped at MAX_SL_PTS
  5. TP = entry ± RR_RATIO × sl_dist
  6. Time stop: 15:55 ET
  7. Max 1 trade per day; no Friday entries

OPTIMAL PARAMETERS (Prompt 28 WF pass — avg_deg=+1.944, pct_pos=70%)
----------------------------------------------------------------------
  IB_ADR_RATIO = 99   (no IB width filter — raw NY open breakout is superior)
  RR_RATIO     = 2.5
  RISK_PCT     = 1.5% Brute config (48.5% pass <21d, 9.9% blow-up)
              = 2.0% for Apex eval phase

SESSION TIMES IN UTC (EDT = UTC-4, Mar-Nov)
-------------------------------------------
  IB window:  13:30-14:30 UTC (09:30-10:30 ET EDT)
  Entry ends: 19:55 UTC     (15:55 ET EDT)
  In EST (UTC-5, Nov-Mar): add 1 hour to IB_S_H, IB_E_H, TS_H.
  Adjust constants before rebuilding when DST changes.

FIXED for cTrader Python API
-----------------------------
  - All platform calls use api. prefix (api.Account, api.Positions, etc.)
  - TimeFrame.Minute5 for M5 bars
  - Class inherits from nothing (no 'object')

DEPLOY
------
  1. cTrader Automate → My Robots → Upload → Select this file
  2. Attach to NAS100 / US TECH 100 M5 chart (verify symbol with broker)
  3. Python cBots do NOT surface a parameter panel — edit SYMBOL, RISK_PCT,
     and other constants in this file, then rebuild before running.
  4. Cloud execution keeps it alive 24/7.

NOTE: NAS100 pip_size = 1.0 (1 point = $1/lot base; verify PipValue at runtime).
"""

import clr
clr.AddReference("cAlgo.API")
from cAlgo.API import *
from robot_wrapper import *


class NAS100IBbreakout():

    # ── CONFIGURATION — edit before building ──────────────────────────────────
    SYMBOL          = "US TECH 100"  # verify with broker: US100, US TECH 100, NAS100, or USTEC
    RISK_PCT        = 1.5        # % risk per trade (1.5 Brute config; 2.0 Apex eval)
    RR_RATIO        = 2.5        # TP distance = RR_RATIO × sl_dist
    DAILY_KILL_PCT  = 3.5        # stop trading when daily equity DD >= this %
    IB_ADR_RATIO    = 99.0       # IB width / ADR filter  (99 = disabled)
    ADR_LOOKBACK    = 5          # days for ADR calculation
    BUFFER_PTS      = 5.0        # breakout buffer beyond IB wall (points)
    MAX_SL_PTS      = 80.0       # maximum SL distance in points
    MIN_SL_PTS      = 15.0       # minimum SL distance in points
    VOL_SMA_PERIOD  = 20         # volume SMA lookback (bars)
    PIP             = 1.0        # NAS100: 1 pip = 1 index point

    # Session times in UTC (EDT = UTC-4, Mar-Nov).
    # In EST (UTC-5, Nov-Mar) add 1 to IB_S_H, IB_E_H, TS_H.
    IB_S_H = 13; IB_S_M = 30   # IB window opens   (9:30 ET EDT)
    IB_E_H = 14; IB_E_M = 30   # IB window closes  (10:30 ET EDT)
    TS_H   = 19; TS_M   = 55   # time stop          (15:55 ET EDT)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_start(self):
        self._day_date   = None
        self._ib_h       = None
        self._ib_l       = None
        self._ib_bars_h  = []
        self._ib_bars_l  = []
        self._ib_done    = False
        self._sig_taken  = False
        self._day_bal    = float(api.Account.Balance)
        self._killed     = False
        self._pos_id     = None
        self._vol_hist   = []
        self._adr        = 0.0
        api.Print("[NAS100IB] Started. Balance={:.2f}  Risk={}%  RR={}".format(
            api.Account.Balance, self.RISK_PCT, self.RR_RATIO))

    def on_bar(self):
        bars = api.MarketData.GetBars(TimeFrame.Minute5, self.SYMBOL)
        if bars.Count < self.VOL_SMA_PERIOD + 2:
            return

        bar   = bars.Last(1)                        # last COMPLETED M5 bar
        t_utc = bar.OpenTime.ToUniversalTime()
        uh    = t_utc.Hour
        um    = t_utc.Minute
        ud    = t_utc.Date
        dow   = int(t_utc.DayOfWeek)               # 0=Sun … 5=Fri … 6=Sat

        # ── Daily reset ──────────────────────────────────────────────────────
        if ud != self._day_date:
            self._new_day(ud, bars)

        if self._killed:
            return

        # ── Kill switch ───────────────────────────────────────────────────────
        equity = float(api.Account.Equity)
        dd_pct = (self._day_bal - equity) / max(self._day_bal, 1.0) * 100.0
        if dd_pct >= self.DAILY_KILL_PCT:
            self._close_all("kill_switch DD={:.2f}%".format(dd_pct))
            self._killed = True
            return

        # ── Volume history ────────────────────────────────────────────────────
        self._vol_hist.append(max(float(bar.TickVolume), 1.0))
        if len(self._vol_hist) > 500:
            self._vol_hist = self._vol_hist[-500:]

        # ── Manage open position (detect SL/TP close) ─────────────────────────
        self._manage_pos()

        # ── Boolean helpers for session windows ───────────────────────────────
        after_ib_start  = (uh > self.IB_S_H) or (uh == self.IB_S_H and um >= self.IB_S_M)
        before_ib_end   = (uh < self.IB_E_H) or (uh == self.IB_E_H and um < self.IB_E_M)
        at_or_after_end = (uh > self.IB_E_H) or (uh == self.IB_E_H and um >= self.IB_E_M)
        at_time_stop    = (uh > self.TS_H)   or (uh == self.TS_H   and um >= self.TS_M)

        # ── Accumulate IB bars ────────────────────────────────────────────────
        if after_ib_start and before_ib_end:
            self._ib_bars_h.append(float(bar.High))
            self._ib_bars_l.append(float(bar.Low))

        # ── Finalise IB at first bar that opens at/after IB_E ────────────────
        if at_or_after_end and not self._ib_done and self._ib_bars_h:
            self._ib_h    = max(self._ib_bars_h)
            self._ib_l    = min(self._ib_bars_l)
            self._ib_done = True
            api.Print("[NAS100IB] IB finalised: {:.1f}-{:.1f}  "
                      "width={:.1f}pts  ADR={:.1f}pts".format(
                          self._ib_l, self._ib_h,
                          self._ib_h - self._ib_l, self._adr))

        # ── No new entries on Friday ──────────────────────────────────────────
        if dow == 5:
            return

        # ── Time stop: close any open position ────────────────────────────────
        if at_time_stop:
            self._close_all("time_stop")
            return

        # ── Entry logic ───────────────────────────────────────────────────────
        in_entry_window = at_or_after_end and not at_time_stop
        if not in_entry_window:
            return

        if self._sig_taken or self._ib_h is None:
            return

        if self._has_open_pos():
            return

        self._try_entry(bar)

    def on_stop(self):
        self._close_all("bot_stopped")
        api.Print("[NAS100IB] Bot stopped.")

    # ── Entry logic ───────────────────────────────────────────────────────────

    def _try_entry(self, bar):
        buf   = self.BUFFER_PTS
        close = float(bar.Close)

        # IB width filter (disabled when IB_ADR_RATIO >= 98)
        if self.IB_ADR_RATIO < 98.0 and self._adr > 0.0:
            ib_width = self._ib_h - self._ib_l
            if ib_width > self.IB_ADR_RATIO * self._adr:
                return   # IB too wide relative to ADR

        # Volume filter
        vol_sma = self._vol_sma()
        vol     = float(bar.TickVolume)

        sig = 0
        if close > self._ib_h + buf and vol >= vol_sma:
            sig = 1
        elif close < self._ib_l - buf and vol >= vol_sma:
            sig = -1
        if sig == 0:
            return

        # SL sizing from opposite IB wall
        if sig == 1:
            sl_dist = close - (self._ib_l - buf)
        else:
            sl_dist = (self._ib_h + buf) - close

        if sl_dist < self.MIN_SL_PTS:
            return
        sl_dist = min(sl_dist, self.MAX_SL_PTS)

        volume = self._size(sl_dist)
        if volume <= 0:
            return

        trade_type = TradeType.Buy if sig == 1 else TradeType.Sell
        result = api.ExecuteMarketOrder(trade_type, self.SYMBOL, volume, "NAS100IB")
        if not result.IsSuccessful:
            api.Print("[NAS100IB] Order rejected: {}".format(result.Error))
            return

        pos   = result.Position
        entry = float(pos.EntryPrice)       # actual fill price
        sl_prc = entry - sig * sl_dist      # recalculate from actual fill
        tp_prc = entry + sig * self.RR_RATIO * sl_dist
        api.ModifyPosition(pos, sl_prc, tp_prc)

        self._pos_id    = pos.Id
        self._sig_taken = True
        api.Print("[NAS100IB] {} {:.0f}u @ {:.1f}  SL={:.1f}  TP={:.1f}".format(
            "LONG" if sig == 1 else "SHORT", volume, entry, sl_prc, tp_prc))

    # ── Position management ───────────────────────────────────────────────────

    def _manage_pos(self):
        """Detect when broker closed the position (hit SL or TP)."""
        if self._pos_id is None:
            return
        for pos in api.Positions:
            if pos.Id == self._pos_id:
                return          # still open
        self._pos_id = None     # no longer in Positions → closed by broker
        api.Print("[NAS100IB] Position closed by broker (SL/TP hit).")

    def _has_open_pos(self):
        if self._pos_id is None:
            return False
        for pos in api.Positions:
            if pos.Id == self._pos_id:
                return True
        self._pos_id = None     # stale — already closed
        return False

    def _close_all(self, reason):
        for pos in api.Positions:
            if pos.SymbolName == self.SYMBOL:
                api.ClosePosition(pos)
        if self._pos_id is not None:
            self._pos_id = None
            api.Print("[NAS100IB] Closed  reason={}".format(reason))

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _new_day(self, ud, bars):
        self._day_date  = ud
        self._ib_h      = None
        self._ib_l      = None
        self._ib_bars_h = []
        self._ib_bars_l = []
        self._ib_done   = False
        self._sig_taken = False
        self._killed    = False
        self._day_bal   = float(api.Account.Balance)
        self._adr       = self._compute_adr(bars, ud)
        api.Print("[NAS100IB] New day  ADR={:.1f}pts".format(self._adr))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_adr(self, bars, today):
        """
        Average daily range over ADR_LOOKBACK prior days.
        Iterates M5 bar history oldest→newest, groups by UTC date,
        skips today's bars.
        today = .NET DateTime (midnight UTC)
        """
        day_hi = {}
        day_lo = {}
        for i in range(bars.Count - 1, -1, -1):
            b    = bars[i]
            d    = b.OpenTime.ToUniversalTime().Date
            if d == today:
                continue
            # Use Ticks as plain Python int key (unique per UTC day)
            key  = int(d.Ticks)
            bh   = float(b.High)
            bl   = float(b.Low)
            if key not in day_hi:
                day_hi[key] = bh
                day_lo[key] = bl
            else:
                if bh > day_hi[key]: day_hi[key] = bh
                if bl < day_lo[key]: day_lo[key] = bl
            if len(day_hi) >= self.ADR_LOOKBACK:
                break

        if not day_hi:
            return 0.0
        return sum(day_hi[k] - day_lo[k] for k in day_hi) / len(day_hi)

    def _vol_sma(self):
        hist = self._vol_hist[-self.VOL_SMA_PERIOD:] \
               if len(self._vol_hist) >= self.VOL_SMA_PERIOD \
               else self._vol_hist
        return sum(hist) / max(len(hist), 1)

    def _size(self, sl_dist):
        """
        Position size in cTrader volume units.
          risk_usd = balance × RISK_PCT / 100
          lots     = risk_usd / (sl_pips × pip_value_per_lot)
          units    = lots × lot_size
        """
        if sl_dist <= 0:
            return 0
        sym      = api.Symbols[self.SYMBOL]
        risk_usd = float(api.Account.Balance) * self.RISK_PCT / 100.0
        sl_pips  = sl_dist / self.PIP          # PIP=1 → sl_pips = sl_dist for NAS100
        pip_val  = float(sym.PipValue)         # USD per pip per 1 standard lot
        if pip_val <= 0:
            return 0
        lots  = risk_usd / (sl_pips * pip_val)
        units = sym.NormalizeVolumeInUnits(lots * float(sym.LotSize))
        return max(units, sym.VolumeInUnitsMin)
