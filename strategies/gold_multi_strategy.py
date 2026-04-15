"""
strategies/gold_multi_strategy.py
===================================
Gold Multi-Strategy Combiner (Modules A + B + C)

Runs all three modules simultaneously. Merges their signals into a single
signal stream respecting:
  - Max 1 position at a time (enforced by backtester's last_exit_idx)
  - Max max_trades_per_day signals emitted per calendar day
  - Module priority: A fires first (London, 03:00-08:00); B+C compete in NY (09:00-14:59)
    Ties broken by bar index (chronological)

The combined strategy forwards the TP1/TP2/trail_distance columns so
run_partial() can handle them correctly.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from strategies.gold_london_h1 import GoldLondonH1
from strategies.gold_vwap_h1 import GoldVwapH1
from strategies.gold_session_pullback_h1 import GoldSessionPullbackH1


class GoldMultiStrategy:
    """
    Combines GoldLondonH1 (A), GoldVwapH1 (B), GoldSessionPullbackH1 (C).

    Parameters forwarded to all three modules:
      rr_tp1, rr_tp2, tp1_pct, trail_atr_mult, risk_per_trade_pct
    """

    name = "gold_multi_strategy"

    def __init__(
        self,
        rr_tp1:             float = 1.0,
        rr_tp2:             float = 2.5,
        tp1_pct:            float = 0.5,
        trail_atr_mult:     float = 1.5,
        risk_per_trade_pct: float = 0.5,
        max_trades_per_day: int   = 3,
        # Module A specific
        a_buffer_pips:      float = 50.0,
        a_min_range_pips:   float = 500.0,
        a_max_range_pips:   float = 2000.0,
        a_sl_cap_pips:      float = 1000.0,
        a_regime_atr_pct:   float = 40.0,
        # Module B specific
        b_sl_buffer_pips:   float = 30.0,
        b_sl_cap_pips:      float = 500.0,
        b_sl_min_pips:      float = 20.0,
        # Module C specific
        c_sl_pips:          float = 200.0,
    ) -> None:
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_trades_per_day = max_trades_per_day

        self._mod_a = GoldLondonH1(
            rr_tp1=rr_tp1, rr_tp2=rr_tp2, tp1_pct=tp1_pct,
            trail_atr_mult=trail_atr_mult,
            buffer_pips=a_buffer_pips, min_range_pips=a_min_range_pips,
            max_range_pips=a_max_range_pips, sl_cap_pips=a_sl_cap_pips,
            regime_atr_pct=a_regime_atr_pct,
            risk_per_trade_pct=risk_per_trade_pct,
        )
        self._mod_b = GoldVwapH1(
            rr_tp1=rr_tp1, rr_tp2=rr_tp2, tp1_pct=tp1_pct,
            trail_atr_mult=trail_atr_mult,
            sl_buffer_pips=b_sl_buffer_pips, sl_cap_pips=b_sl_cap_pips,
            sl_min_pips=b_sl_min_pips,
            risk_per_trade_pct=risk_per_trade_pct,
        )
        self._mod_c = GoldSessionPullbackH1(
            rr_tp1=rr_tp1, rr_tp2=rr_tp2, tp1_pct=tp1_pct,
            trail_atr_mult=trail_atr_mult,
            sl_pips=c_sl_pips,
            risk_per_trade_pct=risk_per_trade_pct,
        )

    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all three modules, merge signals chronologically.
        Enforce max_trades_per_day.
        """
        df_a = self._mod_a.generate_signals(df.copy())
        df_b = self._mod_b.generate_signals(df.copy())
        df_c = self._mod_c.generate_signals(df.copy())

        out = df.copy()
        n   = len(out)
        idx = out.index
        dates = idx.date

        # Initialise output columns
        for col in ("signal", "sl_price", "tp_price", "tp1_price",
                    "tp1_pct", "trail_distance"):
            out[col] = np.nan
        out["signal"]    = 0
        out["time_stop"] = None
        out["time_stop"] = out["time_stop"].astype(object)

        # Build merged signal list: (bar_index, source_df)
        # Priority when same bar: A > B > C
        sig_a = set(np.where(df_a["signal"].fillna(0).to_numpy() != 0)[0])
        sig_b = set(np.where(df_b["signal"].fillna(0).to_numpy() != 0)[0])
        sig_c = set(np.where(df_c["signal"].fillna(0).to_numpy() != 0)[0])

        all_bars = sorted(sig_a | sig_b | sig_c)

        trades_by_day: dict = defaultdict(int)
        used: set = set()

        for i in all_bars:
            d = dates[i]
            if trades_by_day[d] >= self.max_trades_per_day:
                continue

            # Priority: A > B > C (if same bar has signals from multiple modules)
            if i in sig_a:
                src = df_a
            elif i in sig_b:
                src = df_b
            else:
                src = df_c

            if float(src["signal"].iat[i]) == 0:
                continue

            out.iat[i, out.columns.get_loc("signal")]         = src["signal"].iat[i]
            out.iat[i, out.columns.get_loc("sl_price")]       = src["sl_price"].iat[i]
            out.iat[i, out.columns.get_loc("tp_price")]       = src["tp_price"].iat[i]
            out.iat[i, out.columns.get_loc("tp1_price")]      = src["tp1_price"].iat[i]
            out.iat[i, out.columns.get_loc("tp1_pct")]        = src["tp1_pct"].iat[i]
            out.iat[i, out.columns.get_loc("trail_distance")] = src["trail_distance"].iat[i]
            out.iat[i, out.columns.get_loc("time_stop")]      = src["time_stop"].iat[i]
            trades_by_day[d] += 1

        return out
