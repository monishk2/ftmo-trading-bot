"""
Strategy 3: NY Session Breakout
================================

WHY IT WORKS
------------
Between 02:00–09:15 ET, price consolidates in a range set by the tail end of
London morning and the pre-market drift.  At 09:30 ET — the NYSE open — US
banks and funds begin executing accumulated client orders.  This is the single
highest-volume event on the forex calendar and mechanically forces price to
break the overnight range for the same structural reason as the London open:
banks MUST execute client orders at the open regardless of direction.

Identical implementation to LondonOpenBreakout; only the time windows differ:

  Range window  :  02:00–09:15 ET  (pre-NY consolidation)
  Entry window  :  09:30–11:00 ET  (NY open momentum)
  Time stop     :  14:00 ET        (NY midday liquidity dries up)

All filtering, regime gating, SL/TP, and signal column contracts are the same.
The Backtester handles break-even / trailing stops independently.
"""

from __future__ import annotations

from strategies.london_open_breakout import LondonOpenBreakout


class NYSessionBreakout(LondonOpenBreakout):
    """
    NY Session breakout strategy.

    Reuses LondonOpenBreakout entirely — only the name differs so that
    the Backtester can log / save results under the correct label.
    All parameters come from the 'ny_session_breakout' block in
    strategy_params.json via the same setup() interface.
    """

    @property
    def name(self) -> str:
        return "NYSessionBreakout"
