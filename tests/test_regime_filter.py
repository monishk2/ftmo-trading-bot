"""
Tests for strategies/regime_filter.py

Design principle
----------------
We control ATR percentile precisely by constructing synthetic daily OHLCV
where the last N days have known ATR values.  Since ATR percentile =
fraction of the lookback window's ATRs that are LESS THAN today's ATR, we
can place today's ATR above or below a known fraction of the history to
produce an exact percentile.

ATR ≈ high - low for a candle with no gap from previous close (flat candles).
ADX is harder to control exactly, so we use extreme up-only or flat candles
to reliably push ADX high or low.

Test matrix
-----------
 1. HIGH_VOL regime: ATR%ile > 70
 2. NORMAL regime:   ATR%ile in [30, 70]
 3. LOW_VOL regime:  ATR%ile < 30
 4. DEAD regime:     ATR%ile < 15 AND ADX < 20
 5. DEAD not triggered when ADX is high (moving market, low range)
 6. Insufficient history → defaults to NORMAL (no crash)
 7. get_active_strategies HIGH_VOL → [LondonOpenBreakout]
 8. get_active_strategies NORMAL   → [LondonOpenBreakout, FVGRetracement]
 9. get_active_strategies LOW_VOL  → [FVGRetracement]
10. get_active_strategies DEAD     → []
11. Works with 15-min input (resampled internally to daily)
12. Works with pre-resampled daily input
13. Correct strategy names match strategy .name properties
14. ATR%ile = 70 boundary is NORMAL (not HIGH_VOL)
15. ATR%ile = 30 boundary is LOW_VOL (not NORMAL)
"""

import numpy as np
import pandas as pd
import pytest

from strategies.regime_filter import (
    LONDON_BREAKOUT,
    FVG_RETRACEMENT,
    REGIME_DEAD,
    REGIME_HIGH_VOL,
    REGIME_LOW_VOL,
    REGIME_NORMAL,
    RegimeFilter,
)

# ---------------------------------------------------------------------------
# Config (mirrors strategy_params.json regime_filter section)
# ---------------------------------------------------------------------------

CONFIG = {
    "atr_period":             14,
    "atr_lookback_days":      60,
    "high_vol_percentile":    70,
    "low_vol_percentile":     30,
    "dead_percentile_threshold": 15,
    "adx_period":             14,
    "adx_trending_threshold": 25,
    "adx_ranging_threshold":  20,
}


def _rf() -> RegimeFilter:
    r = RegimeFilter()
    r.setup(CONFIG)
    return r


# ---------------------------------------------------------------------------
# Synthetic daily data builders
# ---------------------------------------------------------------------------

def _flat_candle(ts: pd.Timestamp, price: float, atr: float) -> dict:
    """
    A candle with known True Range ≈ atr (high-low).
    No gap from prev close assumed (so TR = high-low).
    """
    return {
        "open":   price,
        "high":   price + atr / 2,
        "low":    price - atr / 2,
        "close":  price,
        "volume": 1000.0,
    }


def _daily_df_with_controlled_atr(
    lookback_atr: float,
    today_atr:    float,
    n_history:    int = 80,   # must be > atr_period + atr_lookback_days = 74
    base_price:   float = 1.10000,
    freq:         str = "B",  # business days
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Build a daily OHLCV DataFrame where:
      - The first `n_history - 1` bars have True Range ≈ `lookback_atr`
      - The final bar (today) has True Range ≈ `today_atr`

    Returns (df, today_ts).

    Because ATR is an EMA, early bars influence today's ATR reading.
    We make lookback candles flat (no trend) and give them the same TR so
    the EMA converges cleanly.  With enough warm-up bars the EMA ≈ TR.
    """
    n_total = n_history + 1   # + today
    dates   = pd.date_range("2022-01-03", periods=n_total, freq=freq)

    rows = []
    for i, ts in enumerate(dates):
        atr = today_atr if i == len(dates) - 1 else lookback_atr
        rows.append(_flat_candle(ts, base_price, atr))

    df = pd.DataFrame(rows, index=dates)
    return df, dates[-1]


def _trending_daily_df(
    n_bars: int = 100,
    step:   float = 0.0010,     # daily close steps up by `step`
    base:   float = 1.10000,
    atr:    float = 0.0010,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Strongly trending market: close steps up every day.
    ADX will be high.  ATR is controlled by the high-low range.
    """
    dates = pd.date_range("2022-01-03", periods=n_bars, freq="B")
    rows  = []
    price = base
    for ts in dates:
        rows.append({
            "open":   price,
            "high":   price + atr / 2,
            "low":    price - atr / 2,
            "close":  price + step,
            "volume": 1000.0,
        })
        price += step
    df = pd.DataFrame(rows, index=dates)
    return df, dates[-1]


# ---------------------------------------------------------------------------
# 1. HIGH_VOL regime
# ---------------------------------------------------------------------------

class TestHighVolRegime:
    """
    60 history days at ATR=0.0010.  Today at ATR=0.0030.
    All 60 history bars (0.0010) < today (0.0030) → ATR%ile = 100% > 70 → HIGH_VOL.
    """

    def setup_method(self):
        df, today = _daily_df_with_controlled_atr(
            lookback_atr=0.0010, today_atr=0.0030, n_history=80
        )
        self.regime    = _rf().get_regime(df, today.date())
        self.today     = today

    def test_regime_is_high_vol(self):
        assert self.regime == REGIME_HIGH_VOL

    def test_not_dead(self):
        assert self.regime != REGIME_DEAD


# ---------------------------------------------------------------------------
# 2. NORMAL regime
# ---------------------------------------------------------------------------

class TestNormalRegime:
    """
    History spans ATR values from low to high.  Today sits at the median.
    Exact: 60 bars, 30 bars have atr=0.0005 and 30 bars have atr=0.0015.
    Today ATR = 0.0010 → 50th percentile → NORMAL.
    """

    def setup_method(self):
        n_history = 80
        n_total   = n_history + 1
        dates     = pd.date_range("2022-01-03", periods=n_total, freq="B")

        # 60 lookback bars: alternating 0.0005 and 0.0015 (30 each)
        # 20 warm-up bars before the lookback window at ATR=0.0010
        rows = []
        for i, ts in enumerate(dates):
            if i < n_history - 60:
                atr = 0.0010
            elif i == len(dates) - 1:
                atr = 0.0010  # today — median of [0.0005, 0.0015] mix
            elif (i - (n_history - 60)) % 2 == 0:
                atr = 0.0005
            else:
                atr = 0.0015
            rows.append(_flat_candle(ts, 1.10000, atr))

        df = pd.DataFrame(rows, index=dates)
        self.regime = _rf().get_regime(df, dates[-1].date())

    def test_regime_is_normal(self):
        assert self.regime == REGIME_NORMAL


# ---------------------------------------------------------------------------
# 3. LOW_VOL regime
# ---------------------------------------------------------------------------

class TestLowVolRegime:
    """
    History at ATR=0.0030.  Today at ATR=0.0005.
    All 60 history bars (0.0030) > today (0.0005) → ATR%ile = 0% < 30 → LOW_VOL.
    (Dead not triggered because ATR%ile=0 < 15 AND… we'll check ADX separately.
    In a flat-candle scenario ADX can be low.  To avoid DEAD, ensure ATR%ile
    is controlled, but not at dead-level — use lookback=0.0010, today=0.0001
    with 16% in the lower part of the distribution, leaving today at ~0 percentile
    which IS below 15.  Instead just use lookback=0.0008, today=0.0003 and verify
    we're in LOW_VOL.  If ADX is also low we might get DEAD, so we use a trending
    history to keep ADX elevated above the threshold.)
    """

    def setup_method(self):
        # Trending history → high ADX → dead-market condition won't trigger
        # even though ATR%ile < 30.
        df, today = _trending_daily_df(n_bars=100, step=0.0010, atr=0.0030)
        # Override the last bar to have very low ATR while history stays high
        df.iloc[-1, df.columns.get_loc("high")] = df.iloc[-1]["close"] + 0.0003 / 2
        df.iloc[-1, df.columns.get_loc("low")]  = df.iloc[-1]["close"] - 0.0003 / 2
        self.regime = _rf().get_regime(df, today.date())

    def test_regime_is_low_vol(self):
        assert self.regime == REGIME_LOW_VOL


# ---------------------------------------------------------------------------
# 4. DEAD regime
# ---------------------------------------------------------------------------

class TestDeadRegime:
    """
    ATR%ile < 15 (very low volatility) AND ADX < 20 (no trend).

    Construct: 60 history bars with ATR=0.0030, today ATR=0.0002.
    ATR%ile = 0% → below 15 dead threshold.
    Flat history with no trend → ADX will be near 0.
    """

    def setup_method(self):
        df, today = _daily_df_with_controlled_atr(
            lookback_atr=0.0030, today_atr=0.0002, n_history=80
        )
        self.regime = _rf().get_regime(df, today.date())

    def test_regime_is_dead(self):
        assert self.regime == REGIME_DEAD


# ---------------------------------------------------------------------------
# 5. Low ATR%ile BUT trending market → LOW_VOL not DEAD
# ---------------------------------------------------------------------------

class TestLowAtrHighAdxIsLowVolNotDead:
    """
    ATR%ile < 15 (very small candles today vs history) BUT the market has
    been strongly trending → ADX >> 20 → condition for DEAD not met.
    Should classify as LOW_VOL.
    """

    def setup_method(self):
        # Strongly trending (step > ATR so trend dominates DM)
        df, today = _trending_daily_df(n_bars=100, step=0.0020, atr=0.0030)
        # Crush today's range to make ATR%ile → 0
        df.iloc[-1, df.columns.get_loc("high")] = df.iloc[-1]["close"] + 0.0001
        df.iloc[-1, df.columns.get_loc("low")]  = df.iloc[-1]["close"] - 0.0001
        self.regime = _rf().get_regime(df, today.date())

    def test_not_dead(self):
        # High ADX prevents DEAD classification
        assert self.regime != REGIME_DEAD

    def test_is_low_vol(self):
        assert self.regime == REGIME_LOW_VOL


# ---------------------------------------------------------------------------
# 6. Insufficient history → NORMAL (safe default)
# ---------------------------------------------------------------------------

class TestInsufficientHistory:
    """Only 10 daily bars — far fewer than atr_period + atr_lookback_days = 74."""

    def setup_method(self):
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        rows  = [_flat_candle(ts, 1.10000, 0.0015) for ts in dates]
        df    = pd.DataFrame(rows, index=dates)
        self.regime = _rf().get_regime(df, dates[-1].date())

    def test_defaults_to_normal(self):
        assert self.regime == REGIME_NORMAL


# ---------------------------------------------------------------------------
# 7–10. get_active_strategies per regime
# ---------------------------------------------------------------------------

class TestActiveStrategiesHighVol:
    def setup_method(self):
        df, today = _daily_df_with_controlled_atr(
            lookback_atr=0.0010, today_atr=0.0030, n_history=80
        )
        self.active = _rf().get_active_strategies(df, today.date())

    def test_london_active(self):
        assert LONDON_BREAKOUT in self.active

    def test_fvg_inactive(self):
        assert FVG_RETRACEMENT not in self.active

    def test_exactly_one(self):
        assert len(self.active) == 1


class TestActiveStrategiesNormal:
    def setup_method(self):
        # Build a DataFrame where today sits at ~50th percentile
        n = 81
        dates = pd.date_range("2022-01-03", periods=n, freq="B")
        rows = []
        for i, ts in enumerate(dates):
            if i == n - 1:
                atr = 0.0010   # median
            elif i < 20:
                atr = 0.0010   # warm-up
            elif (i - 20) % 2 == 0:
                atr = 0.0005
            else:
                atr = 0.0015
            rows.append(_flat_candle(ts, 1.10, atr))
        df = pd.DataFrame(rows, index=dates)
        self.active = _rf().get_active_strategies(df, dates[-1].date())

    def test_both_active(self):
        assert LONDON_BREAKOUT in self.active
        assert FVG_RETRACEMENT in self.active

    def test_exactly_two(self):
        assert len(self.active) == 2


class TestActiveStrategiesLowVol:
    def setup_method(self):
        df, today = _trending_daily_df(n_bars=100, step=0.0010, atr=0.0030)
        df.iloc[-1, df.columns.get_loc("high")] = df.iloc[-1]["close"] + 0.0003 / 2
        df.iloc[-1, df.columns.get_loc("low")]  = df.iloc[-1]["close"] - 0.0003 / 2
        self.active = _rf().get_active_strategies(df, today.date())

    def test_fvg_active(self):
        assert FVG_RETRACEMENT in self.active

    def test_london_inactive(self):
        assert LONDON_BREAKOUT not in self.active

    def test_exactly_one(self):
        assert len(self.active) == 1


class TestActiveStrategiesDead:
    def setup_method(self):
        df, today = _daily_df_with_controlled_atr(
            lookback_atr=0.0030, today_atr=0.0002, n_history=80
        )
        self.active = _rf().get_active_strategies(df, today.date())

    def test_no_strategies(self):
        assert self.active == []


# ---------------------------------------------------------------------------
# 11. Works with 15-min OHLCV input (resampled internally)
# ---------------------------------------------------------------------------

class TestResamplesFrom15Min:
    """
    Feed 15-min bars covering the same high-vol scenario and verify the
    regime is identical to the daily-input version.
    """

    def _make_15min(self, atr_per_day: float, n_days: int, today_atr: float
                    ) -> tuple[pd.DataFrame, pd.Timestamp]:
        """
        Create 15-min bars where each day has the specified daily ATR.
        The daily high-low spread is achieved by setting the first bar's
        high and the last bar's low of each day accordingly.
        """
        freq = pd.tseries.offsets.BusinessDay()
        start = pd.Timestamp("2022-01-03")
        bars  = []
        day_starts = pd.date_range(start, periods=n_days + 1, freq=freq)

        for d, day_start in enumerate(day_starts):
            atr = today_atr if d == len(day_starts) - 1 else atr_per_day
            # 8 bars per day (9:30–11:45 ET) — any count works for resampling
            for j in range(8):
                ts   = day_start + pd.Timedelta(minutes=15 * j)
                high = 1.10000 + atr / 2 if j == 0 else 1.10000 + 0.0001
                low  = 1.10000 - atr / 2 if j == 7 else 1.10000 - 0.0001
                bars.append({
                    "open": 1.10000, "high": high, "low": low,
                    "close": 1.10000, "volume": 100.0,
                    "ts": ts,
                })

        df = pd.DataFrame(
            [{k: v for k, v in b.items() if k != "ts"} for b in bars],
            index=pd.DatetimeIndex([b["ts"] for b in bars]),
        )
        return df, day_starts[-1]

    def setup_method(self):
        df_15, today = self._make_15min(
            atr_per_day=0.0010, n_days=80, today_atr=0.0030
        )
        self.regime = _rf().get_regime(df_15, today.date())

    def test_high_vol_from_15min(self):
        assert self.regime == REGIME_HIGH_VOL


# ---------------------------------------------------------------------------
# 12. Works with pre-resampled daily input
# ---------------------------------------------------------------------------

class TestPreResampled:
    def setup_method(self):
        df, today = _daily_df_with_controlled_atr(
            lookback_atr=0.0010, today_atr=0.0030, n_history=80
        )
        # Confirm it's already daily (1-day gaps)
        assert (df.index[1] - df.index[0]).days >= 1
        self.regime = _rf().get_regime(df, today.date())

    def test_recognized_as_daily(self):
        assert self.regime == REGIME_HIGH_VOL


# ---------------------------------------------------------------------------
# 13. Strategy name constants match strategy .name properties
# ---------------------------------------------------------------------------

class TestStrategyNameConstants:
    """
    Guard against strategy name drift — if someone renames the class the
    constant and the strategy.name must stay in sync.
    """

    def test_london_name_constant(self):
        from strategies.london_open_breakout import LondonOpenBreakout
        assert LondonOpenBreakout().name == LONDON_BREAKOUT

    def test_fvg_name_constant(self):
        from strategies.fvg_retracement import FVGRetracement
        assert FVGRetracement().name == FVG_RETRACEMENT


# ---------------------------------------------------------------------------
# 14. ATR%ile == 70 boundary is NORMAL (not HIGH_VOL)
# ---------------------------------------------------------------------------

class TestHighVolBoundary:
    """
    Verify that the _classify() boundary uses strict > (not >=).
    Test _classify directly so we don't fight EMA smoothing when constructing
    synthetic data that tries to hit exactly 70%.
    """

    def setup_method(self):
        self.rf = _rf()

    def test_exactly_70_is_normal(self):
        # atr_pct = 70.0 exactly → NOT > 70 → NORMAL
        regime = self.rf._classify(atr_pct=70.0, adx_val=30.0)
        assert regime == REGIME_NORMAL

    def test_70_point_1_is_high_vol(self):
        regime = self.rf._classify(atr_pct=70.1, adx_val=30.0)
        assert regime == REGIME_HIGH_VOL

    def test_69_9_is_normal(self):
        regime = self.rf._classify(atr_pct=69.9, adx_val=30.0)
        assert regime == REGIME_NORMAL


# ---------------------------------------------------------------------------
# 15. ATR%ile == 30 boundary is NORMAL (not LOW_VOL)
# ---------------------------------------------------------------------------

class TestLowVolBoundary:
    """
    Verify that _classify() uses strict < for LOW_VOL (not <=).
    Test _classify directly — same reason as test 14.
    """

    def setup_method(self):
        self.rf = _rf()

    def test_exactly_30_is_normal(self):
        # atr_pct = 30.0 exactly → NOT < 30 → NORMAL
        regime = self.rf._classify(atr_pct=30.0, adx_val=30.0)
        assert regime == REGIME_NORMAL

    def test_29_9_is_low_vol(self):
        regime = self.rf._classify(atr_pct=29.9, adx_val=30.0)
        assert regime == REGIME_LOW_VOL

    def test_30_1_is_normal(self):
        regime = self.rf._classify(atr_pct=30.1, adx_val=30.0)
        assert regime == REGIME_NORMAL
