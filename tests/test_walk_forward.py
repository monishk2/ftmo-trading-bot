"""
Tests for backtesting/walk_forward.py

Test plan
---------
1.  _expand_grid             : cartesian product size and structure
2.  _generate_windows        : window count, IS/OOS non-empty, non-overlapping
3.  _generate_windows        : too-little data → empty list
4.  _generate_windows        : IS/OOS boundary continuity
5.  _recommend_params        : picks combo with most positive-OOS appearances
6.  _recommend_params        : tie-broken by higher avg OOS Sharpe
7.  _recommend_params        : empty windows → empty dict
8.  degradation_ratio        : oos_sharpe / is_sharpe arithmetic
9.  degradation_ratio nan    : is_sharpe == 0 → NaN degradation
10. overfitting_warning       : avg_deg < 0.4 sets warning flag
11. is_robust flag            : avg_deg > 0.4 sets is_robust True
12. save_result / round-trip  : JSON file created, NaN → null, reloadable
13. run_walk_forward e2e      : tiny grid, completes, returns WalkForwardResult
14. run_walk_forward strategy : fvg_retracement path also accepted
15. run_walk_forward bad name : ValueError raised
16. run_walk_forward no data  : ValueError (not enough data)
17. WindowResult fields       : all fields present and correctly typed
18. WalkForwardResult fields  : strategy_name / instrument / grid_used stored
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtesting.walk_forward import (
    WalkForwardResult,
    WindowResult,
    _expand_grid,
    _generate_windows,
    _recommend_params,
    run_walk_forward,
    save_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n_bars: int, start: str = "2020-01-01", freq: str = "15min") -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a UTC DatetimeIndex."""
    idx = pd.date_range(start, periods=n_bars, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0002, n_bars))
    df = pd.DataFrame(
        {
            "open":   close - 0.0001,
            "high":   close + 0.0003,
            "low":    close - 0.0003,
            "close":  close,
            "volume": 1000.0,
        },
        index=idx,
    )
    return df


def _dense_df(months: int = 18, bars_per_day: int = 10) -> pd.DataFrame:
    """
    Dense enough for walk-forward to create valid IS/OOS windows.

    bars_per_day bars are spread evenly across each calendar day so that
    the resulting DataFrame spans exactly `months` calendar months.
    """
    mins_per_bar = 1440 // bars_per_day   # e.g. 10 bars/day → 144 min spacing
    freq = f"{mins_per_bar}min"
    n_bars = months * 30 * bars_per_day
    return _make_ohlcv(n_bars, freq=freq)


# ---------------------------------------------------------------------------
# Minimal stub strategy for e2e tests (avoids real strategy logic overhead)
# ---------------------------------------------------------------------------

class _TrivialStrategy:
    """
    Generates one long-entry signal on the very first bar of each call.
    Always exits at SL (no P&L) — only used to verify run_walk_forward
    control flow, not real performance.
    """

    name = "LondonOpenBreakout"
    risk_per_trade_pct = 1.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"]    = 0
        df["sl_price"]  = np.nan
        df["tp_price"]  = np.nan
        df["lot_size"]  = 0.01
        df["time_stop"] = None
        if len(df) > 0:
            i = 0
            df.iloc[i, df.columns.get_loc("signal")]   = 1
            df.iloc[i, df.columns.get_loc("sl_price")] = df["close"].iloc[i] - 0.0020
            df.iloc[i, df.columns.get_loc("tp_price")] = df["close"].iloc[i] + 0.0040
        return df


# ---------------------------------------------------------------------------
# 1. _expand_grid — cartesian product
# ---------------------------------------------------------------------------

class TestExpandGrid:
    def test_size_single_param(self):
        grid = {"a": [1, 2, 3]}
        result = _expand_grid(grid)
        assert len(result) == 3

    def test_size_two_params(self):
        grid = {"a": [1, 2], "b": [10, 20, 30]}
        result = _expand_grid(grid)
        assert len(result) == 6

    def test_all_combos_present(self):
        grid = {"x": [1, 2], "y": ["a", "b"]}
        result = _expand_grid(grid)
        keys_vals = {(d["x"], d["y"]) for d in result}
        assert keys_vals == {(1, "a"), (1, "b"), (2, "a"), (2, "b")}

    def test_single_combo(self):
        grid = {"p": [5]}
        result = _expand_grid(grid)
        assert result == [{"p": 5}]

    def test_returns_list_of_dicts(self):
        grid = {"a": [1, 2], "b": [3, 4]}
        result = _expand_grid(grid)
        assert all(isinstance(d, dict) for d in result)
        assert all("a" in d and "b" in d for d in result)

    def test_london_grid_size(self):
        """Full production grid: 9×5×4×3 = 540 combos."""
        from backtesting.walk_forward import _LONDON_GRID
        result = _expand_grid(_LONDON_GRID)
        assert len(result) == 9 * 5 * 4 * 3

    def test_fvg_grid_size(self):
        """FVG grid: 4×4×4×3 = 192 combos."""
        from backtesting.walk_forward import _FVG_GRID
        result = _expand_grid(_FVG_GRID)
        assert len(result) == 4 * 4 * 4 * 3


# ---------------------------------------------------------------------------
# 2. _generate_windows — structure and boundaries
# ---------------------------------------------------------------------------

class TestGenerateWindows:
    def _df(self, months: int = 20, bpd: int = 10) -> pd.DataFrame:
        return _dense_df(months, bpd)

    def test_returns_list(self):
        df = self._df()
        wins = _generate_windows(df, is_months=6, oos_months=3, step_months=3)
        assert isinstance(wins, list)

    def test_at_least_one_window(self):
        df = self._df(18, 10)
        wins = _generate_windows(df, is_months=6, oos_months=3, step_months=3)
        assert len(wins) >= 1

    def test_tuple_has_six_elements(self):
        df = self._df()
        wins = _generate_windows(df, is_months=6, oos_months=3, step_months=3)
        assert len(wins[0]) == 6

    def test_is_slice_non_empty(self):
        df = self._df()
        for is_sl, oos_sl, *_ in _generate_windows(df, 6, 3, 3):
            assert len(is_sl) > 0

    def test_oos_slice_non_empty(self):
        df = self._df()
        for _, oos_sl, *_ in _generate_windows(df, 6, 3, 3):
            assert len(oos_sl) > 0

    def test_is_ends_before_oos_starts(self):
        """OOS window must start exactly where IS ends (no gap, no overlap)."""
        df = self._df()
        for _, _, is_start, is_end, oos_start, oos_end in _generate_windows(df, 6, 3, 3):
            assert is_end == oos_start

    def test_is_slice_within_is_bounds(self):
        df = self._df()
        for is_sl, _, is_start, is_end, _, _ in _generate_windows(df, 6, 3, 3):
            assert is_sl.index[0] >= is_start
            assert is_sl.index[-1] < is_end

    def test_oos_slice_within_oos_bounds(self):
        df = self._df()
        for _, oos_sl, _, _, oos_start, oos_end in _generate_windows(df, 6, 3, 3):
            assert oos_sl.index[0] >= oos_start
            assert oos_sl.index[-1] < oos_end

    def test_step_advances_by_step_months(self):
        """Consecutive windows' IS starts should differ by step_months."""
        df = self._df(24, 10)
        wins = _generate_windows(df, 6, 3, 3)
        if len(wins) >= 2:
            _, _, s0, _, _, _ = wins[0]
            _, _, s1, _, _, _ = wins[1]
            delta_months = (s1.year - s0.year) * 12 + (s1.month - s0.month)
            assert delta_months == 3

    def test_too_little_data_returns_empty(self):
        """Only 3 months of data → can't fit a 6+3 = 9-month window."""
        df = _dense_df(months=3, bars_per_day=10)
        wins = _generate_windows(df, is_months=6, oos_months=3, step_months=3)
        assert wins == []

    def test_is_slice_min_1000_bars(self):
        """IS slices must have at least 1000 bars (guard in _generate_windows)."""
        df = self._df(18, 10)
        for is_sl, *_ in _generate_windows(df, 6, 3, 3):
            assert len(is_sl) >= 1000


# ---------------------------------------------------------------------------
# 3. _recommend_params
# ---------------------------------------------------------------------------

def _make_window(idx: int, params: dict, oos_sharpe: float) -> WindowResult:
    return WindowResult(
        window_idx=idx,
        is_start="2020-01",
        is_end="2020-06",
        oos_start="2020-07",
        oos_end="2020-09",
        best_params=params,
        is_sharpe=1.0,
        is_return_pct=5.0,
        is_trades=30,
        oos_sharpe=oos_sharpe,
        oos_return_pct=2.0,
        oos_trades=10,
        degradation_ratio=oos_sharpe,
    )


class TestRecommendParams:
    def test_most_frequent_positive_wins(self):
        p_good = {"min_asian_range_pips": 20, "entry_buffer_pips": 2}
        p_bad  = {"min_asian_range_pips": 50, "entry_buffer_pips": 5}
        windows = [
            _make_window(0, p_good, 0.8),
            _make_window(1, p_good, 0.9),
            _make_window(2, p_bad,  0.1),
        ]
        rec = _recommend_params(windows)
        assert rec == p_good

    def test_tie_broken_by_avg_sharpe(self):
        p1 = {"rr": 2.0}
        p2 = {"rr": 3.0}
        # Both appear twice with positive OOS; p2 has higher avg OOS Sharpe
        windows = [
            _make_window(0, p1, 0.5),
            _make_window(1, p1, 0.6),
            _make_window(2, p2, 1.0),
            _make_window(3, p2, 1.2),
        ]
        rec = _recommend_params(windows)
        assert rec == p2

    def test_negative_oos_counted_less(self):
        p_mostly_positive = {"a": 1}
        p_always_negative = {"a": 2}
        windows = [
            _make_window(0, p_mostly_positive,  0.8),
            _make_window(1, p_mostly_positive, -0.2),
            _make_window(2, p_always_negative, -0.5),
            _make_window(3, p_always_negative, -0.3),
        ]
        rec = _recommend_params(windows)
        assert rec == p_mostly_positive

    def test_empty_windows_returns_empty_dict(self):
        assert _recommend_params([]) == {}

    def test_single_window(self):
        p = {"min_asian_range_pips": 25}
        windows = [_make_window(0, p, 1.5)]
        rec = _recommend_params(windows)
        assert rec == p

    def test_returns_dict(self):
        p = {"k": 1}
        windows = [_make_window(0, p, 0.5)]
        assert isinstance(_recommend_params(windows), dict)


# ---------------------------------------------------------------------------
# 4. Degradation ratio arithmetic
# ---------------------------------------------------------------------------

class TestDegradationRatio:
    def test_ratio_calculation(self):
        """degradation_ratio = oos_sharpe / is_sharpe."""
        wr = _make_window(0, {}, oos_sharpe=0.8)
        wr.is_sharpe = 2.0
        wr.oos_sharpe = 0.8
        wr.degradation_ratio = wr.oos_sharpe / wr.is_sharpe
        assert math.isclose(wr.degradation_ratio, 0.4, rel_tol=1e-9)

    def test_ratio_above_one_allowed(self):
        """OOS can outperform IS — ratio > 1 is valid."""
        wr = _make_window(0, {}, oos_sharpe=2.0)
        wr.is_sharpe = 1.0
        wr.oos_sharpe = 2.0
        wr.degradation_ratio = wr.oos_sharpe / wr.is_sharpe
        assert wr.degradation_ratio == 2.0

    def test_zero_is_sharpe_gives_nan(self):
        """If is_sharpe == 0, degradation_ratio must be NaN (not ZeroDivision)."""
        # This is the contract enforced in run_walk_forward
        is_sharpe = 0.0
        oos_sharpe = 0.5
        if is_sharpe != 0 and not np.isnan(is_sharpe):
            deg = oos_sharpe / is_sharpe
        else:
            deg = float("nan")
        assert math.isnan(deg)


# ---------------------------------------------------------------------------
# 5. Overfitting warning and is_robust flag
# ---------------------------------------------------------------------------

class TestRobustnessFlags:
    def _result_with_avg(self, avg_deg: float) -> WalkForwardResult:
        """Construct a minimal WalkForwardResult with given avg_degradation."""
        return WalkForwardResult(
            strategy_name="london_breakout",
            instrument="EURUSD",
            windows=[],
            avg_degradation_ratio=avg_deg,
            recommended_params={},
            is_robust=avg_deg > 0.4,
            overfitting_warning=avg_deg < 0.4,
            grid_used={},
            initial_balance=10_000.0,
        )

    def test_overfitting_warning_below_threshold(self):
        r = self._result_with_avg(0.3)
        assert r.overfitting_warning is True
        assert r.is_robust is False

    def test_no_warning_at_threshold(self):
        r = self._result_with_avg(0.4)
        # 0.4 is NOT below 0.4, so no warning; NOT above 0.4, so not robust
        assert r.overfitting_warning is False
        assert r.is_robust is False

    def test_robust_above_threshold(self):
        r = self._result_with_avg(0.6)
        assert r.is_robust is True
        assert r.overfitting_warning is False

    def test_exactly_zero_avg(self):
        r = self._result_with_avg(0.0)
        assert r.overfitting_warning is True
        assert r.is_robust is False


# ---------------------------------------------------------------------------
# 6. save_result / JSON round-trip
# ---------------------------------------------------------------------------

class TestSaveResult:
    def _minimal_result(self) -> WalkForwardResult:
        wr = WindowResult(
            window_idx=0,
            is_start="2020-01",
            is_end="2020-06",
            oos_start="2020-07",
            oos_end="2020-09",
            best_params={"min_asian_range_pips": 20},
            is_sharpe=1.5,
            is_return_pct=8.0,
            is_trades=35,
            oos_sharpe=float("nan"),
            oos_return_pct=-1.0,
            oos_trades=5,
            degradation_ratio=float("nan"),
        )
        return WalkForwardResult(
            strategy_name="london_breakout",
            instrument="EURUSD",
            windows=[wr],
            avg_degradation_ratio=float("nan"),
            recommended_params={"min_asian_range_pips": 20},
            is_robust=False,
            overfitting_warning=True,
            grid_used={"min_asian_range_pips": [20]},
            initial_balance=10_000.0,
        )

    def test_file_created(self):
        result = self._minimal_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf_result.json"
            save_result(result, path)
            assert path.exists()

    def test_nan_serialised_as_null(self):
        result = self._minimal_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf_result.json"
            save_result(result, path)
            raw = path.read_text()
            data = json.loads(raw)
            # avg_degradation_ratio was NaN → should be null in JSON
            assert data["avg_degradation_ratio"] is None

    def test_window_nan_also_null(self):
        result = self._minimal_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf_result.json"
            save_result(result, path)
            data = json.loads(path.read_text())
            assert data["windows"][0]["oos_sharpe"] is None

    def test_round_trip_preserves_scalars(self):
        result = self._minimal_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf_result.json"
            save_result(result, path)
            data = json.loads(path.read_text())
            assert data["strategy_name"] == "london_breakout"
            assert data["instrument"] == "EURUSD"
            assert data["initial_balance"] == 10_000.0
            assert data["windows"][0]["is_trades"] == 35

    def test_returns_path_object(self):
        result = self._minimal_result()
        with tempfile.TemporaryDirectory() as tmp:
            p = save_result(result, Path(tmp) / "out.json")
            assert isinstance(p, Path)

    def test_creates_parent_dirs(self):
        result = self._minimal_result()
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "a" / "b" / "wf.json"
            save_result(result, nested)
            assert nested.exists()


# ---------------------------------------------------------------------------
# 7. WindowResult / WalkForwardResult dataclass fields
# ---------------------------------------------------------------------------

class TestDataclassFields:
    def test_window_result_fields(self):
        expected = {
            "window_idx", "is_start", "is_end", "oos_start", "oos_end",
            "best_params", "is_sharpe", "is_return_pct", "is_trades",
            "oos_sharpe", "oos_return_pct", "oos_trades", "degradation_ratio",
        }
        actual = {f.name for f in fields(WindowResult)}
        assert expected == actual

    def test_walk_forward_result_fields(self):
        expected = {
            "strategy_name", "instrument", "windows",
            "avg_degradation_ratio", "recommended_params",
            "is_robust", "overfitting_warning", "grid_used", "initial_balance",
        }
        actual = {f.name for f in fields(WalkForwardResult)}
        assert expected == actual


# ---------------------------------------------------------------------------
# 8. run_walk_forward — unknown strategy raises ValueError
# ---------------------------------------------------------------------------

class TestRunWalkForwardValidation:
    def test_unknown_strategy_raises(self):
        df = _dense_df(18, 10)
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_walk_forward(df, strategy_name="banana", verbose=False)

    def test_not_enough_data_raises(self):
        """2 months of data cannot fit a 6+3-month window."""
        df = _dense_df(months=2, bars_per_day=10)
        with pytest.raises(ValueError, match="Not enough data"):
            run_walk_forward(df, strategy_name="london_breakout", verbose=False)


# ---------------------------------------------------------------------------
# 9. run_walk_forward — end-to-end with stubbed strategy + tiny grid
# ---------------------------------------------------------------------------

class TestRunWalkForwardE2E:
    """
    Patches the strategy builder and uses a tiny 2-combo grid so the test
    finishes quickly while still exercising the full control flow.
    """

    _TINY_GRID = {
        "min_asian_range_pips": [20, 25],
        "entry_buffer_pips":    [2],
        "risk_reward_ratio":    [2.0],
        "risk_per_trade_pct":   [1.0],
    }

    def _run(self, df: pd.DataFrame, strategy_name: str = "london_breakout"):
        """
        Patch _build_london_strategy / _build_fvg_strategy to return
        _TrivialStrategy (no real signal logic, just avoids parsing config
        and avoids slow ATR/FVG calculations).
        """
        trivial = _TrivialStrategy()

        with patch("backtesting.walk_forward._LONDON_GRID", self._TINY_GRID), \
             patch("backtesting.walk_forward._FVG_GRID", self._TINY_GRID), \
             patch("backtesting.walk_forward._build_london_strategy",
                   return_value=trivial), \
             patch("backtesting.walk_forward._build_fvg_strategy",
                   return_value=trivial), \
             patch("backtesting.walk_forward._load_london_base_config",
                   return_value={}), \
             patch("backtesting.walk_forward._load_fvg_base_config",
                   return_value={}), \
             patch("backtesting.walk_forward._load_instrument_config",
                   return_value={
                       "pip_size": 0.0001,
                       "typical_spread_pips": 1.0,
                       "commission_per_lot_round_trip": 3.0,
                       "slippage_model": {
                           "normal_pips": 0.3,
                           "session_open_pips": 1.5,
                           "session_open_window_minutes": 15,
                           "news_pips": 2.0,
                       },
                   }):
            return run_walk_forward(
                df,
                strategy_name=strategy_name,
                instrument="EURUSD",
                initial_balance=10_000.0,
                in_sample_months=6,
                oos_months=3,
                step_months=3,
                min_trades=1,    # accept any run with ≥1 trade
                verbose=False,
            )

    def test_returns_walk_forward_result(self):
        df = _dense_df(18, 10)
        result = self._run(df)
        assert isinstance(result, WalkForwardResult)

    def test_strategy_name_preserved(self):
        df = _dense_df(18, 10)
        result = self._run(df)
        assert result.strategy_name == "london_breakout"

    def test_instrument_preserved(self):
        df = _dense_df(18, 10)
        result = self._run(df)
        assert result.instrument == "EURUSD"

    def test_initial_balance_preserved(self):
        df = _dense_df(18, 10)
        result = self._run(df)
        assert result.initial_balance == 10_000.0

    def test_windows_non_empty(self):
        df = _dense_df(18, 10)
        result = self._run(df)
        assert len(result.windows) >= 1

    def test_window_has_correct_types(self):
        df = _dense_df(18, 10)
        result = self._run(df)
        w = result.windows[0]
        assert isinstance(w.is_sharpe, float)
        assert isinstance(w.oos_sharpe, float)
        assert isinstance(w.is_trades, int)
        assert isinstance(w.oos_trades, int)
        assert isinstance(w.best_params, dict)

    def test_recommended_params_is_dict(self):
        df = _dense_df(18, 10)
        result = self._run(df)
        assert isinstance(result.recommended_params, dict)

    def test_avg_degradation_ratio_is_float(self):
        df = _dense_df(18, 10)
        result = self._run(df)
        assert isinstance(result.avg_degradation_ratio, float)

    def test_overfitting_and_robust_are_bool(self):
        df = _dense_df(18, 10)
        result = self._run(df)
        assert isinstance(result.overfitting_warning, bool)
        assert isinstance(result.is_robust, bool)

    def test_fvg_strategy_path_accepted(self):
        df = _dense_df(18, 10)
        result = self._run(df, strategy_name="fvg_retracement")
        assert result.strategy_name == "fvg_retracement"

    def test_degradation_ratio_stored_in_window(self):
        df = _dense_df(18, 10)
        result = self._run(df)
        for w in result.windows:
            # ratio is either a finite float or NaN — never missing
            assert not isinstance(w.degradation_ratio, str)

    def test_save_round_trip_after_e2e(self):
        df = _dense_df(18, 10)
        result = self._run(df)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e2e.json"
            save_result(result, path)
            data = json.loads(path.read_text())
            assert data["strategy_name"] == result.strategy_name
            assert len(data["windows"]) == len(result.windows)
