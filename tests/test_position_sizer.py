"""
Tests for execution/position_sizer.py

Test plan (28 tests across 6 classes)
--------------------------------------
Formula correctness
  1.  Standard EURUSD calculation: balance=10000, risk=1%, sl=20pips → 0.50
  2.  Fractional result rounds DOWN to nearest 0.01
  3.  Result with exact boundary (no fractional pips) is correct
  4.  Custom pip_value_per_lot in instrument_config used
  5.  Small balance produces small lots (not rounded up)

Rounding guarantee (never exceeds risk_pct)
  6.  Verify actual dollar risk never exceeds (risk_pct / 100) × balance
  7.  Verify lot_size × sl_pips × pip_value ≤ balance × risk_pct / 100
  8.  Parameterised over a range of (balance, risk_pct, sl_pips) combos

Minimum lot (0.01) and zero-lot edge cases
  9.  Returns 0.0 when calculated lots < 0.01
  10. Returns 0.01 when calculated lots fall between 0.01 and 0.019...
  11. Very small balance → 0.0 (skip trade)

Input validation
  12. balance <= 0 raises ValueError
  13. risk_pct <= 0 raises ValueError
  14. sl_distance_pips <= 0 raises ValueError
  15. risk_pct = 0 raises ValueError

Known values (manual calculation cross-check)
  16. 10000 balance, 0.75% risk, 15 pip SL → 0.50 lots
  17. 25000 balance, 1.0% risk, 50 pip SL → 0.50 lots
  18. 100000 balance, 0.5% risk, 20 pips SL → 2.50 lots
  19. 10000 balance, 0.5% risk, 100 pip SL → 0.05 lots
  20. 5000 balance, 0.5% risk, 40 pips SL → 0.06 lots

Max lots cap (sanity ceiling)
  21. Pathologically large balance / tiny SL is capped at 100 lots
"""

import math
import pytest

from execution.position_sizer import calculate_lot_size, _DEFAULT_PIP_VALUE, _MAX_LOTS


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _cfg(pip_size: float = 0.0001, pip_value: float = 10.0) -> dict:
    return {"pip_size": pip_size, "pip_value_per_lot": pip_value}


def _eurusd() -> dict:
    return _cfg(pip_size=0.0001, pip_value=10.0)


# ---------------------------------------------------------------------------
# 1–5  Formula correctness
# ---------------------------------------------------------------------------

class TestFormulaCorrectness:
    def test_standard_calculation(self):
        """10000 × 1% / (20 × 10) = 0.50 lots exactly."""
        assert calculate_lot_size(10_000, 1.0, 20, _eurusd()) == 0.50

    def test_rounds_down_not_up(self):
        """
        10000 × 1% / (7 × 10) = 100/70 = 1.4285…
        floor to 0.01 → 1.42, NOT 1.43
        """
        result = calculate_lot_size(10_000, 1.0, 7, _eurusd())
        raw    = (10_000 * 0.01) / (7 * 10)   # 1.4285...
        assert result == math.floor(raw * 100) / 100

    def test_exact_integer_pips(self):
        """10000 × 0.5% / (25 × 10) = 50/250 = 0.20 lots."""
        assert calculate_lot_size(10_000, 0.5, 25, _eurusd()) == 0.20

    def test_custom_pip_value_used(self):
        """
        With pip_value=5 (non-standard), the lot size should double
        compared to pip_value=10.
        """
        lot_10  = calculate_lot_size(10_000, 1.0, 20, _cfg(pip_value=10.0))
        lot_5   = calculate_lot_size(10_000, 1.0, 20, _cfg(pip_value=5.0))
        assert lot_5 > lot_10   # smaller pip value → more lots for same risk

    def test_small_balance_small_lots(self):
        """1000 × 1% / (20 × 10) = 0.05 lots."""
        assert calculate_lot_size(1_000, 1.0, 20, _eurusd()) == 0.05


# ---------------------------------------------------------------------------
# 6–8  Rounding guarantee — never exceed risk_pct
# ---------------------------------------------------------------------------

class TestRoundingGuarantee:
    def _actual_risk(self, balance, risk_pct, sl_pips, cfg) -> float:
        lot = calculate_lot_size(balance, risk_pct, sl_pips, cfg)
        pv  = cfg.get("pip_value_per_lot", 10.0)
        return lot * sl_pips * pv

    def test_actual_risk_never_exceeds_budget(self):
        budget = 10_000 * 0.01   # 1% of 10,000
        actual = self._actual_risk(10_000, 1.0, 7, _eurusd())
        assert actual <= budget + 1e-9

    def test_lot_times_sl_times_pv_leq_risk_dollars(self):
        balance, risk_pct, sl_pips = 25_000, 0.75, 33
        lot    = calculate_lot_size(balance, risk_pct, sl_pips, _eurusd())
        budget = balance * risk_pct / 100.0
        assert lot * sl_pips * 10.0 <= budget + 1e-9

    @pytest.mark.parametrize("balance,risk_pct,sl_pips", [
        (10_000, 1.00,  20),
        (10_000, 0.75,  15),
        (25_000, 0.50, 100),
        (50_000, 1.00,  50),
        ( 5_000, 0.50,  40),
        (10_000, 1.00,   7),
        (10_000, 1.00,  13),
        (10_000, 0.25,   5),
    ])
    def test_never_exceeds_risk_parametrised(self, balance, risk_pct, sl_pips):
        lot    = calculate_lot_size(balance, risk_pct, sl_pips, _eurusd())
        budget = balance * risk_pct / 100.0
        actual = lot * sl_pips * _DEFAULT_PIP_VALUE
        assert actual <= budget + 1e-9, (
            f"balance={balance} risk={risk_pct}% sl={sl_pips}pips "
            f"lot={lot} actual_risk={actual:.4f} budget={budget:.4f}"
        )


# ---------------------------------------------------------------------------
# 9–11  Minimum lot / zero-lot edge cases
# ---------------------------------------------------------------------------

class TestMinimumLot:
    def test_returns_zero_when_below_micro_lot(self):
        """$100 balance, 0.1% risk, 100 pip SL: 100*0.001/(100*10)=0.0001 → 0.0."""
        result = calculate_lot_size(100, 0.1, 100, _eurusd())
        assert result == 0.0

    def test_returns_001_at_exact_boundary(self):
        """
        Need lot exactly 0.01:
          balance × risk / (sl × pv) = 0.01
          balance × risk = 0.01 × sl × pv = 0.01 × 10 × 10 = 1
          balance=1000, risk=0.1% → 1000*0.001=1 → sl=10pips
        """
        result = calculate_lot_size(1_000, 0.1, 10, _eurusd())
        assert result == 0.01

    def test_very_small_balance(self):
        """$10 balance cannot fund 0.01 lot at any reasonable risk."""
        result = calculate_lot_size(10, 1.0, 20, _eurusd())
        assert result == 0.0


# ---------------------------------------------------------------------------
# 12–15  Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_zero_balance_raises(self):
        with pytest.raises(ValueError, match="balance"):
            calculate_lot_size(0, 1.0, 20, _eurusd())

    def test_negative_balance_raises(self):
        with pytest.raises(ValueError, match="balance"):
            calculate_lot_size(-100, 1.0, 20, _eurusd())

    def test_zero_risk_pct_raises(self):
        with pytest.raises(ValueError, match="risk_pct"):
            calculate_lot_size(10_000, 0.0, 20, _eurusd())

    def test_negative_risk_pct_raises(self):
        with pytest.raises(ValueError, match="risk_pct"):
            calculate_lot_size(10_000, -1.0, 20, _eurusd())

    def test_zero_sl_raises(self):
        with pytest.raises(ValueError, match="sl_distance_pips"):
            calculate_lot_size(10_000, 1.0, 0, _eurusd())

    def test_negative_sl_raises(self):
        with pytest.raises(ValueError, match="sl_distance_pips"):
            calculate_lot_size(10_000, 1.0, -5, _eurusd())


# ---------------------------------------------------------------------------
# 16–20  Known values (manually verified)
# ---------------------------------------------------------------------------

class TestKnownValues:
    """Cross-check against hand-calculated results."""

    def test_10k_075pct_15pips(self):
        """10000 × 0.75% / (15 × 10) = 75/150 = 0.50 lots."""
        assert calculate_lot_size(10_000, 0.75, 15, _eurusd()) == 0.50

    def test_25k_1pct_50pips(self):
        """25000 × 1% / (50 × 10) = 250/500 = 0.50 lots."""
        assert calculate_lot_size(25_000, 1.0, 50, _eurusd()) == 0.50

    def test_100k_05pct_20pips(self):
        """100000 × 0.5% / (20 × 10) = 500/200 = 2.50 lots."""
        assert calculate_lot_size(100_000, 0.5, 20, _eurusd()) == 2.50

    def test_10k_05pct_100pips(self):
        """10000 × 0.5% / (100 × 10) = 50/1000 = 0.05 lots."""
        assert calculate_lot_size(10_000, 0.5, 100, _eurusd()) == 0.05

    def test_5k_05pct_40pips(self):
        """5000 × 0.5% / (40 × 10) = 25/400 = 0.0625 → floor = 0.06."""
        assert calculate_lot_size(5_000, 0.5, 40, _eurusd()) == 0.06


# ---------------------------------------------------------------------------
# 21  Max lots cap
# ---------------------------------------------------------------------------

class TestMaxLots:
    def test_capped_at_100_lots(self):
        """100,000,000 balance, 1% risk, 1 pip SL → raw=100,000 → capped at 100."""
        result = calculate_lot_size(100_000_000, 1.0, 1, _eurusd())
        assert result == _MAX_LOTS

    def test_just_below_cap_not_capped(self):
        """Result < 100 lots is returned as-is."""
        result = calculate_lot_size(10_000, 1.0, 20, _eurusd())
        assert result < _MAX_LOTS
