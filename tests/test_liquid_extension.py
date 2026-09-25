"""
test_liquid_extension.py
========================
Unit tests for the liquid-universe extension module.

Covers the LiquidStrategyScheduler constraints, support gate logic,
and matched-control inference documentation.  All tests use synthetic
data; no network calls or disk I/O are required.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Ensure src/ is on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.liquid_universe_extension import (
    LiquidStrategyScheduler,
    apply_support_gate,
    build_matched_controls,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_price_panel(
    tickers: list[str],
    start: str = "2024-01-02",
    n_days: int = 60,
    price: float = 50.0,
) -> pd.DataFrame:
    """Build a wide price panel: index=date, columns=tickers, values=price."""
    dates = pd.date_range(start, periods=n_days, freq="B")
    data = {tkr: np.full(n_days, price) for tkr in tickers}
    return pd.DataFrame(data, index=dates)


def _make_signals(
    tickers: list[str],
    entry_dates: list[str],
    position_size_usd: float = 100_000,
) -> pd.DataFrame:
    """Build a minimal signals DataFrame."""
    return pd.DataFrame(
        {
            "event_id": [f"E{i}" for i in range(len(tickers))],
            "ticker": tickers,
            "entry_date": pd.to_datetime(entry_dates),
            "position_size_usd": [position_size_usd] * len(tickers),
        }
    )


def _make_universe(
    n: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic universe for matched-control tests."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "event_id": [f"U{i}" for i in range(n)],
            "log_market_cap": rng.normal(20.5, 1.5, n),
            "log_addv": rng.normal(14.0, 1.2, n),
            "return_21d": rng.normal(0.0, 0.05, n),
        }
    )


# ---------------------------------------------------------------------------
# Tests: LiquidStrategyScheduler – budget constraint
# ---------------------------------------------------------------------------

class TestSchedulerBudgetConstraint:
    """Total gross exposure must never exceed the configured budget."""

    def test_total_exposure_within_budget(self):
        """With many signals, deployed capital must stay ≤ $5 M budget."""
        budget = 5_000_000
        scheduler = LiquidStrategyScheduler(
            budget=budget,
            max_per_position=250_000,
            holding_sessions=20,
        )
        tickers = [f"T{i:03d}" for i in range(50)]
        prices = _make_price_panel(tickers, n_days=80, price=25.0)
        entry_dates = [prices.index[0].strftime("%Y-%m-%d")] * len(tickers)
        signals = _make_signals(tickers, entry_dates, position_size_usd=200_000)

        inventory = scheduler.process_signals(signals, prices)

        assert (inventory["gross_exposure"] <= budget + 1e-2).all(), (
            f"Gross exposure exceeded budget: max={inventory['gross_exposure'].max():,.0f}"
        )

    def test_cash_deployed_within_budget(self):
        """Cash deployed (cost basis) must never exceed budget."""
        budget = 1_000_000
        scheduler = LiquidStrategyScheduler(
            budget=budget,
            max_per_position=500_000,
            holding_sessions=10,
        )
        tickers = ["A", "B", "C", "D"]
        prices = _make_price_panel(tickers, n_days=30, price=100.0)
        entry_dates = [prices.index[0].strftime("%Y-%m-%d")] * 4
        signals = _make_signals(tickers, entry_dates, position_size_usd=400_000)

        inventory = scheduler.process_signals(signals, prices)
        assert (inventory["cash_deployed"] <= budget + 1e-2).all()


# ---------------------------------------------------------------------------
# Tests: LiquidStrategyScheduler – no overlapping positions
# ---------------------------------------------------------------------------

class TestSchedulerNoOverlap:
    """No two positions in the same issuer should be open simultaneously."""

    def test_no_overlap_same_ticker_sequential(self):
        """Sequential signals in the same ticker should not overlap."""
        scheduler = LiquidStrategyScheduler(
            budget=5_000_000,
            max_per_position=250_000,
            holding_sessions=10,
        )
        prices = _make_price_panel(["AAPL"], n_days=60, price=100.0)
        # Two signals for AAPL on the same day – second should be skipped
        signals = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "ticker": ["AAPL", "AAPL"],
                "entry_date": pd.to_datetime([prices.index[0], prices.index[0]]),
                "position_size_usd": [100_000, 100_000],
            }
        )
        inventory = scheduler.process_signals(signals, prices)
        # Maximum simultaneous positions in AAPL should be 1
        assert (inventory["n_positions"] <= 1).all(), (
            "Scheduler opened overlapping positions in the same issuer"
        )

    def test_second_signal_enters_after_exit(self):
        """A second signal for the same ticker enters only after the first is closed."""
        scheduler = LiquidStrategyScheduler(
            budget=5_000_000,
            max_per_position=250_000,
            holding_sessions=5,
        )
        prices = _make_price_panel(["AAPL"], n_days=30, price=50.0)
        # First entry on day 0, second entry on day 3 (before first exit at day 5)
        signals = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "ticker": ["AAPL", "AAPL"],
                "entry_date": pd.to_datetime([prices.index[0], prices.index[3]]),
                "position_size_usd": [100_000, 100_000],
            }
        )
        inventory = scheduler.process_signals(signals, prices)
        # Never more than 1 AAPL position at the same time
        assert (inventory["n_positions"] <= 1).all()


# ---------------------------------------------------------------------------
# Tests: LiquidStrategyScheduler – max_per_position
# ---------------------------------------------------------------------------

class TestSchedulerMaxPerPosition:
    """Each position must be capped at max_per_position."""

    def test_position_size_capped(self):
        """Even if signal desires a large notional, cost basis must not exceed cap."""
        cap = 250_000
        price = 10.0
        scheduler = LiquidStrategyScheduler(
            budget=5_000_000,
            max_per_position=cap,
            holding_sessions=20,
        )
        prices = _make_price_panel(["XYZ"], n_days=40, price=price)
        # Request 3× the cap
        signals = _make_signals(["XYZ"], [prices.index[0].strftime("%Y-%m-%d")],
                                position_size_usd=cap * 3)
        scheduler.process_signals(signals, prices)
        stats = scheduler.get_summary_stats()
        assert stats["peak_capital_use"] <= cap + price, (
            f"Peak capital use {stats['peak_capital_use']:,.0f} exceeds cap {cap:,}"
        )

    def test_whole_tickets_only(self):
        """Shares purchased must be an integer (no fractional shares)."""
        scheduler = LiquidStrategyScheduler(
            budget=5_000_000,
            max_per_position=50_000,
            holding_sessions=10,
        )
        price = 7.0  # chosen to create non-integer share count from round dollar amount
        prices = _make_price_panel(["FRAC"], n_days=30, price=price)
        signals = _make_signals(["FRAC"], [prices.index[0].strftime("%Y-%m-%d")],
                                position_size_usd=50_001)
        scheduler.process_signals(signals, prices)
        # If deployed, cash used = floor(50_001 / 7) * 7 = 7142 * 7 = 49_994
        # Peak should be a whole multiple of price
        peak = scheduler.get_summary_stats()["peak_capital_use"]
        shares_implicit = peak / price
        assert abs(shares_implicit - round(shares_implicit)) < 1e-6


# ---------------------------------------------------------------------------
# Tests: apply_support_gate
# ---------------------------------------------------------------------------

class TestSupportGateThreshold:
    """Gate passes at ≥80% coverage and fails below."""

    def _make_strategy(self, n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {"event_id": [f"S{i}" for i in range(n)]}
        )

    def _make_controls(self, covered_ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"event_id": covered_ids, "control_event_id": covered_ids})

    def test_full_coverage_passes(self):
        """100% coverage should pass the gate."""
        events = self._make_strategy(10)
        controls = self._make_controls([f"S{i}" for i in range(10)])
        passed, frac = apply_support_gate(events, controls, threshold=0.80)
        assert passed is True
        assert abs(frac - 1.0) < 1e-9

    def test_exactly_80_pct_passes(self):
        """Exactly 80% coverage should pass (≥ threshold)."""
        events = self._make_strategy(10)
        controls = self._make_controls([f"S{i}" for i in range(8)])  # 8 / 10 = 80%
        passed, frac = apply_support_gate(events, controls, threshold=0.80)
        assert passed is True
        assert abs(frac - 0.80) < 1e-9

    def test_below_80_pct_fails(self):
        """79% coverage should fail the gate."""
        events = self._make_strategy(100)
        controls = self._make_controls([f"S{i}" for i in range(79)])  # 79/100 = 79%
        passed, frac = apply_support_gate(events, controls, threshold=0.80)
        assert passed is False
        assert frac < 0.80

    def test_zero_coverage_fails(self):
        """No matches at all should fail with coverage=0."""
        events = self._make_strategy(10)
        controls = pd.DataFrame({"event_id": [], "control_event_id": []})
        passed, frac = apply_support_gate(events, controls, threshold=0.80)
        assert passed is False
        assert frac == 0.0

    def test_coverage_fraction_is_float(self):
        """coverage_fraction must be a float in [0, 1]."""
        events = self._make_strategy(20)
        controls = self._make_controls([f"S{i}" for i in range(15)])
        _, frac = apply_support_gate(events, controls)
        assert isinstance(frac, float)
        assert 0.0 <= frac <= 1.0

    def test_custom_threshold(self):
        """Gate should respect any threshold value, not just 0.80."""
        events = self._make_strategy(10)
        controls = self._make_controls([f"S{i}" for i in range(9)])  # 90%
        # At threshold=0.95 → fail
        passed_95, _ = apply_support_gate(events, controls, threshold=0.95)
        assert passed_95 is False
        # At threshold=0.80 → pass
        passed_80, _ = apply_support_gate(events, controls, threshold=0.80)
        assert passed_80 is True


# ---------------------------------------------------------------------------
# Tests: build_matched_controls – inferred assumption documentation
# ---------------------------------------------------------------------------

class TestMatchedControlsInference:
    """Verify inferred assumptions are documented and matching behaves correctly."""

    def test_each_treated_event_has_at_most_n_matches(self):
        """Each treated event must have at most n_matches controls."""
        n_matches = 1
        treated = _make_universe(n=20, seed=10)
        treated["event_id"] = [f"T{i}" for i in range(20)]
        controls = _make_universe(n=50, seed=20)

        result = build_matched_controls(
            treated, controls, n_matches=n_matches
        )
        counts = result["event_id"].value_counts()
        assert (counts <= n_matches).all(), (
            f"Some events matched more than {n_matches} times: {counts[counts > n_matches]}"
        )

    def test_no_self_matching(self):
        """A treated event must not be matched to itself."""
        universe = _make_universe(n=30, seed=5)
        result = build_matched_controls(universe.head(10), universe)
        if not result.empty:
            same = result[result["event_id"] == result["control_event_id"]]
            assert same.empty, "Self-matching detected"

    def test_output_columns_present(self):
        """Output must contain event_id, control_event_id, and distance."""
        universe = _make_universe(n=30, seed=6)
        result = build_matched_controls(universe.head(5), universe)
        for col in ["event_id", "control_event_id", "distance"]:
            assert col in result.columns, f"Column '{col}' missing from matched_controls"

    def test_distance_non_negative(self):
        """All distances must be ≥ 0."""
        universe = _make_universe(n=40, seed=7)
        result = build_matched_controls(universe.head(10), universe)
        if not result.empty:
            assert (result["distance"] >= 0).all()

    def test_inferred_assumption_comments_present(self):
        """
        Validate that the source code documents inferred assumptions.
        This test reads the source file and checks for the required markers.
        """
        import inspect
        from src import liquid_universe_extension as lue

        source = inspect.getsource(lue)
        count = source.count("# INFERRED ASSUMPTION:")
        assert count >= 5, (
            f"Expected ≥ 5 '# INFERRED ASSUMPTION:' markers in "
            f"liquid_universe_extension.py, found {count}"
        )

    def test_matching_without_replacement_no_duplicate_controls(self):
        """Controls should not be reused (sampling without replacement)."""
        universe = _make_universe(n=50, seed=8)
        treated = universe.head(20).copy()
        treated["event_id"] = [f"TR{i}" for i in range(20)]

        result = build_matched_controls(treated, universe, n_matches=1)
        ctrl_ids = result["control_event_id"].values
        # No control should appear twice
        assert len(ctrl_ids) == len(set(ctrl_ids)), "Duplicate controls detected"
