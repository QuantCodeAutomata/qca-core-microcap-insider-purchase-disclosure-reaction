"""
test_sample_funnel.py
=====================
Unit tests for sample funnel filters and data integrity.

Tests verify that each filter is applied in the documented order and that
edge cases (boundary values, deduplication, anchoring) are handled correctly.
Synthetic DataFrames are constructed directly in each test; no disk I/O is
required.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers for building minimal synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_events(**kwargs) -> pd.DataFrame:
    """Build a minimal Form 4 event DataFrame with sane defaults."""
    n = kwargs.pop("n", 5)
    rng = np.random.default_rng(0)
    base = {
        "event_id": [f"E{i}" for i in range(n)],
        "ticker": [f"TK{i}" for i in range(n)],
        "filing_date": pd.date_range("2021-01-01", periods=n, freq="D"),
        "transaction_date": pd.date_range("2020-12-28", periods=n, freq="D"),
        "reporting_lag_days": [2] * n,
        "market_cap": [100e6] * n,
        "addv": [500_000] * n,
        "actual_close": [10.0] * n,
        "form4_price": [10.0] * n,
        "t0": pd.date_range("2021-01-04", periods=n, freq="B"),
    }
    base.update(kwargs)
    return pd.DataFrame(base)


def _apply_reporting_lag_filter(df: pd.DataFrame, max_lag: int = 90) -> pd.DataFrame:
    return df[df["reporting_lag_days"].between(0, max_lag)].copy()


def _apply_price_basis_gate(df: pd.DataFrame) -> pd.DataFrame:
    ratio = df["actual_close"] / df["form4_price"]
    return df[ratio.between(0.5, 2.0)].copy()


def _apply_market_cap_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["market_cap"].between(30e6, 500e6)].copy()


def _apply_addv_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["addv"] > 200_000].copy()


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep earliest filing per ticker per year."""
    df = df.copy()
    df["year"] = pd.DatetimeIndex(df["filing_date"]).year
    df = df.sort_values("filing_date")
    return df.drop_duplicates(subset=["ticker", "year"], keep="first").drop(columns="year")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLineFiltersOrder:
    """Verify each filter is applied sequentially and reduces the set correctly."""

    def test_line_filters_order(self):
        """Filters must be applied in a fixed order and each must shrink or hold the count."""
        # Construct events that are eliminated at specific filter stages
        df = pd.DataFrame(
            {
                "event_id": ["A", "B", "C", "D", "E"],
                "ticker": ["T1", "T2", "T3", "T4", "T5"],
                "filing_date": pd.to_datetime(
                    ["2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08", "2021-01-09"]
                ),
                "transaction_date": pd.to_datetime(
                    ["2021-01-01", "2021-01-01", "2021-01-01", "2021-01-01", "2021-01-01"]
                ),
                "reporting_lag_days": [4, 91, 4, 4, 4],  # B removed by lag filter
                "market_cap": [100e6, 100e6, 600e6, 100e6, 100e6],  # C removed by mcap
                "addv": [500_000, 500_000, 500_000, 100_000, 500_000],  # D removed by ADDV
                "actual_close": [10.0, 10.0, 10.0, 10.0, 10.0],
                "form4_price": [10.0, 10.0, 10.0, 10.0, 30.0],  # E removed by price gate
            }
        )
        n0 = len(df)

        # Apply in order
        after_lag = _apply_reporting_lag_filter(df)
        assert len(after_lag) == n0 - 1, "Lag filter should remove exactly B"

        after_mcap = _apply_market_cap_filter(after_lag)
        assert len(after_mcap) == len(after_lag) - 1, "Mcap filter should remove exactly C"

        after_addv = _apply_addv_filter(after_mcap)
        assert len(after_addv) == len(after_mcap) - 1, "ADDV filter should remove exactly D"

        after_price = _apply_price_basis_gate(after_addv)
        assert len(after_price) == len(after_addv) - 1, "Price gate should remove exactly E"

        # Only A should remain
        assert list(after_price["event_id"]) == ["A"]


class TestReportingLagFilter:
    """Edge cases for the reporting-lag filter."""

    def test_zero_lag_allowed(self):
        """Same-day filings (lag = 0) must be kept."""
        df = _make_events(n=1)
        df["reporting_lag_days"] = 0
        out = _apply_reporting_lag_filter(df)
        assert len(out) == 1

    def test_90_day_lag_allowed(self):
        """Lag of exactly 90 days must be kept (inclusive upper bound)."""
        df = _make_events(n=1)
        df["reporting_lag_days"] = 90
        out = _apply_reporting_lag_filter(df)
        assert len(out) == 1

    def test_91_day_lag_removed(self):
        """Lag of 91 days must be removed (strictly exceeds limit)."""
        df = _make_events(n=1)
        df["reporting_lag_days"] = 91
        out = _apply_reporting_lag_filter(df)
        assert len(out) == 0

    def test_negative_lag_removed(self):
        """Negative lags (transaction after filing, data error) must be removed."""
        df = _make_events(n=1)
        df["reporting_lag_days"] = -1
        out = _apply_reporting_lag_filter(df)
        assert len(out) == 0


class TestDeduplication:
    """Verify earliest filing is kept when the same ticker files multiple times."""

    def test_earliest_filing_kept(self):
        """When a ticker appears twice in a year, the earlier date survives."""
        df = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "ticker": ["AAPL", "AAPL"],
                "filing_date": pd.to_datetime(["2021-03-01", "2021-06-01"]),
                "reporting_lag_days": [2, 2],
                "market_cap": [100e6, 100e6],
                "addv": [500_000, 500_000],
                "actual_close": [10.0, 10.0],
                "form4_price": [10.0, 10.0],
            }
        )
        out = _deduplicate(df)
        assert len(out) == 1
        assert out.iloc[0]["event_id"] == "E1"

    def test_different_years_both_kept(self):
        """Same ticker in different years should both survive deduplication."""
        df = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "ticker": ["AAPL", "AAPL"],
                "filing_date": pd.to_datetime(["2020-03-01", "2021-06-01"]),
                "reporting_lag_days": [2, 2],
                "market_cap": [100e6, 100e6],
                "addv": [500_000, 500_000],
                "actual_close": [10.0, 10.0],
                "form4_price": [10.0, 10.0],
            }
        )
        out = _deduplicate(df)
        assert len(out) == 2

    def test_different_tickers_both_kept(self):
        """Two distinct tickers with the same date must both survive."""
        df = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "ticker": ["AAPL", "MSFT"],
                "filing_date": pd.to_datetime(["2021-03-01", "2021-03-01"]),
                "reporting_lag_days": [2, 2],
                "market_cap": [100e6, 100e6],
                "addv": [500_000, 500_000],
                "actual_close": [10.0, 10.0],
                "form4_price": [10.0, 10.0],
            }
        )
        out = _deduplicate(df)
        assert len(out) == 2


class TestPriceBasisGate:
    """Verify the actual_close / form4_price ratio filter [0.5, 2.0]."""

    @pytest.mark.parametrize(
        "actual, form4, should_keep",
        [
            (10.0, 10.0, True),    # ratio = 1.0 → keep
            (10.0, 20.0, True),    # ratio = 0.5 → keep (at lower bound)
            (20.0, 10.0, True),    # ratio = 2.0 → keep (at upper bound)
            (9.99, 20.0, False),   # ratio < 0.5 → drop
            (20.01, 10.0, False),  # ratio > 2.0 → drop
            (0.0, 10.0, False),    # ratio = 0.0 → drop
        ],
    )
    def test_price_basis_gate(self, actual, form4, should_keep):
        df = _make_events(n=1)
        df["actual_close"] = actual
        df["form4_price"] = form4
        out = _apply_price_basis_gate(df)
        assert (len(out) == 1) == should_keep, (
            f"actual={actual}, form4={form4}, ratio={actual/form4:.3f}, "
            f"expected keep={should_keep}"
        )


class TestMarketCapFilter:
    """Verify the market-cap filter [30M, 500M]."""

    @pytest.mark.parametrize(
        "mcap, should_keep",
        [
            (30e6, True),        # at lower bound → keep
            (500e6, True),       # at upper bound → keep
            (29_999_999, False), # just below lower → drop
            (500_000_001, False),# just above upper → drop
            (200e6, True),       # mid-range → keep
        ],
    )
    def test_market_cap_filter(self, mcap, should_keep):
        df = _make_events(n=1)
        df["market_cap"] = mcap
        out = _apply_market_cap_filter(df)
        assert (len(out) == 1) == should_keep


class TestAddvFilter:
    """Verify the ADDV filter (strictly > $200K)."""

    @pytest.mark.parametrize(
        "addv, should_keep",
        [
            (200_001, True),    # just above threshold → keep
            (200_000, False),   # at threshold (not strictly greater) → drop
            (199_999, False),   # below threshold → drop
            (1_000_000, True),  # well above → keep
        ],
    )
    def test_addv_filter(self, addv, should_keep):
        df = _make_events(n=1)
        df["addv"] = addv
        out = _apply_addv_filter(df)
        assert (len(out) == 1) == should_keep


class TestEventAnchorT0:
    """Verify that t0 is set to the last trading day on or before filing_date."""

    def _last_trading_day_on_or_before(
        self, filing_date: pd.Timestamp, trading_days: list[pd.Timestamp]
    ) -> pd.Timestamp:
        """Reference implementation: last date in trading_days ≤ filing_date."""
        candidates = [d for d in trading_days if d <= filing_date]
        if not candidates:
            raise ValueError("No trading day on or before filing_date.")
        return max(candidates)

    def test_filing_on_trading_day(self):
        """When filing_date is a trading day, t0 == filing_date."""
        filing = pd.Timestamp("2021-01-04")  # Monday
        trading = pd.date_range("2021-01-01", "2021-01-10", freq="B").tolist()
        t0 = self._last_trading_day_on_or_before(filing, trading)
        assert t0 == filing

    def test_filing_on_weekend(self):
        """When filing_date is Saturday, t0 should be the preceding Friday."""
        filing = pd.Timestamp("2021-01-02")  # Saturday
        trading = pd.date_range("2020-12-28", "2021-01-10", freq="B").tolist()
        t0 = self._last_trading_day_on_or_before(filing, trading)
        assert t0 == pd.Timestamp("2021-01-01")  # preceding Friday (New Year's is 1 Jan here)

    def test_filing_before_all_trading_days_raises(self):
        """Should raise if no trading day precedes filing_date."""
        filing = pd.Timestamp("2019-12-31")
        trading = pd.date_range("2020-01-01", "2020-01-10", freq="B").tolist()
        with pytest.raises(ValueError):
            self._last_trading_day_on_or_before(filing, trading)

    def test_t0_never_after_filing_date(self):
        """t0 must always be ≤ filing_date by construction."""
        rng = np.random.default_rng(99)
        trading = pd.date_range("2020-01-01", "2022-12-31", freq="B").tolist()
        for _ in range(50):
            offset = int(rng.integers(0, 5))  # filing on or near a trading day
            filing = pd.Timestamp("2021-06-01") + pd.Timedelta(days=offset)
            t0 = self._last_trading_day_on_or_before(filing, trading)
            assert t0 <= filing, f"t0={t0} is after filing_date={filing}"
