"""
test_features.py
================
Unit tests for feature engineering logic.

Covers mathematical correctness of each feature, look-ahead contamination
checks, flag construction, and boundary conditions.  All tests use synthetic
data; no disk I/O is required.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Reference implementations of features (mirrors feature_engineering.py)
# ---------------------------------------------------------------------------

def _compute_pct_from_52w_high(
    close_history: pd.Series,
    t0: pd.Timestamp,
) -> float:
    """Return (close_t0 - 52w_high) / 52w_high.

    Uses the 252 trading days ending on t0 (inclusive).  t0 is included in
    the lookback so that when the current price equals the rolling high the
    feature is exactly 0.  Future data (index > t0) is excluded, so there
    is no look-ahead bias.  Result is always ≤ 0.
    """
    up_to_t0 = close_history[close_history.index <= t0]
    if up_to_t0.empty:
        return np.nan
    window = up_to_t0.iloc[-252:]
    high_52w = float(window.max())
    close_t0 = float(up_to_t0.iloc[-1])
    return (close_t0 - high_52w) / high_52w


def _compute_volatility_annualised(
    close_history: pd.Series,
    t0: pd.Timestamp,
    window: int = 21,
) -> float:
    """Annualised volatility from log-returns over the *window* days before t0.

    Uses only data strictly before t0.
    """
    pre_t0 = close_history[close_history.index < t0]
    if len(pre_t0) < window + 1:
        return np.nan
    prices = pre_t0.iloc[-(window + 1):]
    log_ret = np.diff(np.log(prices.values))
    return float(np.std(log_ret, ddof=1) * np.sqrt(252))


def _is_no_lookahead(
    feature_value: float,
    close_history: pd.Series,
    t0: pd.Timestamp,
    compute_fn,
) -> bool:
    """Check that feature_value equals computation restricted to data up to t0."""
    without_future = close_history[close_history.index <= t0]
    recomputed = compute_fn(without_future, t0)
    if np.isnan(feature_value) and np.isnan(recomputed):
        return True
    return abs(feature_value - recomputed) < 1e-9


def _is_first_purchase(
    event_filing_date: pd.Timestamp,
    ticker: str,
    prior_filings: pd.DataFrame,
) -> int:
    """Return 1 if no prior insider purchase exists for ticker before filing_date.

    Uses only *strictly prior* filings (filing_date < event_filing_date).
    """
    earlier = prior_filings[
        (prior_filings["ticker"] == ticker)
        & (prior_filings["filing_date"] < event_filing_date)
        & (prior_filings["transaction_code"] == "P")
    ]
    return int(earlier.empty)


def _compute_biotech_flag(sic_code: str, company_name: str = "") -> int:
    """Return 1 for biotech/pharma companies via SIC code or name matching."""
    biotech_sics = {
        "2836",  # pharmaceutical preparations
        "2835",  # in vitro & in vivo diagnostic substances
        "2830",  # drugs
        "8731",  # commercial physical & biological research
        "8099",  # health services NEC
    }
    # 'bio' catches BioNTech, BioMarin, etc. via substring match
    biotech_keywords = {"bio", "biotech", "pharma", "therapeutics", "biosciences", "genomics"}

    if sic_code in biotech_sics:
        return 1
    if any(kw in company_name.lower() for kw in biotech_keywords):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_price_series(
    n: int = 400,
    start: str = "2020-01-01",
    start_price: float = 20.0,
    drift: float = 0.0003,
    vol: float = 0.02,
    seed: int = 0,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol, n)
    prices = start_price * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=pd.date_range(start, periods=n, freq="B"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPctFrom52wHigh:
    """Verify pct_from_52w_high is always ≤ 0 and computed from pre-t0 data only."""

    def test_at_52w_high_is_zero(self):
        """If current price is the 52w high, feature should be exactly 0."""
        prices = pd.Series(
            [10.0] * 100 + [20.0],
            index=pd.date_range("2020-01-01", periods=101, freq="B"),
        )
        t0 = prices.index[-1]
        val = _compute_pct_from_52w_high(prices, t0)
        assert abs(val) < 1e-10, f"Expected 0 at 52w high, got {val}"

    def test_below_52w_high_is_negative(self):
        """Feature must be ≤ 0 whenever price is at or below the 52w high."""
        prices = _make_price_series(n=300, seed=42)
        # Force the 52w high to be well above current price
        prices.iloc[-1] = prices.iloc[-252:].max() * 0.80
        t0 = prices.index[-1]
        val = _compute_pct_from_52w_high(prices, t0)
        assert val <= 0.0, f"Expected <= 0, got {val}"

    def test_pct_from_52w_high_never_positive(self):
        """Over 200 random events, the feature should never be positive."""
        rng = np.random.default_rng(1)
        prices = _make_price_series(n=500, seed=1)
        for _ in range(200):
            t0_idx = int(rng.integers(252, 499))
            t0 = prices.index[t0_idx]
            val = _compute_pct_from_52w_high(prices, t0)
            if not np.isnan(val):
                assert val <= 1e-9, f"Feature positive ({val:.6f}) at {t0}"

    def test_uses_only_pre_t0_data(self):
        """Future prices after t0 must not influence the feature value."""
        prices = _make_price_series(n=300, seed=7)
        t0 = prices.index[200]
        val_restricted = _compute_pct_from_52w_high(prices[:201], t0)
        val_with_future = _compute_pct_from_52w_high(prices, t0)
        # Both computations should give the same answer
        assert abs(val_restricted - val_with_future) < 1e-10


class TestVolatilityAnnualised:
    """Verify annualised volatility is positive and not look-ahead contaminated."""

    def test_volatility_positive(self):
        """Volatility must be strictly positive for any non-constant price series."""
        prices = _make_price_series(n=300, vol=0.02, seed=3)
        t0 = prices.index[250]
        vol = _compute_volatility_annualised(prices, t0, window=21)
        assert vol > 0.0, f"Volatility should be positive, got {vol}"

    def test_volatility_zero_for_constant_prices(self):
        """Constant price series should yield volatility of exactly 0."""
        prices = pd.Series(
            [10.0] * 100,
            index=pd.date_range("2020-01-01", periods=100, freq="B"),
        )
        t0 = prices.index[-1]
        vol = _compute_volatility_annualised(prices, t0, window=21)
        assert abs(vol) < 1e-10

    def test_higher_vol_series_gives_higher_estimate(self):
        """A series with higher true vol should produce a higher estimated vol."""
        prices_low = _make_price_series(n=300, vol=0.005, seed=10)
        prices_high = _make_price_series(n=300, vol=0.05, seed=10)
        t0 = prices_low.index[250]
        vol_low = _compute_volatility_annualised(prices_low, t0, window=21)
        vol_high = _compute_volatility_annualised(prices_high, t0, window=21)
        assert vol_high > vol_low

    def test_no_lookahead_in_volatility(self):
        """Volatility must only use data up to (not including) t0."""
        prices = _make_price_series(n=300, seed=5)
        t0 = prices.index[200]
        vol = _compute_volatility_annualised(prices, t0, window=21)
        # Compute again with only pre-t0 data to confirm no look-ahead
        vol_restricted = _compute_volatility_annualised(prices[:200], t0, window=21)
        assert abs(vol - vol_restricted) < 1e-10


class TestNoLookahead:
    """General look-ahead contamination checks across features."""

    def test_feature_unchanged_when_future_appended(self):
        """Appending future prices after t0 must not change any feature value."""
        prices_past = _make_price_series(n=260, seed=20)
        t0 = prices_past.index[-1]

        # Compute features using only past data
        pct_past = _compute_pct_from_52w_high(prices_past, t0)
        vol_past = _compute_volatility_annualised(prices_past, t0)

        # Append "future" data
        future = _make_price_series(n=30, start="2021-02-01", seed=99)
        prices_all = pd.concat([prices_past, future]).sort_index()

        pct_all = _compute_pct_from_52w_high(prices_all, t0)
        vol_all = _compute_volatility_annualised(prices_all, t0)

        assert abs(pct_past - pct_all) < 1e-10, "pct_from_52w_high look-ahead detected"
        assert abs(vol_past - vol_all) < 1e-10, "volatility look-ahead detected"


class TestIsFirstPurchase:
    """Verify is_first_purchase uses only strictly prior filings."""

    def test_no_prior_filings_is_first(self):
        """When no prior filings exist, is_first_purchase should be 1."""
        prior_filings = pd.DataFrame(columns=["ticker", "filing_date", "transaction_code"])
        result = _is_first_purchase(pd.Timestamp("2021-06-01"), "AAPL", prior_filings)
        assert result == 1

    def test_prior_purchase_exists(self):
        """When a prior purchase exists, is_first_purchase should be 0."""
        prior_filings = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "filing_date": [pd.Timestamp("2021-01-15")],
                "transaction_code": ["P"],
            }
        )
        result = _is_first_purchase(pd.Timestamp("2021-06-01"), "AAPL", prior_filings)
        assert result == 0

    def test_same_date_filing_not_counted(self):
        """A filing on the same date (not strictly prior) must not disqualify."""
        prior_filings = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "filing_date": [pd.Timestamp("2021-06-01")],  # same date, not strictly prior
                "transaction_code": ["P"],
            }
        )
        result = _is_first_purchase(pd.Timestamp("2021-06-01"), "AAPL", prior_filings)
        assert result == 1

    def test_prior_sale_not_purchase(self):
        """A prior sale (code='S') must not affect is_first_purchase."""
        prior_filings = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "filing_date": [pd.Timestamp("2021-01-15")],
                "transaction_code": ["S"],  # sale, not purchase
            }
        )
        result = _is_first_purchase(pd.Timestamp("2021-06-01"), "AAPL", prior_filings)
        assert result == 1

    def test_different_ticker_not_counted(self):
        """Prior purchases by a different ticker must not affect the flag."""
        prior_filings = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "filing_date": [pd.Timestamp("2021-01-15")],
                "transaction_code": ["P"],
            }
        )
        result = _is_first_purchase(pd.Timestamp("2021-06-01"), "AAPL", prior_filings)
        assert result == 1


class TestBiotechFlag:
    """Verify biotech/pharma identification via SIC and name matching."""

    @pytest.mark.parametrize(
        "sic, name, expected",
        [
            ("2836", "", 1),             # SIC match: pharmaceutical preparations
            ("2835", "", 1),             # SIC match: diagnostics
            ("8731", "", 1),             # SIC match: biological research
            ("5999", "", 0),             # No match: retail misc.
            ("5999", "BioNTech AG", 1),  # Name match: 'bio'
            ("5999", "Pharma Inc", 1),   # Name match: 'pharma'
            ("5999", "Therapeutics LLC", 1),  # Name match: 'therapeutics'
            ("5999", "Apple Inc", 0),    # No match
            ("2836", "Random Corp", 1),  # SIC match even with generic name
        ],
    )
    def test_biotech_flag(self, sic, name, expected):
        result = _compute_biotech_flag(sic, name)
        assert result == expected, (
            f"sic={sic!r}, name={name!r}: expected {expected}, got {result}"
        )

    def test_flag_is_binary(self):
        """Flag must only take values 0 or 1."""
        for sic in ["2836", "5999", "8731", "9999"]:
            val = _compute_biotech_flag(sic, "Test Corp")
            assert val in (0, 1), f"Non-binary flag for sic={sic}: {val}"
