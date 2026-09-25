"""
test_event_return_engine.py
===========================
Unit tests for abnormal-return computation logic.

Tests verify correct FF3 beta estimation, winsorisation, CAR aggregation,
alpha treatment, and extreme-event exclusion rules.  All tests use synthetic
data; no live data or cached files are required.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats


# ---------------------------------------------------------------------------
# Reference implementations of core abnormal-return computations
# (mirrors what event_return_engine.py should implement)
# ---------------------------------------------------------------------------

def _estimate_ff3_betas(
    stock_returns: pd.Series,
    ff3_returns: pd.DataFrame,
    estimation_window: int = 252,
) -> dict:
    """OLS regression of stock excess returns on FF3 factors.

    Returns dict with 'alpha', 'beta_mkt', 'beta_smb', 'beta_hml'.
    """
    y = stock_returns.values[:estimation_window]
    X = ff3_returns[["Mkt-RF", "SMB", "HML"]].values[:estimation_window]
    X_const = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(X_const, y, rcond=None)
    return {
        "alpha": float(coeffs[0]),
        "beta_mkt": float(coeffs[1]),
        "beta_smb": float(coeffs[2]),
        "beta_hml": float(coeffs[3]),
    }


def _compute_expected_return_no_alpha(
    betas: dict,
    ff3_row: pd.Series,
    rf: float,
) -> float:
    """Expected return = RF + beta_mkt*(Mkt-RF) + beta_smb*SMB + beta_hml*HML.

    Note: alpha is intentionally excluded from expected return per study design.
    """
    return (
        rf
        + betas["beta_mkt"] * ff3_row["Mkt-RF"]
        + betas["beta_smb"] * ff3_row["SMB"]
        + betas["beta_hml"] * ff3_row["HML"]
    )


def _winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def _compute_car(ar_series: pd.Series, start: int, end: int) -> float:
    """Sum ARs from index start to end (inclusive, 1-based day numbering)."""
    return float(ar_series.iloc[start - 1 : end].sum())


def _compute_car_alpha(car: float, alpha_hat: float, holding_days: int) -> float:
    """CAR_alpha = CAR - holding_days * alpha_hat."""
    return car - holding_days * alpha_hat


def _drop_extreme_cars(df: pd.DataFrame, car_col: str = "car_1_30") -> pd.DataFrame:
    """Drop events where CAR < -1 (fully wiped out) or price ratio is extreme.

    Price-ratio filter (strict): drop when ratio >= 8 OR ratio <= 1/8.
    Equivalently, keep only events where 1/8 < ratio < 8.
    """
    df = df[df[car_col] >= -1.0].copy()
    if "price_ratio" in df.columns:
        df = df[(df["price_ratio"] > 1 / 8) & (df["price_ratio"] < 8)].copy()
    return df


def _reconstruct_close_with_splits(
    unadjusted_close: pd.Series,
    split_factors: pd.Series,
) -> pd.Series:
    """Cumulative product of split factors converts raw price to adj. close."""
    cum_split = split_factors.cumprod()
    return unadjusted_close * cum_split


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_ff3(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0004, 0.008, n),
            "SMB": rng.normal(0.0001, 0.005, n),
            "HML": rng.normal(0.0001, 0.005, n),
            "RF": np.full(n, 0.00005),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def _make_stock_returns(ff3: pd.DataFrame, betas: dict, noise_std: float = 0.005) -> pd.Series:
    """Generate synthetic stock excess returns consistent with given betas."""
    rng = np.random.default_rng(42)
    er = (
        betas["alpha"]
        + betas["beta_mkt"] * ff3["Mkt-RF"]
        + betas["beta_smb"] * ff3["SMB"]
        + betas["beta_hml"] * ff3["HML"]
        + rng.normal(0, noise_std, len(ff3))
    )
    return er


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFF3BetaEstimation:
    """Verify OLS regression recovers known betas from synthetic data."""

    def test_beta_recovery(self):
        """OLS should recover betas close to their true values."""
        true_betas = {"alpha": 0.0, "beta_mkt": 1.2, "beta_smb": 0.5, "beta_hml": -0.3}
        ff3 = _make_ff3(n=500)
        stock_ret = _make_stock_returns(ff3, true_betas, noise_std=0.001)

        estimated = _estimate_ff3_betas(stock_ret, ff3, estimation_window=500)

        assert abs(estimated["beta_mkt"] - true_betas["beta_mkt"]) < 0.15, (
            f"Mkt-RF beta mismatch: est={estimated['beta_mkt']:.3f}, true={true_betas['beta_mkt']}"
        )
        assert abs(estimated["beta_smb"] - true_betas["beta_smb"]) < 0.15
        assert abs(estimated["beta_hml"] - true_betas["beta_hml"]) < 0.15

    def test_no_alpha_in_expected_return(self):
        """Expected return computation must NOT include alpha_hat."""
        betas = {"alpha": 0.05, "beta_mkt": 1.0, "beta_smb": 0.0, "beta_hml": 0.0}
        ff3_row = pd.Series({"Mkt-RF": 0.01, "SMB": 0.0, "HML": 0.0})
        er = _compute_expected_return_no_alpha(betas, ff3_row, rf=0.0)
        # Expected return should equal beta_mkt * Mkt-RF = 0.01, not 0.05 + 0.01 = 0.06
        assert abs(er - 0.01) < 1e-8, f"Expected 0.01, got {er}"


class TestWinsorization:
    """Verify returns are winsorised at 1st and 99th percentiles."""

    def test_winsorization_clips_extremes(self):
        """Values below p1 and above p99 must be clipped to those percentiles."""
        rng = np.random.default_rng(0)
        base = pd.Series(rng.normal(0, 0.02, 1000))
        # Inject two extreme outliers
        base.iloc[0] = -5.0
        base.iloc[1] = 5.0

        winsorised = _winsorise(base)
        p1 = base.quantile(0.01)
        p99 = base.quantile(0.99)

        assert winsorised.min() >= p1
        assert winsorised.max() <= p99

    def test_winsorization_does_not_affect_interior(self):
        """Values strictly inside [p1, p99] must be unchanged."""
        arr = pd.Series(np.linspace(-0.1, 0.1, 200))
        original_middle = arr.iloc[100]
        winsorised = _winsorise(arr)
        assert abs(winsorised.iloc[100] - original_middle) < 1e-12

    def test_winsorization_preserves_length(self):
        """Winsorisation must not drop any observations."""
        arr = pd.Series(np.random.default_rng(7).normal(0, 1, 500))
        assert len(_winsorise(arr)) == len(arr)


class TestCARComputation:
    """Verify CARs are computed as the correct sum of ARs over the right windows."""

    def test_car_sum_over_window(self):
        """CAR[1,30] must equal the exact sum of ARs on days 1 through 30."""
        ar_vals = np.arange(1.0, 31.0) * 0.001  # days 1..30
        ar_series = pd.Series(ar_vals)
        expected = float(ar_vals.sum())
        computed = _compute_car(ar_series, start=1, end=30)
        assert abs(computed - expected) < 1e-10

    def test_car_sub_window(self):
        """CAR[2,10] must equal the sum of ARs on days 2 through 10."""
        ar_vals = np.ones(30) * 0.002
        ar_series = pd.Series(ar_vals)
        expected = 9 * 0.002  # days 2..10 inclusive = 9 days
        computed = _compute_car(ar_series, start=2, end=10)
        assert abs(computed - expected) < 1e-10

    def test_car_single_day(self):
        """CAR[1,1] must equal AR on day 1."""
        ar_series = pd.Series([0.05, 0.02, -0.01] + [0.0] * 27)
        computed = _compute_car(ar_series, start=1, end=1)
        assert abs(computed - 0.05) < 1e-10


class TestARExcludesAlpha:
    """Primary AR should not include estimated alpha."""

    def test_ar_does_not_include_alpha(self):
        """AR = actual_return - expected_return_no_alpha; alpha_hat must not appear."""
        betas = {"alpha": 0.002, "beta_mkt": 1.0, "beta_smb": 0.0, "beta_hml": 0.0}
        ff3_row = pd.Series({"Mkt-RF": 0.005, "SMB": 0.0, "HML": 0.0})
        rf = 0.0001
        actual_return = 0.008

        # Expected return deliberately excludes alpha
        er = _compute_expected_return_no_alpha(betas, ff3_row, rf=rf)
        ar = actual_return - er

        # With alpha excluded: er = 0.0001 + 1.0*0.005 = 0.0051 → AR = 0.0029
        expected_ar = actual_return - (rf + betas["beta_mkt"] * ff3_row["Mkt-RF"])
        assert abs(ar - expected_ar) < 1e-10

    def test_ar_null_beta_equals_excess_return(self):
        """Under null hypothesis (beta_mkt=1, SMB=HML=0), AR = stock excess return - Mkt-RF."""
        betas = {"alpha": 0.0, "beta_mkt": 1.0, "beta_smb": 0.0, "beta_hml": 0.0}
        ff3_row = pd.Series({"Mkt-RF": 0.01, "SMB": 0.0, "HML": 0.0})
        rf = 0.0
        actual_return = 0.015  # stock excess return (already net of RF because RF=0)

        er = _compute_expected_return_no_alpha(betas, ff3_row, rf=rf)
        ar = actual_return - er

        # Should equal actual - mkt excess = 0.015 - 0.01 = 0.005
        assert abs(ar - 0.005) < 1e-10


class TestCARAlphaIncludesAlpha:
    """CAR_alpha = CAR - n_days * alpha_hat."""

    def test_car_alpha_formula(self):
        """CAR_alpha = CAR - 30 * alpha_hat for a 30-day window."""
        car = 0.12
        alpha_hat = 0.001  # 0.1% per day
        car_alpha = _compute_car_alpha(car, alpha_hat, holding_days=30)
        expected = 0.12 - 30 * 0.001
        assert abs(car_alpha - expected) < 1e-10

    def test_car_alpha_zero_when_alpha_zero(self):
        """When alpha_hat=0, CAR_alpha should equal CAR."""
        car = 0.08
        assert abs(_compute_car_alpha(car, 0.0, 30) - car) < 1e-10

    def test_car_alpha_different_window(self):
        """Verify the formula holds for a 10-day window."""
        car = 0.05
        alpha_hat = 0.002
        assert abs(_compute_car_alpha(car, alpha_hat, 10) - (0.05 - 0.02)) < 1e-10


class TestDropExtremeCARs:
    """Verify events with CAR < -1 or extreme price ratios are excluded."""

    def test_car_below_minus_one_dropped(self):
        """Events with CAR[1,30] < -100% should be removed."""
        df = pd.DataFrame({"event_id": ["A", "B", "C"], "car_1_30": [-1.5, -1.0, 0.05]})
        out = _drop_extreme_cars(df)
        # CAR = -1.5 dropped; CAR = -1.0 is at boundary (>= -1.0, kept)
        assert "A" not in out["event_id"].values
        assert "B" in out["event_id"].values
        assert "C" in out["event_id"].values

    def test_price_ratio_above_eight_dropped(self):
        """Price ratio ≥ 8 should be excluded (stock split contamination)."""
        df = pd.DataFrame(
            {
                "event_id": ["A", "B"],
                "car_1_30": [0.05, 0.05],
                "price_ratio": [8.0, 7.9],
            }
        )
        out = _drop_extreme_cars(df)
        assert "A" not in out["event_id"].values
        assert "B" in out["event_id"].values

    def test_price_ratio_below_one_eighth_dropped(self):
        """Price ratio ≤ 1/8 should be excluded."""
        df = pd.DataFrame(
            {
                "event_id": ["A", "B"],
                "car_1_30": [0.05, 0.05],
                "price_ratio": [1 / 8, 1 / 8 + 0.001],
            }
        )
        out = _drop_extreme_cars(df)
        assert "A" not in out["event_id"].values
        assert "B" in out["event_id"].values


class TestActualCloseReconstruction:
    """Verify split-adjustment logic for reconstructing adjusted closing prices."""

    def test_no_splits(self):
        """With all split factors = 1, adjusted = unadjusted."""
        unadj = pd.Series([10.0, 10.5, 11.0, 10.8])
        splits = pd.Series([1.0, 1.0, 1.0, 1.0])
        adj = _reconstruct_close_with_splits(unadj, splits)
        pd.testing.assert_series_equal(adj, unadj)

    def test_two_for_one_split(self):
        """A 2-for-1 split should halve the pre-split price."""
        unadj = pd.Series([20.0, 20.0, 10.0, 10.0])
        # Factor = 0.5 applied at period 3 (post-split price already halved)
        splits = pd.Series([1.0, 1.0, 0.5, 1.0])
        adj = _reconstruct_close_with_splits(unadj, splits)
        # cum_split = [1, 1, 0.5, 0.5]
        expected = pd.Series([20.0, 20.0, 5.0, 5.0])
        pd.testing.assert_series_equal(adj, expected)

    def test_adjustment_monotonically_applies(self):
        """Cumulative split factors compound over the full series."""
        unadj = pd.Series([100.0, 100.0, 100.0])
        splits = pd.Series([1.0, 0.5, 0.5])  # two 2-for-1 splits
        adj = _reconstruct_close_with_splits(unadj, splits)
        expected_cum = [1.0, 0.5, 0.25]
        for i, (val, cf) in enumerate(zip(adj.values, expected_cum)):
            assert abs(val - 100.0 * cf) < 1e-8, f"Day {i}: adj={val}, expected={100*cf}"


class TestMathematicalProperties:
    """Mathematical property tests for the AR framework."""

    def test_ar_under_null_beta_equals_excess_return(self):
        """
        Under the null (market beta = 1, SMB = HML = 0, alpha = 0):
            AR = R_stock - R_market
        i.e., AR equals the stock's excess return over the market.
        """
        rng = np.random.default_rng(10)
        mkt_ret = rng.normal(0.0004, 0.01, 100)
        stock_ret = mkt_ret + rng.normal(0, 0.005, 100)  # beta ≈ 1 by construction

        betas = {"alpha": 0.0, "beta_mkt": 1.0, "beta_smb": 0.0, "beta_hml": 0.0}
        ff3 = pd.DataFrame(
            {"Mkt-RF": mkt_ret, "SMB": np.zeros(100), "HML": np.zeros(100)}
        )
        ars = []
        for i in range(100):
            er = _compute_expected_return_no_alpha(betas, ff3.iloc[i], rf=0.0)
            ars.append(stock_ret[i] - er)

        np.testing.assert_allclose(np.array(ars), stock_ret - mkt_ret, atol=1e-12)

    def test_car_is_sum_of_ars(self):
        """CAR over any window must exactly equal the sum of the individual ARs."""
        rng = np.random.default_rng(20)
        ar_series = pd.Series(rng.normal(0.001, 0.01, 30))
        for start, end in [(1, 30), (1, 5), (10, 20), (2, 2)]:
            car = _compute_car(ar_series, start, end)
            direct_sum = ar_series.iloc[start - 1 : end].sum()
            assert abs(car - direct_sum) < 1e-12
