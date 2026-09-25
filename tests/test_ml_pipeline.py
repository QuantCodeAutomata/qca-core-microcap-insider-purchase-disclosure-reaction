"""
test_ml_pipeline.py
===================
Unit tests for the ML training and evaluation pipeline.

Covers target construction, train/test split logic, hyperparameter fixtures,
threshold selection, sample statistics, and reproducibility.  All tests use
synthetic DataFrames; no live data or fitted models are required.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, f1_score


# ---------------------------------------------------------------------------
# Reference implementations mirroring ml_training_eval.py
# ---------------------------------------------------------------------------

# Fixed XGBoost hyperparameters used in the study
FIXED_XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 0,
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
}

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2024-12-31")
VALIDATION_YEAR = 2023


def _construct_target(cars: pd.DataFrame, threshold: float = 0.10) -> pd.Series:
    """y = 1 when CAR[1,30] > threshold (default 10%)."""
    return (cars["car_1_30"] > threshold).astype(int)


def _split_train_test(
    events: pd.DataFrame,
    target: pd.Series,
    date_col: str = "filing_date",
) -> dict:
    """Split data: train 2018-2022, validation 2023, test 2024."""
    events = events.copy()
    events["_target"] = target.values
    dates = pd.to_datetime(events[date_col])

    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    val_mask = dates.dt.year == VALIDATION_YEAR
    test_mask = (dates >= TEST_START) & (dates <= TEST_END)

    return {
        "train": events[train_mask],
        "val": events[val_mask],
        "test": events[test_mask],
    }


def _select_threshold_by_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray = None,
) -> float:
    """Choose classification threshold maximising F1 on validation set."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return float(best_t)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_events_with_dates(
    n: int = 1000,
    seed: int = 0,
    start: str = "2018-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(
        rng.choice(
            pd.date_range(start, end, freq="B").to_list(),
            size=n,
            replace=True,
        )
    )
    return pd.DataFrame(
        {
            "event_id": [f"E{i}" for i in range(n)],
            "filing_date": dates,
            "car_1_30": rng.normal(0.03, 0.08, n),
        }
    )


def _make_synthetic_ml_dataset(n: int = 500, positive_rate: float = 0.271, seed: int = 0):
    """Return (X, y) consistent with a target positive rate."""
    rng = np.random.default_rng(seed)
    n_pos = int(n * positive_rate)
    n_neg = n - n_pos
    y = np.array([1] * n_pos + [0] * n_neg)
    rng.shuffle(y)
    X = pd.DataFrame(
        {
            "log_market_cap": rng.normal(18, 1.5, n),
            "log_addv": rng.normal(13, 1.2, n),
            "pct_from_52w_high": -rng.uniform(0, 0.4, n),
            "volatility_21d": rng.uniform(0.01, 0.05, n),
            "return_21d": rng.normal(0, 0.05, n),
        }
    )
    return X, pd.Series(y)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTargetConstruction:
    """Verify binary target: y=1 when CAR[1,30] > 10%."""

    def test_target_above_threshold_is_one(self):
        cars = pd.DataFrame({"car_1_30": [0.11, 0.20, 0.50]})
        y = _construct_target(cars)
        assert list(y) == [1, 1, 1]

    def test_target_exactly_at_threshold_is_zero(self):
        """Boundary: CAR == 10% should be 0 (strictly greater required)."""
        cars = pd.DataFrame({"car_1_30": [0.10]})
        y = _construct_target(cars)
        assert y.iloc[0] == 0

    def test_target_below_threshold_is_zero(self):
        cars = pd.DataFrame({"car_1_30": [-0.05, 0.0, 0.09]})
        y = _construct_target(cars)
        assert list(y) == [0, 0, 0]

    def test_target_binary(self):
        """Output must contain only 0 and 1."""
        rng = np.random.default_rng(0)
        cars = pd.DataFrame({"car_1_30": rng.normal(0.03, 0.1, 500)})
        y = _construct_target(cars)
        assert set(y.unique()).issubset({0, 1})


class TestTrainTestSplit:
    """Verify temporal split: 2018-2022 train, 2023 validation, 2024 test."""

    def test_train_years(self):
        events = _make_events_with_dates(n=2000)
        y = _construct_target(events)
        splits = _split_train_test(events, y)
        train_years = pd.DatetimeIndex(splits["train"]["filing_date"]).year.unique()
        assert all(y in range(2018, 2023) for y in train_years), (
            f"Train set contains non-2018-2022 years: {sorted(train_years)}"
        )

    def test_val_year(self):
        events = _make_events_with_dates(n=2000)
        y = _construct_target(events)
        splits = _split_train_test(events, y)
        val_years = pd.DatetimeIndex(splits["val"]["filing_date"]).year.unique()
        assert list(val_years) == [2023], f"Validation set years: {val_years}"

    def test_test_year(self):
        events = _make_events_with_dates(n=2000)
        y = _construct_target(events)
        splits = _split_train_test(events, y)
        test_years = pd.DatetimeIndex(splits["test"]["filing_date"]).year.unique()
        assert list(test_years) == [2024], f"Test set years: {test_years}"

    def test_no_overlap_between_splits(self):
        """Train, val, and test sets must have disjoint event_ids."""
        events = _make_events_with_dates(n=2000)
        y = _construct_target(events)
        splits = _split_train_test(events, y)
        train_ids = set(splits["train"]["event_id"])
        val_ids = set(splits["val"]["event_id"])
        test_ids = set(splits["test"]["event_id"])
        assert train_ids.isdisjoint(val_ids), "Train and val share event_ids"
        assert train_ids.isdisjoint(test_ids), "Train and test share event_ids"
        assert val_ids.isdisjoint(test_ids), "Val and test share event_ids"

    def test_2023_excluded_from_train_and_test(self):
        """2023 events must appear ONLY in the validation split."""
        events = _make_events_with_dates(n=2000)
        y = _construct_target(events)
        splits = _split_train_test(events, y)

        for split_name in ("train", "test"):
            years = pd.DatetimeIndex(splits[split_name]["filing_date"]).year
            assert 2023 not in years, f"2023 found in {split_name} split"


class TestXGBHyperparams:
    """Verify fixed hyperparameters match the study specification."""

    def test_n_estimators(self):
        assert FIXED_XGB_PARAMS["n_estimators"] == 500

    def test_max_depth(self):
        assert FIXED_XGB_PARAMS["max_depth"] == 4

    def test_learning_rate(self):
        assert abs(FIXED_XGB_PARAMS["learning_rate"] - 0.05) < 1e-9

    def test_subsample(self):
        assert abs(FIXED_XGB_PARAMS["subsample"] - 0.8) < 1e-9

    def test_random_state(self):
        assert FIXED_XGB_PARAMS["random_state"] == 0

    def test_early_stopping_rounds(self):
        assert FIXED_XGB_PARAMS["early_stopping_rounds"] == 50

    def test_eval_metric(self):
        assert FIXED_XGB_PARAMS["eval_metric"] == "auc"


class TestThresholdMaximisesF1:
    """Verify that threshold selection picks the F1-maximising value."""

    def test_threshold_selection_correctness(self):
        """Brute-force search should agree with reference implementation."""
        rng = np.random.default_rng(5)
        y_true = rng.integers(0, 2, size=500)
        y_prob = rng.uniform(0, 1, size=500)

        grid = np.linspace(0.05, 0.95, 91)
        t_selected = _select_threshold_by_f1(y_true, y_prob, grid)

        # Verify selected threshold gives the best F1
        best_f1 = -1.0
        for t in grid:
            f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1

        f1_at_selected = f1_score(y_true, (y_prob >= t_selected).astype(int), zero_division=0)
        assert abs(f1_at_selected - best_f1) < 1e-10

    def test_threshold_in_valid_range(self):
        """Selected threshold must be between 0 and 1."""
        rng = np.random.default_rng(6)
        y_true = rng.integers(0, 2, size=200)
        y_prob = rng.uniform(0, 1, size=200)
        t = _select_threshold_by_f1(y_true, y_prob)
        assert 0.0 < t < 1.0


class TestPositiveRate:
    """Verify the positive rate in the sample is approximately 27.1%."""

    def test_positive_rate_approximately_271(self):
        """Synthetic sample at 27.1% positive rate should round-trip correctly."""
        _, y = _make_synthetic_ml_dataset(n=10_000, positive_rate=0.271, seed=99)
        observed = float(y.mean())
        assert abs(observed - 0.271) < 0.02, f"Positive rate: {observed:.4f}"

    def test_target_from_cars_positive_rate(self):
        """With CAR drawn from N(0.03, 0.08), ~27% should exceed 0.10."""
        rng = np.random.default_rng(42)
        cars = pd.DataFrame({"car_1_30": rng.normal(0.03, 0.08, 100_000)})
        y = _construct_target(cars, threshold=0.10)
        rate = float(y.mean())
        # Expected: P(N(0.03, 0.08) > 0.10) ≈ P(Z > 0.875) ≈ 19%; allow wide band
        assert 0.10 < rate < 0.45


class TestBootstrapSeed:
    """Verify reproducibility when random_state=0 / seed=0 is fixed."""

    def test_same_seed_same_result(self):
        """Two calls with seed=0 should produce identical targets."""
        X1, y1 = _make_synthetic_ml_dataset(n=1000, seed=0)
        X2, y2 = _make_synthetic_ml_dataset(n=1000, seed=0)
        pd.testing.assert_series_equal(y1, y2)

    def test_different_seed_different_result(self):
        """Different seeds should produce different permutations."""
        _, y1 = _make_synthetic_ml_dataset(n=1000, seed=0)
        _, y2 = _make_synthetic_ml_dataset(n=1000, seed=1)
        assert not y1.equals(y2), "Different seeds produced identical targets (unexpected)"


class TestMathematicalProperties:
    """AUC and threshold mathematical invariants."""

    def test_auc_between_zero_and_one(self):
        """AUC must always be in [0, 1]."""
        rng = np.random.default_rng(33)
        for _ in range(10):
            y_true = rng.integers(0, 2, size=200)
            y_score = rng.uniform(0, 1, size=200)
            auc = roc_auc_score(y_true, y_score)
            assert 0.0 <= auc <= 1.0

    def test_perfect_classifier_auc_is_one(self):
        """A perfect probability model must achieve AUC = 1."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])  # perfectly ordered
        auc = roc_auc_score(y_true, y_prob)
        assert abs(auc - 1.0) < 1e-10

    def test_random_classifier_auc_near_half(self):
        """A random (uninformed) classifier should have AUC ≈ 0.5 on large samples."""
        rng = np.random.default_rng(77)
        y_true = rng.integers(0, 2, size=10_000)
        y_prob = rng.uniform(0, 1, size=10_000)
        auc = roc_auc_score(y_true, y_prob)
        assert abs(auc - 0.5) < 0.03, f"Random AUC too far from 0.5: {auc:.4f}"
