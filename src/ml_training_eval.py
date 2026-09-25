"""
ml_training_eval.py

Machine-learning prediction pipeline for the microcap insider purchase
disclosure study.

Pipeline overview
-----------------
Training window   : 2018 – 2022 (inclusive)
Validation window : 2023          (threshold selection)
Test window       : 2024

Binary target
-------------
y = 1  if  CAR_[1,30] > 0.10  else  0

All time splits respect chronological ordering to prevent look-ahead leakage.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────────────────────────────────
# Feature sets
# ────────────────────────────────────────────────────────────────────────────

BASE_FEATURES: list[str] = [
    'pct_from_52w_high',     # (price − 52w-high) / 52w-high  ≤ 0
    'log_mktcap',            # log(market cap in USD)
    'log_adv_30d',           # log(30-day avg daily dollar volume)
    'log_trade_value',       # log(total purchase consideration in USD)
    'shares_owned_pct',      # insider post-trade ownership as % of shares out.
    'filing_lag_days',       # calendar days from transaction date to filing date
    'vol_30d',               # 30-day realised volatility (annualised fraction)
    'is_officer',            # 1 if insider holds an officer role
    'is_director',           # 1 if insider holds a director role
    'is_10pct_owner',        # 1 if insider is a ≥ 10 % beneficial owner
    'mom_20d',               # 20-day raw price return prior to filing
    'purchase_pct_adv',      # trade value / 30-day ADV (liquidity-normalised size)
]

FULL_FEATURES: list[str] = BASE_FEATURES + [
    'mom_60d',               # 60-day raw price return prior to filing
    'mom_252d',              # 252-day raw price return prior to filing (1-year momentum)
    'pct_from_52w_low',      # (price − 52w-low) / 52w-low  ≥ 0
    'amihud_illiq_30d',      # Amihud (2002) illiquidity: mean(|ret| / dollar_vol)
    'bid_ask_pct',           # (ask − bid) / midpoint at filing date
    'beta_ff3_30d',          # market-beta from FF3 estimation on prior 30 trading days
    'alpha_ff3_30d',         # FF3 intercept (daily, annualised) from same window
    'smb_loading',           # SMB factor loading from FF3 estimation
    'hml_loading',           # HML factor loading from FF3 estimation
    'consecutive_purchase_days',  # # trading days with purchases in past 30 days
    'num_concurrent_buyers', # other insiders of same firm filing within ±5 days
    'days_since_last_purchase',   # calendar days since issuer's prior purchase filing
    'log_shares_outstanding',     # log(total shares outstanding)
    'turnover_ratio_30d',    # mean(daily volume / shares outstanding) over 30 days
    'is_preclose',           # 1 if filing was submitted before market close on t0
]

# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return class-1 probability predictions for any supported model type.

    Supports: ``xgb.Booster``, scikit-learn classifiers with
    ``predict_proba``, and scikit-learn ``Pipeline`` objects.
    """
    if isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X.values, feature_names=list(X.columns))
        return model.predict(dmat)
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    raise TypeError(f"Unsupported model type: {type(model)}")


def _year_mask(events: pd.DataFrame, years: list[int]) -> pd.Series:
    """Boolean mask selecting rows whose filing year is in *years*."""
    filing_years = pd.to_datetime(events['filing_date']).dt.year
    return filing_years.isin(years)


# ────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ────────────────────────────────────────────────────────────────────────────

def prepare_ml_dataset(
    events: pd.DataFrame,
    features: pd.DataFrame,
    cars: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct the feature matrix and binary outcome vector.

    Target definition
    -----------------
    y = 1  if  CAR_[1,30] > 0.10  else  0

    Parameters
    ----------
    events:
        Event-level metadata (must contain ``filing_date``).
        Index must be compatible with *features* and *cars*.
    features:
        Wide feature DataFrame whose columns include at least those in
        ``FULL_FEATURES``.  Missing FULL_FEATURES columns are filled with NaN.
    cars:
        Must contain ``CAR_1_30``.

    Returns
    -------
    (X, y) where X contains all FULL_FEATURES columns and y is a binary
    Series of dtype int.  Rows with missing CAR_1_30 are dropped.
    """
    if 'CAR_1_30' not in cars.columns:
        raise ValueError("cars must contain 'CAR_1_30'")

    y_raw = cars['CAR_1_30'].reindex(events.index)
    valid = y_raw.notna()
    y = (y_raw[valid] > 0.10).astype(int)
    y.name = 'target'

    # Build X with all FULL_FEATURES; add missing cols as NaN
    X = features.reindex(events.index).loc[valid, :].copy()
    for col in FULL_FEATURES:
        if col not in X.columns:
            X[col] = np.nan
    X = X[FULL_FEATURES]

    return X, y


# ────────────────────────────────────────────────────────────────────────────
# Cross-validation for boosting rounds
# ────────────────────────────────────────────────────────────────────────────

def time_ordered_cv_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    max_rounds: int = 300,
) -> int:
    """Select the number of XGBoost boosting rounds via time-ordered CV.

    Uses ``TimeSeriesSplit`` so later folds are strictly later in calendar time.
    Within each fold, XGBoost is trained with early stopping monitored against
    validation AUC; the best number of rounds from each fold is recorded and
    the median across folds is returned.

    Parameters
    ----------
    X_train:
        Feature matrix (rows must be in chronological order).
    y_train:
        Binary labels aligned with *X_train*.
    n_splits:
        Number of time-series CV folds (default 5).
    max_rounds:
        Maximum boosting rounds (hard cap; early stopping may select fewer).

    Returns
    -------
    Optimal integer number of boosting rounds (≥ 1).
    """
    params = {
        'max_depth':        5,
        'learning_rate':    0.1,
        'objective':        'binary:logistic',
        'eval_metric':      'auc',
        'seed':             42,
        'nthread':          1,
        'verbosity':        0,
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_rounds: list[int] = []

    X_arr = X_train.values.astype(float)
    y_arr = y_train.values.astype(float)

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_arr)):
        X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            warnings.warn(f"Fold {fold_idx}: single class in train or val; skipping.")
            continue

        d_tr  = xgb.DMatrix(X_tr,  label=y_tr,  feature_names=list(X_train.columns))
        d_val = xgb.DMatrix(X_val, label=y_val, feature_names=list(X_train.columns))

        evals_result: dict = {}
        booster = xgb.train(
            params,
            d_tr,
            num_boost_round=max_rounds,
            evals=[(d_val, 'val')],
            early_stopping_rounds=20,
            evals_result=evals_result,
            verbose_eval=False,
        )
        best_rounds.append(booster.best_iteration + 1)

    if not best_rounds:
        warnings.warn("No valid CV folds; defaulting to max_rounds.")
        return max_rounds

    return int(np.median(best_rounds))


# ────────────────────────────────────────────────────────────────────────────
# Model training
# ────────────────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_rounds: int,
) -> xgb.Booster:
    """Train XGBoost with the paper's fixed hyperparameters.

    Parameters
    ----------
    X_train:
        Feature matrix.  NaN values are handled natively by XGBoost.
    y_train:
        Binary labels.
    n_rounds:
        Number of boosting rounds (typically the output of
        :func:`time_ordered_cv_xgb`).

    Returns
    -------
    Fitted ``xgb.Booster``.
    """
    params = {
        'max_depth':     5,
        'learning_rate': 0.1,
        'objective':     'binary:logistic',
        'eval_metric':   'auc',
        'seed':          42,
        'nthread':       1,
        'verbosity':     0,
    }
    dtrain = xgb.DMatrix(
        X_train.values.astype(float),
        label=y_train.values.astype(float),
        feature_names=list(X_train.columns),
    )
    return xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train a Random Forest classifier with median imputation for missing values.

    Parameters
    ----------
    X_train:
        Feature matrix (NaN allowed; imputed by column median).
    y_train:
        Binary labels.

    Returns
    -------
    Fitted scikit-learn ``Pipeline`` (imputer → RandomForestClassifier).
    Hyperparameters: 100 trees, max_depth=10, random_state=42.
    """
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train an L2-penalised logistic regression with standardisation.

    Parameters
    ----------
    X_train:
        Feature matrix (NaN allowed; imputed by column median before scaling).
    y_train:
        Binary labels.

    Returns
    -------
    Fitted scikit-learn ``Pipeline``
    (imputer → StandardScaler → LogisticRegression).
    Hyperparameters: L2 penalty, C=1, max_iter=1000, solver='lbfgs'.
    """
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf', LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ────────────────────────────────────────────────────────────────────────────
# Threshold selection and evaluation
# ────────────────────────────────────────────────────────────────────────────

def select_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Choose the probability threshold that maximises F1 on the validation set.

    Scans 200 candidate thresholds in (0, 1) and returns the one with the
    highest macro-averaged F1 score.  Falls back to 0.5 when y_val is
    single-class or all predictions are identical.

    Parameters
    ----------
    model:
        Fitted model (``xgb.Booster`` or scikit-learn estimator).
    X_val:
        Validation feature matrix.
    y_val:
        True binary labels for the validation set.

    Returns
    -------
    Optimal threshold in (0, 1).
    """
    proba = _predict_proba(model, X_val)
    y_arr = y_val.values.astype(int)

    if len(np.unique(y_arr)) < 2:
        return 0.5

    thresholds = np.linspace(0.01, 0.99, 200)
    f1_scores = np.array([
        f1_score(y_arr, (proba >= t).astype(int), zero_division=0)
        for t in thresholds
    ])
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    model_name: str,
) -> dict:
    """Compute a comprehensive set of evaluation metrics on the test set.

    Metrics
    -------
    roc_auc          – area under the ROC curve
    pr_auc           – area under the Precision-Recall curve
    precision        – precision at *threshold*
    recall           – recall at *threshold*
    f1               – F1 score at *threshold*
    brier_score      – mean squared probability error
    brier_skill      – 1 − BS / BS_ref  (ref = climatological forecast)
    flagged_count    – number of observations predicted positive
    mean_outcome_flagged   – mean(y | ŷ = 1)
    mean_outcome_rejected  – mean(y | ŷ = 0)

    Parameters
    ----------
    model:
        Fitted model.
    X_test:
        Test feature matrix.
    y_test:
        True binary labels.
    threshold:
        Probability threshold (output of :func:`select_threshold`).
    model_name:
        Human-readable name stored in the returned dict.

    Returns
    -------
    dict with model_name plus all metrics listed above.
    """
    proba = _predict_proba(model, X_test)
    y_arr = y_test.values.astype(int)
    preds = (proba >= threshold).astype(int)

    p_bar = y_arr.mean()
    bs_ref = p_bar * (1.0 - p_bar)
    bs = brier_score_loss(y_arr, proba)

    flagged_mask = preds == 1
    return {
        'model_name':            model_name,
        'roc_auc':               float(roc_auc_score(y_arr, proba))   if len(np.unique(y_arr)) > 1 else np.nan,
        'pr_auc':                float(average_precision_score(y_arr, proba)) if len(np.unique(y_arr)) > 1 else np.nan,
        'precision':             float(precision_score(y_arr, preds, zero_division=0)),
        'recall':                float(recall_score(y_arr, preds, zero_division=0)),
        'f1':                    float(f1_score(y_arr, preds, zero_division=0)),
        'brier_score':           float(bs),
        'brier_skill':           float(1.0 - bs / bs_ref) if bs_ref > 0 else np.nan,
        'threshold':             threshold,
        'flagged_count':         int(flagged_mask.sum()),
        'mean_outcome_flagged':  float(y_arr[flagged_mask].mean())  if flagged_mask.any()  else np.nan,
        'mean_outcome_rejected': float(y_arr[~flagged_mask].mean()) if (~flagged_mask).any() else np.nan,
    }


# ────────────────────────────────────────────────────────────────────────────
# Issuer-cluster bootstrap for AUC
# ────────────────────────────────────────────────────────────────────────────

def issuer_cluster_bootstrap_auc(
    models: list,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    events_test: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Issuer-cluster bootstrap for AUC levels and pairwise differences.

    Each bootstrap iteration resamples unique issuers with replacement,
    then collects all test observations belonging to the sampled issuers.
    AUC is computed for each model on the resample.

    Parameters
    ----------
    models:
        Ordered list of fitted models.  Names are taken from ``model_name``
        keys if the list contains (name, model) tuples, otherwise generic
        indices are used.
    X_test:
        Test feature matrix (index must align with *events_test*).
    y_test:
        True binary labels.
    events_test:
        Must contain ``ticker`` column; index must align with *X_test*.
    n_boot:
        Number of bootstrap replications (default 1000).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict with structure::

        {
            model_name: {
                'auc_point': float,
                'auc_boot_mean': float,
                'ci_lower': float,   # 2.5th percentile
                'ci_upper': float,   # 97.5th percentile
            },
            'pairwise_diffs': {
                'model_A vs model_B': {
                    'delta_point': float,
                    'ci_lower': float,
                    'ci_upper': float,
                    'p_value': float,   # fraction of boot samples where delta ≤ 0
                }
            }
        }
    """
    rng = np.random.default_rng(seed)

    # Normalise model list to (name, model) tuples
    named_models: list[tuple[str, Any]] = []
    for i, m in enumerate(models):
        if isinstance(m, tuple) and len(m) == 2:
            named_models.append((str(m[0]), m[1]))
        else:
            named_models.append((f'model_{i}', m))

    tickers = events_test.reindex(X_test.index)['ticker'].values
    unique_tickers = np.unique(tickers)
    y_arr = y_test.reindex(X_test.index).values.astype(int)

    # Point-estimate AUCs
    point_aucs: dict[str, float] = {}
    for name, model in named_models:
        proba = _predict_proba(model, X_test)
        if len(np.unique(y_arr)) > 1:
            point_aucs[name] = float(roc_auc_score(y_arr, proba))
        else:
            point_aucs[name] = np.nan

    # Pre-compute full-sample probability arrays to avoid repeated prediction
    all_probas = {name: _predict_proba(model, X_test) for name, model in named_models}

    boot_aucs: dict[str, list[float]] = {name: [] for name, _ in named_models}

    for _ in range(n_boot):
        sampled = rng.choice(unique_tickers, size=len(unique_tickers), replace=True)
        # Build index mask: include all obs for sampled tickers (with repetition)
        obs_indices: list[int] = []
        for t in sampled:
            obs_indices.extend(np.where(tickers == t)[0].tolist())

        y_boot = y_arr[obs_indices]
        if len(np.unique(y_boot)) < 2:
            continue

        for name, _ in named_models:
            proba_boot = all_probas[name][obs_indices]
            boot_aucs[name].append(float(roc_auc_score(y_boot, proba_boot)))

    result: dict = {}
    for name, _ in named_models:
        arr = np.array(boot_aucs[name])
        result[name] = {
            'auc_point':     point_aucs[name],
            'auc_boot_mean': float(arr.mean()) if len(arr) > 0 else np.nan,
            'ci_lower':      float(np.percentile(arr, 2.5))  if len(arr) > 0 else np.nan,
            'ci_upper':      float(np.percentile(arr, 97.5)) if len(arr) > 0 else np.nan,
        }

    # Pairwise differences
    pairwise: dict = {}
    names = [n for n, _ in named_models]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            arr_a = np.array(boot_aucs[na])
            arr_b = np.array(boot_aucs[nb])
            n_common = min(len(arr_a), len(arr_b))
            if n_common == 0:
                continue
            deltas = arr_a[:n_common] - arr_b[:n_common]
            delta_point = point_aucs[na] - point_aucs[nb]
            p_val = float(np.mean(deltas <= 0))  # fraction where A not better than B
            pairwise[f'{na} vs {nb}'] = {
                'delta_point': delta_point,
                'ci_lower':    float(np.percentile(deltas, 2.5)),
                'ci_upper':    float(np.percentile(deltas, 97.5)),
                'p_value':     p_val,
            }

    result['pairwise_diffs'] = pairwise
    return result


# ────────────────────────────────────────────────────────────────────────────
# Full ML pipeline
# ────────────────────────────────────────────────────────────────────────────

def run_full_ml_pipeline(
    events: pd.DataFrame,
    features: pd.DataFrame,
    cars: pd.DataFrame,
) -> dict:
    """Orchestrate the complete ML pipeline with fixed train/val/test splits.

    Splits
    ------
    Train      : filing years 2018 – 2022
    Validation : filing year  2023  (threshold selection)
    Test       : filing year  2024

    Both BASE_FEATURES and FULL_FEATURES variants are evaluated for XGBoost.
    RF and Logistic are evaluated on FULL_FEATURES only.

    Returns
    -------
    dict with keys::

        {
            'metrics': pd.DataFrame,          # one row per model
            'bootstrap': dict,                # issuer-cluster bootstrap AUC
            'n_rounds_base': int,             # CV-selected rounds (base features)
            'n_rounds_full': int,             # CV-selected rounds (full features)
            'models': {model_name: fitted_model},
            'splits': {'train': idx, 'val': idx, 'test': idx},
        }
    """
    X_all, y_all = prepare_ml_dataset(events, features, cars)

    train_mask = _year_mask(events.reindex(X_all.index), list(range(2018, 2023)))
    val_mask   = _year_mask(events.reindex(X_all.index), [2023])
    test_mask  = _year_mask(events.reindex(X_all.index), [2024])

    X_tr, y_tr   = X_all.loc[train_mask], y_all.loc[train_mask]
    X_va, y_va   = X_all.loc[val_mask],   y_all.loc[val_mask]
    X_te, y_te   = X_all.loc[test_mask],  y_all.loc[test_mask]

    events_te = events.reindex(X_te.index)

    # ── XGBoost (base features) ──────────────────────────────────────────
    X_tr_b = X_tr[BASE_FEATURES]
    X_va_b = X_va[BASE_FEATURES]
    X_te_b = X_te[BASE_FEATURES]

    n_rounds_base = time_ordered_cv_xgb(X_tr_b, y_tr)
    xgb_base = train_xgboost(X_tr_b, y_tr, n_rounds_base)
    thresh_xgb_base = select_threshold(xgb_base, X_va_b, y_va)
    metrics_xgb_base = evaluate_model(xgb_base, X_te_b, y_te, thresh_xgb_base, 'XGBoost_base')

    # ── XGBoost (full features) ──────────────────────────────────────────
    n_rounds_full = time_ordered_cv_xgb(X_tr, y_tr)
    xgb_full = train_xgboost(X_tr, y_tr, n_rounds_full)
    thresh_xgb_full = select_threshold(xgb_full, X_va, y_va)
    metrics_xgb_full = evaluate_model(xgb_full, X_te, y_te, thresh_xgb_full, 'XGBoost_full')

    # ── Random Forest ────────────────────────────────────────────────────
    rf = train_random_forest(X_tr, y_tr)
    thresh_rf = select_threshold(rf, X_va, y_va)
    metrics_rf = evaluate_model(rf, X_te, y_te, thresh_rf, 'RandomForest')

    # ── Logistic Regression ──────────────────────────────────────────────
    lr = train_logistic(X_tr, y_tr)
    thresh_lr = select_threshold(lr, X_va, y_va)
    metrics_lr = evaluate_model(lr, X_te, y_te, thresh_lr, 'Logistic')

    metrics_df = pd.DataFrame([
        metrics_xgb_base, metrics_xgb_full, metrics_rf, metrics_lr,
    ]).set_index('model_name')

    fitted_models = {
        'XGBoost_base': ('XGBoost_base', xgb_base),
        'XGBoost_full': ('XGBoost_full', xgb_full),
        'RandomForest': ('RandomForest', rf),
        'Logistic':     ('Logistic', lr),
    }

    # Bootstrap on full-feature models for comparability
    boot_models = [
        ('XGBoost_full', xgb_full),
        ('RandomForest', rf),
        ('Logistic', lr),
    ]
    # XGBoost full needs full features; wrap X_te accordingly
    boot_X_te = X_te  # RF and LR use FULL_FEATURES; XGBoost_full also uses FULL_FEATURES

    bootstrap_results = issuer_cluster_bootstrap_auc(
        models=boot_models,
        X_test=boot_X_te,
        y_test=y_te,
        events_test=events_te,
    )

    return {
        'metrics':       metrics_df,
        'bootstrap':     bootstrap_results,
        'n_rounds_base': n_rounds_base,
        'n_rounds_full': n_rounds_full,
        'models':        fitted_models,
        'splits': {
            'train': X_tr.index,
            'val':   X_va.index,
            'test':  X_te.index,
        },
    }


# ────────────────────────────────────────────────────────────────────────────
# Rolling-origin out-of-sample tests
# ────────────────────────────────────────────────────────────────────────────

def run_rolling_origin_tests(
    events: pd.DataFrame,
    features: pd.DataFrame,
    cars: pd.DataFrame,
    test_years: list[int] = None,
) -> pd.DataFrame:
    """Evaluate all models under a rolling-origin (expanding-window) scheme.

    For each *test_year* t:
        - Train  : all events with filing year < t − 1
        - Val    : events with filing year == t − 1  (threshold selection)
        - Test   : events with filing year == t

    Parameters
    ----------
    events:
        Full event table; must contain ``filing_date``.
    features:
        Feature matrix aligned with *events*.
    cars:
        CAR table aligned with *events*.
    test_years:
        List of years to use as the test set.
        Defaults to [2020, 2021, 2022, 2023, 2024].

    Returns
    -------
    DataFrame with one row per (test_year, model_name) and columns for all
    metrics from :func:`evaluate_model` plus ``n_train``, ``n_val``, ``n_test``.
    """
    if test_years is None:
        test_years = [2020, 2021, 2022, 2023, 2024]

    X_all, y_all = prepare_ml_dataset(events, features, cars)
    filing_years = pd.to_datetime(
        events.reindex(X_all.index)['filing_date']
    ).dt.year

    rows: list[dict] = []

    for test_year in test_years:
        val_year = test_year - 1
        train_years = list(range(filing_years.min(), val_year))

        tr_mask  = filing_years.isin(train_years)
        va_mask  = filing_years == val_year
        te_mask  = filing_years == test_year

        if tr_mask.sum() < 10 or va_mask.sum() < 5 or te_mask.sum() < 5:
            warnings.warn(
                f"Skipping test_year={test_year}: insufficient data in one split."
            )
            continue

        X_tr, y_tr = X_all.loc[tr_mask], y_all.loc[tr_mask]
        X_va, y_va = X_all.loc[va_mask], y_all.loc[va_mask]
        X_te, y_te = X_all.loc[te_mask], y_all.loc[te_mask]

        model_specs: list[tuple[str, Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []

        # XGBoost (base)
        try:
            X_tr_b, X_va_b, X_te_b = X_tr[BASE_FEATURES], X_va[BASE_FEATURES], X_te[BASE_FEATURES]
            n_r = time_ordered_cv_xgb(X_tr_b, y_tr)
            m = train_xgboost(X_tr_b, y_tr, n_r)
            model_specs.append(('XGBoost_base', m, X_tr_b, X_va_b, X_te_b))
        except Exception as exc:
            warnings.warn(f"XGBoost_base failed for {test_year}: {exc}")

        # XGBoost (full)
        try:
            n_r = time_ordered_cv_xgb(X_tr, y_tr)
            m = train_xgboost(X_tr, y_tr, n_r)
            model_specs.append(('XGBoost_full', m, X_tr, X_va, X_te))
        except Exception as exc:
            warnings.warn(f"XGBoost_full failed for {test_year}: {exc}")

        # RF
        try:
            m = train_random_forest(X_tr, y_tr)
            model_specs.append(('RandomForest', m, X_tr, X_va, X_te))
        except Exception as exc:
            warnings.warn(f"RandomForest failed for {test_year}: {exc}")

        # Logistic
        try:
            m = train_logistic(X_tr, y_tr)
            model_specs.append(('Logistic', m, X_tr, X_va, X_te))
        except Exception as exc:
            warnings.warn(f"Logistic failed for {test_year}: {exc}")

        for mname, model, _, X_va_m, X_te_m in model_specs:
            try:
                thr = select_threshold(model, X_va_m, y_va)
                met = evaluate_model(model, X_te_m, y_te, thr, mname)
                met.update({
                    'test_year': test_year,
                    'n_train':   int(tr_mask.sum()),
                    'n_val':     int(va_mask.sum()),
                    'n_test':    int(te_mask.sum()),
                })
                rows.append(met)
            except Exception as exc:
                warnings.warn(f"{mname} evaluation failed for {test_year}: {exc}")

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .set_index(['test_year', 'model_name'])
        .sort_index()
    )


# ────────────────────────────────────────────────────────────────────────────
# Visualization
# ────────────────────────────────────────────────────────────────────────────

def plot_roc_pr_curves(
    models: list,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot ROC and Precision-Recall curves side-by-side for all models.

    Parameters
    ----------
    models:
        List of ``(name, fitted_model)`` tuples.
    X_test:
        Test feature matrix.
    y_test:
        True binary labels.
    figsize:
        Figure dimensions.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=figsize)
    palette = sns.color_palette('tab10', len(models))
    y_arr = y_test.values.astype(int)

    for (name, model), color in zip(models, palette):
        proba = _predict_proba(model, X_test)

        fpr, tpr, _ = roc_curve(y_arr, proba)
        auc_val = roc_auc_score(y_arr, proba)
        ax_roc.plot(fpr, tpr, color=color, label=f'{name} (AUC={auc_val:.3f})')

        prec, rec, _ = precision_recall_curve(y_arr, proba)
        pr_auc = average_precision_score(y_arr, proba)
        ax_pr.plot(rec, prec, color=color, label=f'{name} (PR-AUC={pr_auc:.3f})')

    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves')
    ax_roc.legend(fontsize=8)

    ax_pr.axhline(y_arr.mean(), color='k', linestyle='--', linewidth=0.8,
                  label=f'Baseline ({y_arr.mean():.3f})')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves')
    ax_pr.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_feature_importance(
    booster: xgb.Booster,
    top_n: int = 20,
    importance_type: str = 'gain',
    figsize: tuple[float, float] = (8, 7),
) -> plt.Figure:
    """Horizontal bar chart of XGBoost feature importance by *importance_type*.

    Parameters
    ----------
    booster:
        Fitted ``xgb.Booster``.
    top_n:
        Number of top features to display.
    importance_type:
        One of ``'weight'``, ``'gain'``, ``'cover'`` (default ``'gain'``).
    figsize:
        Figure dimensions.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    scores = booster.get_score(importance_type=importance_type)
    if not scores:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No feature importance available.', ha='center', va='center')
        return fig

    imp_df = (
        pd.Series(scores, name='importance')
        .sort_values(ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=figsize)
    imp_df.plot(kind='barh', ax=ax, color='steelblue', alpha=0.85)
    ax.set_xlabel(f'Feature importance ({importance_type})')
    ax.set_title(f'XGBoost top-{top_n} features')
    fig.tight_layout()
    return fig
