"""
event_return_engine.py
======================
Computes abnormal returns, CARs, and BHARs for the microcap insider purchase
event study.

Key design principles (per paper methodology):
- FF3 betas estimated by OLS over [t0-252, t0-1] with intercept.
- Stock returns in estimation window winsorized at stock-specific 1st/99th %.
- PRIMARY abnormal return: AR = stock_excess - [beta_MKT*MktRF + beta_SMB*SMB +
  beta_HML*HML]  -- does NOT include estimated alpha in expected returns.
- INTERCEPT-INCLUSIVE: AR_alpha = AR - alpha_hat.
- Market-adjusted: AR_mkt = R_i - R_mkt.
- Drop events with CAR[1,30] < -1 or any 1-day price ratio >= 8 or <= 1/8.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Beta estimation
# ---------------------------------------------------------------------------

def estimate_ff3_betas(
    returns: pd.Series,
    ff3: pd.DataFrame,
    t0: pd.Timestamp,
    estimation_window: int = 252,
) -> dict:
    """Estimate FF3 factor loadings by OLS over [t0-252, t0-1].

    Parameters
    ----------
    returns : pd.Series
        Daily adjusted-close stock returns indexed by date (NOT excess).
    ff3 : pd.DataFrame
        Fama-French daily factors. Required columns: Mkt-RF, SMB, HML, RF.
        Values should be in decimal (not percentage).
    t0 : pd.Timestamp
        Event anchor date.
    estimation_window : int
        Number of prior trading days to use (default 252).

    Returns
    -------
    dict
        Keys: alpha, beta_mkt, beta_smb, beta_hml, r2, nobs.
        Returns NaN values if estimation fails.
    """
    null_result = {
        "alpha": np.nan,
        "beta_mkt": np.nan,
        "beta_smb": np.nan,
        "beta_hml": np.nan,
        "r2": np.nan,
        "nobs": 0,
    }

    # Estimation window: [t0-252, t0-1]
    prior_dates = returns.index[returns.index < t0]
    if len(prior_dates) < 60:  # Minimum required for reliable estimates
        return null_result

    window_dates = prior_dates[-estimation_window:]
    stock_rets = returns.loc[window_dates].copy()

    # Align FF3 factors
    try:
        ff3_window = ff3.loc[ff3.index.intersection(window_dates)].copy()
    except Exception:
        return null_result

    # Merge on common dates
    common = pd.concat(
        [stock_rets.rename("R_i"), ff3_window[["Mkt-RF", "SMB", "HML", "RF"]]],
        axis=1,
        join="inner",
    ).dropna()

    if len(common) < 30:
        return null_result

    # Compute excess returns
    excess_ret = common["R_i"] - common["RF"]

    # Winsorize stock excess returns at stock-specific 1st/99th percentiles
    p01 = float(excess_ret.quantile(0.01))
    p99 = float(excess_ret.quantile(0.99))
    excess_ret_wins = excess_ret.clip(lower=p01, upper=p99)

    # OLS with intercept
    X = sm.add_constant(common[["Mkt-RF", "SMB", "HML"]])
    try:
        model = sm.OLS(excess_ret_wins, X).fit()
        return {
            "alpha": float(model.params["const"]),
            "beta_mkt": float(model.params["Mkt-RF"]),
            "beta_smb": float(model.params["SMB"]),
            "beta_hml": float(model.params["HML"]),
            "r2": float(model.rsquared),
            "nobs": int(model.nobs),
        }
    except Exception as e:
        logger.debug("OLS failed for t0=%s: %s", t0, e)
        return null_result


# ---------------------------------------------------------------------------
# Abnormal returns for a single event
# ---------------------------------------------------------------------------

def compute_event_abnormal_returns(
    ticker: str,
    t0: pd.Timestamp,
    betas: dict,
    yahoo_panel: dict,
    ff3: pd.DataFrame,
    event_window: int = 61,
) -> Optional[pd.DataFrame]:
    """Compute daily abnormal returns for a single event over the event window.

    Event window: from t0 (day 0) through t0 + event_window trading days.
    Abnormal return series computed for days -1 through +60.

    Parameters
    ----------
    ticker : str
    t0 : pd.Timestamp
        Event anchor (last trading day on or before filing date).
    betas : dict
        Output of estimate_ff3_betas().
    yahoo_panel : dict
        ticker -> pd.DataFrame.
    ff3 : pd.DataFrame
        Fama-French daily factors.
    event_window : int
        Number of post-t0 trading days to compute (default 61 for days 1-60
        plus t0 itself).

    Returns
    -------
    pd.DataFrame | None
        Columns: date, R_i, AR, AR_alpha, AR_mkt, event_day.
        event_day=0 is t0, event_day=1 is the first post-filing trading day.
    """
    if ticker not in yahoo_panel:
        return None

    ydf = yahoo_panel[ticker]
    if ydf is None or len(ydf) == 0:
        return None

    adj_close = ydf.get("Adj_Close", ydf["Close"]).dropna()
    raw_close = ydf["Close"].dropna()

    # Get trading days from t0-1 onward (need t0-1 to compute day 0 return)
    all_dates = adj_close.index.sort_values()
    t0_loc = all_dates.get_loc(t0) if t0 in all_dates else None
    if t0_loc is None:
        # Find nearest prior
        prior = all_dates[all_dates <= t0]
        if len(prior) == 0:
            return None
        t0 = prior[-1]
        t0_loc = all_dates.get_loc(t0)

    # Window: from t0 (day 0) through t0 + event_window
    end_loc = min(t0_loc + event_window, len(all_dates) - 1)
    # Include day -1 for return computation at day 0
    start_loc = max(0, t0_loc - 1)

    window_dates = all_dates[start_loc : end_loc + 1]
    prices = adj_close.loc[window_dates]

    # Raw stock returns
    stock_rets = prices.pct_change().iloc[1:]  # Drop first NaN

    # FF3 factors aligned
    try:
        ff3_window = ff3.loc[ff3.index.intersection(stock_rets.index)].copy()
    except Exception:
        return None

    common = pd.concat(
        [stock_rets.rename("R_i"), ff3_window[["Mkt-RF", "SMB", "HML", "RF"]]],
        axis=1,
        join="inner",
    ).dropna()

    if len(common) == 0:
        return None

    alpha = betas.get("alpha", np.nan)
    b_mkt = betas.get("beta_mkt", np.nan)
    b_smb = betas.get("beta_smb", np.nan)
    b_hml = betas.get("beta_hml", np.nan)

    # Primary AR: loadings-only (no alpha)
    if all(not np.isnan(v) for v in [b_mkt, b_smb, b_hml]):
        expected_excess = (
            b_mkt * common["Mkt-RF"]
            + b_smb * common["SMB"]
            + b_hml * common["HML"]
        )
        common["AR"] = (common["R_i"] - common["RF"]) - expected_excess
    else:
        common["AR"] = np.nan

    # Intercept-inclusive: AR_alpha = AR - alpha_hat
    if not np.isnan(alpha):
        common["AR_alpha"] = common["AR"] - alpha
    else:
        common["AR_alpha"] = common["AR"]

    # Market-adjusted
    common["AR_mkt"] = common["R_i"] - (common["Mkt-RF"] + common["RF"])

    # Event day offset (0 = t0, 1 = first post-filing day, ...)
    t0_pos = common.index.get_loc(t0) if t0 in common.index else None
    if t0_pos is None:
        common["event_day"] = range(1, len(common) + 1)
    else:
        common["event_day"] = range(-t0_pos, len(common) - t0_pos)

    result = common[["R_i", "AR", "AR_alpha", "AR_mkt", "event_day"]].copy()
    result.index.name = "date"
    return result


# ---------------------------------------------------------------------------
# Build AR matrix across all events
# ---------------------------------------------------------------------------

def build_ar_matrix(
    events: pd.DataFrame,
    yahoo_panel: dict,
    ff3: pd.DataFrame,
    betas_map: dict,
    event_window: int = 61,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute AR path matrices for all events.

    Parameters
    ----------
    events : pd.DataFrame
        Final event sample; must have columns ticker, t0.
    yahoo_panel : dict
    ff3 : pd.DataFrame
    betas_map : dict
        event_idx -> betas dict from estimate_ff3_betas().
    event_window : int

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ar_matrix: rows=events, cols=event_day (integer day offsets)
        ar_alpha_matrix: same but using AR_alpha (intercept-inclusive)
    """
    ar_rows = {}
    ar_alpha_rows = {}

    for idx, row in events.iterrows():
        ticker = row["ticker"]
        t0 = pd.Timestamp(row["t0"])
        betas = betas_map.get(idx, {})

        ar_df = compute_event_abnormal_returns(
            ticker, t0, betas, yahoo_panel, ff3, event_window
        )
        if ar_df is None:
            continue

        ar_pivot = ar_df.set_index("event_day")["AR"]
        ar_alpha_pivot = ar_df.set_index("event_day")["AR_alpha"]
        ar_rows[idx] = ar_pivot
        ar_alpha_rows[idx] = ar_alpha_pivot

    ar_matrix = pd.DataFrame(ar_rows).T  # shape: (n_events, n_days)
    ar_alpha_matrix = pd.DataFrame(ar_alpha_rows).T
    return ar_matrix, ar_alpha_matrix


def estimate_all_betas(
    events: pd.DataFrame,
    yahoo_panel: dict,
    ff3: pd.DataFrame,
    estimation_window: int = 252,
) -> dict:
    """Estimate FF3 betas for all events.

    Parameters
    ----------
    events : pd.DataFrame
        Event sample with ticker, t0 columns.
    yahoo_panel : dict
    ff3 : pd.DataFrame
    estimation_window : int

    Returns
    -------
    dict
        event_idx -> betas dict.
    """
    betas_map = {}
    total = len(events)
    logger.info("Estimating betas for %d events...", total)

    for i, (idx, row) in enumerate(events.iterrows()):
        ticker = row["ticker"]
        t0 = pd.Timestamp(row["t0"])

        ydf = yahoo_panel.get(ticker)
        if ydf is None:
            betas_map[idx] = {}
            continue

        adj_close = ydf.get("Adj_Close", ydf["Close"]).dropna()
        stock_rets = adj_close.pct_change().dropna()

        betas_map[idx] = estimate_ff3_betas(stock_rets, ff3, t0, estimation_window)

        if (i + 1) % 500 == 0:
            logger.info("  ... %d / %d betas estimated", i + 1, total)

    logger.info("Beta estimation complete.")
    return betas_map


# ---------------------------------------------------------------------------
# CAR computation and event-level cleaning
# ---------------------------------------------------------------------------

def compute_cars(
    ar_matrix: pd.DataFrame,
    ar_alpha_matrix: pd.DataFrame,
    windows: List[Tuple[int, int]] = [(1, 20), (1, 30), (1, 60)],
) -> pd.DataFrame:
    """Compute CARs over specified windows for all events.

    Parameters
    ----------
    ar_matrix : pd.DataFrame
        Rows=events, columns=event_day integers. AR values (loadings-only).
    ar_alpha_matrix : pd.DataFrame
        Same but intercept-inclusive AR.
    windows : list of (start, end) event-day tuples.

    Returns
    -------
    pd.DataFrame
        One row per event, with columns like CAR_1_30, CAR_alpha_1_30, etc.
    """
    car_records = {}
    for start, end in windows:
        days = list(range(start, end + 1))
        avail_cols = [d for d in days if d in ar_matrix.columns]
        car_col = f"CAR_{start}_{end}"
        car_alpha_col = f"CAR_alpha_{start}_{end}"
        car_records[car_col] = ar_matrix[avail_cols].sum(axis=1, min_count=1)
        if len(ar_alpha_matrix) > 0:
            avail_alpha = [d for d in days if d in ar_alpha_matrix.columns]
            car_records[car_alpha_col] = ar_alpha_matrix[avail_alpha].sum(axis=1, min_count=1)

    return pd.DataFrame(car_records)


def drop_extreme_events(
    events: pd.DataFrame,
    cars: pd.DataFrame,
    yahoo_panel: dict,
    car_col: str = "CAR_1_30",
    car_floor: float = -1.0,
    price_ratio_max: float = 8.0,
    price_ratio_min: float = 1 / 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop events with extreme CAR or price-ratio spikes.

    Rules:
    - Drop if CAR[1,30] < -1.0
    - Drop if any 1-day price ratio >= 8.0 or <= 1/8 in the event window

    Parameters
    ----------
    events : pd.DataFrame
    cars : pd.DataFrame
    yahoo_panel : dict
    car_col : str
    car_floor : float
    price_ratio_max : float
    price_ratio_min : float

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Cleaned events and cars.
    """
    keep_mask = pd.Series(True, index=events.index)

    # Drop events with extreme CAR
    if car_col in cars.columns:
        bad_car = cars[car_col] < car_floor
        keep_mask &= ~bad_car.reindex(events.index, fill_value=False)
        logger.info("Dropping %d events with %s < %.1f", bad_car.sum(), car_col, car_floor)

    # Drop events with extreme intra-window price ratios
    spike_count = 0
    for idx, row in events[keep_mask].iterrows():
        ticker = row["ticker"]
        t0 = pd.Timestamp(row["t0"])
        ydf = yahoo_panel.get(ticker)
        if ydf is None:
            continue

        raw_close = ydf["Close"].dropna()
        future = raw_close[raw_close.index > t0].iloc[:30]
        if len(future) < 2:
            continue

        daily_ratios = future / future.shift(1)
        daily_ratios = daily_ratios.dropna()
        if (daily_ratios >= price_ratio_max).any() or (daily_ratios <= price_ratio_min).any():
            keep_mask.loc[idx] = False
            spike_count += 1

    logger.info("Dropping %d events with extreme price ratios in window", spike_count)

    clean_events = events[keep_mask].copy()
    clean_cars = cars.loc[cars.index.intersection(clean_events.index)].copy()
    return clean_events, clean_cars


# ---------------------------------------------------------------------------
# BHAR computation
# ---------------------------------------------------------------------------

def compute_bhar(
    events: pd.DataFrame,
    yahoo_panel: dict,
    ff3: pd.DataFrame,
    betas_map: dict,
    window_start: int = 1,
    window_end: int = 30,
) -> pd.Series:
    """Compute Buy-and-Hold Abnormal Returns (BHAR) for each event.

    BHAR = prod(1 + R_i) - prod(1 + R_benchmark)
    Benchmark uses loadings-only FF3 expected returns.

    Parameters
    ----------
    events : pd.DataFrame
    yahoo_panel : dict
    ff3 : pd.DataFrame
    betas_map : dict
    window_start : int
        First event day (e.g. 1).
    window_end : int
        Last event day (e.g. 30).

    Returns
    -------
    pd.Series
        BHAR indexed by event index.
    """
    bhar_values = {}
    for idx, row in events.iterrows():
        ticker = row["ticker"]
        t0 = pd.Timestamp(row["t0"])
        betas = betas_map.get(idx, {})

        ar_df = compute_event_abnormal_returns(
            ticker, t0, betas, yahoo_panel, ff3, event_window=window_end + 5
        )
        if ar_df is None:
            bhar_values[idx] = np.nan
            continue

        window_df = ar_df[
            (ar_df["event_day"] >= window_start) & (ar_df["event_day"] <= window_end)
        ]
        if len(window_df) == 0:
            bhar_values[idx] = np.nan
            continue

        cum_stock = float((1 + window_df["R_i"]).prod()) - 1
        # Benchmark: risk-free + factor expected
        b_mkt = betas.get("beta_mkt", np.nan)
        b_smb = betas.get("beta_smb", np.nan)
        b_hml = betas.get("beta_hml", np.nan)

        try:
            ff3_window = ff3.loc[ff3.index.intersection(window_df.index)]
            expected = (
                ff3_window["RF"]
                + b_mkt * ff3_window["Mkt-RF"]
                + b_smb * ff3_window["SMB"]
                + b_hml * ff3_window["HML"]
            )
            cum_bench = float((1 + expected).prod()) - 1
            bhar_values[idx] = cum_stock - cum_bench
        except Exception:
            bhar_values[idx] = np.nan

    return pd.Series(bhar_values, name=f"BHAR_{window_start}_{window_end}")
