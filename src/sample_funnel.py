"""
sample_funnel.py
================
Implements the staged sample-construction pipeline for the microcap insider
purchase disclosure study.  Filters are applied in the exact sequence
described in the paper's methodology.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------

def _log_count(label: str, df: pd.DataFrame, funnel: dict) -> None:
    funnel[label] = len(df)
    logger.info("Funnel step %-55s  n=%d", label, len(df))


def apply_line_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Apply per-line purchase filters in the exact order specified by the paper.

    Filters applied sequentially:
    1. filing_date in [2018-01-01, 2024-12-31] AND transaction_date >= 2018-01-01
    2. reporting_lag in [0, 90] calendar days
    3. transaction_price > 0 AND quantity > 0 AND transaction_value > 0
    4. |transaction_value - price*qty| <= 1% of transaction_value
    5. transaction_value >= $5,000
    6. De-duplicate by (ticker, insider_name, transaction_date, transaction_price,
       quantity, owned), keeping earliest filing_date

    Parameters
    ----------
    df : pd.DataFrame
        Raw OpenInsider purchase DataFrame with columns:
        filing_date, transaction_date, ticker, insider_name, transaction_price,
        quantity, owned, transaction_value.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Filtered DataFrame and a dict mapping step label -> row count.
    """
    funnel: dict = {}
    df = df.copy()

    # Ensure date types
    for col in ("filing_date", "transaction_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    _log_count("0_raw_input", df, funnel)

    # ---- Step 1: date range ---------------------------------------------------
    mask = (
        (df["filing_date"] >= "2018-01-01")
        & (df["filing_date"] <= "2024-12-31")
        & (df["transaction_date"] >= "2018-01-01")
    )
    df = df[mask].copy()
    _log_count("1_date_range", df, funnel)

    # ---- Step 2: reporting lag [0, 90] days ----------------------------------
    df["reporting_lag"] = (df["filing_date"] - df["transaction_date"]).dt.days
    mask = (df["reporting_lag"] >= 0) & (df["reporting_lag"] <= 90)
    df = df[mask].copy()
    _log_count("2_reporting_lag_0_90d", df, funnel)

    # ---- Step 3: positive numeric fields ------------------------------------
    for col in ("transaction_price", "quantity", "transaction_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = (
        (df["transaction_price"] > 0)
        & (df["quantity"] > 0)
        & (df["transaction_value"] > 0)
    )
    df = df[mask].copy()
    _log_count("3_positive_price_qty_value", df, funnel)

    # ---- Step 4: consistency gate |tv - p*q| <= 1% of tv -------------------
    implied = df["transaction_price"] * df["quantity"]
    diff_pct = (df["transaction_value"] - implied).abs() / df["transaction_value"]
    df = df[diff_pct <= 0.01].copy()
    _log_count("4_price_qty_consistency_1pct", df, funnel)

    # ---- Step 5: transaction_value >= $5,000 --------------------------------
    df = df[df["transaction_value"] >= 5_000].copy()
    _log_count("5_min_transaction_value_5k", df, funnel)

    # ---- Step 6: de-duplicate keeping earliest filing -----------------------
    key_cols = [
        "ticker",
        "insider_name",
        "transaction_date",
        "transaction_price",
        "quantity",
        "owned",
    ]
    key_cols = [c for c in key_cols if c in df.columns]
    df = df.sort_values("filing_date").drop_duplicates(subset=key_cols, keep="first")
    _log_count("6_dedup_keep_earliest_filing", df, funnel)

    return df.reset_index(drop=True), funnel


# ---------------------------------------------------------------------------
# Stage 2/3: Build event sample
# ---------------------------------------------------------------------------

def _get_trading_days(yahoo_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return sorted trading-day index from a Yahoo price DataFrame."""
    idx = pd.DatetimeIndex(yahoo_df.index)
    return idx.sort_values()


def _find_t0(filing_date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Return last trading day on or before filing_date, or None if not found."""
    candidates = trading_days[trading_days <= filing_date]
    if len(candidates) == 0:
        return None
    return candidates[-1]


def _reconstruct_actual_close(
    close_series: pd.Series,
    splits_series: pd.Series,
    t0: pd.Timestamp,
) -> float:
    """Reconstruct actual (unadjusted for future splits) close at t0.

    Multiplies Yahoo Close at t0 by cumulative product of all split factors
    strictly after t0.

    Parameters
    ----------
    close_series : pd.Series
        Yahoo Close series indexed by date.
    splits_series : pd.Series
        Yahoo Stock Splits series indexed by date.
    t0 : pd.Timestamp
        Event anchor date.

    Returns
    -------
    float
        Actual close on current share basis at t0.
    """
    if t0 not in close_series.index:
        # Fallback to nearest prior date
        prior = close_series.index[close_series.index <= t0]
        if len(prior) == 0:
            return np.nan
        t0 = prior[-1]

    close_t0 = close_series.loc[t0]
    if pd.isna(close_t0):
        return np.nan

    # Cumulative split factors after t0
    future_splits = splits_series[splits_series.index > t0]
    future_splits = future_splits[future_splits != 0]
    if len(future_splits) > 0:
        cum_split = float(future_splits.prod())
        return float(close_t0) * cum_split
    return float(close_t0)


def _get_pit_shares(
    ticker: str,
    filing_date: pd.Timestamp,
    shares_df: pd.DataFrame,
    yahoo_profiles: pd.DataFrame,
    close_series: pd.Series,
    splits_series: pd.Series,
    t0: pd.Timestamp,
) -> float | None:
    """Determine point-in-time shares outstanding.

    Priority: latest 10-K/10-Q cover filed strictly before filing_date and
    no more than 400 days old. Fallback: Yahoo profile adjusted for split history.

    Parameters
    ----------
    ticker : str
    filing_date : pd.Timestamp
    shares_df : pd.DataFrame
        SEC XBRL shares with columns: ticker, filed_date, shares, form_type.
    yahoo_profiles : pd.DataFrame
        Indexed by ticker; column 'shares_outstanding'.
    close_series : pd.Series
        Yahoo Close indexed by date.
    splits_series : pd.Series
        Yahoo Stock Splits indexed by date.
    t0 : pd.Timestamp

    Returns
    -------
    float | None
    """
    # 1) Try SEC XBRL data
    if shares_df is not None and len(shares_df) > 0:
        ticker_shares = shares_df[
            (shares_df["ticker"] == ticker)
            & (shares_df["filed_date"] < filing_date)
            & (shares_df["form_type"].isin(["10-K", "10-Q"]))
        ].copy()
        if len(ticker_shares) > 0:
            ticker_shares = ticker_shares.sort_values("filed_date", ascending=False)
            latest = ticker_shares.iloc[0]
            age_days = (filing_date - latest["filed_date"]).days
            if age_days <= 400 and latest["shares"] > 0:
                return float(latest["shares"])

    # 2) Fallback: Yahoo profile snapshot adjusted by splits back to t0
    if yahoo_profiles is not None and ticker in yahoo_profiles.index:
        raw_shares = yahoo_profiles.loc[ticker, "shares_outstanding"]
        if pd.notna(raw_shares) and raw_shares > 0:
            # Adjust back: divide by split factors after t0
            future_splits = splits_series[splits_series.index > t0]
            future_splits = future_splits[future_splits != 0]
            if len(future_splits) > 0:
                cum_split = float(future_splits.prod())
                return float(raw_shares) / cum_split
            return float(raw_shares)

    return None


def _compute_addv_30d(
    close_series: pd.Series,
    volume_series: pd.Series,
    t0: pd.Timestamp,
    min_obs: int = 20,
) -> float | None:
    """Compute 30-day average daily dollar volume ending at t0."""
    prior = close_series.index[close_series.index <= t0]
    if len(prior) < min_obs:
        return None
    window_idx = prior[-30:]
    dollar_vol = close_series.loc[window_idx] * volume_series.loc[window_idx]
    obs = dollar_vol.dropna()
    if len(obs) < min_obs:
        return None
    return float(obs.mean())


def build_event_sample(
    filtered_df: pd.DataFrame,
    yahoo_panel: dict,
    ff3: pd.DataFrame,
    shares_df: pd.DataFrame | None,
    yahoo_profiles: pd.DataFrame | None,
    cache_dir: str = "data/cache",
) -> Tuple[pd.DataFrame, dict]:
    """Construct the final event sample from line-filtered purchase records.

    For each candidate purchase line:
    1. Determines event anchor t0 = last trading day on or before filing_date.
    2. Requires ticker presence in yahoo_panel and existence of t0.
    3. Requires >= 252 prior return rows before t0.
    4. Requires >= 30 future trading rows after t0.
    5. Reconstructs actual_close(t0) on current share basis.
    6. Determines point-in-time shares outstanding.
    7. Computes market_cap_pit and filters to [$30M, $500M].
    8. Computes ADDV_30d and filters to > $200K.
    9. Price-basis gate: actual_close(t0) / form4_price in [0.5, 2.0].
    10. Issuer-year same-day ratio median gate in [0.75, 1.33].

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Line-filtered purchase records from apply_line_filters().
    yahoo_panel : dict
        ticker -> pd.DataFrame (OHLCV + Dividends + Stock Splits).
    ff3 : pd.DataFrame
        Fama-French factors with DatetimeIndex.
    shares_df : pd.DataFrame | None
        SEC XBRL shares outstanding data.
    yahoo_profiles : pd.DataFrame | None
        Yahoo profile snapshot indexed by ticker.
    cache_dir : str

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Event DataFrame with added columns and a funnel count dict.
    """
    os.makedirs(cache_dir, exist_ok=True)
    funnel: dict = {}
    records = []

    total = len(filtered_df)
    logger.info("Building event sample from %d candidates...", total)

    for i, row in filtered_df.iterrows():
        ticker = row.get("ticker", "")
        filing_date = pd.Timestamp(row["filing_date"])

        # ---- Gate: ticker in yahoo_panel -------------------------------------
        if ticker not in yahoo_panel:
            continue
        ydf = yahoo_panel[ticker]
        if ydf is None or len(ydf) == 0:
            continue

        close_series = ydf["Close"].dropna()
        volume_series = ydf["Volume"].fillna(0)
        adj_close_series = ydf.get("Adj_Close", ydf["Close"]).dropna()
        high_series = ydf["High"].dropna()
        low_series = ydf["Low"].dropna()
        splits_series = ydf.get("Stock Splits", pd.Series(dtype=float)).fillna(0)

        trading_days = _get_trading_days(close_series.to_frame())

        # ---- Gate: t0 exists -------------------------------------------------
        t0 = _find_t0(filing_date, trading_days)
        if t0 is None:
            continue

        # ---- Gate: >= 252 prior rows -----------------------------------------
        prior_idx = trading_days[trading_days < t0]
        if len(prior_idx) < 252:
            continue

        # ---- Gate: >= 30 future rows -----------------------------------------
        future_idx = trading_days[trading_days > t0]
        if len(future_idx) < 30:
            continue

        # ---- Reconstruct actual close at t0 ---------------------------------
        actual_close_t0 = _reconstruct_actual_close(close_series, splits_series, t0)
        if np.isnan(actual_close_t0) or actual_close_t0 <= 0:
            continue

        # ---- Point-in-time shares -------------------------------------------
        pit_shares = _get_pit_shares(
            ticker,
            filing_date,
            shares_df if shares_df is not None else pd.DataFrame(),
            yahoo_profiles,
            close_series,
            splits_series,
            t0,
        )
        if pit_shares is None or pit_shares <= 0:
            continue

        # ---- Market cap gate [$30M, $500M] ----------------------------------
        market_cap_pit = actual_close_t0 * pit_shares
        if not (30e6 <= market_cap_pit <= 500e6):
            continue

        # ---- ADDV gate > $200K ----------------------------------------------
        addv_30d = _compute_addv_30d(close_series, volume_series, t0)
        if addv_30d is None or addv_30d <= 200_000:
            continue

        # ---- Price-basis gate [0.5, 2.0] ------------------------------------
        form4_price = float(row.get("transaction_price", np.nan))
        if np.isnan(form4_price) or form4_price <= 0:
            continue
        price_ratio = actual_close_t0 / form4_price
        if not (0.5 <= price_ratio <= 2.0):
            continue

        record = row.to_dict()
        record.update(
            {
                "t0": t0,
                "actual_close_t0": actual_close_t0,
                "pit_shares": pit_shares,
                "market_cap_pit": market_cap_pit,
                "addv_30d": addv_30d,
                "price_ratio_t0": price_ratio,
            }
        )
        records.append(record)

    events = pd.DataFrame(records).reset_index(drop=True)
    funnel["7_ticker_t0_prior_future_gates"] = len(events)
    logger.info("After ticker/t0/prior/future gates: %d", len(events))

    # ---- Issuer-year same-day ratio median gate ----------------------------
    if len(events) > 0 and "reporting_lag" in events.columns:
        events["issuer_year"] = (
            events["ticker"].astype(str)
            + "_"
            + events["filing_date"].dt.year.astype(str)
        )
        same_day = events[events["reporting_lag"] <= 2].copy()
        if len(same_day) >= 5:
            median_ratio = same_day.groupby("issuer_year")["price_ratio_t0"].transform(
                "median"
            )
            same_day_counts = events.groupby("issuer_year")["reporting_lag"].transform(
                lambda x: (x <= 2).sum()
            )
            mask_gate = (same_day_counts < 2) | (
                (median_ratio >= 0.75) & (median_ratio <= 1.33)
            )
            # Apply only to rows where we have sufficient same-day obs
            same_day_enough = same_day_counts >= 2
            combined = ~same_day_enough | mask_gate
            events = events[combined].copy()
    funnel["8_same_day_ratio_gate"] = len(events)
    logger.info("After same-day ratio gate: %d", len(events))

    return events.reset_index(drop=True), funnel
