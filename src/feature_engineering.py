"""
feature_engineering.py
=======================
Constructs filing-day information-set features for each event in the
microcap insider purchase sample.

All features are computed using only information available by filing-day
close (t0).  Historical purchase counts use only lines filed strictly before
the current filing.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trailing_window(
    series: pd.Series, t0: pd.Timestamp, n: int, inclusive: bool = True
) -> pd.Series:
    """Return the last n observations of series on or before t0 (inclusive).

    Parameters
    ----------
    series : pd.Series
        Time series indexed by date.
    t0 : pd.Timestamp
    n : int
        Number of trading-day observations to include.
    inclusive : bool
        If True, include t0 itself.
    """
    idx = series.index
    if inclusive:
        mask = idx <= t0
    else:
        mask = idx < t0
    prior = series[mask]
    return prior.iloc[-n:] if len(prior) >= n else prior


def _annualized_vol(log_returns: pd.Series) -> float:
    """Annualised volatility from daily log returns (std * sqrt(252))."""
    return float(log_returns.std() * np.sqrt(252))


def _simple_return(prices: pd.Series, t0: pd.Timestamp, n_days: int) -> Optional[float]:
    """Compute n_days simple return ending at t0.

    Returns (close_t0 / close_t0-n) - 1, or NaN if insufficient data.
    """
    prior = prices[prices.index <= t0]
    if len(prior) < n_days + 1:
        return np.nan
    p_end = float(prior.iloc[-1])
    p_start = float(prior.iloc[-n_days - 1])
    if p_start <= 0:
        return np.nan
    return p_end / p_start - 1


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_features(
    events: pd.DataFrame,
    yahoo_panel: dict,
    raw_purchase_universe: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all filing-day features for each event.

    Features use only information available by t0 (filing-day close).
    Historical purchase counts reference only lines filed strictly before the
    current filing date.

    Parameters
    ----------
    events : pd.DataFrame
        Final event sample with columns: ticker, filing_date, t0,
        actual_close_t0, transaction_price, quantity, owned,
        transaction_value, insider_name, insider_title, market_cap_pit,
        addv_30d.
    yahoo_panel : dict
        ticker -> pd.DataFrame with OHLCV + Dividends + Stock Splits.
    raw_purchase_universe : pd.DataFrame
        Complete raw purchase universe (all scraped lines) for history lookups.

    Returns
    -------
    pd.DataFrame
        events DataFrame with feature columns appended.
    """
    feature_rows = []
    total = len(events)
    logger.info("Computing features for %d events...", total)

    # Pre-process raw universe for historical lookups
    hist = raw_purchase_universe.copy() if raw_purchase_universe is not None else pd.DataFrame()
    if len(hist) > 0:
        hist["filing_date"] = pd.to_datetime(hist["filing_date"], errors="coerce")
        hist["transaction_date"] = pd.to_datetime(
            hist["transaction_date"], errors="coerce"
        )

    for idx, row in events.iterrows():
        ticker = row["ticker"]
        t0 = pd.Timestamp(row["t0"])
        filing_date = pd.Timestamp(row["filing_date"])
        actual_close = float(row.get("actual_close_t0", np.nan))
        form4_price = float(row.get("transaction_price", np.nan))
        insider_name = row.get("insider_name", "")
        insider_title = str(row.get("insider_title", "")).lower()

        feats: dict = {}

        # --- Yahoo data for this ticker ---
        ydf = yahoo_panel.get(ticker)
        if ydf is None or len(ydf) == 0:
            feature_rows.append(feats)
            continue

        adj_close = ydf.get("Adj_Close", ydf["Close"]).dropna()
        raw_close = ydf["Close"].dropna()
        raw_high = ydf["High"].dropna()
        raw_low = ydf["Low"].dropna()
        splits = ydf.get("Stock Splits", pd.Series(dtype=float)).fillna(0)

        # Build split-adjusted high/low for trailing window
        # Denominator uses trailing split-adjusted highs/lows
        def _split_adjust_series(series: pd.Series, ref_date: pd.Timestamp) -> pd.Series:
            """Adjust series values to current share basis by applying split factors after ref_date."""
            # For each date d, multiply value by product of splits after d up to ref_date
            # This converts historical prices to current share equivalent
            result = series.copy().astype(float)
            future_splits = splits[(splits.index > ref_date) & (splits != 0)]
            if len(future_splits) > 0:
                cum_factor = float(future_splits.prod())
                result = result * cum_factor
            return result

        # --- pct_from_52w_high ---
        trailing_high = _trailing_window(raw_high, t0, 252)
        if len(trailing_high) > 0:
            adj_high_series = _split_adjust_series(trailing_high, t0)
            max_high_52w = float(adj_high_series.max())
            feats["pct_from_52w_high"] = (
                actual_close / max_high_52w - 1 if max_high_52w > 0 else np.nan
            )
        else:
            feats["pct_from_52w_high"] = np.nan

        # --- pct_from_52w_low ---
        trailing_low = _trailing_window(raw_low, t0, 252)
        if len(trailing_low) > 0:
            adj_low_series = _split_adjust_series(trailing_low, t0)
            min_low_52w = float(adj_low_series.min())
            feats["pct_from_52w_low"] = (
                actual_close / min_low_52w - 1 if min_low_52w > 0 else np.nan
            )
        else:
            feats["pct_from_52w_low"] = np.nan

        # --- price_deviation ---
        feats["price_deviation"] = (
            actual_close / form4_price - 1
            if form4_price > 0
            else np.nan
        )

        # --- volatility_30d ---
        trail_adj = _trailing_window(adj_close, t0, 31)
        if len(trail_adj) >= 2:
            log_rets = np.log(trail_adj / trail_adj.shift(1)).dropna()
            feats["volatility_30d"] = _annualized_vol(log_rets)
        else:
            feats["volatility_30d"] = np.nan

        # --- return_mtd (calendar month-to-date) ---
        month_start = t0.replace(day=1)
        prior_month_close = adj_close[adj_close.index < month_start]
        cur_month_close = adj_close[adj_close.index <= t0]
        if len(prior_month_close) > 0 and len(cur_month_close) > 0:
            feats["return_mtd"] = (
                float(cur_month_close.iloc[-1]) / float(prior_month_close.iloc[-1]) - 1
            )
        else:
            feats["return_mtd"] = np.nan

        # --- return_5d ---
        feats["return_5d"] = _simple_return(adj_close, t0, 5)

        # --- return_21d ---
        feats["return_21d"] = _simple_return(adj_close, t0, 21)

        # --- Log transaction value ---
        tv = float(row.get("transaction_value", np.nan))
        feats["log_transaction_value"] = np.log(tv) if tv > 0 else np.nan

        # --- market_cap_pit and addv_30d (carry from events) ---
        feats["market_cap_pit"] = row.get("market_cap_pit", np.nan)
        feats["log_market_cap"] = (
            np.log(feats["market_cap_pit"]) if feats["market_cap_pit"] > 0 else np.nan
        )
        feats["addv_30d"] = row.get("addv_30d", np.nan)
        feats["log_addv_30d"] = (
            np.log(feats["addv_30d"]) if (feats["addv_30d"] or 0) > 0 else np.nan
        )

        # --- Insider role dummies ---
        feats["is_ceo"] = int("ceo" in insider_title or "chief executive" in insider_title)
        feats["is_cfo"] = int("cfo" in insider_title or "chief financial" in insider_title)
        feats["is_director"] = int("director" in insider_title)
        feats["is_officer"] = int(
            any(w in insider_title for w in ("officer", "ceo", "cfo", "coo", "cto", "president", "vp", "vice"))
        )

        # --- Historical purchase features from raw universe ---
        if len(hist) > 0:
            prior_filings = hist[
                (hist["ticker"] == ticker)
                & (hist["filing_date"] < filing_date)
            ]

            # is_first_purchase_12m
            cutoff_12m = filing_date - pd.Timedelta(days=365)
            recent = prior_filings[prior_filings["filing_date"] >= cutoff_12m]
            feats["is_first_purchase_12m"] = int(len(recent) == 0)

            # purchase_count_12m
            feats["purchase_count_12m"] = len(recent)

            # days_since_last_purchase
            if len(prior_filings) > 0:
                last_filing = prior_filings["filing_date"].max()
                feats["days_since_last_purchase"] = (filing_date - last_filing).days
            else:
                feats["days_since_last_purchase"] = np.nan

            # other-insider clustering: insiders buying same issuer within 30 days before
            window_30d = hist[
                (hist["ticker"] == ticker)
                & (hist["filing_date"] >= filing_date - pd.Timedelta(days=30))
                & (hist["filing_date"] < filing_date)
                & (hist["insider_name"] != insider_name)
            ]
            feats["other_insider_cluster"] = len(window_30d)

            # value_vs_avg_ratio for same insider
            insider_hist = hist[
                (hist["insider_name"] == insider_name)
                & (hist["filing_date"] < filing_date)
            ]
            if len(insider_hist) > 0 and "transaction_value" in insider_hist.columns:
                avg_val = float(insider_hist["transaction_value"].mean())
                feats["value_vs_avg_ratio"] = tv / avg_val if avg_val > 0 else np.nan
            else:
                feats["value_vs_avg_ratio"] = np.nan
        else:
            feats["is_first_purchase_12m"] = np.nan
            feats["purchase_count_12m"] = np.nan
            feats["days_since_last_purchase"] = np.nan
            feats["other_insider_cluster"] = np.nan
            feats["value_vs_avg_ratio"] = np.nan

        # --- filing_lag_days ---
        feats["filing_lag_days"] = row.get("reporting_lag", np.nan)

        # --- pct_of_holdings ---
        owned = float(row.get("owned", 0) or 0)
        qty = float(row.get("quantity", 0) or 0)
        if owned > qty and qty > 0:
            feats["pct_of_holdings"] = qty / (owned - qty)
        else:
            feats["pct_of_holdings"] = np.nan

        # --- pre_close indicator (filing acceptance before 16:00 ET) ---
        filing_ts = row.get("filing_timestamp", None)
        if filing_ts is not None:
            try:
                ts = pd.Timestamp(filing_ts)
                feats["pre_close"] = int(ts.hour < 16)
            except Exception:
                feats["pre_close"] = np.nan
        else:
            feats["pre_close"] = np.nan

        # --- biotech indicator ---
        sector = str(row.get("sector", "")).lower()
        industry = str(row.get("industry", "")).lower()
        biotech_kws = ("biotech", "biotechnology", "pharmaceutical", "pharma", "bioscience")
        feats["biotech"] = int(any(kw in sector or kw in industry for kw in biotech_kws))

        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows, index=events.index)
    result = pd.concat([events, feat_df], axis=1)
    logger.info("Feature computation complete. Shape: %s", result.shape)
    return result
