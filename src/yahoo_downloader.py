"""
yahoo_downloader.py
===================
Yahoo Finance OHLCV panel downloader for the microcap insider-purchase
disclosure study.

All price data is stored with ``auto_adjust=False`` so that the raw
(unadjusted) close price and the adjusted close price are kept separately,
allowing precise reconstruction of what an investor actually paid or received
at any historical date.

Public API
----------
download_yahoo_ohlcv                – bulk download with parquet cache per ticker
load_yahoo_panel                    – load the cached parquet files
get_splits_for_ticker               – extract the split-factor time series
reconstruct_actual_close            – express a historical close on current-share basis
download_yahoo_profile_snapshots    – shares-outstanding / sector metadata
"""

import logging
import os
import pathlib
import time

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Columns that must appear in every cached OHLCV file.
_OHLCV_COLS = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
# Corporate-action columns stored alongside OHLCV.
_ACTION_COLS = ["Dividends", "Stock Splits"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalise_ohlcv(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten any MultiIndex columns and standardise column names.

    Newer versions of yfinance (≥ 0.2) return a MultiIndex ``(price_type,
    ticker)`` even for single-ticker downloads.  This function collapses those
    to a flat index and renames ``'Adj Close'`` → ``'Adj_Close'``.

    Parameters
    ----------
    raw : pd.DataFrame
        DataFrame returned directly by ``yf.download``.
    ticker : str
        The ticker symbol that was downloaded (used only for logging).

    Returns
    -------
    pd.DataFrame
        DataFrame with a flat column index and the index named ``'date'``.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        # Level 0 = price type, Level 1 = ticker symbol.
        raw.columns = raw.columns.get_level_values(0)

    raw.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    raw.index.name = "date"

    # Drop fully-empty rows (sometimes returned at the boundaries).
    raw = raw.dropna(how="all")

    logger.debug("Normalised %s: %d rows, columns=%s", ticker, len(raw), list(raw.columns))
    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_yahoo_ohlcv(
    tickers: list,
    start: str = "2017-01-01",
    end: str = "2025-03-01",
    cache_dir: str = "data/cache",
) -> dict:
    """Download OHLCV + corporate-action data from Yahoo Finance for each ticker.

    Each ticker's data is fetched with ``auto_adjust=False`` (so that raw and
    dividend-adjusted closes are stored separately) and ``actions=True`` (so
    that dividends and stock-split factors are included).  The result for each
    ticker is written as a parquet file at
    ``<cache_dir>/yahoo/<ticker>.parquet`` and also returned in the dict.

    If a parquet file already exists for a ticker it is **overwritten** with
    fresh data; use :func:`load_yahoo_panel` to retrieve cached data without
    re-downloading.

    Parameters
    ----------
    tickers : list
        List of ticker symbols (strings).  Symbols are uppercased before use.
    start : str
        Earliest date to request, ISO-8601 format ``'YYYY-MM-DD'``
        (default ``'2017-01-01'``).
    end : str
        Latest date to request, ISO-8601 format (default ``'2025-03-01'``).
    cache_dir : str
        Root cache directory.  The parquet files land in the ``yahoo/``
        subdirectory.

    Returns
    -------
    dict
        Mapping ``{ticker: pd.DataFrame}``.  Each DataFrame has a
        ``DatetimeIndex`` named ``'date'`` and columns that include at minimum
        ``Open``, ``High``, ``Low``, ``Close``, ``Adj_Close``, ``Volume``,
        ``Dividends``, and ``Stock Splits``.  Tickers with no data map to an
        empty DataFrame.
    """
    yahoo_cache = pathlib.Path(cache_dir) / "yahoo"
    os.makedirs(yahoo_cache, exist_ok=True)

    result: dict = {}

    for ticker in tickers:
        ticker = str(ticker).strip().upper()
        cache_file = yahoo_cache / f"{ticker}.parquet"

        logger.info("Downloading %s from Yahoo Finance (%s → %s) …", ticker, start, end)
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
            )

            if raw is None or raw.empty:
                logger.warning("Yahoo Finance returned no data for %s.", ticker)
                result[ticker] = pd.DataFrame()
                time.sleep(0.5)
                continue

            df = _normalise_ohlcv(raw, ticker)
            df.to_parquet(cache_file)
            logger.info("Cached %d rows for %s → %s", len(df), ticker, cache_file)
            result[ticker] = df

        except Exception as exc:
            logger.error("Failed to download %s: %s", ticker, exc)
            result[ticker] = pd.DataFrame()

        # Polite delay to avoid hitting Yahoo rate limits.
        time.sleep(0.5)

    return result


def load_yahoo_panel(
    tickers: list,
    cache_dir: str = "data/cache",
) -> dict:
    """Load previously cached OHLCV parquet files for the requested tickers.

    Parameters
    ----------
    tickers : list
        List of ticker symbols.  Symbols are uppercased before lookup.
    cache_dir : str
        Root cache directory that contains the ``yahoo/`` subdirectory.

    Returns
    -------
    dict
        Mapping ``{ticker: pd.DataFrame}``.  Tickers whose cache file does not
        exist map to an empty DataFrame; a warning is logged for each such
        ticker.
    """
    yahoo_cache = pathlib.Path(cache_dir) / "yahoo"
    result: dict = {}

    for ticker in tickers:
        ticker = str(ticker).strip().upper()
        cache_file = yahoo_cache / f"{ticker}.parquet"

        if cache_file.exists():
            result[ticker] = pd.read_parquet(cache_file)
            logger.debug("Loaded %s from %s (%d rows).", ticker, cache_file, len(result[ticker]))
        else:
            logger.warning("No cached data found for %s at %s.", ticker, cache_file)
            result[ticker] = pd.DataFrame()

    return result


def get_splits_for_ticker(
    ticker: str,
    cache_dir: str = "data/cache",
) -> pd.Series:
    """Return the non-trivial stock-split factor series for a ticker.

    Reads the ``'Stock Splits'`` column from the cached parquet file and
    returns only rows where the split factor is non-zero (i.e., an actual
    split event occurred on that date).

    Parameters
    ----------
    ticker : str
        Ticker symbol (case-insensitive).
    cache_dir : str
        Root cache directory containing the ``yahoo/`` subdirectory.

    Returns
    -------
    pd.Series
        DatetimeIndex → float split factor (e.g., 2.0 for a 2-for-1 split).
        Returns an empty Series if no splits exist or the cache file is
        missing.
    """
    ticker = str(ticker).strip().upper()
    cache_file = pathlib.Path(cache_dir) / "yahoo" / f"{ticker}.parquet"

    if not cache_file.exists():
        logger.warning("No cached data for %s; returning empty split series.", ticker)
        return pd.Series(dtype=float, name="Stock Splits")

    df = pd.read_parquet(cache_file)

    if "Stock Splits" not in df.columns:
        logger.debug("No 'Stock Splits' column in cached data for %s.", ticker)
        return pd.Series(dtype=float, name="Stock Splits")

    splits = df["Stock Splits"].dropna()
    splits = splits[splits != 0.0]
    splits.name = "Stock Splits"
    return splits


def reconstruct_actual_close(
    ticker: str,
    t0: pd.Timestamp,
    cache_dir: str = "data/cache",
) -> float:
    """Compute the actual (unadjusted) close at *t0* expressed on the current-share basis.

    Yahoo Finance stores raw (unadjusted) close prices when downloaded with
    ``auto_adjust=False``.  After a forward stock split the share count
    increases and the per-share price falls.  To express the *t0* price in the
    same per-share units that apply *today* (after all subsequent splits), we
    multiply the raw close at *t0* by the cumulative product of every split
    factor that occurred **after** *t0*.

    For example, if the raw close at *t0* was $200 and a 2-for-1 split
    occurred later, the reconstructed price is $200 × 2 = $400 – the price
    that would be quoted had the current share structure been in effect all
    along.

    Parameters
    ----------
    ticker : str
        Ticker symbol (case-insensitive).
    t0 : pd.Timestamp
        Target date.  If the exact date is not in the cache the most recent
        available trading day on or before *t0* is used.
    cache_dir : str
        Root cache directory containing the ``yahoo/`` subdirectory.

    Returns
    -------
    float
        Reconstructed close price on the current-share basis.

    Raises
    ------
    FileNotFoundError
        If no cached parquet file exists for *ticker*.
    ValueError
        If the cache contains no data on or before *t0*.
    """
    ticker = str(ticker).strip().upper()
    cache_file = pathlib.Path(cache_dir) / "yahoo" / f"{ticker}.parquet"

    if not cache_file.exists():
        raise FileNotFoundError(
            f"No cached OHLCV data for {ticker} at {cache_file}. "
            "Run download_yahoo_ohlcv first."
        )

    df = pd.read_parquet(cache_file)

    # Resolve the target date to the nearest available trading day.
    if t0 not in df.index:
        candidates = df.index[df.index <= t0]
        if candidates.empty:
            raise ValueError(
                f"No trading data on or before {t0} in the cache for {ticker}."
            )
        t0 = candidates[-1]
        logger.debug("Resolved t0 to nearest prior trading day: %s for %s.", t0, ticker)

    close_at_t0 = float(df.loc[t0, "Close"])

    # Collect all split events that occurred strictly after t0.
    splits = get_splits_for_ticker(ticker, cache_dir)
    future_splits = splits[splits.index > t0]

    if future_splits.empty:
        return close_at_t0

    cumulative_factor = float(future_splits.prod())
    reconstructed = close_at_t0 * cumulative_factor
    logger.debug(
        "%s: Close at %s = %.4f; cumulative split factor (post-t0) = %.4f; "
        "reconstructed = %.4f",
        ticker,
        t0.date(),
        close_at_t0,
        cumulative_factor,
        reconstructed,
    )
    return reconstructed


def download_yahoo_profile_snapshots(
    tickers: list,
    cache_path: str = "data/cache/yahoo_profiles.parquet",
) -> pd.DataFrame:
    """Download and cache key fundamental metadata for each ticker from Yahoo Finance.

    Calls ``yfinance.Ticker.info`` for each ticker to retrieve shares
    outstanding, sector, industry, market capitalisation, and company name.
    Missing or unavailable fields are stored as ``NaN`` / empty string so that
    the output is always a complete rectangular table.

    Parameters
    ----------
    tickers : list
        List of ticker symbols (case-insensitive).
    cache_path : str
        Full path to the parquet output file.  The parent directory is created
        automatically.  Any existing file at this path is **overwritten**.

    Returns
    -------
    pd.DataFrame
        One row per ticker.  Columns:
        ``ticker``, ``shares_outstanding``, ``sector``, ``industry``,
        ``market_cap``, ``company_name``.
    """
    cache = pathlib.Path(cache_path)
    os.makedirs(cache.parent, exist_ok=True)

    records: list[dict] = []

    for ticker in tickers:
        ticker = str(ticker).strip().upper()
        logger.info("Fetching Yahoo Finance profile for %s …", ticker)

        record: dict = {
            "ticker": ticker,
            "shares_outstanding": np.nan,
            "sector": "",
            "industry": "",
            "market_cap": np.nan,
            "company_name": "",
        }

        try:
            info = yf.Ticker(ticker).info
            record["shares_outstanding"] = info.get("sharesOutstanding", np.nan)
            record["sector"] = info.get("sector", "") or ""
            record["industry"] = info.get("industry", "") or ""
            record["market_cap"] = info.get("marketCap", np.nan)
            record["company_name"] = (
                info.get("longName") or info.get("shortName") or ""
            )
            logger.debug(
                "%s: shares=%.0f, sector='%s', industry='%s'",
                ticker,
                record["shares_outstanding"] if not np.isnan(record["shares_outstanding"]) else -1,
                record["sector"],
                record["industry"],
            )
        except Exception as exc:
            logger.warning(
                "Could not retrieve profile for %s (will use NaN/empty): %s",
                ticker,
                exc,
            )

        records.append(record)
        time.sleep(0.3)

    df = pd.DataFrame(records)
    df.to_parquet(cache, index=False)
    logger.info("Cached %d ticker profiles → %s", len(df), cache)

    return df
