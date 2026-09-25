"""
raw_data_ingest.py
==================
OpenInsider scraper and Fama-French factor loader for the microcap insider-
purchase disclosure study.

Public API
----------
scrape_openinsider_monthly          – one month of insider purchases/sales
scrape_openinsider_sales_monthly    – convenience wrapper for code-S sales
build_raw_purchase_universe         – multi-year purchase panel with parquet cache
build_raw_sales_universe            – multi-year sales panel with parquet cache
load_ff3_factors                    – Fama-French 3-factor daily returns
"""

import io
import logging
import os
import pathlib
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OpenInsider screener URL template.  Parameters:
#   {from_date} / {to_date}   – ISO-8601 dates bounding the filing-date range
#   {fd} / {td}               – same dates for the second occurrence in the
#                               query string (OpenInsider uses them as filters
#                               for both the filing-date and trade-date fields)
#   {tcp}                     – transaction code ('P' = purchase, 'S' = sale)
_OPENINSIDER_URL = (
    "https://openinsider.com/screener?"
    "s=&o=&pl=&ph=&ll=&lh="
    "&fd={from_date}&td={to_date}"
    "&td=&tcp={tcp}"
    "&is=&myOnly=0"
    "&fd={fd}&td={td}"
    "&sh=&ch=&vl=&vh=&ocl=&och="
    "&isofficer=&isdirector=&is10percent=&isother="
    "&bf=&bh="
    "&format=csv"
)

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Mapping of OpenInsider CSV column names to the study's canonical names.
_COL_MAP = {
    "Filing Date": "filing_date",
    "Trade Date": "transaction_date",
    "Ticker": "ticker",
    "Company Name": "company_name",
    "Insider Name": "insider_name",
    "Title": "insider_title",
    "Trade Type": "trade_type",
    "Price": "transaction_price",
    "Qty": "quantity",
    "Owned": "owned",
    "\u0394Own": "delta_own",   # ΔOwn
    "Value": "transaction_value",
}

# Columns that every output DataFrame must expose.
_REQUIRED_OUT = [
    "filing_date",
    "transaction_date",
    "ticker",
    "insider_name",
    "insider_title",
    "transaction_price",
    "quantity",
    "owned",
    "transaction_value",
    "filing_timestamp",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _last_day_of_month(year: int, month: int) -> int:
    """Return the last calendar day (int) of the given year/month."""
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    return (next_month_first - timedelta(days=1)).day


def _fetch_with_retry(
    url: str,
    max_retries: int = 5,
    backoff_base: float = 2.0,
    timeout: int = 30,
) -> requests.Response:
    """GET *url* with exponential-backoff retries on transient failures.

    Parameters
    ----------
    url : str
        Fully-qualified URL to fetch.
    max_retries : int
        Maximum number of attempts before re-raising the last exception.
    backoff_base : float
        Base for the exponential back-off: ``backoff_base ** attempt`` seconds.
    timeout : int
        Per-request socket timeout in seconds.

    Returns
    -------
    requests.Response
        Successful (2xx) response object.

    Raises
    ------
    requests.exceptions.RequestException
        After *max_retries* consecutive failures.
    """
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            wait = backoff_base ** attempt
            logger.warning(
                "HTTP request failed (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)
    raise last_exc


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip currency symbols, commas, and plus signs, then cast to float.

    Parameters
    ----------
    series : pd.Series
        Raw string series from the OpenInsider CSV.

    Returns
    -------
    pd.Series
        Float series; unparseable values become ``NaN``.
    """
    return (
        series.astype(str)
        .str.replace(r"[$,+%()]", "", regex=True)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )


def _is_html(text: str) -> bool:
    """Return True if *text* looks like an HTML page rather than CSV data."""
    stripped = text.lstrip()
    return stripped.startswith("<") or "<html" in stripped[:200].lower()


def _parse_openinsider_response(
    content: str,
    transaction_code: str,
) -> pd.DataFrame:
    """Parse the raw text returned by an OpenInsider CSV export.

    Applies column renaming, type coercion, numeric cleaning, and filters to
    rows matching *transaction_code*.

    Parameters
    ----------
    content : str
        Raw text content from OpenInsider (expected to be CSV).
    transaction_code : str
        Expected trade-type code (e.g., ``'P'`` or ``'S'``); rows not matching
        this code are dropped as a belt-and-suspenders check.

    Returns
    -------
    pd.DataFrame
        Cleaned frame with exactly the columns listed in ``_REQUIRED_OUT``.
        Returns an empty DataFrame on parse failure.
    """
    if _is_html(content):
        logger.error("OpenInsider returned an HTML page instead of CSV (possible rate-limit).")
        return pd.DataFrame(columns=_REQUIRED_OUT)

    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception as exc:
        logger.error("CSV parse error: %s", exc)
        return pd.DataFrame(columns=_REQUIRED_OUT)

    if df.empty:
        return pd.DataFrame(columns=_REQUIRED_OUT)

    # Normalise column names (strip surrounding whitespace and BOM chars).
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # Drop the leading row-number column OpenInsider prepends (often named 'X').
    if df.columns[0] in ("X", ""):
        df = df.iloc[:, 1:]

    # Belt-and-suspenders: keep only rows whose trade type matches the request.
    if "Trade Type" in df.columns:
        mask = df["Trade Type"].astype(str).str.strip() == transaction_code
        df = df.loc[mask].copy()

    if df.empty:
        return pd.DataFrame(columns=_REQUIRED_OUT)

    # Rename to canonical column names.
    df.rename(
        columns={k: v for k, v in _COL_MAP.items() if k in df.columns},
        inplace=True,
    )

    # Parse date columns.
    for col in ("filing_date", "transaction_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # filing_timestamp preserves the full datetime (including intraday time if
    # OpenInsider includes it) while filing_date is normalised to midnight.
    if "filing_date" in df.columns:
        df["filing_timestamp"] = df["filing_date"]
        df["filing_date"] = df["filing_date"].dt.normalize()

    # Normalise ticker symbols.
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # Clean numeric fields.
    for col in ("transaction_price", "quantity", "owned", "transaction_value"):
        if col in df.columns:
            df[col] = _clean_numeric(df[col])

    # Guarantee all required output columns are present.
    for col in _REQUIRED_OUT:
        if col not in df.columns:
            df[col] = np.nan

    return df[_REQUIRED_OUT].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API – OpenInsider scrapers
# ---------------------------------------------------------------------------

def scrape_openinsider_monthly(
    year: int,
    month: int,
    transaction_code: str = "P",
) -> pd.DataFrame:
    """Fetch insider-transaction data for a single calendar month from OpenInsider.

    Constructs a date-bounded CSV export URL for the OpenInsider screener,
    downloads the response with retry logic, and returns a cleaned DataFrame.

    Parameters
    ----------
    year : int
        Four-digit calendar year (e.g., 2023).
    month : int
        Calendar month, 1–12.
    transaction_code : str
        OpenInsider trade-type filter.  ``'P'`` = open-market purchase
        (default); ``'S'`` = open-market sale.

    Returns
    -------
    pd.DataFrame
        Cleaned frame with columns defined in ``_REQUIRED_OUT``.  Returns an
        empty DataFrame (with the correct columns) on network or parse failure.
    """
    first_day = date(year, month, 1).strftime("%Y-%m-%d")
    last_day = date(year, month, _last_day_of_month(year, month)).strftime("%Y-%m-%d")

    url = _OPENINSIDER_URL.format(
        from_date=first_day,
        to_date=last_day,
        fd=first_day,
        td=last_day,
        tcp=transaction_code,
    )
    logger.info(
        "Scraping OpenInsider %d-%02d (code=%s): %s",
        year,
        month,
        transaction_code,
        url,
    )

    try:
        resp = _fetch_with_retry(url)
    except requests.exceptions.RequestException as exc:
        logger.error(
            "Failed to fetch OpenInsider data for %d-%02d after all retries: %s",
            year,
            month,
            exc,
        )
        return pd.DataFrame(columns=_REQUIRED_OUT)

    return _parse_openinsider_response(resp.text, transaction_code)


def scrape_openinsider_sales_monthly(year: int, month: int) -> pd.DataFrame:
    """Fetch open-market sale data for a single calendar month from OpenInsider.

    Thin wrapper around :func:`scrape_openinsider_monthly` with
    ``transaction_code='S'``.

    Parameters
    ----------
    year : int
        Four-digit calendar year.
    month : int
        Calendar month, 1–12.

    Returns
    -------
    pd.DataFrame
        Cleaned frame with columns defined in ``_REQUIRED_OUT`` for sale
        transactions.  Returns an empty DataFrame on failure.
    """
    return scrape_openinsider_monthly(year, month, transaction_code="S")


# ---------------------------------------------------------------------------
# Public API – Universe builders
# ---------------------------------------------------------------------------

def build_raw_purchase_universe(
    start_year: int = 2018,
    start_month: int = 1,
    end_year: int = 2024,
    end_month: int = 12,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """Build the full multi-year panel of insider open-market purchases.

    Iterates over every calendar month in the specified range, calling
    :func:`scrape_openinsider_monthly` for months not yet cached and reading
    the parquet cache for months that are.  Each month's data is stored as an
    individual parquet file named ``openinsider_P_{year}_{month:02d}.parquet``
    under *cache_dir*.

    Parameters
    ----------
    start_year : int
        First year to include (default 2018).
    start_month : int
        First month to include, 1–12 (default 1 = January).
    end_year : int
        Last year to include (default 2024).
    end_month : int
        Last month to include, 1–12 (default 12 = December).
    cache_dir : str
        Directory used for per-month parquet caches.

    Returns
    -------
    pd.DataFrame
        Concatenated frame of all monthly purchase records with columns from
        ``_REQUIRED_OUT``.  Ticker symbols are uppercased and whitespace-
        stripped.  Returns an empty DataFrame if no data could be retrieved.
    """
    os.makedirs(cache_dir, exist_ok=True)

    frames: list[pd.DataFrame] = []
    year, month = start_year, start_month

    while (year, month) <= (end_year, end_month):
        cache_file = pathlib.Path(cache_dir) / f"openinsider_P_{year}_{month:02d}.parquet"

        if cache_file.exists():
            logger.info("Loading cached purchase data: %s", cache_file)
            df = pd.read_parquet(cache_file)
        else:
            df = scrape_openinsider_monthly(year, month, transaction_code="P")
            if not df.empty:
                df.to_parquet(cache_file, index=False)
                logger.info("Cached %d rows → %s", len(df), cache_file)
            # Polite crawl delay only when a live request was made.
            time.sleep(1.5)

        if not df.empty:
            frames.append(df)

        # Advance to next month.
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    if not frames:
        logger.warning("build_raw_purchase_universe: no data retrieved for the requested range.")
        return pd.DataFrame(columns=_REQUIRED_OUT)

    result = pd.concat(frames, ignore_index=True)
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()
    logger.info(
        "Purchase universe: %d rows, %d unique tickers",
        len(result),
        result["ticker"].nunique(),
    )
    return result


def build_raw_sales_universe(
    start_year: int = 2018,
    end_year: int = 2024,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """Build the full multi-year panel of insider open-market sales.

    Iterates over every calendar month from January of *start_year* through
    December of *end_year*, calling :func:`scrape_openinsider_sales_monthly`
    for months not yet cached and reading the parquet cache for months that
    are.  Each month's data is stored as
    ``openinsider_S_{year}_{month:02d}.parquet`` under *cache_dir*.

    Parameters
    ----------
    start_year : int
        First year to include (default 2018).
    end_year : int
        Last year to include (default 2024).
    cache_dir : str
        Directory used for per-month parquet caches.

    Returns
    -------
    pd.DataFrame
        Concatenated frame of all monthly sale records with columns from
        ``_REQUIRED_OUT``.  Ticker symbols are uppercased and whitespace-
        stripped.  Returns an empty DataFrame if no data could be retrieved.
    """
    os.makedirs(cache_dir, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            cache_file = pathlib.Path(cache_dir) / f"openinsider_S_{year}_{month:02d}.parquet"

            if cache_file.exists():
                logger.info("Loading cached sale data: %s", cache_file)
                df = pd.read_parquet(cache_file)
            else:
                df = scrape_openinsider_sales_monthly(year, month)
                if not df.empty:
                    df.to_parquet(cache_file, index=False)
                    logger.info("Cached %d rows → %s", len(df), cache_file)
                time.sleep(1.5)

            if not df.empty:
                frames.append(df)

    if not frames:
        logger.warning("build_raw_sales_universe: no data retrieved for the requested range.")
        return pd.DataFrame(columns=_REQUIRED_OUT)

    result = pd.concat(frames, ignore_index=True)
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()
    logger.info(
        "Sales universe: %d rows, %d unique tickers",
        len(result),
        result["ticker"].nunique(),
    )
    return result


# ---------------------------------------------------------------------------
# Public API – Fama-French factor loader
# ---------------------------------------------------------------------------

def load_ff3_factors(
    cache_path: str = "data/cache/ff3_daily.parquet",
) -> pd.DataFrame:
    """Download and cache the Fama-French 3-factor daily return series.

    Fetches data from Ken French's data library via
    ``pandas_datareader`` (source ``'famafrench'``, dataset
    ``'F-F_Research_Data_Factors_daily'``).  Raw values arrive as
    percentages; this function converts them to decimal fractions before
    returning or caching.

    Parameters
    ----------
    cache_path : str
        Full path to the parquet cache file.  Parent directory is created
        automatically if it does not exist.  If the file already exists the
        download is skipped and the cached copy is returned.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (``datetime64[ns]``), ``Mkt-RF``, ``SMB``, ``HML``,
        ``RF`` – all as decimal fractions (e.g., 0.01 for a 1 % return).
        Sorted ascending by date with no duplicate dates.

    Raises
    ------
    Exception
        Re-raises any exception from ``pandas_datareader`` after logging it.
    """
    cache = pathlib.Path(cache_path)

    if cache.exists():
        logger.info("Loading FF3 daily factors from cache: %s", cache)
        return pd.read_parquet(cache)

    os.makedirs(cache.parent, exist_ok=True)
    logger.info("Downloading Fama-French 3-factor daily data from Ken French library …")

    try:
        raw = pdr.get_data_famafrench(
            "F-F_Research_Data_Factors_daily",
            start="2010-01-01",
        )
    except Exception as exc:
        logger.error("pandas_datareader failed to fetch FF3 factors: %s", exc)
        raise

    # The famafrench reader returns a dict; key 0 is the factors table.
    df = raw[0].copy()

    # Values are in percentage points – convert to decimal fractions.
    df = df / 100.0

    # Convert PeriodIndex (daily) to DatetimeIndex.
    if hasattr(df.index, "to_timestamp"):
        df.index = df.index.to_timestamp()
    else:
        df.index = pd.to_datetime(df.index)

    df.index.name = "date"
    df = df.reset_index()

    # Normalise column names (strip any stray whitespace).
    df.columns = [str(c).strip() for c in df.columns]

    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)

    df.to_parquet(cache, index=False)
    logger.info("Cached FF3 daily factors (%d rows) → %s", len(df), cache)

    return df
