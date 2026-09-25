"""
placebo_and_comparators.py

Placebo and comparator identification tests for the microcap insider purchase
disclosure study.

Placebo design
--------------
For each sample issuer × calendar year, 24 pseudo-event dates are drawn
uniformly at random from dates that satisfy the same data-availability,
liquidity, and exclusion windows as the real purchase sample.  Outcomes are
computed on the same event-study code path and the resulting distribution is
compared against the real purchase gradient.

Comparators
-----------
1. Insider *sales* (Form 4 code S, open-market, non-exercise).
2. Quarterly earnings surprises (8-K Item 2.02 filings, XBRL EPS).

Data conventions
----------------
yahoo_panel : dict[str, pd.DataFrame]
    Maps ticker → OHLCV DataFrame with a DatetimeIndex and at minimum the
    columns ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.

ff3 : pd.DataFrame
    Daily Fama-French three-factor returns.  DatetimeIndex; columns
    ``Mkt-RF``, ``SMB``, ``HML``, ``RF`` (all as decimal fractions, not %).

shares_df : pd.DataFrame
    Shares-outstanding panel.  Must contain ``ticker``, ``date``,
    ``shares_outstanding`` (in units of shares).
"""

from __future__ import annotations

import json
import time
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.stats as sps
import seaborn as sns
import statsmodels.api as sm

# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

_EDGAR_BASE = "https://data.sec.gov"
_EFTS_BASE  = "https://efts.sec.gov"
_HEADERS     = {"User-Agent": "research-study contact@example.com",
                "Accept-Encoding": "gzip, deflate"}

_MIN_ESTIMATION_DAYS = 30   # minimum prior-window trading days for FF3 estimation
_MIN_FUTURE_DAYS     = 60   # minimum future trading days for CAR computation
_PLACEBO_GAP_DAYS    = 30   # required distance from any real purchase filing
_MIN_ADDV_USD        = 50_000  # minimum average daily dollar volume gate


def _get_trading_days(price_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return sorted DatetimeIndex of trading days from a yahoo_panel entry."""
    if isinstance(price_df.index, pd.DatetimeIndex):
        return price_df.index.sort_values()
    if 'Date' in price_df.columns:
        return pd.DatetimeIndex(pd.to_datetime(price_df['Date'])).sort_values()
    raise ValueError("price_df must have a DatetimeIndex or a 'Date' column")


def _price_series(price_df: pd.DataFrame, col: str = 'Close') -> pd.Series:
    """Return a named price Series indexed by trading date."""
    idx = _get_trading_days(price_df)
    if isinstance(price_df.index, pd.DatetimeIndex):
        return price_df[col].sort_index()
    return pd.Series(price_df[col].values, index=idx, name=col)


def _daily_return(close: pd.Series) -> pd.Series:
    """Simple daily return (close-to-close)."""
    return close.pct_change()


def _addv(price_df: pd.DataFrame, n: int = 30) -> pd.Series:
    """30-day trailing average daily dollar volume (per trading day)."""
    close = _price_series(price_df, 'Close')
    volume = _price_series(price_df, 'Volume')
    dv = close * volume
    return dv.rolling(n, min_periods=max(1, n // 2)).mean()


def _estimate_ff3(
    stock_ret: pd.Series,
    ff3: pd.DataFrame,
) -> Optional[dict]:
    """OLS estimation of FF3 model on aligned data.

    Returns dict with keys: alpha, beta_mkt, beta_smb, beta_hml, rsq.
    Returns None if fewer than 15 overlapping observations.
    """
    shared = stock_ret.index.intersection(ff3.index)
    if len(shared) < 15:
        return None

    y = stock_ret.loc[shared].astype(float)
    rf = ff3.loc[shared, 'RF'].astype(float)
    X = sm.add_constant(ff3.loc[shared, ['Mkt-RF', 'SMB', 'HML']].astype(float))
    y_excess = y - rf

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = sm.OLS(y_excess, X).fit()
    except Exception:
        return None

    return {
        'alpha':    float(res.params.get('const', np.nan)),
        'beta_mkt': float(res.params.get('Mkt-RF', np.nan)),
        'beta_smb': float(res.params.get('SMB', np.nan)),
        'beta_hml': float(res.params.get('HML', np.nan)),
        'rsq':      float(res.rsquared),
    }


def _compute_car(
    stock_ret: pd.Series,
    ff3: pd.DataFrame,
    params: dict,
    future_dates: pd.DatetimeIndex,
    t_start: int,
    t_end: int,
) -> float:
    """Cumulative abnormal return over a window relative to the event date.

    Parameters
    ----------
    stock_ret:
        Full stock return series.
    ff3:
        Factor return panel.
    params:
        Output of ``_estimate_ff3``.
    future_dates:
        Ordered trading dates after the event (index 0 = t0+1).
    t_start, t_end:
        Inclusive window indices (1-based, relative to event date).
    """
    window_dates = future_dates[t_start - 1 : t_end]
    if len(window_dates) < (t_end - t_start + 1):
        return np.nan

    car = 0.0
    for d in window_dates:
        if d not in stock_ret.index or d not in ff3.index:
            return np.nan
        r = float(stock_ret.loc[d])
        rf_d = float(ff3.loc[d, 'RF'])
        expected = (
            rf_d
            + params['alpha']
            + params['beta_mkt'] * float(ff3.loc[d, 'Mkt-RF'])
            + params['beta_smb'] * float(ff3.loc[d, 'SMB'])
            + params['beta_hml'] * float(ff3.loc[d, 'HML'])
        )
        car += r - expected
    return car


def _quintile_assign(values: pd.Series, cutoffs: np.ndarray) -> pd.Series:
    """Assign *values* to quintiles 1–5 using pre-computed *cutoffs*.

    ``cutoffs`` should be a 6-element array (bin edges including −∞ and +∞)
    as returned by ``pd.qcut(..., retbins=True)``.
    """
    bins = np.concatenate([[-np.inf], cutoffs[1:-1], [np.inf]])
    return pd.cut(values, bins=bins, labels=[1, 2, 3, 4, 5],
                  include_lowest=True).astype(float).astype('Int64')


def _cluster_se_of_mean(values: np.ndarray, groups: np.ndarray) -> float:
    """Issuer-clustered standard error of the mean (CGM one-way)."""
    n = len(values)
    if n == 0:
        return np.nan
    mu = values.mean()
    resids = values - mu
    df_tmp = pd.DataFrame({'r': resids, 'g': groups})
    g_sums = df_tmp.groupby('g')['r'].sum()
    n_c = len(g_sums)
    # G/(G-1) finite-cluster correction
    v = float((g_sums ** 2).sum()) / (n ** 2) * (n_c / max(n_c - 1, 1))
    return float(np.sqrt(max(v, 0.0)))


# ────────────────────────────────────────────────────────────────────────────
# Placebo generation
# ────────────────────────────────────────────────────────────────────────────

def generate_placebo_sample(
    events: pd.DataFrame,
    yahoo_panel: dict,
    ff3: pd.DataFrame,
    shares_df: pd.DataFrame,
    n_per_issuer_year: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate pseudo-event placebo observations for significance testing.

    For each (ticker, year) pair present in *events*, sample up to
    *n_per_issuer_year* random trading days that satisfy:

    - Price and volume data available for the full estimation and event windows.
    - At least ``_MIN_ESTIMATION_DAYS`` prior trading days available.
    - At least ``_MIN_FUTURE_DAYS`` future trading days available.
    - Trailing 30-day average daily dollar volume ≥ ``_MIN_ADDV_USD``.
    - FF3 factor data available for estimation window.
    - The candidate date lies > ``_PLACEBO_GAP_DAYS`` trading days away from
      every actual purchase-filing date for the same issuer.

    FF3 parameters are estimated on the same code path as the real sample.
    Placebo CARs are computed for windows [1,30] and [1,60].
    The column ``pct_from_52w_high`` is computed from the prior 252 trading-day
    high; quintile assignment is left to the caller (see :func:`run_placebo_analysis`).

    Parameters
    ----------
    events:
        Real purchase events.  Must contain ``ticker``, ``filing_date``.
    yahoo_panel:
        Mapping of ticker → OHLCV DataFrame (see module docstring).
    ff3:
        Fama-French 3-factor daily returns.
    shares_df:
        Shares-outstanding panel (see module docstring).
    n_per_issuer_year:
        Target number of placebo dates per (ticker, year).
    seed:
        NumPy random seed.

    Returns
    -------
    DataFrame with one row per accepted placebo observation; columns include
    ``ticker``, ``event_date``, ``year``, ``pct_from_52w_high``,
    ``CAR_1_30``, ``CAR_1_60``, ``ff3_alpha``, ``vol_30d``.
    """
    rng = np.random.default_rng(seed)
    filing_dates = pd.to_datetime(events['filing_date'])
    events_with_dates = events.assign(filing_date_dt=filing_dates)

    records: list[dict] = []

    for ticker, grp in events_with_dates.groupby('ticker'):
        if ticker not in yahoo_panel:
            continue
        price_df = yahoo_panel[ticker]
        trading_days = _get_trading_days(price_df)
        close = _price_series(price_df, 'Close')
        volume = _price_series(price_df, 'Volume')
        ret = _daily_return(close)
        adv = _addv(price_df)

        # Actual filing dates for this issuer (as trading-day positions)
        actual_dates = pd.DatetimeIndex(grp['filing_date_dt'].dropna().unique())

        years = filing_dates.dt.year.unique()
        for year in sorted(years):
            year_days = trading_days[
                (trading_days.year == year) &
                (trading_days >= trading_days.min() + pd.Timedelta(days=_MIN_ESTIMATION_DAYS * 2)) &
                (trading_days <= trading_days.max() - pd.Timedelta(days=_MIN_FUTURE_DAYS * 2))
            ]
            if len(year_days) == 0:
                continue

            # Filter by gap from any actual purchase filing
            def _is_far_enough(d: pd.Timestamp) -> bool:
                td_pos = int(np.searchsorted(trading_days, d))
                for act in actual_dates:
                    act_pos = int(np.searchsorted(trading_days, act))
                    if abs(td_pos - act_pos) <= _PLACEBO_GAP_DAYS:
                        return False
                return True

            candidates = [d for d in year_days if _is_far_enough(d)]
            if not candidates:
                continue

            rng.shuffle(candidates)
            accepted = 0

            for event_date in candidates:
                if accepted >= n_per_issuer_year:
                    break

                td_pos = int(np.searchsorted(trading_days, event_date))

                # Prior-window check
                prior_start = max(0, td_pos - _MIN_ESTIMATION_DAYS)
                prior_end   = td_pos
                if (prior_end - prior_start) < _MIN_ESTIMATION_DAYS:
                    continue

                # Future-window check
                future_end = min(len(trading_days) - 1, td_pos + _MIN_FUTURE_DAYS)
                if (future_end - td_pos) < _MIN_FUTURE_DAYS:
                    continue

                # ADDV gate
                if event_date not in adv.index or adv.loc[event_date] < _MIN_ADDV_USD:
                    continue

                # FF3 data gate
                prior_dates = trading_days[prior_start:prior_end]
                if not all(d in ff3.index for d in prior_dates[:5]):
                    continue

                # Estimate FF3
                stock_ret_prior = ret.reindex(prior_dates).dropna()
                ff3_params = _estimate_ff3(stock_ret_prior, ff3)
                if ff3_params is None:
                    continue

                # Future dates for CAR
                future_dates = trading_days[td_pos + 1 : td_pos + 1 + _MIN_FUTURE_DAYS + 1]

                car_1_30 = _compute_car(ret, ff3, ff3_params, future_dates, 1, 30)
                car_1_60 = _compute_car(ret, ff3, ff3_params, future_dates, 1, 60)
                if np.isnan(car_1_30):
                    continue

                # 52-week high (prior 252 trading days)
                prior_252_start = max(0, td_pos - 252)
                high_252 = close.iloc[prior_252_start:td_pos].max()
                price_now = close.loc[event_date] if event_date in close.index else np.nan
                pct_from_52w_high = (price_now - high_252) / high_252 if high_252 > 0 else np.nan

                # vol_30d (annualised)
                vol_30d = float(ret.iloc[max(0, td_pos - 30):td_pos].std() * np.sqrt(252))

                records.append({
                    'ticker':           ticker,
                    'event_date':       event_date,
                    'year':             year,
                    'pct_from_52w_high': pct_from_52w_high,
                    'CAR_1_30':         car_1_30,
                    'CAR_1_60':         car_1_60 if not np.isnan(car_1_60) else np.nan,
                    'ff3_alpha':        ff3_params['alpha'],
                    'vol_30d':          vol_30d,
                })
                accepted += 1

    return pd.DataFrame(records).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# Placebo analysis
# ────────────────────────────────────────────────────────────────────────────

def run_placebo_analysis(
    placebo_events: pd.DataFrame,
    purchase_quintile_cutoffs: np.ndarray,
    primary_cars: pd.DataFrame,
    alpha_inclusive_cars: pd.DataFrame,
) -> pd.DataFrame:
    """Compare placebo gradients under primary and intercept-inclusive benchmarks.

    Quintile assignments for placebo observations use the purchase-sample
    cutoffs to ensure comparability.

    Parameters
    ----------
    placebo_events:
        Output of :func:`generate_placebo_sample`.  Must contain
        ``pct_from_52w_high``, ``ticker``, ``CAR_1_30``.
    purchase_quintile_cutoffs:
        Bin edges from the real purchase sample (stored in
        ``events.attrs['quintile_cutoffs']`` after :func:`form_quintiles`).
    primary_cars:
        Loadings-only CARs for the real purchase sample.  Must contain
        ``CAR_1_30``; index aligns with *events* (not placebo).
    alpha_inclusive_cars:
        Intercept-inclusive CARs for the real purchase sample.

    Returns
    -------
    DataFrame comparing mean CAR_1_30 by quintile for:
    - placebo (primary benchmark)
    - purchase primary
    - purchase alpha-inclusive
    with issuer-clustered SEs and a Welch t-test for each placebo cell.
    """
    placebo = placebo_events.copy()
    placebo['quintile'] = _quintile_assign(
        placebo['pct_from_52w_high'], purchase_quintile_cutoffs
    ).astype('Int64')
    placebo = placebo.dropna(subset=['quintile'])

    rows: list[dict] = []
    for q in range(1, 6):
        sub = placebo.loc[placebo['quintile'] == q]
        car = sub['CAR_1_30'].dropna()
        tickers = sub.loc[car.index, 'ticker'].values

        n = len(car)
        mean_c = float(car.mean()) if n > 0 else np.nan
        se_c = _cluster_se_of_mean(car.values, tickers) if n > 1 else np.nan
        t_stat, p_val = (
            (mean_c / se_c, 2 * sps.t.sf(abs(mean_c / se_c),
                                           df=max(len(np.unique(tickers)) - 1, 1)))
            if (se_c and se_c > 0) else (np.nan, np.nan)
        )
        welch_t, welch_p = (
            sps.ttest_1samp(car.dropna(), 0.0)
            if len(car.dropna()) > 1 else (np.nan, np.nan)
        )

        rows.append({
            'quintile':             q,
            'n_placebo':            n,
            'mean_placebo_primary': mean_c,
            'se_placebo':           se_c,
            't_placebo':            t_stat,
            'p_placebo':            p_val,
            'welch_t_placebo':      float(welch_t),
            'welch_p_placebo':      float(welch_p),
        })

    result = pd.DataFrame(rows).set_index('quintile')

    # Merge purchase-sample quintile means (primary)
    if 'quintile' in primary_cars.columns and 'CAR_1_30' in primary_cars.columns:
        purch_primary = (
            primary_cars.groupby('quintile')['CAR_1_30']
            .agg(mean_purchase_primary='mean', n_purchase='count')
        )
        result = result.join(purch_primary, how='left')

    if 'quintile' in alpha_inclusive_cars.columns and 'CAR_1_30' in alpha_inclusive_cars.columns:
        purch_incl = (
            alpha_inclusive_cars.groupby('quintile')['CAR_1_30']
            .agg(mean_purchase_inclusive='mean')
        )
        result = result.join(purch_incl, how='left')

    return result


# ────────────────────────────────────────────────────────────────────────────
# Insider sales comparator
# ────────────────────────────────────────────────────────────────────────────

def build_sales_comparator(
    sales_raw: pd.DataFrame,
    yahoo_panel: dict,
    ff3: pd.DataFrame,
) -> pd.DataFrame:
    """Build an insider sales comparator event study.

    Filtering
    ---------
    - Keep only ``transaction_code == 'S'`` (open-market sales).
    - Exclude rows flagged as option-exercise dispositions
      (``is_option_exercise == True`` if present, or ``footnote`` containing
      "option" / "exercise").
    - Collapse multiple trades on the same filing to one event per
      (ticker, filing_timestamp) by aggregating transaction value.
    - Restrict main comparison to after-close filings
      (``is_preclose == False`` or ``is_preclose`` missing).

    Event study
    -----------
    FF3 parameters are estimated on the 30 prior trading days (same as
    purchase sample).  CARs are computed for windows [1,30] and [1,60].
    The ``pct_from_52w_high`` feature is computed from the prior 252-day high.

    Parameters
    ----------
    sales_raw:
        Form 4 sales-leg DataFrame.  Required columns:
        ``ticker``, ``filing_timestamp``, ``transaction_code``,
        ``transaction_value_usd``.  Optional: ``is_option_exercise``,
        ``footnote``, ``is_preclose``.
    yahoo_panel:
        Ticker → OHLCV DataFrame.
    ff3:
        Daily Fama-French factor returns.

    Returns
    -------
    Event-level DataFrame with columns matching the purchase event table
    (ticker, filing_date, CAR_1_30, CAR_1_60, pct_from_52w_high, etc.).
    """
    df = sales_raw.copy()

    # Keep open-market sales only
    df = df.loc[df['transaction_code'] == 'S'].copy()

    # Exclude option-exercise dispositions
    if 'is_option_exercise' in df.columns:
        df = df.loc[~df['is_option_exercise'].fillna(False)]
    if 'footnote' in df.columns:
        fn = df['footnote'].fillna('').str.lower()
        df = df.loc[~(fn.str.contains('option') | fn.str.contains('exercise'))]

    # Collapse to one event per (ticker, filing_timestamp)
    df['filing_timestamp'] = pd.to_datetime(df['filing_timestamp'])
    df['filing_date'] = df['filing_timestamp'].dt.normalize()

    agg = (
        df.groupby(['ticker', 'filing_date'])
        .agg(total_value=('transaction_value_usd', 'sum'),
             n_trades=('transaction_code', 'count'))
        .reset_index()
    )

    # After-close filings (main comparison)
    if 'is_preclose' in df.columns:
        preclose_map = (
            df.groupby(['ticker', 'filing_date'])['is_preclose']
            .first()
            .reset_index()
        )
        agg = agg.merge(preclose_map, on=['ticker', 'filing_date'], how='left')
        agg = agg.loc[~agg['is_preclose'].fillna(False)]

    records: list[dict] = []
    for _, row in agg.iterrows():
        ticker = row['ticker']
        filing_date = row['filing_date']

        if ticker not in yahoo_panel:
            continue
        price_df = yahoo_panel[ticker]
        trading_days = _get_trading_days(price_df)
        close = _price_series(price_df, 'Close')
        ret = _daily_return(close)

        td_pos = int(np.searchsorted(trading_days, filing_date))
        if td_pos < _MIN_ESTIMATION_DAYS or td_pos + _MIN_FUTURE_DAYS >= len(trading_days):
            continue

        prior_dates = trading_days[td_pos - _MIN_ESTIMATION_DAYS : td_pos]
        ff3_params = _estimate_ff3(ret.reindex(prior_dates).dropna(), ff3)
        if ff3_params is None:
            continue

        future_dates = trading_days[td_pos + 1 : td_pos + 1 + _MIN_FUTURE_DAYS + 1]
        car_1_30 = _compute_car(ret, ff3, ff3_params, future_dates, 1, 30)
        car_1_60 = _compute_car(ret, ff3, ff3_params, future_dates, 1, 60)
        if np.isnan(car_1_30):
            continue

        prior_252_start = max(0, td_pos - 252)
        high_252 = close.iloc[prior_252_start:td_pos].max()
        price_now = close.loc[filing_date] if filing_date in close.index else np.nan
        pct_52w = (price_now - high_252) / high_252 if high_252 > 0 else np.nan
        vol_30d = float(ret.iloc[max(0, td_pos - 30):td_pos].std() * np.sqrt(252))

        records.append({
            'ticker':           ticker,
            'filing_date':      filing_date,
            'total_value':      row['total_value'],
            'n_trades':         row['n_trades'],
            'CAR_1_30':         car_1_30,
            'CAR_1_60':         car_1_60,
            'ff3_alpha':        ff3_params['alpha'],
            'pct_from_52w_high': pct_52w,
            'vol_30d':          vol_30d,
        })

    return pd.DataFrame(records).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# Earnings comparator
# ────────────────────────────────────────────────────────────────────────────

def _edgar_get(url: str, retries: int = 3, backoff: float = 1.5) -> Optional[dict]:
    """GET a JSON resource from the SEC EDGAR REST API with retry/back-off."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                time.sleep(backoff * (2 ** attempt))
            else:
                return None
        except Exception:
            time.sleep(backoff * (2 ** attempt))
    return None


def _fetch_8k_item202_filings(
    ticker: str,
    cik: str,
    start_dt: str = '2018-01-01',
    end_dt: str = '2024-12-31',
) -> list[dict]:
    """Return list of 8-K Item 2.02 filing metadata for one issuer."""
    url = (
        f"{_EFTS_BASE}/LATEST/search-index"
        f"?q=%222.02%22&forms=8-K"
        f"&dateRange=custom&startdt={start_dt}&enddt={end_dt}"
        f"&entity={ticker}"
    )
    data = _edgar_get(url)
    if data is None:
        return []

    hits = data.get('hits', {}).get('hits', [])
    filings: list[dict] = []
    for hit in hits:
        src = hit.get('_source', {})
        period = src.get('period_of_report', '')
        filed  = src.get('file_date', '')
        accession = src.get('accession_no', '').replace('-', '')
        if not filed or not period:
            continue
        filings.append({
            'ticker':      ticker,
            'cik':         cik,
            'filed_date':  pd.to_datetime(filed),
            'period':      pd.to_datetime(period),
            'accession_no': accession,
        })
    return filings


def _fetch_xbrl_eps(cik: str) -> pd.DataFrame:
    """Fetch quarterly EPS (diluted) from the SEC EDGAR XBRL company-facts API.

    Returns DataFrame with columns: end_date (DatetimeIndex), eps_diluted.
    """
    url = f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    data = _edgar_get(url)
    if data is None:
        return pd.DataFrame()

    facts = data.get('facts', {})
    # Try US-GAAP EarningsPerShareDiluted first, then Basic
    for concept in ('EarningsPerShareDiluted', 'EarningsPerShareBasic'):
        eps_data = (
            facts.get('us-gaap', {})
            .get(concept, {})
            .get('units', {})
            .get('USD/shares', [])
        )
        if eps_data:
            break

    if not eps_data:
        return pd.DataFrame()

    rows = [
        {
            'end_date': pd.to_datetime(r['end']),
            'eps':      r['val'],
            'form':     r.get('form', ''),
            'filed':    pd.to_datetime(r.get('filed', pd.NaT)),
        }
        for r in eps_data
        if r.get('form', '') in ('10-Q', '10-K')
    ]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates('end_date').set_index('end_date').sort_index()
    df.index = pd.DatetimeIndex(df.index)
    return df[['eps', 'filed']]


def build_earnings_comparator(
    sample_tickers: list,
    cache_dir: str = 'data/cache',
) -> pd.DataFrame:
    """Build an earnings-announcement comparator using SEC EDGAR.

    Procedure
    ---------
    1. Resolve CIKs for each ticker via the EDGAR submissions API.
    2. Fetch 8-K Item 2.02 filing dates.
    3. Fetch quarterly EPS from the XBRL company-facts API.
    4. Compute signed seasonal-random-walk (SRW) surprise:
       ``surprise = EPS_t − EPS_{t-4}``
       (same quarter one year prior; sign-standardised by absolute prior value
       when |EPS_{t-4}| > 0.01).
    5. Restrict to observations with ≥ 4 prior quarters available.
    6. Abnormal returns are not computed here (require yahoo_panel); the
       returned DataFrame is suitable for merging with an event-study runner.

    Parameters
    ----------
    sample_tickers:
        List of ticker symbols to query.
    cache_dir:
        Local directory for caching EDGAR responses (created if absent).

    Returns
    -------
    DataFrame with one row per usable earnings event, columns:
    ``ticker``, ``cik``, ``filed_date``, ``period``, ``eps``,
    ``eps_prior_year``, ``srw_surprise``, ``srw_surprise_scaled``.
    """
    import os
    os.makedirs(cache_dir, exist_ok=True)

    # ── Step 1: resolve CIK for each ticker ─────────────────────────────
    ticker_cik_url = f"{_EDGAR_BASE}/files/company_tickers.json"
    cik_map_raw = _edgar_get(ticker_cik_url) or {}
    cik_map: dict[str, str] = {}
    for entry in cik_map_raw.values():
        t = str(entry.get('ticker', '')).upper()
        c = str(entry.get('cik_str', ''))
        if t and c:
            cik_map[t] = c

    records: list[dict] = []

    for ticker in sample_tickers:
        ticker_u = ticker.upper()
        cik = cik_map.get(ticker_u)
        if cik is None:
            warnings.warn(f"CIK not found for ticker {ticker}; skipping.")
            continue

        # ── Step 2: 8-K Item 2.02 filings ───────────────────────────────
        cache_path = os.path.join(cache_dir, f"8k_{ticker_u}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    filings = json.load(f)
                # Restore datetime types
                for fi in filings:
                    fi['filed_date'] = pd.to_datetime(fi['filed_date'])
                    fi['period']     = pd.to_datetime(fi['period'])
            except Exception:
                filings = _fetch_8k_item202_filings(ticker_u, cik)
        else:
            filings = _fetch_8k_item202_filings(ticker_u, cik)
            try:
                with open(cache_path, 'w') as f:
                    serialisable = [
                        {**fi,
                         'filed_date': str(fi['filed_date']),
                         'period':     str(fi['period'])}
                        for fi in filings
                    ]
                    json.dump(serialisable, f)
            except Exception:
                pass

        if not filings:
            continue

        # ── Step 3: XBRL EPS ─────────────────────────────────────────────
        eps_cache_path = os.path.join(cache_dir, f"eps_{ticker_u}.json")
        if os.path.exists(eps_cache_path):
            try:
                eps_df = pd.read_json(eps_cache_path)
                if 'end_date' in eps_df.columns:
                    eps_df = eps_df.set_index('end_date')
                    eps_df.index = pd.DatetimeIndex(eps_df.index)
            except Exception:
                eps_df = _fetch_xbrl_eps(cik)
        else:
            eps_df = _fetch_xbrl_eps(cik)
            if not eps_df.empty:
                try:
                    eps_df.reset_index().to_json(eps_cache_path)
                except Exception:
                    pass

        if eps_df.empty:
            continue

        # ── Steps 4-5: compute SRW surprise ──────────────────────────────
        for filing in filings:
            period = filing['period']
            # Find EPS for this quarter
            eps_match = eps_df.loc[
                eps_df.index[abs((eps_df.index - period).days) <= 5]
            ]
            if eps_match.empty:
                continue
            eps_t = float(eps_match['eps'].iloc[0])

            # Same quarter prior year
            prior_period = period - pd.DateOffset(years=1)
            eps_prior_match = eps_df.loc[
                eps_df.index[abs((eps_df.index - prior_period).days) <= 10]
            ]
            if eps_prior_match.empty:
                continue
            eps_t4 = float(eps_prior_match['eps'].iloc[0])

            # Require at least 4 prior quarters in eps_df
            n_prior = int((eps_df.index < period).sum())
            if n_prior < 4:
                continue

            srw_raw = eps_t - eps_t4
            srw_scaled = (
                srw_raw / abs(eps_t4) if abs(eps_t4) > 0.01 else np.nan
            )

            records.append({
                'ticker':             ticker_u,
                'cik':               cik,
                'filed_date':        filing['filed_date'],
                'period':            period,
                'eps':               eps_t,
                'eps_prior_year':    eps_t4,
                'srw_surprise':      srw_raw,
                'srw_surprise_scaled': srw_scaled,
            })

        # Rate-limit courtesy
        time.sleep(0.05)

    return pd.DataFrame(records).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# Comparator significance tests
# ────────────────────────────────────────────────────────────────────────────

def run_comparator_tests(
    purchase_events: pd.DataFrame,
    sales_events: pd.DataFrame,
    earnings_events: pd.DataFrame,
    quintile_cutoffs: np.ndarray,
) -> dict:
    """Run main comparator significance tests vs the purchase sample.

    Tests performed
    ---------------
    1. **Purchase vs Placebo** (via :func:`run_placebo_analysis`):
       Welch t-test on Q1 mean CAR (H₀: purchase Q1 mean = placebo Q1 mean).

    2. **Purchase vs Sales**:
       For each quintile, two-sample Welch t-test comparing mean CAR_1_30.
       Also tests whether the Q1–Q5 gradient is reversed for sales.

    3. **Purchase vs Earnings**:
       If EPS surprise quartiles overlap with our CAR quintiles, compare
       mean post-announcement CARs.  Welch t-test for the bottom-quintile
       cell (most negative surprise vs. farthest-from-high purchase).

    Parameters
    ----------
    purchase_events:
        Must contain ``quintile``, ``ticker``, ``CAR_1_30``.
    sales_events:
        Output of :func:`build_sales_comparator`.  Must contain
        ``pct_from_52w_high`` (used to assign quintiles) and ``CAR_1_30``.
    earnings_events:
        Output of :func:`build_earnings_comparator`.  Must contain
        ``srw_surprise``.  If a ``CAR_1_30`` column is present it is used;
        otherwise only surprise statistics are reported.
    quintile_cutoffs:
        Bin edges from the purchase sample (6-element array).

    Returns
    -------
    dict with keys:
        ``purchase_quintile_means``  – pd.Series
        ``sales_quintile_means``     – pd.Series (NaN if empty)
        ``purchase_vs_sales``        – pd.DataFrame (quintile-level Welch tests)
        ``earnings_surprise_stats``  – pd.DataFrame (EPS surprise by quintile)
        ``purchase_vs_earnings``     – dict (t-stat, p-value for Q1 comparison)
        ``gradient_tests``           – pd.DataFrame (Q5−Q1 gap tests per group)
    """
    required = {'quintile', 'ticker', 'CAR_1_30'}
    missing = required - set(purchase_events.columns)
    if missing:
        raise ValueError(
            f"purchase_events is missing columns {missing}. "
            "Join with the cars DataFrame before calling: "
            "purchase_events = events_q.join(cars[['CAR_1_30']])"
        )

    result: dict = {}

    # ── Purchase quintile means ──────────────────────────────────────────
    purch_grp = purchase_events.groupby('quintile')['CAR_1_30']
    purch_means = purch_grp.mean()
    result['purchase_quintile_means'] = purch_means

    # ── Sales: assign quintiles and compute means ────────────────────────
    if not sales_events.empty and 'pct_from_52w_high' in sales_events.columns:
        sales_with_q = sales_events.copy()
        sales_with_q['quintile'] = _quintile_assign(
            sales_with_q['pct_from_52w_high'], quintile_cutoffs
        )
        sales_grp = sales_with_q.dropna(subset=['quintile', 'CAR_1_30']).groupby('quintile')

        sales_means = sales_grp['CAR_1_30'].mean()
        result['sales_quintile_means'] = sales_means

        # Quintile-level Welch t-test: purchase vs sales
        pvs_rows: list[dict] = []
        for q in range(1, 6):
            p_vals = purchase_events.loc[purchase_events['quintile'] == q, 'CAR_1_30'].dropna()
            s_vals = (
                sales_with_q
                .loc[sales_with_q['quintile'] == q, 'CAR_1_30']
                .dropna()
            ) if q in sales_grp.groups else pd.Series(dtype=float)

            if len(p_vals) > 1 and len(s_vals) > 1:
                t_stat, p_val = sps.ttest_ind(p_vals, s_vals, equal_var=False)
            else:
                t_stat, p_val = np.nan, np.nan

            pvs_rows.append({
                'quintile':      q,
                'n_purchase':    len(p_vals),
                'n_sales':       len(s_vals),
                'mean_purchase': float(p_vals.mean()) if len(p_vals) > 0 else np.nan,
                'mean_sales':    float(s_vals.mean()) if len(s_vals) > 0 else np.nan,
                'diff':          float(p_vals.mean() - s_vals.mean()) if (len(p_vals) > 0 and len(s_vals) > 0) else np.nan,
                't_stat':        float(t_stat),
                'p_value':       float(p_val),
            })
        result['purchase_vs_sales'] = pd.DataFrame(pvs_rows).set_index('quintile')
    else:
        result['sales_quintile_means'] = pd.Series(dtype=float)
        result['purchase_vs_sales']    = pd.DataFrame()

    # ── Earnings: surprise statistics and CAR comparison ────────────────
    if not earnings_events.empty and 'srw_surprise' in earnings_events.columns:
        earn = earnings_events.copy()

        # Quartile on surprise (Q1 = most negative, Q4 = most positive)
        if earn['srw_surprise'].notna().sum() >= 4:
            earn['surprise_quartile'] = pd.qcut(
                earn['srw_surprise'], q=4, labels=[1, 2, 3, 4], duplicates='drop'
            ).astype(float)
        else:
            earn['surprise_quartile'] = np.nan

        earn_stats = earn.groupby('surprise_quartile').agg(
            n=('srw_surprise', 'count'),
            mean_surprise=('srw_surprise', 'mean'),
            mean_car=('CAR_1_30', 'mean') if 'CAR_1_30' in earn.columns else ('srw_surprise', lambda x: np.nan),
        )
        result['earnings_surprise_stats'] = earn_stats

        # Compare purchase Q1 vs earnings bottom-surprise quartile
        purch_q1 = purchase_events.loc[purchase_events['quintile'] == 1, 'CAR_1_30'].dropna()
        earn_q1  = earn.loc[earn['surprise_quartile'] == 1, 'CAR_1_30'].dropna() \
            if 'CAR_1_30' in earn.columns else pd.Series(dtype=float)

        if len(purch_q1) > 1 and len(earn_q1) > 1:
            t_stat, p_val = sps.ttest_ind(purch_q1, earn_q1, equal_var=False)
            result['purchase_vs_earnings'] = {
                'mean_purchase_Q1': float(purch_q1.mean()),
                'mean_earnings_Q1': float(earn_q1.mean()),
                't_stat': float(t_stat),
                'p_value': float(p_val),
            }
        else:
            result['purchase_vs_earnings'] = {
                'mean_purchase_Q1': float(purch_q1.mean()) if len(purch_q1) > 0 else np.nan,
                'mean_earnings_Q1': np.nan,
                't_stat': np.nan,
                'p_value': np.nan,
            }
    else:
        result['earnings_surprise_stats'] = pd.DataFrame()
        result['purchase_vs_earnings']    = {}

    # ── Q5−Q1 gradient tests across groups ──────────────────────────────
    grad_rows: list[dict] = []

    def _gradient_test(grp_df: pd.DataFrame, label: str) -> None:
        """Append Q5−Q1 gradient row to grad_rows."""
        q1v = grp_df.loc[grp_df['quintile'] == 1, 'CAR_1_30'].dropna()
        q5v = grp_df.loc[grp_df['quintile'] == 5, 'CAR_1_30'].dropna()
        if len(q1v) > 1 and len(q5v) > 1:
            t_s, p_v = sps.ttest_ind(q5v, q1v, equal_var=False)
        else:
            t_s, p_v = np.nan, np.nan
        grad_rows.append({
            'group':  label,
            'Q1_mean': float(q1v.mean()) if len(q1v) > 0 else np.nan,
            'Q5_mean': float(q5v.mean()) if len(q5v) > 0 else np.nan,
            'gap':     float(q5v.mean() - q1v.mean()) if (len(q1v) > 0 and len(q5v) > 0) else np.nan,
            't_stat':  float(t_s),
            'p_value': float(p_v),
        })

    _gradient_test(purchase_events, 'Purchase')

    if not sales_events.empty and 'pct_from_52w_high' in sales_events.columns:
        sales_with_q2 = sales_events.copy()
        sales_with_q2['quintile'] = _quintile_assign(
            sales_with_q2['pct_from_52w_high'], quintile_cutoffs
        )
        _gradient_test(sales_with_q2.dropna(subset=['quintile']), 'Sales')

    result['gradient_tests'] = pd.DataFrame(grad_rows).set_index('group')

    return result


# ────────────────────────────────────────────────────────────────────────────
# Visualization
# ────────────────────────────────────────────────────────────────────────────

def plot_comparator_car_paths(
    purchase_ar_paths: pd.DataFrame,
    purchase_events: pd.DataFrame,
    sales_ar_paths: Optional[pd.DataFrame] = None,
    sales_events: Optional[pd.DataFrame] = None,
    placebo_car_df: Optional[pd.DataFrame] = None,
    quintile: int = 1,
    window: tuple[int, int] = (1, 60),
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Compare cumulative-AR paths for purchase, placebo, and sales in one quintile.

    Parameters
    ----------
    purchase_ar_paths:
        Wide AR-path DataFrame for the real purchase sample.
    purchase_events:
        Must contain ``quintile`` column.
    sales_ar_paths:
        Optional wide AR-path DataFrame for the insider sales comparator.
    sales_events:
        Optional event DataFrame for sales with ``quintile`` column.
    placebo_car_df:
        Optional DataFrame with ``event_date``, ``ticker``, ``quintile``,
        and AR-path columns (integer offsets).
    quintile:
        Quintile to display (default 1 = farthest from high).
    window:
        (t_start, t_end) relative to t0.
    figsize:
        Figure dimensions.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    t_start, t_end = window
    day_cols = [c for c in range(t_start, t_end + 1) if c in purchase_ar_paths.columns]

    fig, ax = plt.subplots(figsize=figsize)

    def _plot_path(ar_df: pd.DataFrame, ev_df: pd.DataFrame,
                   label: str, color: str, linestyle: str = '-') -> None:
        mask = ev_df['quintile'] == quintile
        sub = ar_df.loc[mask, day_cols]
        if sub.empty:
            return
        cum = sub.cumsum(axis=1)
        mean_p = cum.mean()
        n = mask.sum()
        se = cum.std() / np.sqrt(n)
        ax.plot(day_cols, mean_p.values, color=color, linestyle=linestyle, label=label)
        ax.fill_between(day_cols, (mean_p - 1.96 * se).values,
                        (mean_p + 1.96 * se).values, color=color, alpha=0.12)

    _plot_path(purchase_ar_paths, purchase_events, f'Purchase Q{quintile}',
               color='#2196F3', linestyle='-')

    if sales_ar_paths is not None and sales_events is not None:
        _plot_path(sales_ar_paths, sales_events, f'Sales Q{quintile}',
                   color='#F44336', linestyle='--')

    if placebo_car_df is not None and 'quintile' in placebo_car_df.columns:
        sub_p = placebo_car_df.loc[placebo_car_df['quintile'] == quintile]
        placebo_cols = [c for c in day_cols if c in placebo_car_df.columns]
        if placebo_cols and not sub_p.empty:
            cum_p = sub_p[placebo_cols].cumsum(axis=1)
            mean_pp = cum_p.mean()
            se_pp = cum_p.std() / np.sqrt(len(sub_p))
            ax.plot(placebo_cols, mean_pp.values, color='#4CAF50',
                    linestyle=':', label=f'Placebo Q{quintile}')
            ax.fill_between(placebo_cols, (mean_pp - 1.96 * se_pp).values,
                            (mean_pp + 1.96 * se_pp).values, color='#4CAF50', alpha=0.12)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Trading day relative to filing (t₀)')
    ax.set_ylabel('Mean cumulative abnormal return')
    ax.set_title(f'Comparator CAR paths — Q{quintile} (farthest from 52-week high)')
    ax.legend(framealpha=0.7)
    fig.tight_layout()
    return fig
