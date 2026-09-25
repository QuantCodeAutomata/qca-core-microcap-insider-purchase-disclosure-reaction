"""
event_study_tables.py

Event study analysis and table generation for the microcap insider purchase
disclosure study.

Data conventions
----------------
ar_paths : pd.DataFrame
    Wide-format abnormal-return paths.  Index aligns with ``events.index``
    (one row per event).  Columns are integer day offsets relative to t0
    (e.g. -30, -29, ..., 0, 1, ..., 60).  Values are *daily* abnormal
    returns (not cumulated).

cars : pd.DataFrame
    Pre-computed cumulative-abnormal-return aggregates.  Index aligns with
    ``events.index``.  Expected columns (others are ignored):
        CAR_1_30   – CAR over days [t0+1, t0+30]
        CAR_1_60   – CAR over days [t0+1, t0+60]
        CAR_2_30   – CAR over days [t0+2, t0+30]  (drift)
        CAR_2_60   – CAR over days [t0+2, t0+60]  (drift)
        AR_0       – abnormal return on filing day t0
        AR_1       – abnormal return on t0+1

events : pd.DataFrame
    One row per event.  Expected columns include (at minimum):
        ticker, filing_date, pct_from_52w_high, vol_30d,
        is_preclose (bool), filing_yearmonth (str "YYYY-MM")
"""

from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
import statsmodels.api as sm

# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _car_window(ar_paths: pd.DataFrame, t_start: int, t_end: int) -> pd.Series:
    """Return CAR summed from *t_start* to *t_end* (both inclusive)."""
    cols = [c for c in range(t_start, t_end + 1) if c in ar_paths.columns]
    return ar_paths[cols].sum(axis=1)


def _cluster_var_of_mean(values: np.ndarray, group_ids: np.ndarray) -> float:
    """One-way clustered variance of the sample mean.

    V = (1/n²) * Σ_g (Σ_{i∈g} eᵢ)²    where eᵢ = yᵢ − ȳ
    """
    n = len(values)
    if n == 0:
        return np.nan
    mu = values.mean()
    resids = values - mu
    df_tmp = pd.DataFrame({'r': resids, 'g': group_ids})
    cluster_sums = df_tmp.groupby('g')['r'].sum()
    return float((cluster_sums ** 2).sum()) / (n ** 2)


def _welch_ttest_vs_zero(series: pd.Series) -> tuple[float, float]:
    """Return (t-statistic, two-sided p-value) for H₀: μ = 0."""
    n = series.dropna().__len__()
    if n < 2:
        return np.nan, np.nan
    result = sps.ttest_1samp(series.dropna(), 0.0)
    return float(result.statistic), float(result.pvalue)


def _finite_cluster_correction(n_clusters: int) -> float:
    """Small-sample degrees-of-freedom multiplier  G/(G−1)."""
    if n_clusters <= 1:
        return 1.0
    return n_clusters / (n_clusters - 1)


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def form_quintiles(
    events: pd.DataFrame,
    feature_col: str = 'pct_from_52w_high',
) -> pd.DataFrame:
    """Assign each event to a sample-quantile-based quintile.

    Q1 = farthest below the 52-week high (most negative *pct_from_52w_high*).
    Q5 = nearest to the 52-week high (least negative, closest to 0).

    Parameters
    ----------
    events:
        Event-level DataFrame that must contain *feature_col*.
    feature_col:
        Column used to rank events.  Must be numeric.

    Returns
    -------
    Copy of *events* with an added integer column ``quintile`` ∈ {1, 2, 3, 4, 5}
    and a ``quintile_cutoffs`` attribute (5-element array of bin edges stored
    as ``events.attrs['quintile_cutoffs']``).
    """
    df = events.copy()
    labels = [1, 2, 3, 4, 5]
    quintile_col, bins = pd.qcut(
        df[feature_col], q=5, labels=labels,
        retbins=True, duplicates='drop',
    )
    df['quintile'] = quintile_col.astype(int)
    df.attrs['quintile_cutoffs'] = bins
    return df


def quintile_summary_table(
    events: pd.DataFrame,
    ar_paths: pd.DataFrame,
    cars: pd.DataFrame,
) -> pd.DataFrame:
    """Generate per-quintile summary statistics for the event study.

    For each quintile the table reports:
    - Count of events
    - Mean and median CAR_[1,30]
    - Mean AR on t0+1 (``AR_1``)
    - Mean drift [2, 30]  and drift [2, 60]
    - Positive-tail rate  (CAR_[1,30] > +10 %)
    - Negative-tail rate  (CAR_[1,30] < −10 %)
    - 10th / 25th / 75th / 90th percentiles of CAR_[1,30]
    - Issuer-clustered SE and t-stat for mean CAR_[1,30]

    Parameters
    ----------
    events:
        Must contain ``quintile`` and ``ticker`` columns.
    ar_paths:
        Wide AR-path DataFrame (see module docstring).
    cars:
        Pre-computed CARs (see module docstring).

    Returns
    -------
    DataFrame indexed by quintile (1–5) with one column per metric.
    """
    required_cols = {'quintile', 'ticker'}
    missing = required_cols - set(events.columns)
    if missing:
        raise ValueError(f"events is missing columns: {missing}")

    merged = (
        events[['quintile', 'ticker']]
        .join(cars[['CAR_1_30', 'CAR_1_60', 'CAR_2_30', 'CAR_2_60',
                     'AR_0', 'AR_1']], how='left')
    )

    # Rebuild drift columns from ar_paths if not in cars
    if 'CAR_2_30' not in cars.columns:
        merged['CAR_2_30'] = _car_window(ar_paths, 2, 30)
    if 'CAR_2_60' not in cars.columns:
        merged['CAR_2_60'] = _car_window(ar_paths, 2, 60)
    if 'AR_1' not in cars.columns:
        if 1 in ar_paths.columns:
            merged['AR_1'] = ar_paths[1]

    rows = []
    for q in range(1, 6):
        mask = merged['quintile'] == q
        sub = merged.loc[mask].copy()
        car = sub['CAR_1_30'].dropna()
        tickers = sub.loc[car.index, 'ticker'].values

        n = len(car)
        mean_car = car.mean()
        median_car = car.median()

        # Issuer-clustered SE
        if n > 1 and len(np.unique(tickers)) > 1:
            n_clust = len(np.unique(tickers))
            v_clust = _cluster_var_of_mean(car.values, tickers) * _finite_cluster_correction(n_clust)
            se_clust = np.sqrt(v_clust)
        else:
            se_clust = car.std(ddof=1) / np.sqrt(max(n, 1))

        t_clust = mean_car / se_clust if se_clust > 0 else np.nan
        p_clust = 2 * sps.t.sf(abs(t_clust), df=max(len(np.unique(tickers)) - 1, 1))

        welch_t, welch_p = _welch_ttest_vs_zero(car)

        rows.append({
            'quintile':         q,
            'n_events':         n,
            'mean_CAR_1_30':    mean_car,
            'median_CAR_1_30':  median_car,
            'se_CAR_1_30':      se_clust,
            't_CAR_1_30':       t_clust,
            'p_CAR_1_30':       p_clust,
            'welch_t':          welch_t,
            'welch_p':          welch_p,
            'mean_AR_1':        sub['AR_1'].mean() if 'AR_1' in sub else np.nan,
            'mean_drift_2_30':  sub['CAR_2_30'].mean() if 'CAR_2_30' in sub else np.nan,
            'mean_drift_2_60':  sub['CAR_2_60'].mean() if 'CAR_2_60' in sub else np.nan,
            'pos_tail_rate':    (car > 0.10).mean(),
            'neg_tail_rate':    (car < -0.10).mean(),
            'p10':              car.quantile(0.10),
            'p25':              car.quantile(0.25),
            'p75':              car.quantile(0.75),
            'p90':              car.quantile(0.90),
        })

    return pd.DataFrame(rows).set_index('quintile')


def announcement_decomposition(
    events: pd.DataFrame,
    ar_paths: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose the announcement-window return into its components.

    Components computed per event
    ------------------------------
    - ``AR_t0``        : filing-day abnormal return (pre-close filings only;
                         NaN for after-close filings).
    - ``AR_t1``        : abnormal return on t0+1 (all events).
    - ``overnight_gap``: raw open-to-prior-close gap = open(t0+1)/close(t0) − 1.
    - ``intraday_t1``  : raw intraday return = close(t0+1)/open(t0+1) − 1.
    - ``drift_2_30``   : cumulative AR over days [t0+2, t0+30].
    - ``drift_2_60``   : cumulative AR over days [t0+2, t0+60].

    Parameters
    ----------
    events:
        Must contain ``is_preclose`` (bool) and optionally ``open_t1``,
        ``close_t0``, ``close_t1`` for the raw overnight / intraday splits.
    ar_paths:
        Wide AR-path DataFrame (integer column offsets, see module docstring).

    Returns
    -------
    DataFrame indexed like *events* with one column per component.
    """
    result = pd.DataFrame(index=events.index)

    # Filing-day AR: only meaningful for pre-close filings
    if 0 in ar_paths.columns:
        ar_t0 = ar_paths[0].copy().astype(float)
        if 'is_preclose' in events.columns:
            ar_t0.loc[~events['is_preclose'].astype(bool)] = np.nan
        result['AR_t0'] = ar_t0

    # AR on first post-filing day (always)
    if 1 in ar_paths.columns:
        result['AR_t1'] = ar_paths[1]

    # Raw overnight gap and intraday return (require OHLC columns in events)
    for raw_col, evt_col in [
        ('overnight_gap', ('open_t1', 'close_t0')),
        ('intraday_t1',   ('close_t1', 'open_t1')),
    ]:
        num_col, den_col = evt_col
        if num_col in events.columns and den_col in events.columns:
            result[raw_col] = (
                events[num_col].astype(float) / events[den_col].astype(float) - 1
            )
        else:
            result[raw_col] = np.nan

    # Post-announcement drift
    result['drift_2_30'] = _car_window(ar_paths, 2, 30)
    result['drift_2_60'] = _car_window(ar_paths, 2, 60)

    return result


def volatility_decomposition(
    events: pd.DataFrame,
    cars: pd.DataFrame,
) -> pd.DataFrame:
    """Express announcement returns as volatility-standardised z-scores by quintile.

    Metrics
    -------
    z_day1 = AR_(t0+1) / (vol_30d / √252)
        Standardises the day-1 abnormal return by daily volatility.
    z_30   = CAR_[1,30] / (vol_30d * √(30/252))
        Standardises the 30-day CAR by the annualised volatility scaled to
        the 30-trading-day horizon.

    Parameters
    ----------
    events:
        Must contain ``quintile`` and ``vol_30d`` (annualised, as a fraction).
    cars:
        Must contain ``AR_1`` and ``CAR_1_30``.

    Returns
    -------
    DataFrame indexed by quintile (1–5) with columns for mean z_day1, mean z_30,
    and issuer-clustered SEs.
    """
    if 'vol_30d' not in events.columns:
        raise ValueError("events must contain 'vol_30d'")

    daily_vol = events['vol_30d'] / np.sqrt(252)
    horizon_vol = events['vol_30d'] * np.sqrt(30 / 252)

    merged = events[['quintile', 'ticker', 'vol_30d']].copy()
    if 'AR_1' in cars.columns:
        merged['z_day1'] = cars['AR_1'] / daily_vol
    else:
        merged['z_day1'] = np.nan

    if 'CAR_1_30' in cars.columns:
        merged['z_30'] = cars['CAR_1_30'] / horizon_vol
    else:
        merged['z_30'] = np.nan

    rows = []
    for q in range(1, 6):
        sub = merged.loc[merged['quintile'] == q]
        tickers = sub['ticker'].values if 'ticker' in sub else np.arange(len(sub))

        def _mean_se(col_name: str) -> tuple[float, float]:
            vals = sub[col_name].dropna()
            if vals.empty:
                return np.nan, np.nan
            mean_v = vals.mean()
            t_ids = sub.loc[vals.index, 'ticker'].values if 'ticker' in sub else np.arange(len(vals))
            n_c = len(np.unique(t_ids))
            v = _cluster_var_of_mean(vals.values, t_ids) * _finite_cluster_correction(n_c)
            return mean_v, float(np.sqrt(v))

        mean_z1, se_z1 = _mean_se('z_day1')
        mean_z30, se_z30 = _mean_se('z_30')

        rows.append({
            'quintile':    q,
            'mean_z_day1': mean_z1,
            'se_z_day1':   se_z1,
            't_z_day1':    mean_z1 / se_z1 if se_z1 > 0 else np.nan,
            'mean_z_30':   mean_z30,
            'se_z_30':     se_z30,
            't_z_30':      mean_z30 / se_z30 if se_z30 > 0 else np.nan,
        })

    return pd.DataFrame(rows).set_index('quintile')


def benchmark_sensitivity(
    events: pd.DataFrame,
    cars: pd.DataFrame,
    alpha_hats: pd.Series,
    window: int = 30,
) -> pd.DataFrame:
    """Compare loadings-only vs intercept-inclusive CAR gradients.

    Definitions
    -----------
    CAR_primary    (loadings-only): already stored in ``cars['CAR_1_30']``.
    CAR_inclusive  = CAR_primary − α̂ × window
        Subtracting the alpha carry gives the "fairer" benchmark.

    Q5−Q1 gap decomposition
    -----------------------
    raw_gap       = mean(CAR_primary, Q5) − mean(CAR_primary, Q1)
    carry_diff    = −window × [mean(α̂, Q5) − mean(α̂, Q1)]
    adjusted_gap  = raw_gap + carry_diff

    Parameters
    ----------
    events:
        Must contain ``quintile`` and ``ticker``.
    cars:
        Must contain ``CAR_1_30``.
    alpha_hats:
        Series indexed like *events* with per-event estimated FF3 intercept
        (annualised or in daily units consistent with the CARs).
    window:
        Trading days used for the CAR (default 30).

    Returns
    -------
    DataFrame with one row per quintile plus a summary row for Q5−Q1,
    reporting mean primary CAR, mean inclusive CAR, mean alpha carry, and
    clustered SEs.
    """
    if 'CAR_1_30' not in cars.columns:
        raise ValueError("cars must contain 'CAR_1_30'")

    merged = events[['quintile', 'ticker']].copy()
    merged['CAR_primary'] = cars['CAR_1_30']
    merged['alpha_hat'] = alpha_hats
    merged['alpha_carry'] = -alpha_hats * window
    merged['CAR_inclusive'] = merged['CAR_primary'] + merged['alpha_carry']

    rows = []
    quintile_means: dict[int, dict] = {}
    for q in range(1, 6):
        sub = merged.loc[merged['quintile'] == q].dropna(subset=['CAR_primary'])
        tickers = sub['ticker'].values if 'ticker' in sub else np.arange(len(sub))
        n_c = len(np.unique(tickers))

        def _stat(col: str) -> tuple[float, float]:
            vals = sub[col].dropna().values
            if len(vals) == 0:
                return np.nan, np.nan
            mean_v = vals.mean()
            t_ids = sub.loc[sub[col].dropna().index, 'ticker'].values if 'ticker' in sub else np.arange(len(vals))
            v = _cluster_var_of_mean(vals, t_ids) * _finite_cluster_correction(len(np.unique(t_ids)))
            return mean_v, float(np.sqrt(v))

        mean_prim, se_prim       = _stat('CAR_primary')
        mean_incl, se_incl       = _stat('CAR_inclusive')
        mean_carry, se_carry     = _stat('alpha_carry')
        alpha_mean               = sub['alpha_hat'].mean()

        quintile_means[q] = dict(
            mean_primary=mean_prim, mean_inclusive=mean_incl,
            mean_alpha=alpha_mean, mean_carry=mean_carry,
        )
        rows.append({
            'quintile':       q,
            'n':              len(sub),
            'mean_primary':   mean_prim,
            'se_primary':     se_prim,
            'mean_inclusive': mean_incl,
            'se_inclusive':   se_incl,
            'mean_alpha_hat': alpha_mean,
            'mean_carry':     mean_carry,
        })

    # Q5 − Q1 gap row
    q5, q1 = quintile_means.get(5, {}), quintile_means.get(1, {})
    raw_gap   = q5.get('mean_primary', np.nan)   - q1.get('mean_primary', np.nan)
    carry_diff = q5.get('mean_carry', np.nan)     - q1.get('mean_carry', np.nan)
    adj_gap   = q5.get('mean_inclusive', np.nan)  - q1.get('mean_inclusive', np.nan)

    rows.append({
        'quintile':       'Q5-Q1',
        'n':              np.nan,
        'mean_primary':   raw_gap,
        'se_primary':     np.nan,
        'mean_inclusive': adj_gap,
        'se_inclusive':   np.nan,
        'mean_alpha_hat': np.nan,
        'mean_carry':     carry_diff,
    })

    return pd.DataFrame(rows).set_index('quintile')


def run_two_way_clustered_ttest(
    data: pd.DataFrame,
    outcome: str,
    group1: str = 'ticker',
    group2: str = 'filing_yearmonth',
) -> dict:
    """Two-way clustered t-test for H₀: E[*outcome*] = 0.

    Uses the Cameron–Gelbach–Miller (2011) variance estimator:

        V_2way = V_{g1} + V_{g2} − V_{g1∩g2}

    with finite-cluster (G/(G−1)) corrections applied independently to
    each one-way component.  The intersection clusters (unique
    (g1, g2) pairs) receive their own correction.

    Parameters
    ----------
    data:
        DataFrame containing *outcome* and both grouping columns.
    outcome:
        Numeric column whose mean is tested against zero.
    group1:
        First clustering dimension (default: issuer ticker).
    group2:
        Second clustering dimension (default: year-month string "YYYY-MM").

    Returns
    -------
    dict with keys:
        estimate, se_g1, se_g2, se_intersection, se_2way,
        t_stat, p_value, n, n_g1, n_g2, n_intersection,
        welch_t, welch_p.
    """
    sub = data[[outcome, group1, group2]].dropna()
    if sub.empty:
        return {k: np.nan for k in ['estimate', 'se_g1', 'se_g2', 'se_intersection',
                                     'se_2way', 't_stat', 'p_value', 'n',
                                     'n_g1', 'n_g2', 'n_intersection',
                                     'welch_t', 'welch_p']}

    y = sub[outcome].values.astype(float)
    g1 = sub[group1].values
    g2 = sub[group2].values
    g12 = np.array([f"{a}__{b}" for a, b in zip(g1, g2)])

    n = len(y)
    mu = y.mean()

    n_g1  = len(np.unique(g1))
    n_g2  = len(np.unique(g2))
    n_g12 = len(np.unique(g12))

    v_g1  = _cluster_var_of_mean(y, g1)  * _finite_cluster_correction(n_g1)
    v_g2  = _cluster_var_of_mean(y, g2)  * _finite_cluster_correction(n_g2)
    v_g12 = _cluster_var_of_mean(y, g12) * _finite_cluster_correction(n_g12)

    v_2way = v_g1 + v_g2 - v_g12
    # Guard against numerical noise producing tiny negative variance
    v_2way = max(v_2way, 0.0)
    se_2way = np.sqrt(v_2way)

    # Use min(G1, G2) − 1 for df (conservative)
    df = min(n_g1, n_g2) - 1
    t_stat = mu / se_2way if se_2way > 0 else np.nan
    p_value = float(2 * sps.t.sf(abs(t_stat), df=max(df, 1)))

    welch_t, welch_p = _welch_ttest_vs_zero(pd.Series(y))

    return dict(
        estimate=mu,
        se_g1=np.sqrt(v_g1),
        se_g2=np.sqrt(v_g2),
        se_intersection=np.sqrt(v_g12),
        se_2way=se_2way,
        t_stat=t_stat,
        p_value=p_value,
        n=n,
        n_g1=n_g1,
        n_g2=n_g2,
        n_intersection=n_g12,
        welch_t=welch_t,
        welch_p=welch_p,
    )


# ────────────────────────────────────────────────────────────────────────────
# Visualization
# ────────────────────────────────────────────────────────────────────────────

def plot_car_paths_by_quintile(
    ar_paths: pd.DataFrame,
    events: pd.DataFrame,
    window: tuple[int, int] = (-5, 60),
    figsize: tuple[float, float] = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot mean cumulative-AR paths by quintile with 95 % confidence bands.

    Parameters
    ----------
    ar_paths:
        Wide AR-path DataFrame (integer column offsets).
    events:
        Must contain ``quintile`` column.
    window:
        (start_day, end_day) inclusive, relative to t0.
    figsize:
        Figure size passed to ``plt.subplots`` when *ax* is None.
    ax:
        Existing Axes to draw on; a new figure is created if None.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    t_start, t_end = window
    day_cols = [c for c in range(t_start, t_end + 1) if c in ar_paths.columns]

    palette = sns.color_palette('viridis', 5)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for q in range(1, 6):
        mask = events['quintile'] == q
        sub = ar_paths.loc[mask, day_cols]
        # Accumulate from the start of the window
        cum = sub.cumsum(axis=1)
        mean_path = cum.mean()
        n = mask.sum()
        stderr = cum.std() / np.sqrt(n)

        ax.plot(day_cols, mean_path.values, color=palette[q - 1], label=f'Q{q}')
        ax.fill_between(
            day_cols,
            (mean_path - 1.96 * stderr).values,
            (mean_path + 1.96 * stderr).values,
            color=palette[q - 1], alpha=0.15,
        )

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', label='t₀ (filing)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Trading day relative to filing (t₀)')
    ax.set_ylabel('Mean cumulative abnormal return')
    ax.set_title('CAR paths by distance-to-52-week-high quintile')
    ax.legend(title='Quintile', framealpha=0.7)
    fig.tight_layout()
    return fig


def plot_quintile_summary_bar(
    summary_df: pd.DataFrame,
    metric: str = 'mean_CAR_1_30',
    se_metric: Optional[str] = 'se_CAR_1_30',
    figsize: tuple[float, float] = (7, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Bar chart of a summary metric across quintiles with optional error bars.

    Parameters
    ----------
    summary_df:
        Output of :func:`quintile_summary_table` (indexed by quintile 1–5).
    metric:
        Column in *summary_df* to plot.
    se_metric:
        Column containing standard errors; set to None to omit error bars.
    figsize:
        Figure size.
    ax:
        Existing Axes to draw on.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    quintiles = summary_df.index.tolist()
    values = summary_df[metric].values
    yerr = summary_df[se_metric].values * 1.96 if se_metric and se_metric in summary_df else None

    colors = sns.color_palette('viridis', len(quintiles))
    bars = ax.bar(quintiles, values, color=colors, alpha=0.85,
                  yerr=yerr, capsize=4, error_kw={'elinewidth': 1.2})
    ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
    ax.set_xticks(quintiles)
    ax.set_xticklabels([f'Q{q}' for q in quintiles])
    ax.set_xlabel('Distance-to-52-week-high quintile')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(f'{metric.replace("_", " ")} by quintile (±95 % CI)')
    fig.tight_layout()
    return fig


def plot_announcement_decomposition(
    decomp_df: pd.DataFrame,
    events: pd.DataFrame,
    figsize: tuple[float, float] = (9, 5),
) -> plt.Figure:
    """Stacked bar chart of mean announcement-window components by quintile.

    Parameters
    ----------
    decomp_df:
        Output of :func:`announcement_decomposition`, indexed like *events*.
    events:
        Must contain ``quintile`` column, aligned with *decomp_df*.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    components = ['AR_t0', 'AR_t1', 'drift_2_30']
    labels = ['Filing-day AR (t₀)', 'Day+1 AR (t₁)', 'Drift [2,30]']
    colours = ['#4C72B0', '#DD8452', '#55A868']

    merged = decomp_df.join(events[['quintile']])
    group = merged.groupby('quintile')[components].mean()

    fig, ax = plt.subplots(figsize=figsize)
    bottoms = np.zeros(len(group))
    for col, label, colour in zip(components, labels, colours):
        vals = group[col].fillna(0).values
        ax.bar(group.index, vals, bottom=bottoms, label=label, color=colour, alpha=0.85)
        bottoms += vals

    ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
    ax.set_xticks(group.index)
    ax.set_xticklabels([f'Q{q}' for q in group.index])
    ax.set_xlabel('Distance-to-52-week-high quintile')
    ax.set_ylabel('Mean return component')
    ax.set_title('Announcement return decomposition by quintile')
    ax.legend(framealpha=0.7)
    fig.tight_layout()
    return fig
