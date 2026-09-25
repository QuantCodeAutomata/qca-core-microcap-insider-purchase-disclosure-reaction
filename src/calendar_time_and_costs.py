"""
calendar_time_and_costs.py
==========================
Calendar-time portfolio analyses and transaction-cost / capacity diagnostics
for the microcap insider-purchase disclosure study.

All functions accept tidy pandas DataFrames and return tidy results so that
callers can compose them freely.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annualised_bdays(n: int = 252) -> int:
    return n


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """Raise ValueError when *required* columns are absent from *df*."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


# ---------------------------------------------------------------------------
# 1. build_calendar_time_portfolio
# ---------------------------------------------------------------------------

def build_calendar_time_portfolio(
    events: pd.DataFrame,
    ar_paths: pd.DataFrame,
    holding_start: int = 1,
    holding_end: int = 30,
    weight_scheme: str = "equal",
) -> pd.DataFrame:
    """Build a calendar-time portfolio of post-filing positions.

    For every calendar day present in *ar_paths*, the function identifies all
    events whose holding window ``[t0 + holding_start, t0 + holding_end]``
    brackets that day, then aggregates individual abnormal returns into a
    single daily portfolio return.

    A Q1-only variant (events whose transaction-value quintile == 1) is
    included as an additional column.

    Parameters
    ----------
    events : pd.DataFrame
        Must contain columns ``event_id``, ``t0`` (datetime), and optionally
        ``market_cap`` (float, USD) for value-weighting, ``quintile`` (int).
    ar_paths : pd.DataFrame
        Tidy frame with columns ``event_id``, ``date`` (datetime), ``ar``
        (abnormal return, float).  Each row is one event × one calendar day.
    holding_start : int
        First holding day relative to *t0* (inclusive).  Default 1.
    holding_end : int
        Last holding day relative to *t0* (inclusive).  Default 30.
    weight_scheme : str
        ``'equal'`` (default) or ``'value'`` (market-cap weighted).

    Returns
    -------
    pd.DataFrame
        Index: calendar date.
        Columns: ``port_return`` (float), ``port_return_q1`` (float),
        ``n_positions`` (int), ``n_positions_q1`` (int).

    Raises
    ------
    ValueError
        If required columns are missing or *weight_scheme* is unrecognised.
    """
    if weight_scheme not in {"equal", "value"}:
        raise ValueError(f"weight_scheme must be 'equal' or 'value', got '{weight_scheme}'")

    _validate_columns(events, ["event_id", "t0"], "events")
    _validate_columns(ar_paths, ["event_id", "date", "ar"], "ar_paths")

    events = events.copy()
    events["t0"] = pd.to_datetime(events["t0"])
    events["entry_date"] = events["t0"] + pd.to_timedelta(holding_start, unit="D")
    events["exit_date"] = events["t0"] + pd.to_timedelta(holding_end, unit="D")

    has_quintile = "quintile" in events.columns
    has_mktcap = "market_cap" in events.columns

    # Merge ar_paths with event metadata once
    merged = ar_paths.merge(
        events[
            ["event_id", "entry_date", "exit_date"]
            + (["quintile"] if has_quintile else [])
            + (["market_cap"] if has_mktcap else [])
        ],
        on="event_id",
        how="left",
    )
    merged["date"] = pd.to_datetime(merged["date"])

    # Keep only rows inside the holding window
    mask = (merged["date"] >= merged["entry_date"]) & (merged["date"] <= merged["exit_date"])
    active = merged.loc[mask].copy()

    logger.info(
        "Calendar-time portfolio: %d event-day observations across %d events",
        len(active),
        active["event_id"].nunique(),
    )

    def _aggregate(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["date", f"port_return_{label}", f"n_positions_{label}"])

        if weight_scheme == "equal":
            daily = (
                frame.groupby("date")["ar"]
                .agg(port_return="mean", n_positions="count")
                .reset_index()
            )
        else:  # value-weighted
            if not has_mktcap:
                raise ValueError("weight_scheme='value' requires 'market_cap' column in events")
            frame = frame.copy()
            frame["w"] = frame.groupby("date")["market_cap"].transform(lambda x: x / x.sum())
            daily = frame.groupby("date").apply(
                lambda g: pd.Series(
                    {
                        "port_return": (g["ar"] * g["w"]).sum(),
                        "n_positions": len(g),
                    }
                )
            ).reset_index()

        daily = daily.rename(
            columns={
                "port_return": f"port_return_{label}",
                "n_positions": f"n_positions_{label}",
            }
        )
        return daily

    all_dates = pd.DataFrame({"date": sorted(active["date"].unique())})

    # All-events portfolio
    agg_all = _aggregate(active, "all")
    result = all_dates.merge(agg_all, on="date", how="left")

    # Q1-only portfolio
    if has_quintile:
        q1_active = active[active["quintile"] == 1]
        agg_q1 = _aggregate(q1_active, "q1")
        result = result.merge(agg_q1, on="date", how="left")
    else:
        logger.warning("No 'quintile' column in events; skipping Q1 portfolio.")
        result["port_return_q1"] = np.nan
        result["n_positions_q1"] = 0

    result = result.rename(
        columns={
            "port_return_all": "port_return",
            "n_positions_all": "n_positions",
        }
    )
    result = result.set_index("date").sort_index()
    result[["port_return", "port_return_q1"]] = result[
        ["port_return", "port_return_q1"]
    ].fillna(0.0)
    result[["n_positions", "n_positions_q1"]] = result[
        ["n_positions", "n_positions_q1"]
    ].fillna(0).astype(int)

    logger.info("Portfolio built: %d calendar days", len(result))
    return result


# ---------------------------------------------------------------------------
# 2. estimate_calendar_alpha
# ---------------------------------------------------------------------------

def estimate_calendar_alpha(
    portfolio_returns: pd.DataFrame,
    ff3: pd.DataFrame,
    use_umd: bool = False,
    hac_lags: int = 5,
) -> dict:
    """Regress portfolio excess returns on Fama-French 3-factor (+ UMD) model.

    Uses Newey-West heteroskedasticity- and autocorrelation-consistent (HAC)
    standard errors.  The intercept (alpha) is estimated in daily units and
    then scaled to a monthly figure by multiplying by 21 trading days.

    Parameters
    ----------
    portfolio_returns : pd.DataFrame
        Must contain columns ``port_return`` (float) and ``RF`` (daily risk-free
        rate, float).  Index must be calendar date.
    ff3 : pd.DataFrame
        Fama-French factor returns.  Must contain ``Mkt-RF``, ``SMB``, ``HML``
        (and ``UMD`` if *use_umd* is True) plus ``RF``.  Index is date.
    use_umd : bool
        If True, include UMD (momentum) factor.  Default False.
    hac_lags : int
        Number of Newey-West lags.  Default 5.

    Returns
    -------
    dict
        Keys: ``alpha_daily``, ``alpha_monthly``, ``t_stat``, ``p_value``,
        ``r2``, ``n_obs``, ``factors``.

    Raises
    ------
    ValueError
        If required columns are missing after merging.
    """
    _validate_columns(portfolio_returns, ["port_return"], "portfolio_returns")
    factors = ["Mkt-RF", "SMB", "HML"]
    if use_umd:
        factors.append("UMD")
    _validate_columns(ff3, factors + ["RF"], "ff3")

    port = portfolio_returns[["port_return"]].copy()
    ff = ff3[factors + ["RF"]].copy()

    data = port.join(ff, how="inner").dropna()
    if data.empty:
        raise ValueError("No overlapping dates between portfolio_returns and ff3.")

    logger.info("Calendar-alpha regression: %d observations, factors=%s, HAC lags=%d",
                len(data), factors, hac_lags)

    data["excess_return"] = data["port_return"] - data["RF"]
    y = data["excess_return"]
    X = sm.add_constant(data[factors])

    model = sm.OLS(y, X)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    alpha_daily = float(res.params["const"])
    alpha_monthly = alpha_daily * 21
    t_stat = float(res.tvalues["const"])
    p_value = float(res.pvalues["const"])
    r2 = float(res.rsquared)

    logger.info(
        "Alpha: daily=%.6f, monthly=%.4f, t=%.3f, p=%.4f, R²=%.4f",
        alpha_daily, alpha_monthly, t_stat, p_value, r2,
    )

    return {
        "alpha_daily": alpha_daily,
        "alpha_monthly": alpha_monthly,
        "t_stat": t_stat,
        "p_value": p_value,
        "r2": r2,
        "n_obs": len(data),
        "factors": factors,
    }


# ---------------------------------------------------------------------------
# 3. skip_day_one_variant
# ---------------------------------------------------------------------------

def skip_day_one_variant(
    events: pd.DataFrame,
    ar_paths: pd.DataFrame,
) -> pd.DataFrame:
    """Build a calendar-time portfolio that skips the first day after filing.

    Constructs the portfolio over ``[t0+2, t0+31]`` (equal-weighted) to
    assess whether the day-1 price gap drives the full-sample alpha.

    Parameters
    ----------
    events : pd.DataFrame
        Same schema as :func:`build_calendar_time_portfolio`.
    ar_paths : pd.DataFrame
        Same schema as :func:`build_calendar_time_portfolio`.

    Returns
    -------
    pd.DataFrame
        Daily portfolio returns indexed by calendar date, same schema as
        :func:`build_calendar_time_portfolio`.
    """
    logger.info("Building skip-day-one portfolio (t0+2 to t0+31) …")
    return build_calendar_time_portfolio(
        events=events,
        ar_paths=ar_paths,
        holding_start=2,
        holding_end=31,
        weight_scheme="equal",
    )


# ---------------------------------------------------------------------------
# 4. transaction_cost_analysis
# ---------------------------------------------------------------------------

def transaction_cost_analysis(
    events: pd.DataFrame,
    cars: pd.DataFrame,
    cost_bps_grid: list[int] = None,
) -> pd.DataFrame:
    """Show net-of-cost performance across a grid of round-trip cost assumptions.

    Round-trip transaction costs (entry + exit spread/commission) are deducted
    from each event's CAR[1,30].  Summary statistics across all events are
    reported for each cost level.

    Parameters
    ----------
    events : pd.DataFrame
        Must contain ``event_id`` and optionally ``quintile``.
    cars : pd.DataFrame
        Must contain ``event_id`` and ``car_1_30`` (cumulative abnormal return
        over the [1, 30] window, as a decimal fraction).
    cost_bps_grid : list[int]
        Round-trip cost in basis points to evaluate.
        Default ``[0, 10, 25, 50, 100]``.

    Returns
    -------
    pd.DataFrame
        Rows: one per cost level.
        Columns: ``cost_bps``, ``mean_net_car``, ``median_net_car``,
        ``pct_positive``, ``sharpe_proxy``.
    """
    if cost_bps_grid is None:
        cost_bps_grid = [0, 10, 25, 50, 100]

    _validate_columns(cars, ["event_id", "car_1_30"], "cars")
    _validate_columns(events, ["event_id"], "events")

    data = events[["event_id"]].merge(cars[["event_id", "car_1_30"]], on="event_id", how="inner")
    logger.info("Transaction-cost analysis: %d events", len(data))

    rows = []
    for bps in cost_bps_grid:
        rt_cost = bps / 10_000
        net_car = data["car_1_30"] - rt_cost
        rows.append(
            {
                "cost_bps": bps,
                "mean_net_car": float(net_car.mean()),
                "median_net_car": float(net_car.median()),
                "pct_positive": float((net_car > 0).mean() * 100),
                "sharpe_proxy": float(net_car.mean() / net_car.std()) if net_car.std() > 0 else np.nan,
                "n_events": len(net_car),
            }
        )

    result = pd.DataFrame(rows)
    logger.info("Transaction-cost grid:\n%s", result.to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# 5. capacity_diagnostics
# ---------------------------------------------------------------------------

def capacity_diagnostics(events: pd.DataFrame) -> pd.DataFrame:
    """Compute capacity and market-impact diagnostics for the strategy.

    Reports, for each event, the hypothetical position size as a share of
    average daily dollar volume (ADDV), plus aggregate inventory and
    concentration statistics.

    Parameters
    ----------
    events : pd.DataFrame
        Must contain ``event_id``, ``addv`` (average daily dollar volume,
        USD), and ``position_size`` (intended dollar notional per event, USD).
        Optionally ``quintile`` for quintile-level breakdown.

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        ``pct_of_addv_mean``, ``pct_of_addv_max``, ``pct_of_addv_p95``,
        ``avg_inventory_usd``, ``peak_inventory_usd``,
        ``herfindahl_concentration``, ``n_events``.
        One row per quintile (if available) plus an "All" row.
    """
    _validate_columns(events, ["event_id", "addv", "position_size"], "events")

    df = events.copy()
    df["pct_of_addv"] = df["position_size"] / df["addv"]

    has_quintile = "quintile" in df.columns

    def _stats(frame: pd.DataFrame) -> dict:
        pct = frame["pct_of_addv"]
        pos = frame["position_size"]
        # Herfindahl index on position weights
        total = pos.sum()
        weights = pos / total if total > 0 else pos * 0
        herf = float((weights ** 2).sum())
        return {
            "pct_of_addv_mean": float(pct.mean() * 100),
            "pct_of_addv_max": float(pct.max() * 100),
            "pct_of_addv_p95": float(pct.quantile(0.95) * 100),
            "avg_inventory_usd": float(pos.mean()),
            "peak_inventory_usd": float(pos.max()),
            "herfindahl_concentration": herf,
            "n_events": len(frame),
        }

    rows = []
    if has_quintile:
        for q, grp in df.groupby("quintile"):
            row = {"group": f"Q{q}"}
            row.update(_stats(grp))
            rows.append(row)
    row_all = {"group": "All"}
    row_all.update(_stats(df))
    rows.append(row_all)

    result = pd.DataFrame(rows).set_index("group")
    logger.info("Capacity diagnostics:\n%s", result.to_string())
    return result
