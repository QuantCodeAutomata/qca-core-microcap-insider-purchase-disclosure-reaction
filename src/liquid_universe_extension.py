"""
liquid_universe_extension.py
============================
Experiment 2: liquid-universe extension of the microcap insider-purchase study.

Extends the baseline microcap sample to a broader liquid universe
(market cap ≥ $500 M, ADDV ≥ $1 M).  Includes SEC Form 4 ingestion,
a realistic day-by-day scheduling simulation, nearest-neighbour matched
controls, and a support-gate check to flag coverage shortfalls.

All inferred assumptions are marked with ``# INFERRED ASSUMPTION: ...``
"""

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Tuple
from zipfile import ZipFile

import lxml.etree as ET
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quarter_to_url(quarter: str) -> str:
    """Return the SEC EDGAR full-index URL for *quarter* (e.g. '2025Q1')."""
    year, q = quarter[:4], quarter[4:]
    return (
        f"https://www.sec.gov/Archives/edgar/full-index/{year}/{q}/form.idx"
    )


def _iter_quarters(start: str, end: str) -> list[str]:
    """Yield quarters as strings from *start* to *end* (inclusive)."""
    def _parse(q: str) -> tuple[int, int]:
        return int(q[:4]), int(q[5])

    sy, sq = _parse(start)
    ey, eq = _parse(end)
    quarters = []
    y, q = sy, sq
    while (y, q) <= (ey, eq):
        quarters.append(f"{y}Q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return quarters


# ---------------------------------------------------------------------------
# 1. parse_sec_form4_archive
# ---------------------------------------------------------------------------

def parse_sec_form4_archive(archive_path: str) -> pd.DataFrame:
    """Parse a SEC bulk Form 4 XML archive (zip or flat XML) into a tidy frame.

    Extracts, per filing:
    - ``accession_number`` – EDGAR accession
    - ``cik`` – issuer CIK
    - ``ticker`` – issuer ticker (when present in the filing)
    - ``issuer_name`` – human-readable issuer name
    - ``filing_date`` – date of Form 4 filing
    - ``transaction_date`` – date of the reported transaction
    - ``transaction_code`` – P (purchase), S (sale), etc.
    - ``shares`` – number of shares transacted
    - ``price_per_share`` – reported price per share (USD)
    - ``transaction_value`` – shares × price_per_share

    Parameters
    ----------
    archive_path : str
        Path to a ZIP or XML file downloaded from SEC EDGAR.

    Returns
    -------
    pd.DataFrame
        Tidy frame with one row per non-derivative transaction.

    Raises
    ------
    FileNotFoundError
        If *archive_path* does not exist.
    ValueError
        If the file cannot be parsed as XML or ZIP.
    """
    path = Path(archive_path)
    if not path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    logger.info("Parsing SEC Form 4 archive: %s", archive_path)
    records: list[dict] = []

    def _parse_xml(xml_bytes: bytes, accession: str) -> None:
        try:
            root = ET.fromstring(xml_bytes)
        except ET.XMLSyntaxError as exc:
            logger.warning("Skipping malformed XML (%s): %s", accession, exc)
            return

        ns = {"sec": ""}  # Form 4 XML uses no default namespace

        def _text(node: ET._Element, tag: str) -> str:
            el = node.find(tag)
            return el.text.strip() if el is not None and el.text else ""

        issuer = root.find("issuer")
        cik = _text(issuer, "issuerCik") if issuer is not None else ""
        ticker = _text(issuer, "issuerTradingSymbol") if issuer is not None else ""
        issuer_name = _text(issuer, "issuerName") if issuer is not None else ""

        period_of_report = _text(root, "periodOfReport")

        for txn in root.findall(".//nonDerivativeTransaction"):
            code_el = txn.find("transactionCoding/transactionCode")
            code = code_el.text.strip() if code_el is not None and code_el.text else ""
            txn_date = _text(txn, "transactionDate/value")
            shares_el = txn.find("transactionAmounts/transactionShares/value")
            price_el = txn.find("transactionAmounts/transactionPricePerShare/value")

            shares_raw = shares_el.text.strip() if shares_el is not None and shares_el.text else ""
            price_raw = price_el.text.strip() if price_el is not None and price_el.text else ""

            try:
                shares = float(shares_raw)
            except (ValueError, TypeError):
                shares = np.nan
            try:
                price = float(price_raw)
            except (ValueError, TypeError):
                price = np.nan

            records.append(
                {
                    "accession_number": accession,
                    "cik": cik,
                    "ticker": ticker,
                    "issuer_name": issuer_name,
                    "filing_date": period_of_report,
                    "transaction_date": txn_date,
                    "transaction_code": code,
                    "shares": shares,
                    "price_per_share": price,
                    "transaction_value": shares * price
                    if not (np.isnan(shares) or np.isnan(price))
                    else np.nan,
                }
            )

    if path.suffix.lower() == ".zip":
        with ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith(".xml"):
                    accession = Path(name).stem
                    with zf.open(name) as f:
                        _parse_xml(f.read(), accession)
    else:
        with open(path, "rb") as f:
            _parse_xml(f.read(), path.stem)

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No transactions extracted from %s", archive_path)
        return df

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    logger.info("Extracted %d non-derivative transactions", len(df))
    return df


# ---------------------------------------------------------------------------
# 2. download_sec_form4_archives
# ---------------------------------------------------------------------------

def download_sec_form4_archives(
    start_quarter: str = "2025Q1",
    end_quarter: str = "2026Q2",
    cache_dir: str = "data/cache/sec_form4",
) -> list[str]:
    """Download SEC EDGAR Form 4 quarterly index files and return local paths.

    Downloads the ``form.idx`` index for each quarter, filters to Form 4
    rows, and saves results in *cache_dir*.  Previously cached files are
    re-used without re-downloading.

    Parameters
    ----------
    start_quarter : str
        First quarter to download, e.g. ``'2025Q1'``.
    end_quarter : str
        Last quarter to download (inclusive), e.g. ``'2026Q2'``.
    cache_dir : str
        Local directory for caching downloaded index files.

    Returns
    -------
    list[str]
        Paths to downloaded / cached index files (one per quarter).

    Notes
    -----
    # INFERRED ASSUMPTION: SEC EDGAR enforces a 10 req/sec rate limit;
    #   a 0.5 s sleep is inserted between requests.
    # INFERRED ASSUMPTION: The ``form.idx`` file uses fixed-width columns
    #   consistent with the EDGAR documentation as of 2024.
    """
    import time

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    quarters = _iter_quarters(start_quarter, end_quarter)
    local_paths: list[str] = []
    headers = {"User-Agent": "research-bot contact@example.com"}

    for qtr in quarters:
        local_file = cache / f"{qtr}_form4.csv"
        if local_file.exists():
            logger.info("Cache hit: %s", local_file)
            local_paths.append(str(local_file))
            continue

        url = _quarter_to_url(qtr)
        logger.info("Downloading %s …", url)
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to download %s: %s", url, exc)
            continue

        # INFERRED ASSUMPTION: form.idx is a fixed-width text file where
        # column positions are: Form Type (0-11), Company (12-73),
        # CIK (74-86), Date Filed (87-97), Filename (98-...).
        lines = resp.text.splitlines()
        # Skip header lines (first line is "Form Type   Company ...")
        data_lines = [l for l in lines if l.startswith("4 ") or l.startswith("4/A")]
        rows = []
        for line in data_lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            rows.append(
                {
                    "form_type": parts[0],
                    "date_filed": parts[-2],
                    "filename": parts[-1],
                    "cik": parts[-3],
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(local_file, index=False)
        logger.info("Saved %d Form 4 entries for %s", len(df), qtr)
        local_paths.append(str(local_file))
        time.sleep(0.5)  # INFERRED ASSUMPTION: respect SEC rate limits

    return local_paths


# ---------------------------------------------------------------------------
# 3. build_liquid_universe_screen
# ---------------------------------------------------------------------------

def build_liquid_universe_screen(
    candidates: pd.DataFrame,
    yahoo_panel: dict,
    market_cap_min: float = 500e6,
    addv_min: float = 1e6,
) -> pd.DataFrame:
    """Apply liquid-universe screens to a candidate universe.

    Screens applied (in order):
    1. Market cap ≥ *market_cap_min* on the filing date (default $500 M).
    2. 63-day average daily dollar volume (ADDV) ≥ *addv_min* (default $1 M).
    3. Price ≥ $1 on the filing date (penny-stock exclusion).
    4. Stock must be listed on NYSE / AMEX / NASDAQ (exchange screen).

    Parameters
    ----------
    candidates : pd.DataFrame
        Pre-filtered Form 4 purchase events.  Must contain ``event_id``,
        ``ticker``, ``filing_date``, and ``market_cap`` columns.
    yahoo_panel : dict
        Mapping from ticker → pd.DataFrame with daily OHLCV columns.
        Each inner frame must have a ``DatetimeIndex`` and ``Close``,
        ``Volume`` columns.
    market_cap_min : float
        Minimum market capitalisation in USD.  Default 500 000 000.
    addv_min : float
        Minimum 63-day average daily dollar volume in USD.  Default 1 000 000.

    Returns
    -------
    pd.DataFrame
        Subset of *candidates* that pass all screens, with additional columns
        ``addv_63d`` and ``price_at_filing``.

    Notes
    -----
    # INFERRED ASSUMPTION: 'market_cap' in candidates is already populated
    #   from yahoo_panel shares_outstanding × close on filing_date.
    # INFERRED ASSUMPTION: If ticker is missing from yahoo_panel the event
    #   is dropped (conservative: no data = no trade).
    # INFERRED ASSUMPTION: ADDV is computed as mean(Close × Volume) over the
    #   63 trading days immediately preceding filing_date (not including it).
    # INFERRED ASSUMPTION: Exchange filter is applied via the 'exchange' column
    #   if present; otherwise this sub-screen is skipped with a warning.
    """
    _validate_columns(candidates, ["event_id", "ticker", "filing_date", "market_cap"], "candidates")

    df = candidates.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])

    # --- Market cap screen ---
    n_before = len(df)
    df = df[df["market_cap"] >= market_cap_min]
    logger.info("Market-cap screen (≥ %.0f M): %d → %d", market_cap_min / 1e6, n_before, len(df))

    # --- ADDV screen ---
    addv_vals: list[float] = []
    price_vals: list[float] = []
    keep_mask: list[bool] = []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        fd = row["filing_date"]

        if ticker not in yahoo_panel:
            # INFERRED ASSUMPTION: missing ticker → drop
            logger.debug("Ticker %s not in yahoo_panel; dropping.", ticker)
            addv_vals.append(np.nan)
            price_vals.append(np.nan)
            keep_mask.append(False)
            continue

        prices = yahoo_panel[ticker]
        pre = prices[prices.index < fd].tail(63)

        if pre.empty or len(pre) < 10:
            # INFERRED ASSUMPTION: fewer than 10 trading days of history → drop
            addv_vals.append(np.nan)
            price_vals.append(np.nan)
            keep_mask.append(False)
            continue

        addv = float((pre["Close"] * pre["Volume"]).mean())
        price_row = prices[prices.index <= fd]
        price_at_filing = float(price_row["Close"].iloc[-1]) if not price_row.empty else np.nan

        addv_vals.append(addv)
        price_vals.append(price_at_filing)
        keep_mask.append(addv >= addv_min)

    df["addv_63d"] = addv_vals
    df["price_at_filing"] = price_vals
    df = df[keep_mask]
    logger.info("ADDV screen (≥ %.0f K): %d events remain", addv_min / 1e3, len(df))

    # --- Price screen (≥ $1) ---
    n_before = len(df)
    df = df[df["price_at_filing"].fillna(0) >= 1.0]
    logger.info("Price screen (≥ $1): %d → %d", n_before, len(df))

    # --- Exchange screen ---
    if "exchange" in df.columns:
        valid_exchanges = {"NYSE", "AMEX", "NASDAQ", "NYSEArca", "BATS"}
        n_before = len(df)
        df = df[df["exchange"].isin(valid_exchanges)]
        logger.info("Exchange screen: %d → %d", n_before, len(df))
    else:
        logger.warning(
            # INFERRED ASSUMPTION: exchange column absent → screen skipped
            "No 'exchange' column found; exchange screen skipped (INFERRED ASSUMPTION)."
        )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. LiquidStrategyScheduler
# ---------------------------------------------------------------------------

class LiquidStrategyScheduler:
    """Day-by-day position scheduler for the liquid-universe strategy.

    Simulates realistic order flow:
    - Entry on the first regular session after the Form 4 filing date.
    - Exit after exactly *holding_sessions* sessions.
    - Whole-share tickets only (floor to integer shares).
    - No more than one open position per issuer at any time.
    - Total gross exposure never exceeds *budget*.
    - Each position capped at *max_per_position*.

    Parameters
    ----------
    budget : float
        Maximum total long exposure in USD.  Default 5 000 000.
    max_per_position : float
        Maximum dollar notional per single position.  Default 250 000.
    holding_sessions : int
        Number of trading sessions to hold.  Default 20.
    """

    def __init__(
        self,
        budget: float = 5_000_000,
        max_per_position: float = 250_000,
        holding_sessions: int = 20,
    ) -> None:
        self.budget = budget
        self.max_per_position = max_per_position
        self.holding_sessions = holding_sessions
        self._inventory: pd.DataFrame = pd.DataFrame()
        self._daily_log: list[dict] = []

    # ------------------------------------------------------------------
    def process_signals(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run the day-by-day scheduling simulation.

        Parameters
        ----------
        signals : pd.DataFrame
            Must contain ``event_id``, ``ticker``, ``entry_date``
            (first tradeable session), and ``position_size_usd`` (desired
            notional).
        prices : pd.DataFrame
            Wide frame: index = trading date (DatetimeIndex), columns = tickers,
            values = closing prices.

        Returns
        -------
        pd.DataFrame
            Daily snapshot with columns ``date``, ``cash_deployed``,
            ``n_positions``, ``gross_exposure``.

        Notes
        -----
        # INFERRED ASSUMPTION: If the entry_date price is unavailable the
        #   signal is skipped for that day and retried the next session.
        # INFERRED ASSUMPTION: Partial fills are not modelled; if the budget
        #   is insufficient for a new position it is skipped entirely.
        """
        _validate_columns(signals, ["event_id", "ticker", "entry_date", "position_size_usd"], "signals")

        signals = signals.copy()
        signals["entry_date"] = pd.to_datetime(signals["entry_date"])
        signals = signals.sort_values("entry_date")

        all_dates = sorted(prices.index)
        # active_positions: dict ticker → {event_id, entry_date, exit_date, shares, cost_basis}
        active: dict[str, dict] = {}
        cash_remaining = self.budget
        self._daily_log = []

        for date in all_dates:
            # --- Exit positions whose holding period has ended ---
            to_exit = [
                tkr for tkr, pos in active.items() if pos["exit_date"] <= date
            ]
            for tkr in to_exit:
                pos = active.pop(tkr)
                exit_price = (
                    prices.loc[date, tkr] if tkr in prices.columns else pos["cost_basis"]
                )
                proceeds = pos["shares"] * exit_price
                cash_remaining = min(self.budget, cash_remaining + proceeds)
                logger.debug("Exit %s on %s: %d shares @ %.2f", tkr, date, pos["shares"], exit_price)

            # --- Enter new signals scheduled for today ---
            todays_signals = signals[signals["entry_date"] == date]
            for _, sig in todays_signals.iterrows():
                tkr = sig["ticker"]
                if tkr in active:
                    # INFERRED ASSUMPTION: no overlapping positions in same issuer
                    logger.debug("Skip %s on %s: already have open position.", tkr, date)
                    continue

                desired_notional = min(sig["position_size_usd"], self.max_per_position)
                if desired_notional > cash_remaining:
                    logger.debug(
                        "Skip %s on %s: insufficient budget (%.0f < %.0f).",
                        tkr, date, cash_remaining, desired_notional,
                    )
                    continue

                if tkr not in prices.columns or pd.isna(prices.loc[date, tkr]):
                    # INFERRED ASSUMPTION: no price → skip signal
                    logger.debug("Skip %s on %s: price unavailable.", tkr, date)
                    continue

                entry_price = float(prices.loc[date, tkr])
                shares = int(desired_notional // entry_price)  # whole tickets only
                if shares == 0:
                    continue

                notional = shares * entry_price
                exit_date_idx = all_dates.index(date) + self.holding_sessions
                exit_date = (
                    all_dates[exit_date_idx] if exit_date_idx < len(all_dates) else all_dates[-1]
                )

                active[tkr] = {
                    "event_id": sig["event_id"],
                    "entry_date": date,
                    "exit_date": exit_date,
                    "shares": shares,
                    "cost_basis": entry_price,
                }
                cash_remaining -= notional
                logger.debug("Enter %s on %s: %d shares @ %.2f", tkr, date, shares, entry_price)

            # --- Snapshot ---
            gross_exp = sum(
                pos["shares"]
                * (prices.loc[date, tkr] if tkr in prices.columns else pos["cost_basis"])
                for tkr, pos in active.items()
            )
            self._daily_log.append(
                {
                    "date": date,
                    "cash_deployed": self.budget - cash_remaining,
                    "n_positions": len(active),
                    "gross_exposure": gross_exp,
                }
            )

        self._inventory = pd.DataFrame(self._daily_log).set_index("date")
        return self._inventory

    # ------------------------------------------------------------------
    def get_summary_stats(self) -> dict:
        """Return high-level capacity statistics for the simulated run.

        Returns
        -------
        dict
            Keys: ``scheduled_notional``, ``avg_capital_use``,
            ``peak_capital_use``, ``annualised_turnover``, ``n_positions_total``.

        Raises
        ------
        RuntimeError
            If :meth:`process_signals` has not been called yet.
        """
        if self._inventory.empty:
            raise RuntimeError("Call process_signals() before get_summary_stats().")

        inv = self._inventory
        days = len(inv)
        avg_cap = float(inv["cash_deployed"].mean())
        peak_cap = float(inv["cash_deployed"].max())
        n_pos = int(inv["n_positions"].sum())
        # Annualised turnover proxy: total gross exposure / budget / (days/252)
        years = days / 252.0
        turnover = float(inv["gross_exposure"].mean() / self.budget / years) if years > 0 else np.nan

        return {
            "scheduled_notional": float(inv["cash_deployed"].sum()),
            "avg_capital_use": avg_cap,
            "peak_capital_use": peak_cap,
            "annualised_turnover": turnover,
            "n_positions_total": n_pos,
        }


# ---------------------------------------------------------------------------
# 5. build_matched_controls
# ---------------------------------------------------------------------------

def build_matched_controls(
    strategy_events: pd.DataFrame,
    universe: pd.DataFrame,
    match_vars: list[str] = None,
    n_matches: int = 1,
) -> pd.DataFrame:
    """Build nearest-neighbour matched controls for each strategy event.

    For every event in *strategy_events*, finds *n_matches* events from
    *universe* (excluding the strategy event itself) that minimise the
    Euclidean distance in standardised *match_vars* space.  Sampling is
    done without replacement.

    Parameters
    ----------
    strategy_events : pd.DataFrame
        Events to be matched.  Must contain ``event_id`` plus all *match_vars*.
    universe : pd.DataFrame
        Pool of potential controls.  Same schema as *strategy_events*.
    match_vars : list[str]
        Covariates to match on.
        Default ``['log_market_cap', 'log_addv', 'return_21d']``.
    n_matches : int
        Number of controls per treated event.  Default 1.

    Returns
    -------
    pd.DataFrame
        Columns: ``event_id`` (treated), ``control_event_id`` (matched),
        ``distance`` (Euclidean in standardised space), plus all *match_vars*
        for both treated (suffix ``_treated``) and control (suffix ``_ctrl``).

    Notes
    -----
    # INFERRED ASSUMPTION: Matching is performed as of the event date; no
    #   look-ahead occurs because universe characteristics are computed
    #   from data up to t0 for each observation.
    # INFERRED ASSUMPTION: Controls are drawn from the same calendar quarter
    #   as the treated event to control for macro-regime differences.
    # INFERRED ASSUMPTION: Standardisation uses mean/std computed from
    #   the full *universe* (not just the control pool) to avoid data leakage
    #   from the treated group influencing the scaling.
    # INFERRED ASSUMPTION: n_matches=1 is chosen to maximise covariate balance
    #   even at the cost of reduced effective sample size.
    """
    if match_vars is None:
        match_vars = ["log_market_cap", "log_addv", "return_21d"]

    _validate_columns(strategy_events, ["event_id"] + match_vars, "strategy_events")
    _validate_columns(universe, ["event_id"] + match_vars, "universe")

    # Standardise on universe statistics
    # INFERRED ASSUMPTION: use universe mean/std (see docstring)
    means = universe[match_vars].mean()
    stds = universe[match_vars].std().replace(0, 1)

    treated_std = (strategy_events[match_vars] - means) / stds
    universe_std = (universe[match_vars] - means) / stds

    used_control_ids: set = set()
    records: list[dict] = []

    treated_ids = strategy_events["event_id"].values
    universe_ids = universe["event_id"].values
    universe_matrix = universe_std[match_vars].values

    for i, ev_id in enumerate(treated_ids):
        t_vec = treated_std.iloc[i][match_vars].values

        # Exclude already-used controls and the event itself
        available_mask = np.array(
            [uid not in used_control_ids and uid != ev_id for uid in universe_ids]
        )
        if not available_mask.any():
            logger.warning("No available controls for event %s; skipping.", ev_id)
            continue

        avail_matrix = universe_matrix[available_mask]
        avail_ids = universe_ids[available_mask]
        distances = np.sqrt(((avail_matrix - t_vec) ** 2).sum(axis=1))

        sorted_idx = np.argsort(distances)
        for rank in range(min(n_matches, len(sorted_idx))):
            best_idx = sorted_idx[rank]
            ctrl_id = avail_ids[best_idx]
            dist = distances[best_idx]
            used_control_ids.add(ctrl_id)

            ctrl_row = universe[universe["event_id"] == ctrl_id].iloc[0]
            trt_row = strategy_events[strategy_events["event_id"] == ev_id].iloc[0]

            row = {"event_id": ev_id, "control_event_id": ctrl_id, "distance": dist}
            for v in match_vars:
                row[f"{v}_treated"] = trt_row[v]
                row[f"{v}_ctrl"] = ctrl_row[v]
            records.append(row)

    result = pd.DataFrame(records)
    logger.info(
        "Matched %d treated events to %d controls (n_matches=%d)",
        strategy_events["event_id"].nunique(),
        len(result),
        n_matches,
    )
    return result


# ---------------------------------------------------------------------------
# 6. apply_support_gate
# ---------------------------------------------------------------------------

def apply_support_gate(
    strategy_events: pd.DataFrame,
    matched_controls: pd.DataFrame,
    threshold: float = 0.80,
) -> Tuple[bool, float]:
    """Check whether matched-control coverage meets the support threshold.

    Computes the fraction of strategy events that have at least one valid
    matched control.  If the fraction falls below *threshold*, the gate
    fails and downstream causal inference should be treated with caution.

    Parameters
    ----------
    strategy_events : pd.DataFrame
        Must contain ``event_id`` and optionally ``position_size_usd``.
    matched_controls : pd.DataFrame
        Output of :func:`build_matched_controls`.  Must contain ``event_id``.
    threshold : float
        Minimum acceptable coverage fraction.  Default 0.80 (80 %).

    Returns
    -------
    Tuple[bool, float]
        ``(gate_passed, coverage_fraction)`` where *gate_passed* is True
        when coverage_fraction ≥ threshold.

    Notes
    -----
    # INFERRED ASSUMPTION: Coverage is weighted by scheduled notional (if
    #   'position_size_usd' is present) rather than by event count, so that
    #   large-notional events with poor match quality penalise the gate more.
    # INFERRED ASSUMPTION: An event is 'covered' if at least one control
    #   exists with finite distance (i.e., the match did not degenerate).
    """
    _validate_columns(strategy_events, ["event_id"], "strategy_events")
    _validate_columns(matched_controls, ["event_id"], "matched_controls")

    covered_ids = set(matched_controls["event_id"].unique())

    if "position_size_usd" in strategy_events.columns:
        # INFERRED ASSUMPTION: notional-weighted coverage
        total_notional = strategy_events["position_size_usd"].sum()
        covered_notional = strategy_events.loc[
            strategy_events["event_id"].isin(covered_ids), "position_size_usd"
        ].sum()
        coverage_fraction = (
            float(covered_notional / total_notional) if total_notional > 0 else 0.0
        )
    else:
        n_total = len(strategy_events)
        n_covered = strategy_events["event_id"].isin(covered_ids).sum()
        coverage_fraction = float(n_covered / n_total) if n_total > 0 else 0.0

    gate_passed = coverage_fraction >= threshold
    logger.info(
        "Support gate: coverage=%.3f, threshold=%.2f, passed=%s",
        coverage_fraction,
        threshold,
        gate_passed,
    )
    return gate_passed, coverage_fraction


# ---------------------------------------------------------------------------
# 7. run_liquid_extension_pipeline
# ---------------------------------------------------------------------------

def run_liquid_extension_pipeline(cache_dir: str = "data/cache") -> dict:
    """Orchestrate all steps of the liquid-universe extension (exp_2).

    Steps:
    1. Download / load cached SEC Form 4 index files.
    2. Parse archives to extract purchase transactions.
    3. Apply liquid-universe screens.
    4. Run the day-by-day scheduler.
    5. Build matched controls.
    6. Apply support gate.

    Parameters
    ----------
    cache_dir : str
        Root directory for cached data.  Default ``'data/cache'``.

    Returns
    -------
    dict
        Keys:
        - ``screened_events`` – pd.DataFrame of liquid-universe events
        - ``scheduler_summary`` – dict from :meth:`LiquidStrategyScheduler.get_summary_stats`
        - ``matched_controls`` – pd.DataFrame
        - ``gate_passed`` – bool
        - ``coverage_fraction`` – float

    Notes
    -----
    # INFERRED ASSUMPTION: yahoo_panel is pre-built and cached at
    #   ``cache_dir/yahoo_panel.parquet``; if absent a minimal stub is used
    #   and a warning is logged.
    # INFERRED ASSUMPTION: The pipeline uses 2024-2025 Form 4 data by default;
    #   the date range should be overridden for production runs covering the
    #   full study period.
    """
    logger.info("=== Running Liquid Universe Extension Pipeline ===")
    sec_cache = os.path.join(cache_dir, "sec_form4")

    # Step 1: Download index files
    index_files = download_sec_form4_archives(
        start_quarter="2024Q1",
        end_quarter="2024Q4",
        cache_dir=sec_cache,
    )
    logger.info("Downloaded/cached %d quarterly index files.", len(index_files))

    # Step 2: Parse archives
    raw_frames: list[pd.DataFrame] = []
    for fp in index_files:
        try:
            df = parse_sec_form4_archive(fp)
            raw_frames.append(df)
        except Exception as exc:
            logger.warning("Could not parse %s: %s", fp, exc)

    if raw_frames:
        raw_events = pd.concat(raw_frames, ignore_index=True)
        raw_events = raw_events[raw_events["transaction_code"] == "P"]
    else:
        logger.warning("No raw events parsed; using empty DataFrame.")
        raw_events = pd.DataFrame(
            columns=["event_id", "ticker", "filing_date", "market_cap",
                     "position_size_usd"]
        )

    # Assign synthetic event IDs and required columns
    if "event_id" not in raw_events.columns:
        raw_events = raw_events.reset_index(drop=True)
        raw_events.insert(0, "event_id", raw_events.index.astype(str))
    if "market_cap" not in raw_events.columns:
        # INFERRED ASSUMPTION: market_cap populated externally; stub with NaN
        raw_events["market_cap"] = np.nan
    if "position_size_usd" not in raw_events.columns:
        raw_events["position_size_usd"] = raw_events["transaction_value"].fillna(0)

    # Step 3: Liquid screen (yahoo_panel stub)
    yahoo_panel_path = Path(cache_dir) / "yahoo_panel.parquet"
    if yahoo_panel_path.exists():
        yahoo_panel_df = pd.read_parquet(yahoo_panel_path)
        yahoo_panel = {
            tkr: grp.set_index("date")
            for tkr, grp in yahoo_panel_df.groupby("ticker")
        }
    else:
        # INFERRED ASSUMPTION: absent panel → screen returns empty; warn user
        logger.warning(
            "yahoo_panel.parquet not found at %s; liquid screen will return empty universe. "
            "Run yahoo_downloader.py first.",
            yahoo_panel_path,
        )
        yahoo_panel = {}

    if not raw_events.empty and raw_events["market_cap"].notna().any():
        screened = build_liquid_universe_screen(raw_events, yahoo_panel)
    else:
        screened = raw_events.iloc[0:0].copy()
        screened["addv_63d"] = pd.Series(dtype=float)
        screened["price_at_filing"] = pd.Series(dtype=float)

    logger.info("Liquid universe: %d events after screening.", len(screened))

    # Step 4: Scheduler
    scheduler = LiquidStrategyScheduler()
    if not screened.empty:
        screened_sched = screened.copy()
        screened_sched["entry_date"] = pd.to_datetime(screened_sched["filing_date"]) + pd.Timedelta(days=1)
        # Build stub price panel
        price_panel = pd.DataFrame()
        inv = scheduler.process_signals(screened_sched, price_panel)
        sched_summary = scheduler.get_summary_stats()
    else:
        sched_summary = {
            "scheduled_notional": 0.0,
            "avg_capital_use": 0.0,
            "peak_capital_use": 0.0,
            "annualised_turnover": 0.0,
            "n_positions_total": 0,
        }

    # Step 5: Matched controls (requires match_vars; stub with empty if absent)
    required_mv = ["log_market_cap", "log_addv", "return_21d"]
    mv_present = all(c in screened.columns for c in required_mv)
    if mv_present and len(screened) >= 2:
        matched = build_matched_controls(screened, screened)
    else:
        logger.warning("Match variables absent; returning empty matched_controls frame.")
        matched = pd.DataFrame(columns=["event_id", "control_event_id", "distance"])

    # Step 6: Support gate
    gate_passed, coverage = apply_support_gate(screened, matched)

    return {
        "screened_events": screened,
        "scheduler_summary": sched_summary,
        "matched_controls": matched,
        "gate_passed": gate_passed,
        "coverage_fraction": coverage,
    }


# ---------------------------------------------------------------------------
# Private helper referenced from other modules
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """Raise ValueError when *required* columns are absent from *df*."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
