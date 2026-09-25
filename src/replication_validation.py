"""
replication_validation.py
=========================
Validation report generation for the microcap insider-purchase disclosure study.

Checks computed results against published benchmark figures and produces a
structured markdown report summarising pass/fail status for each metric.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tolerances used for soft comparisons
# ---------------------------------------------------------------------------

_ABS_TOL_COUNT = 500        # ± counts for raw/final tallies
_REL_TOL_RATE = 0.05        # ± 5 percentage points for rate metrics
_REL_TOL_RETURN = 0.02      # ± 2 percentage points for return metrics
_REL_TOL_STAT = 0.05        # ± 5 % relative for statistics like AUC


def _check(
    label: str,
    actual: float,
    expected: float,
    tol: float,
    relative: bool = False,
) -> dict[str, Any]:
    """Return a result dict indicating whether the check passed.

    Parameters
    ----------
    label : str
        Human-readable name of the metric.
    actual : float
        Value computed by this run.
    expected : float
        Published benchmark value.
    tol : float
        Allowed deviation (absolute units or relative fraction).
    relative : bool
        If True, tolerance is applied as a fraction of *expected*.

    Returns
    -------
    dict
        Keys: ``metric``, ``expected``, ``actual``, ``deviation``,
        ``tolerance``, ``passed``.
    """
    if relative:
        deviation = abs(actual - expected) / (abs(expected) + 1e-12)
    else:
        deviation = abs(actual - expected)
    passed = bool(deviation <= tol)
    return {
        "metric": label,
        "expected": expected,
        "actual": actual,
        "deviation": deviation,
        "tolerance": tol,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# 1. validate_sample_counts
# ---------------------------------------------------------------------------

def validate_sample_counts(funnel_dict: dict) -> dict:
    """Check sample-funnel counts against published benchmarks.

    Parameters
    ----------
    funnel_dict : dict
        Must contain:
        - ``raw_count`` (int) – raw Form 4 purchase records before any filter.
        - ``final_count`` (int) – events remaining after all filters.
        - ``ticker_count`` (int) – distinct tickers in the final sample.

    Returns
    -------
    dict
        Keys: ``checks`` (list of check dicts), ``all_passed`` (bool).
    """
    required = ["raw_count", "final_count", "ticker_count"]
    missing = [k for k in required if k not in funnel_dict]
    if missing:
        raise ValueError(f"funnel_dict missing keys: {missing}")

    checks = [
        _check("raw_count", float(funnel_dict["raw_count"]), 101_093, _ABS_TOL_COUNT),
        _check("final_count", float(funnel_dict["final_count"]), 13_534, _ABS_TOL_COUNT),
        _check("ticker_count", float(funnel_dict["ticker_count"]), 1_192, _ABS_TOL_COUNT),
    ]

    all_passed = all(c["passed"] for c in checks)
    logger.info(
        "Sample count validation: %d/%d checks passed",
        sum(c["passed"] for c in checks),
        len(checks),
    )
    for c in checks:
        status = "✓" if c["passed"] else "✗"
        logger.info(
            "  %s %s: expected=%.0f actual=%.0f deviation=%.0f",
            status, c["metric"], c["expected"], c["actual"], c["deviation"],
        )

    return {"checks": checks, "all_passed": all_passed}


# ---------------------------------------------------------------------------
# 2. validate_ml_targets
# ---------------------------------------------------------------------------

def validate_ml_targets(ml_results: dict) -> dict:
    """Check ML-pipeline metrics against published benchmarks.

    Parameters
    ----------
    ml_results : dict
        Must contain:
        - ``positive_rate`` (float) – fraction of events with CAR[1,30] > 10 %.
        - ``auc_2024`` (float) – out-of-sample AUC on the 2024 test set.
        - ``threshold`` (float) – F1-maximising classification threshold.

    Returns
    -------
    dict
        Keys: ``checks`` (list), ``all_passed`` (bool).
    """
    required = ["positive_rate", "auc_2024", "threshold"]
    missing = [k for k in required if k not in ml_results]
    if missing:
        raise ValueError(f"ml_results missing keys: {missing}")

    checks = [
        _check(
            "positive_rate",
            float(ml_results["positive_rate"]),
            0.271,
            _REL_TOL_RATE,
        ),
        _check(
            "auc_2024",
            float(ml_results["auc_2024"]),
            0.676,
            _REL_TOL_STAT,
            relative=True,
        ),
        _check(
            "threshold",
            float(ml_results["threshold"]),
            0.184,
            _REL_TOL_STAT,
            relative=True,
        ),
    ]

    all_passed = all(c["passed"] for c in checks)
    logger.info(
        "ML target validation: %d/%d checks passed",
        sum(c["passed"] for c in checks),
        len(checks),
    )
    for c in checks:
        status = "✓" if c["passed"] else "✗"
        logger.info(
            "  %s %s: expected=%.4f actual=%.4f", status, c["metric"], c["expected"], c["actual"],
        )

    return {"checks": checks, "all_passed": all_passed}


# ---------------------------------------------------------------------------
# 3. validate_gradient
# ---------------------------------------------------------------------------

def validate_gradient(quintile_table: pd.DataFrame) -> dict:
    """Check return-gradient metrics across quintiles.

    Validates that:
    - Q1 CAR[1,30] ≈ 8.6 %
    - Q5 CAR[1,30] ≈ 1.7 %
    - Q1 day-1 AR ≈ 4.13 %
    - Q5 day-1 AR ≈ 0.86 %

    Parameters
    ----------
    quintile_table : pd.DataFrame
        Must have index (or column ``quintile``) = 1 … 5 and columns
        ``car_1_30`` and ``ar_day1`` (both as decimal fractions).

    Returns
    -------
    dict
        Keys: ``checks`` (list), ``all_passed`` (bool).
    """
    # Normalise: accept either numeric index or 'quintile' column
    qt = quintile_table.copy()
    if "quintile" in qt.columns:
        qt = qt.set_index("quintile")

    required_cols = ["car_1_30", "ar_day1"]
    missing_cols = [c for c in required_cols if c not in qt.columns]
    if missing_cols:
        raise ValueError(f"quintile_table missing columns: {missing_cols}")

    def _get(q: int, col: str) -> float:
        if q not in qt.index:
            raise ValueError(f"Quintile {q} not found in quintile_table index.")
        return float(qt.loc[q, col])

    checks = [
        _check("Q1_car_1_30",  _get(1, "car_1_30"), 0.086, _REL_TOL_RETURN),
        _check("Q5_car_1_30",  _get(5, "car_1_30"), 0.017, _REL_TOL_RETURN),
        _check("Q1_ar_day1",   _get(1, "ar_day1"),  0.0413, _REL_TOL_RETURN),
        _check("Q5_ar_day1",   _get(5, "ar_day1"),  0.0086, _REL_TOL_RETURN),
    ]

    all_passed = all(c["passed"] for c in checks)
    logger.info(
        "Gradient validation: %d/%d checks passed",
        sum(c["passed"] for c in checks),
        len(checks),
    )
    for c in checks:
        status = "✓" if c["passed"] else "✗"
        logger.info(
            "  %s %s: expected=%.4f actual=%.4f",
            status, c["metric"], c["expected"], c["actual"],
        )

    return {"checks": checks, "all_passed": all_passed}


# ---------------------------------------------------------------------------
# 4. generate_validation_report
# ---------------------------------------------------------------------------

def generate_validation_report(all_results: dict) -> str:
    """Produce a comprehensive markdown validation report.

    Parameters
    ----------
    all_results : dict
        Top-level results dictionary produced by the exp_1 pipeline.
        Expected keys (all optional; missing sections are noted in the report):
        - ``funnel`` – dict for :func:`validate_sample_counts`
        - ``ml`` – dict for :func:`validate_ml_targets`
        - ``quintile_table`` – pd.DataFrame for :func:`validate_gradient`
        - ``calendar_alpha`` – dict with ``alpha_monthly``, ``t_stat``
        - ``cost_analysis`` – pd.DataFrame from
          :func:`~calendar_time_and_costs.transaction_cost_analysis`

    Returns
    -------
    str
        Markdown-formatted validation report.
    """
    lines: list[str] = []

    lines += [
        "# Replication Validation Report",
        "",
        "> Auto-generated by `replication_validation.generate_validation_report()`",
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------
    # Section 1 – Sample counts
    # ------------------------------------------------------------------
    lines += ["## 1. Sample Funnel Counts", ""]
    if "funnel" in all_results:
        try:
            vc = validate_sample_counts(all_results["funnel"])
            lines += _render_checks_table(vc["checks"])
            status = "✅ PASSED" if vc["all_passed"] else "❌ FAILED"
            lines += [f"**Overall: {status}**", ""]
        except Exception as exc:
            lines += [f"⚠️  Error running validation: {exc}", ""]
    else:
        lines += ["_Section skipped: `funnel` key not present in results._", ""]

    # ------------------------------------------------------------------
    # Section 2 – ML targets
    # ------------------------------------------------------------------
    lines += ["## 2. ML Pipeline Metrics", ""]
    if "ml" in all_results:
        try:
            vm = validate_ml_targets(all_results["ml"])
            lines += _render_checks_table(vm["checks"])
            status = "✅ PASSED" if vm["all_passed"] else "❌ FAILED"
            lines += [f"**Overall: {status}**", ""]
        except Exception as exc:
            lines += [f"⚠️  Error running validation: {exc}", ""]
    else:
        lines += ["_Section skipped: `ml` key not present in results._", ""]

    # ------------------------------------------------------------------
    # Section 3 – Return gradient
    # ------------------------------------------------------------------
    lines += ["## 3. Quintile Return Gradient", ""]
    if "quintile_table" in all_results:
        try:
            vg = validate_gradient(all_results["quintile_table"])
            lines += _render_checks_table(vg["checks"])
            status = "✅ PASSED" if vg["all_passed"] else "❌ FAILED"
            lines += [f"**Overall: {status}**", ""]
        except Exception as exc:
            lines += [f"⚠️  Error running validation: {exc}", ""]
    else:
        lines += ["_Section skipped: `quintile_table` key not present in results._", ""]

    # ------------------------------------------------------------------
    # Section 4 – Calendar-time alpha
    # ------------------------------------------------------------------
    lines += ["## 4. Calendar-Time Alpha", ""]
    if "calendar_alpha" in all_results:
        ca = all_results["calendar_alpha"]
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Alpha (monthly, bps) | {ca.get('alpha_monthly', 'N/A'):.1f} |"
            if isinstance(ca.get('alpha_monthly'), (int, float)) else
            f"| Alpha (monthly, bps) | N/A |",
            f"| t-statistic | {ca.get('t_stat', 'N/A'):.3f} |"
            if isinstance(ca.get('t_stat'), (int, float)) else
            "| t-statistic | N/A |",
            f"| p-value | {ca.get('p_value', 'N/A'):.4f} |"
            if isinstance(ca.get('p_value'), (int, float)) else
            "| p-value | N/A |",
            f"| R² | {ca.get('r2', 'N/A'):.4f} |"
            if isinstance(ca.get('r2'), (int, float)) else
            "| R² | N/A |",
            "",
        ]
    else:
        lines += ["_Section skipped: `calendar_alpha` key not present in results._", ""]

    # ------------------------------------------------------------------
    # Section 5 – Transaction costs
    # ------------------------------------------------------------------
    lines += ["## 5. Transaction Cost Analysis", ""]
    if "cost_analysis" in all_results:
        ca_df = all_results["cost_analysis"]
        if isinstance(ca_df, pd.DataFrame) and not ca_df.empty:
            lines += [ca_df.to_markdown(index=False), ""]
        else:
            lines += ["_No cost analysis data available._", ""]
    else:
        lines += ["_Section skipped: `cost_analysis` key not present in results._", ""]

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    import datetime
    lines += [
        "---",
        f"_Report generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC_",
    ]

    report = "\n".join(lines)
    logger.info("Validation report generated (%d lines).", len(lines))
    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_checks_table(checks: list[dict]) -> list[str]:
    """Render a list of check dicts as a markdown table."""
    header = ["| Metric | Expected | Actual | Deviation | Tolerance | Status |",
              "|--------|----------|--------|-----------|-----------|--------|"]
    rows = []
    for c in checks:
        icon = "✅" if c["passed"] else "❌"
        rows.append(
            f"| {c['metric']} | {c['expected']:.6g} | {c['actual']:.6g} "
            f"| {c['deviation']:.6g} | {c['tolerance']:.6g} | {icon} |"
        )
    return header + rows + [""]
