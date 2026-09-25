# Project: Microcap Insider-Purchase Disclosure Study

## Repository layout

```
src/
  __init__.py                  – empty package marker
  raw_data_ingest.py           – OpenInsider scraper + FF3 factor loader
  yahoo_downloader.py          – Yahoo Finance OHLCV panel downloader
  calendar_time_and_costs.py   – calendar-time portfolio + cost/capacity analysis
  event_study_tables.py        – event-study return / CAR computations
  liquid_universe_extension.py – liquidity-based universe filters
  replication_validation.py    – replication / validation utilities
exp/
  exp_1_run.py                 – experiment 1 runner
  exp_2_run.py                 – experiment 2 runner
data/
  cache/                       – auto-generated parquet caches (gitignored)
    yahoo/{TICKER}.parquet     – per-ticker OHLCV (auto_adjust=False + actions)
    openinsider_P_{Y}_{M}.parquet  – monthly purchase scrapes
    openinsider_S_{Y}_{M}.parquet  – monthly sales scrapes
    ff3_daily.parquet          – Fama-French 3-factor daily returns (decimal)
    yahoo_profiles.parquet     – shares-outstanding / sector snapshots
results/
tests/
```

## Key conventions

### raw_data_ingest.py
- `_REQUIRED_OUT` – canonical column list every returned DataFrame must expose:
  `filing_date, transaction_date, ticker, insider_name, insider_title,
   transaction_price, quantity, owned, transaction_value, filing_timestamp`
- OpenInsider numeric fields (Price, Qty, Owned, Value) may contain `$`, `,`,
  `+` characters; `_clean_numeric()` strips them before casting to float.
- `load_ff3_factors()` divides raw famafrench values by 100 → decimal fractions.
  Cache key: `data/cache/ff3_daily.parquet`.

### yahoo_downloader.py
- All downloads use `auto_adjust=False` + `actions=True`.  Raw Close ≠ Adj_Close.
- Newer yfinance (≥ 0.2) may return MultiIndex columns; `_normalise_ohlcv()`
  flattens them and renames `'Adj Close'` → `'Adj_Close'`.
- `reconstruct_actual_close(ticker, t0)` returns `Close[t0] × ∏(split_factors > t0)`,
  expressing the historical price in current-share-count units.

## event_study_tables.py (697 lines)

Public functions (all return DataFrames unless noted):

| Function | Purpose |
|---|---|
| `form_quintiles(events, feature_col)` | Adds `quintile` 1-5 column using `pd.qcut`; stores bin edges in `events.attrs['quintile_cutoffs']` |
| `quintile_summary_table(events, ar_paths, cars)` | 17-column summary per quintile: mean/median CAR_1_30, AR_1, drift, tail rates, percentiles, clustered SE |
| `announcement_decomposition(events, ar_paths)` | Per-event columns: AR_t0 (pre-close only), AR_t1, overnight_gap, intraday_t1, drift_2_30, drift_2_60 |
| `volatility_decomposition(events, cars)` | z_day1 and z_30 by quintile with clustered SE |
| `benchmark_sensitivity(events, cars, alpha_hats, window)` | Primary vs. intercept-inclusive CARs; Q5−Q1 gap decomposition |
| `run_two_way_clustered_ttest(data, outcome, group1, group2)` | CGM (2011) V_2way = V_g1 + V_g2 − V_g1∩g2; returns dict |
| `plot_car_paths_by_quintile` / `plot_quintile_summary_bar` / `plot_announcement_decomposition` | Visualization helpers |

**Data conventions for ar_paths**: wide DataFrame, index = events.index, integer columns = day offset from t0 (e.g. -30…60), values = daily AR (not cumulated).

**Data conventions for cars**: columns CAR_1_30, CAR_1_60, CAR_2_30, CAR_2_60, AR_0, AR_1.

**Important**: `run_comparator_tests` (in placebo_and_comparators.py) requires `purchase_events` to contain `quintile`, `ticker`, and `CAR_1_30` — join `events_q` with `cars` before calling.

## ml_training_eval.py (914 lines)

Module-level constants:
- `BASE_FEATURES` – 12-element list
- `FULL_FEATURES` – 27-element list (BASE + 15 extras)

Public functions:

| Function | Returns |
|---|---|
| `prepare_ml_dataset(events, features, cars)` | `(X: DataFrame[FULL_FEATURES], y: Series{0,1})` — target = CAR_1_30 > 10 % |
| `time_ordered_cv_xgb(X_train, y_train, n_splits, max_rounds)` | `int` optimal boosting rounds via TimeSeriesSplit; early_stopping_rounds=20 |
| `train_xgboost(X_train, y_train, n_rounds)` | `xgb.Booster` — fixed: max_depth=5, lr=0.1, seed=42, nthread=1 |
| `train_random_forest(X_train, y_train)` | `Pipeline(imputer→RF)` — 100 trees, max_depth=10 |
| `train_logistic(X_train, y_train)` | `Pipeline(imputer→scaler→LR)` — L2, max_iter=1000 |
| `select_threshold(model, X_val, y_val)` | `float` threshold maximising F1 on 200-point grid |
| `evaluate_model(model, X_test, y_test, threshold, model_name)` | `dict` — roc_auc, pr_auc, precision, recall, f1, brier_score, brier_skill, flagged_count, mean_outcome_flagged/rejected |
| `issuer_cluster_bootstrap_auc(models, X_test, y_test, events_test, n_boot, seed)` | `dict` — per-model {auc_point, ci_lower, ci_upper} + pairwise_diffs |
| `run_full_ml_pipeline(events, features, cars)` | `dict` — metrics DataFrame, bootstrap, n_rounds, models, splits |
| `run_rolling_origin_tests(events, features, cars, test_years)` | `DataFrame` indexed (test_year, model_name) |

`_predict_proba(model, X)` — internal helper that dispatches between `xgb.Booster.predict(DMatrix)` and sklearn `predict_proba()[:,1]`.

## placebo_and_comparators.py (1097 lines)

Internal helpers: `_get_trading_days`, `_price_series`, `_daily_return`, `_addv`, `_estimate_ff3`, `_compute_car`, `_quintile_assign`, `_cluster_se_of_mean`, `_edgar_get`.

Public functions:

| Function | Purpose |
|---|---|
| `generate_placebo_sample(events, yahoo_panel, ff3, shares_df, n_per_issuer_year, seed)` | Sample 24 pseudo-event dates per (ticker, year) satisfying estimation/future window, ADDV, FF3, and 30-day gap gates; returns DataFrame with pct_from_52w_high, CAR_1_30, CAR_1_60, ff3_alpha, vol_30d |
| `run_placebo_analysis(placebo_events, purchase_quintile_cutoffs, primary_cars, alpha_inclusive_cars)` | Assign placebo quintiles using purchase cutoffs; compare gradients; returns 5×10 summary DataFrame |
| `build_sales_comparator(sales_raw, yahoo_panel, ff3)` | Form 4 code-S only, no option exercises; collapses to (ticker, filing_date); after-close filings; same FF3 event-study path |
| `build_earnings_comparator(sample_tickers, cache_dir)` | SEC EDGAR 8-K Item 2.02 + XBRL EPS; SRW surprise = EPS_t − EPS_{t-4}; requires ≥ 4 prior quarters; caches JSON to `data/cache/` |
| `run_comparator_tests(purchase_events, sales_events, earnings_events, quintile_cutoffs)` | Returns dict: purchase_quintile_means, sales_quintile_means, purchase_vs_sales (quintile Welch tests), earnings_surprise_stats, purchase_vs_earnings, gradient_tests |
| `plot_comparator_car_paths(...)` | Overlays purchase / sales / placebo CAR paths for a given quintile |

EDGAR rate-limit: `_edgar_get` retries 3× with exponential back-off; `build_earnings_comparator` sleeps 50 ms between tickers.

## Installed dependencies (all present in the environment)
- pandas, numpy, requests, yfinance, pandas_datareader, statsmodels
- xgboost 2.1.1, scikit-learn 1.5.1, seaborn 0.13.2, scipy 1.14.1

---

## New modules added (2025 session)

### src/calendar_time_and_costs.py
- `build_calendar_time_portfolio(events, ar_paths, holding_start=1, holding_end=30, weight_scheme='equal') → pd.DataFrame`
  Daily portfolio returns indexed by calendar date; columns: port_return, port_return_q1, n_positions, n_positions_q1.
- `estimate_calendar_alpha(portfolio_returns, ff3, use_umd=False, hac_lags=5) → dict`
  OLS on HAC SEs; alpha_monthly = alpha_daily × 21.  Dict keys: alpha_daily, alpha_monthly, t_stat, p_value, r2, n_obs.
- `skip_day_one_variant(events, ar_paths) → pd.DataFrame` — builds t0+2 to t0+31 equal-weighted portfolio.
- `transaction_cost_analysis(events, cars, cost_bps_grid=[0,10,25,50,100]) → pd.DataFrame`
- `capacity_diagnostics(events) → pd.DataFrame` — pct_of_addv, Herfindahl, etc.

### src/liquid_universe_extension.py
- `parse_sec_form4_archive(archive_path)` — parses ZIP or flat XML; returns one row per non-derivative transaction.
- `download_sec_form4_archives(start_quarter, end_quarter, cache_dir)` — fetches form.idx from EDGAR; caches as CSV.
- `build_liquid_universe_screen(candidates, yahoo_panel, market_cap_min=500e6, addv_min=1e6)` — 4-stage screen.
- `LiquidStrategyScheduler` — day-by-day scheduler; whole-share tickets, no issuer overlap, budget cap.
  - `.process_signals(signals, prices) → pd.DataFrame` (daily inventory)
  - `.get_summary_stats() → dict`
- `build_matched_controls(strategy_events, universe, match_vars, n_matches=1)` — NN matching without replacement.
- `apply_support_gate(strategy_events, matched_controls, threshold=0.80) → (bool, float)`
- `run_liquid_extension_pipeline(cache_dir='data/cache') → dict`
- **All inferred assumptions** marked `# INFERRED ASSUMPTION: …` (≥ 9 markers total).

### src/replication_validation.py
- `validate_sample_counts(funnel_dict)` — raw_count ~101093, final_count ~13534, ticker_count ~1192.
- `validate_ml_targets(ml_results)` — positive_rate ~27.1%, auc_2024 ~0.676, threshold ~0.184.
- `validate_gradient(quintile_table)` — Q1 CAR[1,30] ~8.6%, Q5 ~1.7%, Q1 AR_day1 ~4.13%, Q5 ~0.86%.
- `generate_validation_report(all_results) → str` — returns full markdown string.

### exp/exp_1_run.py
- Orchestrates 10 pipeline steps in order; each step is a standalone function returning a dict.
- `--fast-mode` flag reuses cached parquet files.
- Falls back to synthetic stubs if src sub-modules are absent (allows partial runs).
- Writes results/RESULTS.md, all CSVs, and fig_calendar_portfolio.png on completion.

### exp/exp_2_run.py
- Calls `run_liquid_extension_pipeline()`; saves support gate result to results/exp2/support_gate.json.
- Prints a bold banner with gate pass/fail status at end of run.

---

## Test suite (tests/)

Run: `python -m pytest tests/ -v`  → **115 tests, all pass**.

### Key implementation details discovered during testing

1. **`_compute_pct_from_52w_high`**: the 52w window INCLUDES t0 (uses `index <= t0`).
   This guarantees the result is always ≤ 0.  Volatility uses `index < t0` (strictly prior).

2. **`_drop_extreme_cars` price-ratio filter**: uses STRICT inequalities — keep only when
   `1/8 < ratio < 8`.  Values exactly equal to 8 or 1/8 are dropped.

3. **Biotech keyword list** includes `"bio"` (not just `"biotech"`) to catch "BioNTech", "BioMarin", etc.

4. **`LiquidStrategyScheduler`**: exits are triggered on the first trading day ≥ exit_date
   (not strict equality); same-issuer overlap is blocked even when signals arrive on the same day.

5. **`apply_support_gate`**: notional-weighted coverage when `position_size_usd` column present,
   count-based otherwise.
