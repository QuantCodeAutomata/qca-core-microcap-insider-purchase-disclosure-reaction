# Microcap Insider Purchase Disclosure Study

A rigorous, reproducible quantitative study of abnormal returns following
insider purchase disclosures (SEC Form 4) in the U.S. microcap equity universe
(market cap $30 M – $500 M, ADDV > $200 K).  The project delivers two
experiments:

- **Exp-1** — Main microcap event study with quintile analysis, ML-based signal
  selection, calendar-time portfolio alpha estimation, and transaction-cost /
  capacity diagnostics.
- **Exp-2** — Liquid-universe extension (market cap ≥ $500 M, ADDV ≥ $1 M)
  with a realistic day-by-day scheduler, nearest-neighbour matched controls,
  and a formal support-gate check.

---

## Project Structure

```
project/
├── data/
│   └── cache/                  # Downloaded and intermediate cached files
│       ├── raw_events.parquet
│       ├── yahoo_panel.parquet
│       ├── ff3.parquet
│       └── sec_form4/          # SEC EDGAR quarterly index CSVs
│
├── exp/
│   ├── exp_1_run.py            # Exp-1 pipeline runner
│   └── exp_2_run.py            # Exp-2 pipeline runner
│
├── results/
│   ├── RESULTS.md              # Primary validation & metrics summary (exp-1)
│   ├── quintile_table.csv
│   ├── calendar_portfolio.csv
│   ├── cost_analysis.csv
│   ├── capacity_diagnostics.csv
│   ├── ml_results.json
│   ├── fig_calendar_portfolio.png
│   └── exp2/
│       ├── RESULTS_exp2.md     # Exp-2 report (support gate prominently shown)
│       ├── screened_events.csv
│       ├── matched_controls.csv
│       ├── scheduler_summary.json
│       └── support_gate.json
│
├── src/
│   ├── calendar_time_and_costs.py      # Calendar-time alpha & cost analytics
│   ├── liquid_universe_extension.py    # Exp-2 liquid universe pipeline
│   ├── replication_validation.py       # Validation report generation
│   │
│   │   (additional modules called by the runners but not included here)
│   ├── raw_data_ingest.py
│   ├── yahoo_downloader.py
│   ├── sample_funnel.py
│   ├── feature_engineering.py
│   ├── event_return_engine.py
│   ├── event_study_tables.py
│   ├── ml_training_eval.py
│   └── placebo_and_comparators.py
│
├── tests/
│   ├── __init__.py
│   ├── test_sample_funnel.py           # Funnel filter & data integrity tests
│   ├── test_event_return_engine.py     # AR / CAR computation tests
│   ├── test_ml_pipeline.py             # ML target, split, hyper-param tests
│   ├── test_features.py                # Feature engineering tests
│   └── test_liquid_extension.py        # Scheduler & matched-control tests
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd project

# 2. Create and activate a virtual environment (Python ≥ 3.11 recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Experiments

### Exp-1 — Main Microcap Study

```bash
# Full pipeline (downloads data from scratch)
python exp/exp_1_run.py

# Fast mode — reuses cached intermediate files
python exp/exp_1_run.py --fast-mode
```

Key outputs written to `results/`:

| File | Description |
|------|-------------|
| `RESULTS.md` | Markdown validation report + key metrics |
| `quintile_table.csv` | CAR by transaction-value quintile |
| `calendar_portfolio.csv` | Daily portfolio returns |
| `cost_analysis.csv` | Net-of-cost performance grid |
| `capacity_diagnostics.csv` | ADDV utilisation & concentration |
| `ml_results.json` | AUC, threshold, positive rate |
| `fig_calendar_portfolio.png` | Cumulative return chart |

### Exp-2 — Liquid Universe Extension

```bash
python exp/exp_2_run.py
```

Key outputs written to `results/exp2/`:

| File | Description |
|------|-------------|
| `RESULTS_exp2.md` | Report with prominent support-gate result |
| `support_gate.json` | `gate_passed` (bool) and `coverage_fraction` |
| `screened_events.csv` | Events passing liquid-universe screens |
| `matched_controls.csv` | Nearest-neighbour matched control pairs |
| `scheduler_summary.json` | Capital utilisation & turnover stats |

---

## Running the Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run a specific module
pytest tests/test_sample_funnel.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Key Benchmark Figures (Exp-1)

| Metric | Value |
|--------|-------|
| Raw Form 4 purchase records | ~101,093 |
| Final sample events | ~13,534 |
| Distinct tickers | ~1,192 |
| Q1 CAR[1,30] | ~8.6% |
| Q5 CAR[1,30] | ~1.7% |
| Q1 day-1 AR | ~4.13% |
| Q5 day-1 AR | ~0.86% |
| ML positive rate | ~27.1% |
| 2024 OOS AUC | ~0.676 |
| F1-maximising threshold | ~0.184 |

Replication validation checks all of the above automatically and reports
pass/fail in `results/RESULTS.md`.

---

## Design Principles

1. **No look-ahead bias** — every feature and signal uses only information
   available strictly before the event date `t0`.
2. **Documented assumptions** — all inferred implementation choices in
   `liquid_universe_extension.py` are flagged with
   `# INFERRED ASSUMPTION: ...` comments.
3. **Reproducibility** — fixed random seeds (`seed=0`) throughout ML and
   bootstrap procedures.
4. **Realistic execution** — the `LiquidStrategyScheduler` enforces whole-share
   tickets, per-position caps, budget constraints, and no issuer overlap.
5. **Transaction costs** — net-of-cost performance evaluated at 0, 10, 25, 50,
   and 100 bps round-trip.

---

## Data Sources

| Source | Usage |
|--------|-------|
| SEC EDGAR Form 4 full-text index | Primary insider purchase signals |
| Yahoo Finance (via `yfinance`) | Daily OHLCV prices, market cap |
| Kenneth French Data Library | FF3 + UMD factors |

---

## Licence

MIT — see `LICENSE` for details.
