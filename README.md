# thai-egp-anomaly-xai

Code for two studies of Thailand's electronic government procurement (e-GP) contract records,
fiscal years 2015–2025 (B.E. 2558–2568):

1. **Discretionary thresholds and contract splitting** — difference-in-bunching at the THB 500,000 ceiling,
   same-vendor splitting clusters, and a difference-in-differences estimate of the cost in forgone discounts.
2. **Anomaly screening with machine learning and explainability** — Isolation Forest, ECOD, HBOS,
   autoencoder and LOF scores with SHAP explanations, evaluated against rule-based red flags.

This repository contains **code only**. No data are redistributed: every input is public and is listed,
with its source URL, in [`DATA_SOURCES.md`](DATA_SOURCES.md) and resource by resource in
[`DATA_MANIFEST.csv`](DATA_MANIFEST.csv).

## Files

| Path | Purpose |
|---|---|
| `src/egp_download.py` | Download the e-GP datasets from data.go.th (CKAN API, resumable) into `data/raw/` |
| `src/profile_raw.py` | Profile raw files: encodings, layouts, field quality |
| `src/clean.py` | Stage, parse, deduplicate and validate; writes Parquet to `data/interim/` |
| `src/features.py` | Buying-unit and vendor keys, contract features → `data/processed/` |
| `src/bunching.py` | Bunching, splitting clusters, coverage, cluster bootstrap, specification curve, DiD on discounts |
| `src/model_iforest.py` | Isolation Forest + SHAP |
| `src/model_compare.py`, `src/model_compare2.py` | ECOD, HBOS, autoencoder, LOF, ensembles, rule-lift evaluation |
| `src/model_compare3.py`, `src/summarize_compare3.py` | Round 3: consistent data-level injection test, model comparison across FY2561–2568 |
| `src/check_specific_above500k.py` | Check of specific/selection-method contracts above THB 500,000 after the 2017 Act |
| `src/make_giq_figures.py` | Figures 1–3 of the threshold paper (vector PDF/EPS) |
| `run_*.bat` | Windows batch runs used for the long jobs (logs to `logs/`) |
| `environment.yml` | Conda environment `egp-xai` (Python 3.11, DuckDB, Polars, scikit-learn, SHAP) |

Outputs are written to `data/`, `reports/` and `models/`, which are not tracked.

## Reproduce

```bat
conda env create -f environment.yml
conda activate egp-xai
set Y=2558 2559 2560 2561 2562 2563 2564 2565 2566 2567 2568
python src\egp_download.py --years all --dest data\raw --extra summary_cgdcontract
python src\profile_raw.py --years %Y%
python src\clean.py --years %Y% --db "%TEMP%\egp_clean.duckdb"
python src\features.py --years %Y% --db "%TEMP%\egp_feat.duckdb"
```

Threshold paper (`src/bunching.py <step>`; each `*_aggregate` step must run before its estimator):

| Result | Steps |
|---|---|
| Bunching, difference-in-bunching, monthly event study | `aggregate`, `estimate`, `dib`, `event` |
| Inference and robustness | `unit_bins_aggregate`, `cluster_bootstrap`, `dib_spec_curve`, `robust`, `datadriven` |
| Splitting clusters and placebo thresholds | `split_aggregate`, `split`, `split_placebo`, `split_placebo_T_aggregate`, `split_placebo_T` |
| Coverage and missing mass | `unit_bands`, `coverage`, `consolidation_aggregate`, `consolidation` |
| Cost of discretion | `did_aggregate`, `did_discount`, `harm_cells_aggregate`, `harm_fe`, `split_harm_aggregate`, `split_harm` |

`src/make_giq_figures.py` draws Figures 1–3 from the estimates written to `reports/`.
Anomaly screening: `python src\model_iforest.py --year 2568`, then `model_compare.py` and `model_compare2.py`
with the same `--year` (see `run_*.bat`).

The raw CSV files total about 46 GB. All steps run out of core with DuckDB/Polars on a desktop computer
without a GPU; point the DuckDB database (`--db`) and its temp directory to a fast local disk.
