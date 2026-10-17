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
[`DATA_SHA256.csv`](DATA_SHA256.csv) gives the size and SHA-256 checksum of each of the 112 source data files
(46.6 GB) and of the download manifest, as downloaded for the paper, so that a replicator can confirm that a fresh download holds the same bytes
(`src/raw_checksums.py`). The only data in the repository are the 200 split clusters of the validation sample and
their codes (`reports/split_audit_sample.csv`, `reports/split_audit_codes.csv`, `reports/validation/`), taken from the
same open data (vendor names withheld).

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

`src/make_giq_figures.py` draws Figures 1–5 from the estimates written to `reports/`.

Scripts behind each part of the threshold paper (run from `src/`; each writes a Markdown report to `reports/`):

| Section of the paper | Scripts |
|---|---|
| 3 Data (coverage, descriptive statistics) | `coverage_check.py`, `descriptives.py` |
| 5.1–5.2 Bunching and timing, agency-clustered intervals | `bunching.py` (above), `agency_bootstrap.py`, `agency_event.py` |
| 5.3 Where the missing contracts went | `excess_sources.py`, `value_reallocation.py`, `budget_at_threshold.py` |
| 5.4 Splitting at the four thresholds | `bunching.py split*`, `split_transition_months.py` |
| 5.5 Cost of discretion | `bunching.py did_* harm_* split_harm*`, `did_inference.py`, `did_ref_agency.py`, `did_by_gov.py`, `price_equality.py` |
| 6 Robustness: inflation, composition, work-type mix | `inflation_adjust.py`, `composition_check.py`, `worktype_mix.py` |
| 6 Robustness: other checks | `revision_checks.py`, `carril_test.py`, `split_keys.py`, `split_cross_vendor.py`, `check_specific_above500k.py` |
| 6 Validation sample of split clusters | `split_audit_sample.py` (draws the sample), `split_audit_sheets.py` (coding sheets; `kappa` reports agreement) |

Validation sample. `reports/split_audit_codes.csv` holds one code per cluster (A = one purchase divided,
B = separate needs, C = cannot tell). `reports/validation/split_audit_verify.xlsx` lists all 200 clusters with their
project names for checking each code; `python src\split_audit_sheets.py kappa` writes
`reports/validation/split_audit_agreement.md` with the verified shares and, once an independent coder has filled in
`second_coder_blind.xlsx` (50 clusters, period cues masked; not tracked until coding is complete), the agreement and
Cohen's kappa.
Anomaly screening: `python src\model_iforest.py --year 2568`, then `model_compare.py` and `model_compare2.py`
with the same `--year` (see `run_*.bat`).

The raw CSV files total about 46.6 GB. All steps run out of core with DuckDB/Polars on a desktop computer
without a GPU; point the DuckDB database (`--db`) and its temp directory to a fast local disk.
