"""Write queue/trip.tsv: the unattended queue for 17-23 Oct 2026 (papers 1-2 + foundation for 3-9).

  python queue/make_trip.py        (then: egpq start queue/trip.tsv)

Lanes run in parallel; jobs in a lane run in order. Paths come from $EGP/egpq.env:
  EGP (~/egp), REPO ($EGP/repo), PY_CPU (egp-xai python), PY_GPU (egp-gpu python).
Work trees (made by queue/setup_trip.sh): $EGP/work/p2 and $EGP/work/p1repro, each a git worktree with its
own data/processed (and data/interim for p1), so reruns never overwrite the main data or the PC results.
"""
from pathlib import Path

P2, P1 = "$EGP/work/p2", "$EGP/work/p1repro"
PC = "$EGP/work/pc_snapshot/reports"
FDN = "$EGP/data/shared/foundation_v1"
Y_ALL = [str(y) for y in range(2558, 2569)]
Y_MODEL = [str(y) for y in range(2568, 2560, -1)]           # 2568 ... 2561, as on the PC
BUNCHING = ["aggregate", "estimate", "dib", "event",
            "unit_bins_aggregate", "cluster_bootstrap", "dib_spec_curve", "robust", "datadriven",
            "split_aggregate", "split", "split_placebo", "split_placebo_T_aggregate", "split_placebo_T",
            "unit_bands", "coverage", "consolidation_aggregate", "consolidation",
            "did_aggregate", "did_discount", "harm_cells_aggregate", "harm_fe", "split_harm_aggregate", "split_harm"]
SECTION = ["coverage_check", "descriptives", "agency_bootstrap", "agency_event", "excess_sources",
           "value_reallocation", "budget_at_threshold", "split_transition_months", "split_keys", "split_keys_hash",
           "carril_test", "split_cross_vendor", "did_inference", "did_ref_agency", "did_by_gov", "price_equality",
           "inflation_adjust", "composition_check", "worktype_mix", "revision_checks", "check_specific_above500k",
           "make_giq_figures"]
WAIT_R4 = ("timeout 5h bash -c 'while pgrep -u \"$USER\" -f [r]un_round4.sh >/dev/null; do sleep 60; done'; "
           "echo round 4 finished or 5 h passed; true")

rows = []


def job(jid, lane, after, hours, cwd, cmd):
    rows.append("\t".join([jid, lane, ",".join(after) if after else "-", str(hours), cwd, cmd]))


# ---- lane cpu_p2: paper 2 rerun from features to screener, all in the Chopin environment ----
job("wait_round4", "cpu_p2", [], 5.5, "$EGP", WAIT_R4)
for y in Y_MODEL:
    job(f"p2_features_{y}", "cpu_p2", ["wait_round4"], 3, P2,
        f'"$PY_CPU" src/features.py --years {y} --threads 4 --memory 8GB --db "$TMPDIR/feat.duckdb"')
    job(f"p2_iforest_{y}", "cpu_p2", [f"p2_features_{y}"], 8, P2, f'"$PY_CPU" src/model_iforest.py --year {y}')
    job(f"p2_compare_{y}", "cpu_p2", [f"p2_iforest_{y}"], 6, f"{P2}/src", f'"$PY_CPU" model_compare.py --year {y}')
    job(f"p2_compare2_{y}", "cpu_p2", [f"p2_compare_{y}"], 8, f"{P2}/src", f'"$PY_CPU" model_compare2.py --year {y}')
    job(f"p2_compare3_{y}", "cpu_p2", [f"p2_compare2_{y}"], 8, f"{P2}/src", f'"$PY_CPU" model_compare3.py --year {y}')
    job(f"p2_screen_{y}", "cpu_p2", [f"p2_compare3_{y}"], 6, f"{P2}/src", f'"$PY_CPU" screen_group.py --year {y}')
# summaries and comparisons use whatever finished (no dependencies, so they run even if a year failed)
job("p2_summary", "cpu_p2", [], 1, f"{P2}/src",
    '"$PY_CPU" summarize_compare3.py; a=$?; "$PY_CPU" summarize_screen_group.py; b=$?; exit $((a + b))')
job("p2_vs_pc", "cpu_p2", [], 0.5, "$REPO",
    f'"$PY_CPU" foundation/compare_reports.py {PC} {P2}/reports --out {P2}/reports/_vs_pc.md')
job("p2_vs_round4", "cpu_p2", [], 0.5, "$REPO",
    f'"$PY_CPU" foundation/compare_reports.py $REPO/reports {P2}/reports --only model_compare3 screen_group '
    f'--out {P2}/reports/_vs_round4.md')

# ---- lane cpu_p1: paper 1 reproduction from raw CSV (check only; the submitted numbers stay as they are) ----
job("p1_profile", "cpu_p1", ["wait_round4"], 16, P1,
    f'"$PY_CPU" src/profile_raw.py --years {" ".join(Y_ALL)} --threads 4 --memory 10GB --db "$TMPDIR/profile.duckdb"')
job("p1_clean", "cpu_p1", ["p1_profile"], 16, P1,
    f'"$PY_CPU" src/clean.py --years {" ".join(Y_ALL)} --threads 4 --memory 10GB --db "$TMPDIR/clean.duckdb"')
for y in Y_ALL:
    job(f"p1_features_{y}", "cpu_p1", ["p1_clean"], 3, P1,
        f'"$PY_CPU" src/features.py --years {y} --threads 4 --memory 8GB --db "$TMPDIR/feat.duckdb"')
# bunching and section scripts depend only on p1_clean: lane order already runs features first,
# and a failed year should not block the other analyses (each failure is logged on its own)
for s in BUNCHING:
    job(f"p1_bunch_{s}", "cpu_p1", ["p1_clean"], 10, f"{P1}/src", f'"$PY_CPU" bunching.py {s}')
for s in SECTION:
    job(f"p1_{s}", "cpu_p1", ["p1_clean"], 8, f"{P1}/src", f'"$PY_CPU" {s}.py')
job("p1_vs_pc", "cpu_p1", [], 0.5, "$REPO",
    f'"$PY_CPU" foundation/compare_reports.py {PC} {P1}/reports --out {P1}/reports/_vs_pc.md')

# ---- lane cpu_fdn: foundation tables for papers 3-9 ----
job("fdn_names", "cpu_fdn", [], 4, "$REPO", f'"$PY_CPU" foundation/fdn_tables.py names --out {FDN}')
job("fdn_edges", "cpu_fdn", ["fdn_names"], 6, "$REPO", f'"$PY_CPU" foundation/fdn_tables.py edges --out {FDN}')
job("fdn_panels", "cpu_fdn", ["fdn_edges"], 2, "$REPO", f'"$PY_CPU" foundation/fdn_tables.py panels --out {FDN}')

# ---- lane gpu: name embeddings with three multilingual models (vendor names first: small, quick check) ----
for s, hours in (("vendor_names", 3), ("project_names", 30)):
    for m in ("bge-m3", "e5-large", "qwen3-emb-0.6b"):
        job(f"emb_{s.split('_')[0]}_{m}", "gpu", ["fdn_names"], hours, "$REPO",
            f'"$PY_GPU" foundation/fdn_embed.py --set {s} --model {m} --fdn {FDN}')

head = [
    "# Unattended queue 17-23 Oct 2026. Generated by queue/make_trip.py; edit that file, not this one.",
    "# columns (TAB): id lane after timeout_h cwd command",
    "#!lane cpu_p2 threads=4 cpus=0-3",
    "#!lane cpu_p1 threads=4 cpus=4-7",
    "#!lane cpu_fdn threads=3 cpus=8-10",
    "#!lane gpu threads=2 cpus=11-12 gpu=0",
]
out = Path(__file__).with_name("trip.tsv")
out.write_text("\n".join(head + rows) + "\n", encoding="utf-8")
print(f"{out}: {len(rows)} jobs")
