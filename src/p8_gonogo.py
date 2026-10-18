"""Paper 8, step 6: go/no-go test on FY2568 — does vendor identity (tax ID vs name) move integrity indicators?

Implements the locked plan in the project doc claude/p8-gonogo-plan.md (19 Oct 2026). Do not change the sample,
definitions or thresholds here without recording the change there.

  python src/p8_gonogo.py [--year 2568]          -> reports/p8_gonogo.md, reports/p8_gonogo.json

Sample: FY contracts with tax_id_kind = 'full', vendor_name_key present, agreed_price > 0, sign_date present.
Identities: ID = tax_id; NAME = name entity as in features.py (vendor_name_key, + |province for non-firms).
Indicators: I1 HHI per unit (units with >= 10 contracts), I2 repeat share, I3 same-day split clusters
(>= 2 contracts, all <= 500k, total in (500k, 2M]), I4 active vendors, I5 ranking of units by split share.
Decision (GO if any): M1 |dS3|/S3 >= 5%; M2 share |dHHI| >= 0.05 is >= 10% or median |dHHI| >= 0.01;
M3 |d vendors|/vendors >= 5%; M4 Kendall tau-b < 0.90 or top-200 overlap < 80%.
Also (descriptive, RQ1): by fiscal year, tax_id_kind shares, checksum validity, many-to-many ID/name structure.
Outputs are aggregates only (no names, no tax IDs).
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
T, T_HI, MIN_N, TOPK = 500_000, 2_000_000, 10, 200
ENTITY = """CASE WHEN winner_type IN ('company', 'partnership', 'government') THEN vendor_name_key
                 ELSE vendor_name_key || '|' || coalesce(province, '') END"""
# Thai 13-digit ID: check digit = (11 - sum(d_i * (14 - i), i = 1..12) mod 11) mod 10
CHECKSUM = ("((11 - (" + " + ".join(f"CAST(substr(tax_id, {i}, 1) AS INT) * {14 - i}" for i in range(1, 13))
            + ") % 11) % 10) = CAST(substr(tax_id, 13, 1) AS INT)")


def connect():
    con = duckdb.connect()
    tmp = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "duckdb_p8"
    con.execute(f"SET temp_directory='{tmp.as_posix()}'; SET threads={int(os.environ.get('EGPQ_THREADS', '4'))}; "
                f"SET memory_limit='{os.environ.get('EGP_DUCKDB_MEMORY', '8GB')}'; SET preserve_insertion_order=false")
    return con


def src(year):
    return (INTERIM / f"contracts_{year}" / "*.parquet").as_posix()


def by_identity(con, v):
    """Unit-level indicators and system totals under identity column v of table s."""
    U = con.execute(f"""
        WITH pr AS (SELECT unit, {v} AS ven, count(*) n, sum(p) val FROM s GROUP BY 1, 2),
             cl AS (SELECT unit, {v} AS ven, d, count(*) k, max(p) mx, sum(p) tot FROM s GROUP BY 1, 2, 3),
             sp AS (SELECT unit, count(*) n_cl, sum(k) n_in FROM cl
                    WHERE k >= 2 AND mx <= {T} AND tot > {T} AND tot <= {T_HI} GROUP BY 1),
             u AS (SELECT unit, sum(n) n, sum(val) val, count(*) n_ven,
                          sum(n) FILTER (WHERE n >= 2) n_rep FROM pr GROUP BY 1),
             h AS (SELECT pr.unit, sum((pr.val / u.val) ^ 2) hhi FROM pr JOIN u USING (unit)
                   WHERE u.val > 0 GROUP BY 1)
        SELECT u.unit, u.n, u.n_ven, coalesce(u.n_rep, 0) / u.n AS repeat_share, h.hhi,
               coalesce(sp.n_cl, 0) AS split_clusters, coalesce(sp.n_in, 0) / u.n AS split_share
        FROM u LEFT JOIN h USING (unit) LEFT JOIN sp USING (unit)""").df()
    sysv = con.execute(f"SELECT count(DISTINCT {v}) FROM s").fetchone()[0]
    return U, {"split_clusters": int(U.split_clusters.sum()), "vendors": int(sysv)}


def decide(A, B, sA, sB):
    m = A.merge(B, on="unit", suffixes=("_id", "_name"))
    big = m[m.n_id >= MIN_N].copy()
    big["dhhi"] = big.hhi_name - big.hhi_id
    d_hhi = big.dhhi.abs()
    r_id = big.sort_values(["split_share_id", "unit"], ascending=[False, True]).unit.head(TOPK)
    r_nm = big.sort_values(["split_share_name", "unit"], ascending=[False, True]).unit.head(TOPK)
    tau = kendalltau(big.split_share_id, big.split_share_name, variant="b").statistic
    res = {
        "units_ge10": int(len(big)),
        "S3_id": sA["split_clusters"], "S3_name": sB["split_clusters"],
        "M1_rel_dS3": (sB["split_clusters"] - sA["split_clusters"]) / max(sA["split_clusters"], 1),
        "M2_share_abs_dHHI_ge_005": float((d_hhi >= 0.05).mean()), "M2_median_abs_dHHI": float(d_hhi.median()),
        "mean_dHHI": float(big.dhhi.mean()),
        "M3_vendors_id": sA["vendors"], "M3_vendors_name": sB["vendors"],
        "M3_rel_dvendors": (sB["vendors"] - sA["vendors"]) / max(sA["vendors"], 1),
        "M4_kendall_tau_b": float(tau), "M4_top200_overlap": len(set(r_id) & set(r_nm)) / TOPK,
        "repeat_share_median_id": float(big.repeat_share_id.median()),
        "repeat_share_median_name": float(big.repeat_share_name.median()),
        "repeat_share_median_abs_d": float((big.repeat_share_name - big.repeat_share_id).abs().median()),
    }
    hits = {"M1": abs(res["M1_rel_dS3"]) >= 0.05,
            "M2": res["M2_share_abs_dHHI_ge_005"] >= 0.10 or res["M2_median_abs_dHHI"] >= 0.01,
            "M3": abs(res["M3_rel_dvendors"]) >= 0.05,
            "M4": (res["M4_kendall_tau_b"] < 0.90) or (res["M4_top200_overlap"] < 0.80)}
    res["criteria_met"] = hits
    res["verdict"] = "GO" if any(hits.values()) else "STOP"
    return res


def rq1(con):
    rows = []
    for p in sorted(INTERIM.glob("contracts_25??")):
        y = p.name.split("_")[1]
        f = src(y)
        kinds = dict(con.execute(f"SELECT tax_id_kind, count(*) FROM read_parquet('{f}') GROUP BY 1").fetchall())
        n = sum(kinds.values())
        st = con.execute(f"""
            WITH c AS (SELECT tax_id, {ENTITY} AS ne FROM read_parquet('{f}')
                       WHERE tax_id_kind = 'full' AND vendor_name_key IS NOT NULL),
                 a AS (SELECT ne, count(DISTINCT tax_id) k FROM c GROUP BY 1),
                 b AS (SELECT tax_id, count(DISTINCT ne) k FROM c GROUP BY 1)
            SELECT (SELECT count(*) FROM c), (SELECT avg(({CHECKSUM})::INT) FROM c),
                   (SELECT count(*) FROM b), (SELECT count(*) FROM a),
                   (SELECT avg((k > 1)::INT) FROM a), (SELECT avg((k > 1)::INT) FROM b)""").fetchone()
        rows.append({"fy": y, "contracts": n, **{f"share_{k}": v / n for k, v in kinds.items()},
                     "full_with_name": st[0], "checksum_valid": st[1], "distinct_ids": st[2],
                     "distinct_name_entities": st[3], "names_with_gt1_id": st[4], "ids_with_gt1_name": st[5]})
    return pd.DataFrame(rows).fillna(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2568")
    ap.add_argument("--skip-rq1", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    con = connect()
    con.execute(f"""CREATE TEMP TABLE s AS
        SELECT agency || ' || ' || coalesce(sub_agency, '') AS unit, try_cast(sign_date AS DATE) AS d,
               agreed_price AS p, tax_id AS id_v, {ENTITY} AS name_v
        FROM read_parquet('{src(a.year)}')
        WHERE tax_id_kind = 'full' AND vendor_name_key IS NOT NULL AND agreed_price > 0 AND sign_date IS NOT NULL""")
    n_s, n_all = con.execute(f"SELECT (SELECT count(*) FROM s), (SELECT count(*) FROM read_parquet('{src(a.year)}'))").fetchone()
    A, sA = by_identity(con, "id_v")
    B, sB = by_identity(con, "name_v")
    res = decide(A, B, sA, sB)
    res.update(year=a.year, sample_contracts=int(n_s), all_contracts=int(n_all), sample_share=n_s / n_all)
    R = None if a.skip_rq1 else rq1(con)
    res["seconds"] = round(time.time() - t0)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "p8_gonogo.json").write_text(json.dumps({"gonogo": res, "rq1": None if R is None else R.to_dict("records")},
                                                       indent=1, default=float), encoding="utf-8")
    c = res["criteria_met"]
    L = [f"# Paper 8 step 6: go/no-go (FY{a.year}) — **{res['verdict']}**", "",
         "Locked plan: project doc `claude/p8-gonogo-plan.md` (19 Oct 2026). Aggregates only.", "",
         f"Sample: {n_s:,} of {n_all:,} contracts ({n_s / n_all:.1%}) with a full tax ID and a name; "
         f"{res['units_ge10']:,} buying units with ≥ {MIN_N} contracts.", "",
         "| criterion | ID identity | NAME identity | value | threshold | met |", "|---|---|---|---|---|---|",
         f"| M1 same-day split clusters | {res['S3_id']:,} | {res['S3_name']:,} | {res['M1_rel_dS3']:+.1%} | ≥ ±5% | {'yes' if c['M1'] else 'no'} |",
         f"| M2 units with \\|ΔHHI\\| ≥ 0.05 | | | {res['M2_share_abs_dHHI_ge_005']:.1%} | ≥ 10% | {'yes' if c['M2'] else 'no'} |",
         f"| M2 median \\|ΔHHI\\| | | | {res['M2_median_abs_dHHI']:.4f} | ≥ 0.01 | |",
         f"| M3 active vendors | {res['M3_vendors_id']:,} | {res['M3_vendors_name']:,} | {res['M3_rel_dvendors']:+.1%} | ≥ ±5% | {'yes' if c['M3'] else 'no'} |",
         f"| M4 Kendall τ-b (unit split share) | | | {res['M4_kendall_tau_b']:.3f} | < 0.90 | {'yes' if c['M4'] else 'no'} |",
         f"| M4 top-{TOPK} overlap | | | {res['M4_top200_overlap']:.0%} | < 80% | |", "",
         f"Mean ΔHHI (NAME − ID): {res['mean_dHHI']:+.4f}. Median repeat share: ID {res['repeat_share_median_id']:.3f}, "
         f"NAME {res['repeat_share_median_name']:.3f} (median |Δ| {res['repeat_share_median_abs_d']:.3f}).", ""]
    if R is not None:
        cols = ["fy", "contracts"] + [x for x in R.columns if x.startswith("share_")] + \
               ["checksum_valid", "distinct_ids", "distinct_name_entities", "names_with_gt1_id", "ids_with_gt1_name"]
        T_ = R[cols].copy()
        for x in cols[2:]:
            T_[x] = T_[x].map(lambda v: f"{v:,.0f}" if x.startswith("distinct") else f"{v:.3f}")
        T_["contracts"] = T_["contracts"].map(lambda v: f"{v:,.0f}")
        L += ["## RQ1 (descriptive): tax ID availability and ID–name structure by fiscal year", "",
              "Shares of contracts by tax_id_kind; checksum and many-to-many shares among contracts with a full ID and a name.", "",
              T_.to_markdown(index=False), ""]
    L.append(f"Run time {res['seconds']} s.")
    (REPORTS / "p8_gonogo.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:16]), flush=True)


if __name__ == "__main__":
    main()
