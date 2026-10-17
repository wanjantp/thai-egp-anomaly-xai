#!/usr/bin/env python3
"""
How much splitting does the same-vendor measure miss? (v3 of the GIQ manuscript; Caires et al. (2025) find that
Portuguese buyers spread split contracts across sellers.)

Cluster definitions (a purchase = contracts of one buying unit grouped as below):
  vendor_day     unit × vendor × signing day          (the manuscript's measure)
  kind_day       unit × type of work × signing day    (any vendor; type of work from the project name, KIND_SQL)
  unit_day       unit × signing day                    (any vendor, any type)
  vendor_month   unit × vendor × month
  kind_month     unit × type of work × month
A cluster is split at THB 500,000 if it has 2+ contracts, every contract ≤ THB 500,000 and a total of
THB 0.5–2 million. Excess split clusters = split clusters − (fiscal 2015 split rate × clusters with a THB 0.5–2 million
total). "Explained" = excess split clusters in fiscal 2018–2025 / contracts missing above THB 500,000 (anchor 2–3M,
from data/processed/_revision/anchor_bands.parquet, built by `revision_checks.py anchor`).

Definitions without the vendor also catch unrelated purchases made on the same day; subtracting the fiscal 2015
rate removes such coincidences only if they are as common after the reform as before. Read kind_day and unit_day
as upper bounds and vendor_day as a lower bound.

Usage (Anaconda Prompt, env egp-xai):  python src\\split_cross_vendor.py
Output: reports/split_cross_vendor.md (+ .csv). Cache: data/processed/_revision/xv_YYYY.parquet.
"""
import sys
import time

import pandas as pd

from bunching import KIND_SQL, MAX_PRICE, REPORTS, connect, log
from revision_checks import DEDUP, OUT_DIR, src, years

DEFS = {"vendor_day": ("vkey", "sign_date"), "kind_day": ("kind", "sign_date"), "unit_day": (None, "sign_date"),
        "vendor_month": ("vkey", "date_trunc('month', sign_date)"), "kind_month": ("kind", "date_trunc('month', sign_date)")}


def aggregate(con):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for y in years():
        f = OUT_DIR / f"xv_{y}.parquet"
        if not f.exists():
            t0 = time.time()
            con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
                SELECT {DEDUP} sign_date, agreed_price, agency || '|' || coalesce(sub_agency, '') AS unit,
                       vendor_name_key || '|' || coalesce(province, '') AS vkey, {KIND_SQL} AS kind
                FROM read_parquet('{src(y)}')
                WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE} AND vendor_name_key IS NOT NULL""")
            rows = []
            for name, (key, t) in DEFS.items():
                g = f"unit, {key}, {t}" if key else f"unit, {t}"
                rows.append(con.execute(f"""
                    WITH g AS (SELECT count(*) k, sum(agreed_price) v, max(agreed_price) mx, count(DISTINCT vkey) nv
                               FROM c GROUP BY {g})
                    SELECT {y} AS fy, '{name}' AS def,
                           sum(CASE WHEN v > 500000 AND v <= 2000000 THEN 1 ELSE 0 END)::BIGINT AS n_cl,
                           sum(CASE WHEN v > 500000 AND v <= 2000000 AND k >= 2 AND mx <= 500000 THEN 1 ELSE 0 END)::BIGINT AS n_split,
                           sum(CASE WHEN v > 500000 AND v <= 2000000 AND k >= 2 AND mx <= 500000 AND nv >= 2 THEN 1 ELSE 0 END)::BIGINT AS n_split_multi
                    FROM g""").df())
            pd.concat(rows).to_parquet(f, index=False)
            log(f"xv {y}: {time.time() - t0:.0f}s")
        parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True)


def missing():
    A = pd.read_parquet(OUT_DIR / "anchor_bands.parquet")
    N = A.groupby(["fy", "band"])["n"].sum().unstack().fillna(0)
    r15 = N.loc[2558, "0.5-2M"] / N.loc[2558, "2-3M"]
    m = r15 * N["2-3M"] - N["0.5-2M"]
    return m[[y for y in m.index if 2561 <= y <= 2568]].sum()


def main():
    con = connect()
    con.execute("SET preserve_insertion_order = false")
    X = aggregate(con)
    X.to_csv(REPORTS / "split_cross_vendor.csv", index=False)
    try:
        M = missing()
    except FileNotFoundError:
        M = float("nan")
        log("anchor_bands.parquet missing: run `python revision_checks.py anchor` for the explained share")
    rows = []
    for d, g in X.groupby("def"):
        g = g.set_index("fy").sort_index()
        rate15 = g.loc[2558, "n_split"] / g.loc[2558, "n_cl"]
        act = g.loc[[y for y in g.index if 2561 <= y <= 2568]]
        excess = (act.n_split - rate15 * act.n_cl).sum()
        rows.append({"definition": d, "split_share_2015": rate15, "split_share_2018_25": act.n_split.sum() / act.n_cl.sum(),
                     "multi_vendor_share_of_splits_2018_25": act.n_split_multi.sum() / act.n_split.sum(),
                     "excess_split_2018_25": excess, "explained": excess / M})
    R = pd.DataFrame(rows).set_index("definition").loc[list(DEFS)]
    L = ["# Splitting across vendors: how much does the same-vendor measure miss?", "",
         f"Contracts missing above THB 500,000 in fiscal 2018–2025 (anchor THB 2–3 million): {M:,.0f}.",
         "Definitions without the vendor are upper bounds (they also group unrelated same-day purchases).", "",
         "| definition | split share FY2015 | split share FY2018–25 | splits with 2+ vendors | excess split clusters 2018–25 | explained |",
         "|---|---|---|---|---|---|"]
    for d, r in R.iterrows():
        L.append(f"| {d} | {r.split_share_2015:.1%} | {r.split_share_2018_25:.1%} | {r.multi_vendor_share_of_splits_2018_25:.1%} | "
                 f"{r.excess_split_2018_25:,.0f} | {r.explained:.0%} |")
    out = REPORTS / "split_cross_vendor.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
