#!/usr/bin/env python3
"""
Why do fiscal 2016–2017 (B.E. 2559–2560) hold fewer contract records? (Section 3 and Section 8 of the GIQ manuscript)

  units    distinct agencies and buying units (agency × sub-unit) and records per unit, by fiscal year and central/local
  bands    records by fiscal year × price band × central/local
  numbers  e-GP project numbers: YY MM S NNNNNN (year, month, series digit, six-digit running number). For each
           YYMM × series, the share of the number range (max − min + 1) that appears in the published contract data,
           aggregated to the fiscal year of the project number
  summary  Comptroller General's summary tables (data/raw/summary_cgdcontract): projects and budget by year

Usage (Anaconda Prompt, env egp-xai):  python src\\coverage_check.py [units bands numbers summary]
Output: reports/coverage_check.md (+ reports/coverage_*.csv). Reads data/interim and data/raw/summary_cgdcontract only.
"""
import sys
from pathlib import Path

import pandas as pd

from bunching import GOV_SQL, INTERIM, REPORTS, connect, log

RAW_SUMMARY = INTERIM.parent / "raw" / "summary_cgdcontract"
BAND = """CASE WHEN agreed_price IS NULL OR agreed_price <= 0 THEN 'n/a' WHEN agreed_price <= 10000 THEN '≤10k'
               WHEN agreed_price <= 30000 THEN '10–30k' WHEN agreed_price <= 100000 THEN '30–100k'
               WHEN agreed_price <= 500000 THEN '100–500k' WHEN agreed_price <= 2000000 THEN '0.5–2M' ELSE '>2M' END"""
SECTIONS = {}


def years():
    return sorted(int(p.name.split("_")[1]) for p in INTERIM.glob("contracts_25??") if p.is_dir())


def src(y):
    return (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()


def units(con):
    U = pd.concat([con.execute(f"""SELECT {y} AS fy, {GOV_SQL} AS gov, count(DISTINCT agency) agencies,
            count(DISTINCT agency || '|' || coalesce(sub_agency, '')) units, count(*) records
            FROM read_parquet('{src(y)}') GROUP BY ALL""").df() for y in years()])
    U["records_per_unit"] = U.records / U.units
    U.to_csv(REPORTS / "coverage_units.csv", index=False)
    P = U.pivot(index="fy", columns="gov", values=["units", "records", "records_per_unit"])
    SECTIONS["units"] = ["## Buying units and records per unit", "", P.round(1).to_markdown(), ""]


def bands(con):
    B = pd.concat([con.execute(f"""SELECT {y} AS fy, {GOV_SQL} AS gov, {BAND} AS band, count(*) records
            FROM read_parquet('{src(y)}') GROUP BY ALL""").df() for y in years()])
    B.to_csv(REPORTS / "coverage_bands.csv", index=False)
    P = (B.pivot_table(index=["gov", "band"], columns="fy", values="records", aggfunc="sum") / 1000).round(0)
    SECTIONS["bands"] = ["## Records by price band (thousands)", "", P.to_markdown(), ""]


def numbers(con):
    N = pd.concat([con.execute(f"""SELECT left(project_id, 4) yymm, substr(project_id, 5, 1) s,
            count(DISTINCT project_id) n, min(CAST(substr(project_id, 6) AS BIGINT)) lo, max(CAST(substr(project_id, 6) AS BIGINT)) hi
            FROM read_parquet('{src(y)}') WHERE length(project_id) = 11 AND regexp_matches(project_id, '^[0-9]+$')
            GROUP BY ALL""").df() for y in years()])
    N = N.groupby(["yymm", "s"]).agg(n=("n", "sum"), lo=("lo", "min"), hi=("hi", "max")).reset_index()
    N["yy"], N["mm"] = N.yymm.str[:2].astype(int), N.yymm.str[2:].astype(int)
    N = N[(N.mm.between(1, 12)) & (N.n >= 500)].copy()
    N["fy"] = 2500 + N.yy + (N.mm >= 10).astype(int)
    N["numbers"] = N.hi - N.lo + 1
    N.to_csv(REPORTS / "coverage_numbers.csv", index=False)
    G = N.groupby("fy").agg(published=("n", "sum"), numbers=("numbers", "sum"))
    G["share_published"] = (G.published / G.numbers).round(3)
    for c in ("published", "numbers"):
        G[c] = G[c].map(lambda x: f"{int(x):,}")
    G = G[G.index.isin(years())]
    Mo = N.groupby("yymm").agg(published=("n", "sum"), numbers=("numbers", "sum"))
    Mo["share_published"] = (Mo.published / Mo.numbers).round(3)
    Mo = Mo.loc["5807":"5906"]
    for c in ("published", "numbers"):
        Mo[c] = Mo[c].map(lambda x: f"{int(x):,}")
    SECTIONS["months"] = ["## Monthly share around the reform (YYMM of the project number, B.E. year)", "",
                          "5809 = September 2015; 5810–5811 = October–November 2015, the first months under the new ceiling.", "",
                          Mo.to_markdown(), ""]
    SECTIONS["numbers"] = ["## e-GP project numbers that appear in the published contract data", "",
                           "Fiscal year of the project number (YYMM); series with fewer than 500 published projects dropped.", "",
                           G.to_markdown(), ""]


def summary(con=None):
    rows = []
    for f in sorted(Path(RAW_SUMMARY).glob("*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
        n = pd.to_numeric(d["จำนวนโครงการ"].str.replace(",", ""), errors="coerce")
        v = pd.to_numeric(d["วงเงินงบประมาณ (บาท)"].str.replace(",", ""), errors="coerce")
        rows.append({"fy": int(f.name[:4]), "projects_million": n.sum() / 1e6, "budget_billion": v.sum() / 1e9,
                     "units_with_projects": int((n > 0).sum())})
    S = pd.DataFrame(rows).set_index("fy")
    S.to_csv(REPORTS / "coverage_summary.csv")
    SECTIONS["summary"] = ["## Comptroller General's summary tables", "", S.round(2).to_markdown(), ""]


def write():
    L = ["# Coverage of fiscal 2016–2017 (B.E. 2559–2560)", ""]
    for k in ("units", "bands", "numbers", "months", "summary"):
        L += SECTIONS.get(k, [])
    out = REPORTS / "coverage_check.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    con = connect()
    con.execute("SET preserve_insertion_order = false")   # lower memory for the distinct counts
    steps = sys.argv[1:] or ["units", "bands", "numbers", "summary"]
    for s in steps:
        {"units": units, "bands": bands, "numbers": numbers, "summary": summary}[s](con)
        log(f"{s} done")
    write()
