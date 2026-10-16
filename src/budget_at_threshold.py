#!/usr/bin/env python3
"""
Contracts in the top bin below THB 500,000 (agreed price THB 495,001–500,000): how many record a budget or an
agreed price of exactly THB 500,000, and how many record the agreed price at the budget? Post-Act against the
pre-reform reference, for all contracts and for single contracts (the only contract of the unit–vendor pair that
day, not part of a same-month split), as in excess_sources.py (consistency matrix, GIQ v24).

Usage (Anaconda Prompt, env egp-xai):  python src\\budget_at_threshold.py   (re-run until all years are cached)
Output: reports/budget_at_threshold.md; cache data/processed/_revision/bat_YYYY.parquet.
"""
import sys
import time

import pandas as pd

from bunching import MAX_PRICE, REGIME_SQL, REPORTS, connect, log
from revision_checks import OUT_DIR, src, years


def main(budget_s=150):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()
    con.execute("SET preserve_insertion_order = false")
    t_all = time.time()
    for y in years():
        f = OUT_DIR / f"bat_{y}.parquet"
        if f.exists():
            continue
        if time.time() - t_all > budget_s:
            log("time budget reached; run again to continue")
            return
        t0 = time.time()
        con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
            SELECT sign_date, date_trunc('month', sign_date) mo, agreed_price p, budget,
                   agency || '|' || coalesce(sub_agency, '') || '|' || vendor_name_key || '|' || coalesce(province, '') AS uv,
                   {REGIME_SQL} AS regime
            FROM read_parquet('{src(y)}')
            WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE} AND vendor_name_key IS NOT NULL""")
        con.execute("CREATE OR REPLACE TEMP TABLE d AS SELECT uv, sign_date, count(*) k FROM c GROUP BY ALL")
        con.execute("CREATE OR REPLACE TEMP TABLE m AS SELECT uv, mo, count(*) k, sum(p) v, max(p) mx FROM c GROUP BY ALL")
        con.execute(f"""COPY (
            SELECT {y} AS fy, c.regime,
                   (d.k = 1 AND NOT (m.k >= 2 AND m.mx <= 500000 AND m.v > 500000 AND m.v <= 2000000)) AS single,
                   count(*)::INT n,
                   sum(CASE WHEN c.budget = 500000 THEN 1 ELSE 0 END)::INT n_budget_500k,
                   sum(CASE WHEN c.p = 500000 THEN 1 ELSE 0 END)::INT n_price_500k,
                   sum(CASE WHEN c.p = c.budget THEN 1 ELSE 0 END)::INT n_price_eq_budget
            FROM c JOIN d USING (uv, sign_date) JOIN m USING (uv, mo)
            WHERE c.p > 495000 AND c.p <= 500000
            GROUP BY ALL) TO '{f.as_posix()}' (FORMAT parquet)""")
        con.execute("DROP TABLE c; DROP TABLE d; DROP TABLE m")
        log(f"bat {y}: {time.time() - t0:.0f}s")
    X = pd.concat([pd.read_parquet(OUT_DIR / f"bat_{y}.parquet") for y in years()], ignore_index=True)
    X = X[X.regime.isin(["pre", "act"])]
    L = ["# Top bin below THB 500,000 (agreed price THB 495,001–500,000)", "",
         "| period | contracts | group | n | budget = THB 500,000 | agreed price = THB 500,000 | agreed price = budget |",
         "|---|---|---|---|---|---|---|"]
    for reg in ("pre", "act"):
        for lab, sub in (("all", X[X.regime == reg]), ("single", X[(X.regime == reg) & X.single])):
            n = sub.n.sum()
            L.append(f"| {reg} | top bin | {lab} | {n:,} | {sub.n_budget_500k.sum() / n:.1%} | "
                     f"{sub.n_price_500k.sum() / n:.1%} | {sub.n_price_eq_budget.sum() / n:.1%} |")
    out = REPORTS / "budget_at_threshold.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
