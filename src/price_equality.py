#!/usr/bin/env python3
"""
Share of THB 0.1–0.5 million contracts whose agreed price equals the budget or the reference price, by period and method
class (numerical consistency audit, GIQ v19). Source of the "where the discount goes" shares in Section 5.5.

Periods as in Table 2: pre = signed before 1 Oct 2015; transition = to 22 Aug 2017; act = from 23 Aug 2017.
Usage (Anaconda Prompt, env egp-xai):  python src\\price_equality.py
Output: reports/price_equality.md (+ .csv; per-year parts cached in data/processed/_revision/price_eq_<year>.parquet)
"""
import sys
import time

import pandas as pd

from bunching import MAX_PRICE, METHOD_CLS, REPORTS, connect, log
from revision_checks import DEDUP, OUT_DIR, src


def main(budget_s=150):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()
    t_all = time.time()
    for y in range(2558, 2569):
        f = OUT_DIR / f"price_eq_{y}.parquet"
        if f.exists():
            continue
        if time.time() - t_all > budget_s:
            log("time budget reached; run again to continue")
            return
        con.execute(f"""COPY (
            WITH c AS (SELECT {DEDUP} sign_date, agreed_price, budget, ref_price, {METHOD_CLS} AS mcls
                       FROM read_parquet('{src(y)}')
                       WHERE sign_date IS NOT NULL AND agreed_price > 100000 AND agreed_price <= 500000)
            SELECT CASE WHEN sign_date < DATE '2015-10-01' THEN 'pre' WHEN sign_date < DATE '2017-08-23' THEN 'transition'
                        ELSE 'act' END AS period, mcls, count(*)::BIGINT n,
                   sum(CASE WHEN budget > 0 THEN 1 ELSE 0 END)::BIGINT n_b,
                   sum(CASE WHEN budget > 0 AND agreed_price = budget THEN 1 ELSE 0 END)::BIGINT eq_b,
                   sum(CASE WHEN ref_price > 0 THEN 1 ELSE 0 END)::BIGINT n_r,
                   sum(CASE WHEN ref_price > 0 AND agreed_price = ref_price THEN 1 ELSE 0 END)::BIGINT eq_r
            FROM c GROUP BY ALL) TO '{f.as_posix()}' (FORMAT parquet)""")
        log(f"price_eq {y} done")
    D = pd.concat([pd.read_parquet(OUT_DIR / f"price_eq_{y}.parquet") for y in range(2558, 2569)], ignore_index=True)
    G = D.groupby(["period", "mcls"])[["n", "n_b", "eq_b", "n_r", "eq_r"]].sum().reset_index()
    G["share_eq_budget"] = G.eq_b / G.n_b
    G["share_eq_ref"] = G.eq_r / G.n_r
    G.to_csv(REPORTS / "price_equality.csv", index=False)
    L = ["# THB 0.1–0.5 million contracts recorded at exactly the budget or the reference price", "",
         "Denominators: contracts with a positive budget (budget column) or a positive reference price (reference column).", "",
         "| period | method | contracts | = budget | = reference price |", "|---|---|---|---|---|"]
    for r in G.sort_values(["mcls", "period"]).itertuples():
        L.append(f"| {r.period} | {r.mcls} | {r.n:,} | {r.share_eq_budget:.1%} | {r.share_eq_ref:.1%} |")
    out = REPORTS / "price_equality.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
