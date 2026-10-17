#!/usr/bin/env python3
"""
Descriptive statistics of the analysis sample (GIQ manuscript, Table 2 from v7; statistical review item 6).

For contracts with an agreed price of THB 1–3 million, by regime (assigned by signing date) × central/local ×
method class, and for the THB 0.1–0.5 million band that the reform moved:
  contracts, median and mean agreed price, share signed by a discretionary method,
  discount relative to budget (1 − agreed/budget, contracts with 0.3 ≤ agreed/budget ≤ 1.5): mean and SD,
  share of contracts with the agreed price at or above the budget.
Aggregated year by year (sums, sums of squares, approximate medians), then combined.

Usage (Anaconda Prompt, env egp-xai):  python src\\descriptives.py
Output: reports/descriptives.md (+ descriptives.csv); cache data/processed/_revision/desc_YYYY.parquet.
"""
import sys
import time

import numpy as np
import pandas as pd

from bunching import GOV_SQL, MAX_PRICE, METHOD_CLS, REGIME_SQL, REPORTS, connect, log
from revision_checks import OUT_DIR, src, years


def aggregate(con):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for y in years():
        f = OUT_DIR / f"desc_{y}.parquet"
        if not f.exists():
            t0 = time.time()
            con.execute(f"""COPY (
                WITH c AS (SELECT agreed_price p, budget, {GOV_SQL} AS gov, {REGIME_SQL} AS regime, {METHOD_CLS} AS mcls,
                                  CASE WHEN agreed_price > 100000 AND agreed_price <= 500000 THEN '0.1-0.5M' ELSE 'other' END AS band,
                                  budget > 0 AND agreed_price / budget BETWEEN 0.3 AND 1.5 AS ok_b
                           FROM read_parquet('{src(y)}')
                           WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE})
                SELECT {y} AS fy, regime, gov, mcls, band, count(*)::BIGINT n, sum(p) sp, approx_quantile(p, 0.5) med,
                       sum(CASE WHEN ok_b THEN 1 ELSE 0 END)::BIGINT nb,
                       sum(CASE WHEN ok_b THEN 1 - p / budget END) sd, sum(CASE WHEN ok_b THEN (1 - p / budget) ^ 2 END) sd2,
                       sum(CASE WHEN ok_b AND p >= budget THEN 1 ELSE 0 END)::BIGINT n_ge
                FROM c GROUP BY ALL) TO '{f.as_posix()}' (FORMAT parquet)""")
            log(f"desc {y}: {time.time() - t0:.0f}s")
        parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True)


def summarize(g):
    n, nb = g.n.sum(), g.nb.sum()
    m = g.sd.sum() / nb
    sd = np.sqrt(max(g.sd2.sum() / nb - m * m, 0))
    return pd.Series({"contracts": n, "median_price": np.average(g.med, weights=g.n),   # contract-weighted mean of yearly medians
                      "mean_price": g.sp.sum() / n,
                      "share_discretionary": g.loc[g.mcls == "discretion", "n"].sum() / n,
                      "discount_mean": m, "discount_sd": sd, "share_at_or_above_budget": g.n_ge.sum() / nb})


def main():
    con = connect()
    con.execute("SET preserve_insertion_order = false")
    D = aggregate(con)
    rows = []
    for band, B in (("THB 1–3,000,000", D), ("THB 0.1–0.5 million", D[D.band == "0.1-0.5M"])):
        for (rg, gv), g in B.groupby(["regime", "gov"]):
            rows.append({"band": band, "regime": rg, "gov": gv, **summarize(g)})
        for (rg, mc), g in B[B.mcls.isin(["discretion", "competitive"])].groupby(["regime", "mcls"]):
            rows.append({"band": band, "regime": rg, "gov": f"all, {mc}", **summarize(g)})
    R = pd.DataFrame(rows)
    order = {"pre": 0, "transition": 1, "act": 2}
    R = R.sort_values(["band", "regime", "gov"], key=lambda s: s.map(order) if s.name == "regime" else s)
    R.to_csv(REPORTS / "descriptives.csv", index=False)
    L = ["# Descriptive statistics by regime (agreed price; discount = 1 − agreed/budget)", "",
         "Regimes by signing date: pre = before 1 Oct 2015; transition = 1 Oct 2015 – 22 Aug 2017; act = from 23 Aug 2017. "
         "Median = contract-weighted mean of yearly approximate medians.", "",
         "| band | regime | group | contracts | median price | mean price | discretionary | discount mean (SD) | at/above budget |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        L.append(f"| {r.band} | {r.regime} | {r.gov} | {r.contracts:,.0f} | {r.median_price:,.0f} | {r.mean_price:,.0f} | "
                 f"{r.share_discretionary:.0%} | {r.discount_mean:.1%} ({r.discount_sd:.1%}) | {r.share_at_or_above_budget:.0%} |")
    out = REPORTS / "descriptives.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
