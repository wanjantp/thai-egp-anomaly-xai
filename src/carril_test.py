#!/usr/bin/env python3
"""
Carril (2021)-style splitting test for the GIQ manuscript (v5).

Carril (2021, Sec. 3.1) asks whether bunching below the U.S. simplified acquisition threshold survives when the
sample is restricted to office–firm pairs with a single transaction in the fiscal year; it does, so he rules out
splitting. We run the same test on Thai e-GP data. If splitting drives the bunching below THB 500,000, the
response should shrink sharply in samples that exclude repeat transactions between a buying unit and a vendor.

Samples (each applied to both the reference period and the post-Act period):
  all             every contract (the manuscript's baseline)
  single_pair     contracts of unit–vendor pairs with exactly one contract in the fiscal year (Carril's test)
  no_same_day     contracts that are the only one of their unit–vendor pair on that signing day
Estimator: the manuscript's difference-in-bunching (THB 5,000 bins, window THB 450,000–500,000, scaling band
THB 200,000–400,000, reference = contracts signed before 15 September 2015), post-Act period.

Usage (Anaconda Prompt, env egp-xai):  python src\\carril_test.py
Output: reports/carril_test.md (+ .csv). Cache: data/processed/_revision/carril_YYYY.parquet.
"""
import sys
import time

import pandas as pd

from bunching import MAX_PRICE, REGIME_SQL, REPORTS, _bins5k, connect, dib_estimate, log
from revision_checks import DEDUP, OUT_DIR, src, years

W, LO, HI, T, ZL = 5000, 100000, 2000000, 500000, 50000
SAMPLES = ("all", "single_pair", "no_same_day")


def aggregate(con):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for y in years():
        f = OUT_DIR / f"carril_{y}.parquet"
        if not f.exists():
            t0 = time.time()
            con.execute(f"""COPY (
                WITH c AS (SELECT {DEDUP} sign_date, agreed_price, agency || '|' || coalesce(sub_agency, '') AS unit,
                                  vendor_name_key || '|' || coalesce(province, '') AS vkey
                           FROM read_parquet('{src(y)}')
                           WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE}
                             AND vendor_name_key IS NOT NULL),
                     x AS (SELECT *, count(*) OVER (PARTITION BY unit, vkey) AS npair,
                                     count(*) OVER (PARTITION BY unit, vkey, sign_date) AS nday FROM c)
                SELECT {y} AS fy, {REGIME_SQL} AS regime, (ceil(agreed_price / 1000) * 1000)::INT AS bin,
                       count(*)::INT AS n_all,
                       sum(CASE WHEN npair = 1 THEN 1 ELSE 0 END)::INT AS n_single_pair,
                       sum(CASE WHEN nday = 1 THEN 1 ELSE 0 END)::INT AS n_no_same_day
                FROM x WHERE agreed_price <= {HI} GROUP BY ALL) TO '{f.as_posix()}' (FORMAT parquet)""")
            log(f"carril {y}: {time.time() - t0:.0f}s")
        parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True)


def main():
    con = connect()
    con.execute("SET preserve_insertion_order = false")
    B = aggregate(con)
    rows = []
    for s in SAMPLES:
        col = "n_" + s
        ref = B[B.regime == "pre"].groupby("bin")[col].sum()
        act = B[B.regime == "act"].groupby("bin")[col].sum()
        r_ref, r_act = _bins5k(ref, W, LO, HI), _bins5k(act, W, LO, HI)
        r = dib_estimate(r_act.values, r_ref.values, r_ref.index.values, T, W, 200000, 400000, ZL, [1500000], "const")
        rows.append({"sample": s, "contracts_act_100k_2M": float(r_act.sum()), "share_of_all_act": None,
                     "dz_over_T": r["dz"] / T, "excess": r["excess_B"]})
    R = pd.DataFrame(rows).set_index("sample")
    R["share_of_all_act"] = R.contracts_act_100k_2M / R.loc["all", "contracts_act_100k_2M"]
    R["relative_to_all"] = R.dz_over_T / R.loc["all", "dz_over_T"]
    R.to_csv(REPORTS / "carril_test.csv")
    L = ["# Carril (2021)-style test: does bunching below THB 500,000 survive without repeat unit–vendor transactions?", "",
         "Difference-in-bunching, post-Act period vs contracts signed before 15 September 2015; same sample restriction in both.", "",
         "| sample | contracts THB 0.1–2M (post-Act) | share of all | Δz/T | relative to all | excess contracts |",
         "|---|---|---|---|---|---|"]
    for s, r in R.iterrows():
        L.append(f"| {s} | {r.contracts_act_100k_2M:,.0f} | {r.share_of_all_act:.0%} | {r.dz_over_T:.3f} | "
                 f"{r.relative_to_all:.0%} | {r.excess:,.0f} |")
    L += ["", "single_pair = unit–vendor pairs with one contract in the fiscal year (Carril 2021); "
              "no_same_day = contracts that are alone for their unit–vendor pair on the signing day."]
    out = REPORTS / "carril_test.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
