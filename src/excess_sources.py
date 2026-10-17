#!/usr/bin/env python3
"""
Where does the excess mass below THB 500,000 come from? (results review, GIQ v8)

The difference-in-bunching excess is counted in the window THB 450,000–500,000, but the pieces of split purchases
spread across the whole range below the threshold. This script splits every contract of THB 0.1–2 million into
mutually exclusive groups and decomposes the excess additively. The counterfactual is linear in the reference
(ĉ = k·r, k from the THB 200,000–400,000 band over all groups), so
    excess_g = Σ_window (y_g − k·r_g)   and   Σ_g excess_g = total excess.
Groups (buying unit × vendor key; vendor key = normalized name, plus province for shops and individuals):
  split_day     piece of a same-day split purchase at THB 500,000 (2+ contracts, each ≤ THB 500,000, total 0.5–2 million)
  split_month   piece of a same-month split purchase that is not a same-day one
  multi_day     other same-day groups of 2+ contracts (totals below THB 500,000 or above THB 2 million, or a piece above it)
  single        the only contract of its unit–vendor pair that day, not part of a same-month split
For the single contracts, the script also reports how many lie in the top bin (THB 495,001–500,000) and how many
record the agreed price equal to the budget.

Usage (Anaconda Prompt, env egp-xai):  python src\\excess_sources.py
Output: reports/excess_sources.md (+ .csv); cache data/processed/_revision/exs_YYYY.parquet.
"""
import sys
import time

import numpy as np
import pandas as pd

from bunching import MAX_PRICE, REGIME_SQL, REPORTS, connect, log
from revision_checks import OUT_DIR, src, years

W, LO, HI, T, ZL = 5000, 100000, 2000000, 500000, 50000
GROUPS = ("split_day", "split_month", "multi_day", "single")


def aggregate(con):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for y in years():
        f = OUT_DIR / f"exs_{y}.parquet"
        if not f.exists():
            t0 = time.time()
            con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
                SELECT sign_date, date_trunc('month', sign_date) mo, agreed_price p, budget,
                       agency || '|' || coalesce(sub_agency, '') || '|' || vendor_name_key || '|' || coalesce(province, '') AS uv,
                       {REGIME_SQL} AS regime
                FROM read_parquet('{src(y)}')
                WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE} AND vendor_name_key IS NOT NULL""")
            con.execute("""CREATE OR REPLACE TEMP TABLE d AS
                SELECT uv, sign_date, count(*) k, sum(p) v, max(p) mx FROM c GROUP BY ALL""")
            con.execute("""CREATE OR REPLACE TEMP TABLE m AS
                SELECT uv, mo, count(*) k, sum(p) v, max(p) mx FROM c GROUP BY ALL""")
            con.execute(f"""COPY (
                SELECT {y} AS fy, c.regime,
                       CASE WHEN d.k >= 2 AND d.mx <= 500000 AND d.v > 500000 AND d.v <= 2000000 THEN 'split_day'
                            WHEN m.k >= 2 AND m.mx <= 500000 AND m.v > 500000 AND m.v <= 2000000 THEN 'split_month'
                            WHEN d.k >= 2 THEN 'multi_day' ELSE 'single' END AS grp,
                       (ceil(c.p / 5000) * 5000)::INT AS bin, count(*)::INT n,
                       sum(CASE WHEN c.p = c.budget THEN 1 ELSE 0 END)::INT n_eq_budget
                FROM c JOIN d USING (uv, sign_date) JOIN m USING (uv, mo)
                WHERE c.p > 100000 AND c.p <= 2000000
                GROUP BY ALL) TO '{f.as_posix()}' (FORMAT parquet)""")
            con.execute("DROP TABLE c; DROP TABLE d; DROP TABLE m")
            log(f"exs {y}: {time.time() - t0:.0f}s")
        parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True)


def main():
    con = connect()
    con.execute("SET preserve_insertion_order = false")
    X = aggregate(con)
    grid = np.arange(LO + W, HI + W, W)
    piv = lambda d: d.groupby(["grp", "bin"]).n.sum().unstack(0).reindex(index=grid, columns=list(GROUPS)).fillna(0)
    r, y = piv(X[X.regime == "pre"]), piv(X[X.regime == "act"])
    S = (grid > 200000) & (grid <= 400000)
    win = (grid > T - ZL) & (grid <= T)
    k = y[S].values.sum() / r[S].values.sum()
    ex = (y[win] - k * r[win]).sum()
    tot = ex.sum()
    cf_bar = (k * r[win]).values.sum() / win.sum()
    rows = []
    for g in GROUPS:
        rows.append({"group": g, "contracts_act_0.1_2M": y[g].sum(), "share_of_act_contracts": y[g].sum() / y.values.sum(),
                     "in_window_act": y.loc[win, g].sum(), "share_of_group_in_window": y.loc[win, g].sum() / y[g].sum(),
                     "share_of_group_in_scaling_band": y.loc[S, g].sum() / y[g].sum(),
                     "excess_in_window": ex[g], "share_of_excess": ex[g] / tot})
    R = pd.DataFrame(rows)
    R.to_csv(REPORTS / "excess_sources.csv", index=False)
    top = (grid > T - W) & (grid <= T)
    sx = X[(X.regime == "act") & (X.grp == "single") & (X.bin == T)]
    L = ["# Sources of the excess mass below THB 500,000 (post-Act vs pre-reform reference)", "",
         f"Scaling factor k = {k:.3f}; total excess in THB 450,001–500,000 = {tot:,.0f} contracts "
         f"(Δz/T = {tot / (cf_bar / W) / T:.3f}). Contracts without a vendor key are excluded.", "",
         "| group | post-Act contracts THB 0.1–2M | share | in window | share of group in window | share in scaling band | excess in window | share of excess |",
         "|---|---|---|---|---|---|---|---|"]
    for q in R.itertuples():
        L.append(f"| {q.group} | {q._2:,.0f} | {q.share_of_act_contracts:.1%} | {q.in_window_act:,.0f} | {q.share_of_group_in_window:.1%} | "
                 f"{q.share_of_group_in_scaling_band:.1%} | {q.excess_in_window:,.0f} | {q.share_of_excess:.0%} |")
    L += ["", f"Single contracts in the top bin (THB 495,001–500,000), post-Act: {sx.n.sum():,.0f}; "
          f"excess in the top bin over all groups: {(y[top] - k * r[top]).values.sum():,.0f}; "
          f"single contracts in the top bin recorded at exactly the budget: {sx.n_eq_budget.sum() / max(sx.n.sum(), 1):.0%}."]
    out = REPORTS / "excess_sources.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
