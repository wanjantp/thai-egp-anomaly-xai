#!/usr/bin/env python3
"""
Does the rise in bunching over time come from the mix of work types? (final pre-submission audit, GIQ v25)

The yearly difference-in-bunching response (Δz/T at THB 500,000, reference = contracts signed before 15 September 2015,
same estimator as the manuscript) is re-estimated with each year's contracts reweighted so that the mix of work types
(classified from project names with bunching.KIND_SQL: medical, works, food, goods, services, other) is held at its
fiscal 2016 level, as composition_check.py does for the central/local mix. A last pair of rows holds both mixes
(work type × central/local) at fiscal 2016. Agency-clustered Poisson bootstrap (weights shared across years).

Usage (Anaconda Prompt, env egp-xai):  python src\\worktype_mix.py   (re-run until all years are cached)
Input: data/interim/contracts_YYYY/*.parquet (read-only).
Cache: data/processed/_unit_kind_bins5k/YYYY.parquet.  Output: reports/worktype_mix.md (+ .csv).
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

from bunching import INTERIM, KIND_SQL, PROCESSED, REGIME_SQL, REPORTS, connect, dib_estimate, log
from composition_check import is_local

W, LO, HI, T = 5000, 100000, 2000000, 500000
YEARS = list(range(2559, 2569))
KB_DIR = PROCESSED / "_unit_kind_bins5k"


def aggregate(budget_s=150):
    KB_DIR.mkdir(exist_ok=True)
    con = connect()
    con.execute("SET preserve_insertion_order=false")
    t_all = time.time()
    for y in range(2558, 2569):
        out = KB_DIR / f"{y}.parquet"
        if out.exists():
            continue
        if time.time() - t_all > budget_s:
            log("time budget reached; run again to continue")
            return False
        t0 = time.time()
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        con.execute(f"""COPY (
            WITH c AS (SELECT DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)
                              agency || '|' || coalesce(sub_agency, '') AS unit, sign_date, agreed_price, project_name
                       FROM read_parquet('{src}')
                       WHERE sign_date IS NOT NULL AND agreed_price > 100000 AND agreed_price <= 2000000)
            SELECT {y} AS fiscal_year, unit, {REGIME_SQL} AS regime, {KIND_SQL} AS kind,
                   (ceil(agreed_price / 5000) * 5000)::INT AS bin, count(*)::INT AS n
            FROM c GROUP BY ALL) TO '{out.as_posix()}' (FORMAT parquet)""")
        log(f"unit_kind_bins {y}: {time.time() - t0:.0f}s")
    return True


def main(R=500, seed=2026):
    if not aggregate():
        return
    U = pd.concat([pd.read_parquet(f) for f in sorted(KB_DIR.glob("*.parquet"))], ignore_index=True)
    grid = np.arange(LO + W, HI + W, W)
    gi = {g: i for i, g in enumerate(grid)}
    U = U[U["bin"].isin(gi)].copy()
    U["local"] = U["unit"].str.split("|", n=1).str[0].map(is_local)
    U["cell"] = U["kind"] + np.where(U["local"], "|local", "|central")
    kinds = sorted(U["kind"].unique())
    cells = sorted(U["cell"].unique())
    uid, units = pd.factorize(U["unit"].str.split("|", n=1).str[0])
    gid = U["bin"].map(gi).values
    nU, nG = len(units), len(grid)
    rng = np.random.default_rng(seed)
    Ws = sparse.csr_matrix(np.vstack([np.ones(nU), rng.poisson(1.0, size=(R, nU))]))
    est = lambda yt, yr: dib_estimate(yt, yr, grid, T, W, 200000, 400000, 50000, [1500000], "const")["dz"] / T
    is_ref = (U.regime == "pre").values
    in_year = {y: ((U.fiscal_year == y) & (U.regime != "pre")).values for y in YEARS}

    def agg(mask):
        return np.asarray((Ws @ sparse.csr_matrix((U.n.values[mask].astype(float), (uid[mask], gid[mask])),
                                                  shape=(nU, nG))).todense())

    ref = agg(is_ref)
    rows = []

    def add(g, y, post):
        vals = np.array([est(post[i], ref[i]) for i in range(post.shape[0])])
        rows.append({"group": g, "fiscal_year": y - 543, "dz_T": vals[0], "lo": np.percentile(vals[1:], 2.5),
                     "hi": np.percentile(vals[1:], 97.5), "draws": vals[1:], "contracts": float(post[0].sum())})

    by = {}
    for col, levels in (("kind", kinds), ("cell", cells)):
        for y in YEARS:
            for k in levels:
                by[col, k, y] = agg(in_year[y] & (U[col] == k).values)
    shares = {}
    for y in YEARS:
        tot = sum(by["kind", k, y] for k in kinds)
        add("all", y, tot)
        shares[y] = {k: by["kind", k, y][0].sum() / tot[0].sum() for k in kinds}
    for col, levels, lab in (("kind", kinds, "work-type mix held at 2016"),
                             ("cell", cells, "work-type × central/local mix held at 2016")):
        s16 = {k: by[col, k, 2559].sum(1) / sum(by[col, j, 2559].sum(1) for j in levels) for k in levels}
        for y in YEARS:
            parts = {k: by[col, k, y] for k in levels}
            n = sum(p.sum(1) for p in parts.values())
            post = sum(p * np.divide(s16[k] * n, p.sum(1), out=np.zeros_like(n), where=p.sum(1) > 0)[:, None]
                       for k, p in parts.items())
            add(lab, y, post)
        log(f"{lab} done")
    R_ = pd.DataFrame(rows)
    R_.drop(columns="draws").to_csv(REPORTS / "worktype_mix.csv", index=False)
    groups = list(dict.fromkeys(R_.group))

    L = ["# Bunching response by year with the work-type mix held at fiscal 2016 (Δz/T at THB 500,000)", "",
         f"Reference: contracts signed before 15 September 2015 (all work types). 95% intervals from a Poisson bootstrap "
         f"over agencies ({R} draws, weights shared across years). Work type from project names (bunching.KIND_SQL).", "",
         "| fiscal year | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
    for y in YEARS:
        cs = []
        for g in groups:
            r = R_[(R_.group == g) & (R_.fiscal_year == y - 543)].iloc[0]
            cs.append(f"{r.dz_T:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
        L.append(f"| {y - 543} | " + " | ".join(cs) + " |")
    L += ["", "## Change from fiscal 2016 to fiscal 2025", "", "| group | 2016 | 2025 | change | 95% CI of change |",
          "|---|---|---|---|---|"]
    for g in groups:
        a = R_[(R_.group == g) & (R_.fiscal_year == 2016)].iloc[0]
        b = R_[(R_.group == g) & (R_.fiscal_year == 2025)].iloc[0]
        d = b.draws - a.draws
        L.append(f"| {g} | {a.dz_T:.3f} | {b.dz_T:.3f} | {b.dz_T - a.dz_T:+.3f} | "
                 f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] |")
    L += ["", "## Work-type shares of contracts THB 0.1–2 million (post-reform part of each fiscal year)", "",
          "| fiscal year | " + " | ".join(kinds) + " |", "|---" * (len(kinds) + 1) + "|"]
    for y in YEARS:
        L.append(f"| {y - 543} | " + " | ".join(f"{shares[y][k]:.1%}" for k in kinds) + " |")
    out = REPORTS / "worktype_mix.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
