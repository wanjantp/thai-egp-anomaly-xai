#!/usr/bin/env python3
"""
Monthly difference-in-bunching with agency-clustered intervals (results review, GIQ v8).

The event study in bunching.py (`event`) attaches contract-level Poisson errors plus shape error. This script
re-estimates the same monthly series with a Poisson bootstrap over agencies (weights shared across months, so the
reference and every month move together in each draw), plus the same shape error, for
  THB 500,000: 5,000 bins, THB 100,000–1,000,000, reference = pre-reform months (leave-one-month-out), scaling
               band THB 200,000–400,000, window THB 50,000;
  THB 100,000: 1,000 bins, THB 30,000–200,000, reference = post-Act months, scaling band THB 40,000–85,000,
               window THB 10,000.
It also reports, with draw-based intervals: the October minus September 2015 jump, the spread of pre-reform months,
the September effect in each post-Act fiscal year, and period means at the THB 100,000 line.

Usage (Anaconda Prompt, env egp-xai):  python src\\agency_event.py
Output: reports/agency_event.md, reports/bunching_event_monthly_agency.csv; cache data/processed/_revision/aev_YYYY.parquet.
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

from bunching import MAX_PRICE, REGIME_SQL, REPORTS, connect, log
from revision_checks import OUT_DIR, src, years

CFG = {500_000: dict(scale_tag="5k", W=5_000, lo=100_000, hi=1_000_000, zL=50_000, sc=(200_000, 400_000), ref="pre"),
       100_000: dict(scale_tag="1k", W=1_000, lo=30_000, hi=200_000, zL=10_000, sc=(40_000, 85_000), ref="act")}


def aggregate(con):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for y in years():
        f = OUT_DIR / f"aev_{y}.parquet"
        if not f.exists():
            t0 = time.time()
            con.execute(f"""COPY (
                WITH c AS (SELECT agency, strftime(sign_date, '%Y-%m') ym, {REGIME_SQL} AS regime, agreed_price p
                           FROM read_parquet('{src(y)}')
                           WHERE sign_date IS NOT NULL AND agreed_price > 30000 AND agreed_price <= 1000000)
                SELECT agency, ym, regime, '5k' AS scale, (ceil(p / 5000) * 5000)::INT AS bin, count(*)::INT n
                FROM c WHERE p > 100000 GROUP BY ALL
                UNION ALL
                SELECT agency, ym, regime, '1k' AS scale, (ceil(p / 1000) * 1000)::INT AS bin, count(*)::INT n
                FROM c WHERE p <= 200000 GROUP BY ALL) TO '{f.as_posix()}' (FORMAT parquet)""")
            log(f"aev {y}: {time.time() - t0:.0f}s")
        parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True)


def one(yt, yr, grid, T, W, zL, mask):
    """Δz for each row of yt, yr (draws × bins)."""
    k = yt[:, mask].sum(1) / np.maximum(yr[:, mask].sum(1), 1e-9)
    cf = yr * k[:, None]
    w = (grid > T - zL) & (grid <= T)
    h0 = cf[:, w].mean(1) / W
    return np.where(h0 > 0, (yt[:, w] - cf[:, w]).sum(1) / np.where(h0 > 0, h0, 1), np.nan)


def series(A, T, c, R, rng):
    d = A[A.scale == c["scale_tag"]]
    grid = np.arange(c["lo"] + c["W"], c["hi"] + c["W"], c["W"])
    gi = {g: i for i, g in enumerate(grid)}
    d = d[d.bin.isin(gi)]
    aid, ags = pd.factorize(d.agency, sort=True)  # sorted: same bootstrap draws on every machine
    gid = d.bin.map(gi).values
    Ws = sparse.csr_matrix(np.vstack([np.ones(len(ags)), rng.poisson(1.0, size=(R, len(ags)))]))
    months = sorted(m for m in d.ym.unique() if "2014-10" <= m <= "2025-09")
    M = {}
    for m, g in d.groupby("ym"):
        idx = g.index
        M[m] = np.asarray((Ws @ sparse.csr_matrix((g.n.values.astype(float), (aid[d.index.get_indexer(idx)], gid[d.index.get_indexer(idx)])),
                                                  shape=(len(ags), len(grid)))).todense())
    ref_months = sorted(set(d.loc[d.regime == c["ref"], "ym"]))
    # months that straddle a regime boundary contribute only their reference-regime contracts to the reference
    Mref = {}
    for m, g in d[d.regime == c["ref"]].groupby("ym"):
        idx = d.index.get_indexer(g.index)
        Mref[m] = np.asarray((Ws @ sparse.csr_matrix((g.n.values.astype(float), (aid[idx], gid[idx])),
                                                     shape=(len(ags), len(grid)))).todense())
    ref_tot = sum(Mref.values())
    sm = (grid > c["sc"][0]) & (grid <= c["sc"][1])
    rows, D = [], {}
    for m in months:
        yt = M[m]
        if yt[0].sum() < 500:
            continue
        yr = ref_tot - Mref[m] if m in Mref else ref_tot
        dz = one(yt, yr, grid, T, c["W"], c["zL"], sm)
        pl = []
        for Tp in range(c["sc"][0] + c["zL"], c["sc"][1] + 1, c["W"]):
            msk = sm & ~((grid > Tp - c["zL"]) & (grid <= Tp))
            pl.append(one(yt[:1], yr[:1], grid, Tp, c["W"], c["zL"], msk)[0])
        psd = np.nanstd(pl, ddof=1)
        tot = dz[1:] + rng.normal(0, psd, R)
        D[m] = np.concatenate([[dz[0]], tot]) / T
        rows.append({"T": T, "ym": m, "dz_over_T": dz[0] / T, "se_agency": np.nanstd(tot, ddof=1) / T,
                     "ci_lo": np.nanpercentile(tot, 2.5) / T, "ci_hi": np.nanpercentile(tot, 97.5) / T,
                     "placebo_sd": psd / T, "n_scale": yt[0, sm].sum()})
    return pd.DataFrame(rows), D


def ci(x):
    return f"{x[0]:+.3f} [{np.nanpercentile(x[1:], 2.5):+.3f}, {np.nanpercentile(x[1:], 97.5):+.3f}]"


def main(R=200, seed=2026):
    con = connect()
    con.execute("SET preserve_insertion_order = false")
    A = aggregate(con).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    E5, D5 = series(A, 500_000, CFG[500_000], R, rng)
    log("500k done")
    E1, D1 = series(A, 100_000, CFG[100_000], R, rng)
    log("100k done")
    E = pd.concat([E5, E1], ignore_index=True)
    E.to_csv(REPORTS / "bunching_event_monthly_agency.csv", index=False)
    L = [f"# Monthly difference-in-bunching, agency-clustered ({R} Poisson draws over agencies + shape error)", "",
         "## THB 500,000, months around the reform", "", "| month | Δz/T | SE | 95% CI |", "|---|---|---|---|"]
    for r in E5[(E5.ym >= "2015-06") & (E5.ym <= "2016-01")].itertuples():
        L.append(f"| {r.ym} | {r.dz_over_T:.3f} | {r.se_agency:.3f} | [{r.ci_lo:.3f}, {r.ci_hi:.3f}] |")
    jump = D5["2015-10"] - D5["2015-09"]
    pre = [m for m in D5 if m < "2015-09"]
    P = np.vstack([D5[m] for m in pre])
    L += ["", f"- October minus September 2015: {ci(jump)}",
          f"- pre-reform months (Oct 2014 – Aug 2015, n = {len(pre)}): range {P[:, 0].min():+.3f} to {P[:, 0].max():+.3f}, "
          f"SD {P[:, 0].std(ddof=1):.3f}; months whose 95% CI excludes zero: "
          f"{sum((np.nanpercentile(D5[m][1:], 2.5) > 0) or (np.nanpercentile(D5[m][1:], 97.5) < 0) for m in pre)}",
          "", "## September effect (THB 500,000): September minus mean of other months, by fiscal year", "",
          "| fiscal year | difference [95% CI] |", "|---|---|"]
    for fy in range(2018, 2026):
        sep = f"{fy}-09"
        oth = [m for m in D5 if (m >= f"{fy - 1}-10") and (m <= f"{fy}-08")]
        if sep in D5 and oth:
            L.append(f"| {fy} | {ci(D5[sep] - np.mean([D5[m] for m in oth], axis=0))} |")
    L += ["", "## THB 100,000 line, excess above its post-Act level: period means", "", "| period | mean Δz/T [95% CI] |", "|---|---|"]
    for name, a, b in (("pre (Oct 2014 – Sep 2015)", "2014-10", "2015-09"), ("transition (Oct 2015 – Jul 2017)", "2015-10", "2017-07"),
                       ("first Act year (Sep 2017 – Sep 2018)", "2017-09", "2018-09"), ("Act (Oct 2018 – Sep 2025)", "2018-10", "2025-09")):
        ms = [m for m in D1 if a <= m <= b]
        L.append(f"| {name} | {ci(np.nanmean([D1[m] for m in ms], axis=0))} |")
    out = REPORTS / "agency_event.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
