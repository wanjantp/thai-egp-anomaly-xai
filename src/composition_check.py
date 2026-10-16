#!/usr/bin/env python3
"""
Does the rise in bunching over time come from buyers or from the mix of buyers? (methodology audit M1, GIQ v6)

The number of local buying units in the published data grows after fiscal 2017, and local governments bunch more
than central agencies. This script re-estimates the yearly difference-in-bunching response (Δz/T at THB 500,000,
reference = contracts signed before 15 September 2015, same estimator as the manuscript) separately for
  all units, central agencies, local governments,
  the balanced panel of buying units present in every fiscal year (from bunching_unit_bands.parquet),
and the balanced panel split into central and local. Each group uses its own reference distribution. A last pair of
rows holds the central/local mix of contracts at its fiscal 2016 level, which isolates the part of the trend due to mix.
Agency-clustered Poisson bootstrap (whole agencies resampled, weights shared across years; cluster="unit" resamples
buying units instead) gives 95% intervals and the
change from fiscal 2016 to fiscal 2025.

Usage (Anaconda Prompt, env egp-xai):  python src\\composition_check.py
Input: data/processed/_unit_bins5k/*.parquet (from `bunching.py unit_bins_aggregate`), bunching_unit_bands.parquet.
Output: reports/composition_check.md (+ .csv).
"""
import sys

import numpy as np
import pandas as pd
from scipy import sparse

from bunching import REPORTS, UNIT_BANDS, UNIT_BINS_DIR, dib_estimate, log

W, LO, HI, T = 5000, 100000, 2000000, 500000
LOCAL_PREFIX = ("องค์การบริหารส่วน", "เทศบาล")
LOCAL_EXACT = ("กรุงเทพมหานคร", "เมืองพัทยา")
YEARS = list(range(2559, 2569))


def is_local(agency):
    return agency.startswith(LOCAL_PREFIX) or agency in LOCAL_EXACT


def main(R=500, seed=2026, cluster="agency"):
    U = pd.concat([pd.read_parquet(f) for f in sorted(UNIT_BINS_DIR.glob("*.parquet"))], ignore_index=True)
    grid = np.arange(LO + W, HI + W, W)
    gi = {g: i for i, g in enumerate(grid)}
    U = U[U["bin"].isin(gi)].copy()
    U["local"] = U["unit"].str.split("|", n=1).str[0].map(is_local)
    B = pd.read_parquet(UNIT_BANDS, columns=["fiscal_year", "agency", "sub_agency"])
    B["unit"] = B["agency"] + "|" + B["sub_agency"].fillna("")
    pres = B.groupby("unit")["fiscal_year"].nunique()
    balanced = set(pres[pres == pres.max()].index)
    U["balanced"] = U["unit"].isin(balanced)
    log(f"units {U.unit.nunique():,}; balanced {len(balanced):,} (present in {pres.max()} fiscal years)")

    groups = {"all": np.ones(len(U), bool), "central": ~U.local.values, "local": U.local.values,
              "balanced": U.balanced.values, "balanced central": (U.balanced & ~U.local).values,
              "balanced local": (U.balanced & U.local).values}
    uid, units = pd.factorize(U["unit"].str.split("|", n=1).str[0] if cluster == "agency" else U["unit"])
    gid = U["bin"].map(gi).values
    nU, nG = len(units), len(grid)
    rng = np.random.default_rng(seed)
    Wt = np.vstack([np.ones(nU), rng.poisson(1.0, size=(R, nU))])          # row 0 = point estimate
    est = lambda yt, yr: dib_estimate(yt, yr, grid, T, W, 200000, 400000, 50000, [1500000], "const")["dz"] / T
    Ws = sparse.csr_matrix(Wt)
    is_ref = (U.regime == "pre").values
    in_year = {y: ((U.fiscal_year == y) & (U.regime != "pre")).values for y in YEARS}

    def agg(mask):                                    # (R+1) × bins, contracts weighted by each draw
        return np.asarray((Ws @ sparse.csr_matrix((U.n.values[mask].astype(float), (uid[mask], gid[mask])),
                                                  shape=(nU, nG))).todense())

    rows = []

    def add(g, y, post, ref):
        vals = np.array([est(post[i], ref[i]) for i in range(Wt.shape[0])])
        rows.append({"group": g, "fiscal_year": y - 543, "dz_T": vals[0], "lo": np.percentile(vals[1:], 2.5),
                     "hi": np.percentile(vals[1:], 97.5), "draws": vals[1:], "contracts": float(post[0].sum())})

    posts = {}
    for g, m in groups.items():
        ref = agg(m & is_ref)
        for y in YEARS:
            posts[g, y] = agg(m & in_year[y])
            add(g, y, posts[g, y], ref)
        log(f"{g} done")
    # mix held at fiscal 2016: reweight central and local contracts in each year to their fiscal 2016 shares
    for base, c, l in (("all", "central", "local"), ("balanced", "balanced central", "balanced local")):
        ref = agg(groups[base] & is_ref)
        sh16 = posts[l, 2559].sum(1) / (posts[c, 2559].sum(1) + posts[l, 2559].sum(1))
        for y in YEARS:
            pc, pl = posts[c, y], posts[l, y]
            n = pc.sum(1) + pl.sum(1)
            post = pc * ((1 - sh16) * n / pc.sum(1))[:, None] + pl * (sh16 * n / pl.sum(1))[:, None]
            add(f"{base}, 2016 mix", y, post, ref)
        groups[f"{base}, 2016 mix"] = None
        log(f"{base}, 2016 mix done")
    R_ = pd.DataFrame(rows)
    R_.drop(columns="draws").to_csv(REPORTS / "composition_check.csv", index=False)

    L = ["# Bunching response by year: buyers or the mix of buyers? (Δz/T at THB 500,000)", "",
         "Each group uses its own pre-reform reference. 95% intervals from a Poisson bootstrap over "
         f"{'agencies' if cluster == 'agency' else 'buying units'} ({R} draws, weights shared across years).", "",
         "| fiscal year | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
    for y in YEARS:
        cells = []
        for g in groups:
            r = R_[(R_.group == g) & (R_.fiscal_year == y - 543)].iloc[0]
            cells.append(f"{r.dz_T:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
        L.append(f"| {y - 543} | " + " | ".join(cells) + " |")
    L += ["", "## Change from fiscal 2016 to fiscal 2025", "", "| group | 2016 | 2025 | change | 95% CI of change | share local, 2016 → 2025 |",
          "|---|---|---|---|---|---|"]
    for g in groups:
        a = R_[(R_.group == g) & (R_.fiscal_year == 2016)].iloc[0]
        b = R_[(R_.group == g) & (R_.fiscal_year == 2025)].iloc[0]
        d = b.draws - a.draws
        share = ""
        if g in ("all", "balanced", "all, 2016 mix", "balanced, 2016 mix"):
            loc = "local" if g.startswith("all") else "balanced local"
            if g.endswith("mix"):
                L.append(f"| {g} | {a.dz_T:.3f} | {b.dz_T:.3f} | {b.dz_T - a.dz_T:+.3f} | "
                         f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] | held at 2016 |")
                continue
            s16 = R_[(R_.group == loc) & (R_.fiscal_year == 2016)].contracts.iloc[0] / a.contracts
            s25 = R_[(R_.group == loc) & (R_.fiscal_year == 2025)].contracts.iloc[0] / b.contracts
            share = f"{s16:.0%} → {s25:.0%}"
        L.append(f"| {g} | {a.dz_T:.3f} | {b.dz_T:.3f} | {b.dz_T - a.dz_T:+.3f} | "
                 f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] | {share} |")
    L += ["", "\"2016 mix\": each year's central and local contracts reweighted to their fiscal 2016 shares (all units: "
          "local share of contracts THB 0.1–2 million), estimated against the group's own reference. "
          "The gap between a group and its 2016-mix row is the part of the trend due to the shift toward local buyers."]
    out = REPORTS / "composition_check.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
