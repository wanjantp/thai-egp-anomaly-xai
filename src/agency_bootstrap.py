#!/usr/bin/env python3
"""
Agency-clustered inference for the GIQ manuscript (statistical review, v7).

Earlier intervals resampled buying units (agency × sub-unit). Sub-units of one agency share rules, budgets and
managers, so this script resamples whole agencies (the `agency` field: a central department, a local government,
a state enterprise, ...) with Poisson(1) weights shared across periods, and redoes
  [A] difference-in-bunching at THB 500,000 (data/processed/_unit_bins5k): transition, post-Act, each fiscal year,
      central and local agencies (each against its own reference) and their difference;
      intervals combine sampling and shape error: each bootstrap draw gets a normal deviate with the SD of the
      placebo estimates at fake thresholds inside the scaling band (reports/bunching_dib.csv, placebo_dz_sd);
  [B] splitting measures (data/processed/_revision/boot_YYYY.parquet, from `revision_checks.py boot`): split shares,
      excess split clusters, missing contracts (anchor THB 2–3 million), explained shares, share of flags above the
      pre-reform rate.

Usage (Anaconda Prompt, env egp-xai):  python src\\agency_bootstrap.py
Output: reports/agency_bootstrap.md, reports/bunching_cluster_boot_agency.csv (same columns as
bunching_cluster_boot.csv, read by make_giq_figures.py for Figure 2), reports/revision_boot_agency.csv.
"""
import sys

import numpy as np
import pandas as pd
from scipy import sparse

from bunching import PROCESSED, REPORTS, UNIT_BINS_DIR, dib_estimate, log

W, LO, HI, T = 5000, 100000, 2000000, 500000
LOCAL_PREFIX = ("องค์การบริหารส่วน", "เทศบาล")
LOCAL_EXACT = ("กรุงเทพมหานคร", "เมืองพัทยา")


def agency_of(unit):
    return unit.str.split("|", n=1).str[0]


def is_local(agency):
    return agency.str.startswith(LOCAL_PREFIX) | agency.isin(LOCAL_EXACT)


def bunching_part(R, seed):
    U = pd.concat([pd.read_parquet(f) for f in sorted(UNIT_BINS_DIR.glob("*.parquet"))], ignore_index=True)
    grid = np.arange(LO + W, HI + W, W)
    gi = {g: i for i, g in enumerate(grid)}
    U = U[U["bin"].isin(gi)].copy()
    U["agency"] = agency_of(U["unit"])
    loc = is_local(U["agency"]).values
    cid, ags = pd.factorize(U["agency"])
    gid = U["bin"].map(gi).values
    nA, nG = len(ags), len(grid)
    rng = np.random.default_rng(seed)
    Ws = sparse.csr_matrix(np.vstack([np.ones(nA), rng.poisson(1.0, size=(R, nA))]))

    def agg(m):
        return np.asarray((Ws @ sparse.csr_matrix((U.n.values[m].astype(float), (cid[m], gid[m])), shape=(nA, nG))).todense())

    def est(post, ref):
        o = [dib_estimate(post[i], ref[i], grid, T, W, 200000, 400000, 50000, [1500000], "const") for i in range(post.shape[0])]
        return np.array([x["dz"] for x in o]), np.array([x["excess_B"] for x in o])

    pre = (U.regime == "pre").values
    periods = {"transition (FY2559–60)": (U.regime == "transition").values, "act (FY2560–68)": (U.regime == "act").values}
    for y in range(2559, 2569):
        periods[f"FY{y}"] = ((U.fiscal_year == y) & (U.regime != "pre")).values
    old = pd.read_csv(REPORTS / "bunching_dib.csv").set_index("label")
    psd_act = old.loc["act (FY2560–68)", "placebo_dz_sd"]
    ref = agg(pre)
    rows, D = [], {}
    for k, m in periods.items():
        dz, ex = est(agg(m), ref)
        psd = old.loc[k, "placebo_dz_sd"] if k in old.index else psd_act
        tot = dz[1:] + rng.normal(0, psd, R)
        D[k] = dz
        rows.append({"label": k, "dz": dz[0], "dz_over_T": dz[0] / T, "se_cluster": dz[1:].std(ddof=1), "placebo_sd": psd,
                     "se_total_cluster": tot.std(ddof=1),
                     "ci95_lo_T": np.percentile(tot, 2.5) / T, "ci95_hi_T": np.percentile(tot, 97.5) / T,
                     "ci95_lo_T_sampling": np.percentile(dz[1:], 2.5) / T, "ci95_hi_T_sampling": np.percentile(dz[1:], 97.5) / T,
                     "excess_B": ex[0], "excess_se_cluster": ex[1:].std(ddof=1)})
        log(f"[A] {k}")
    G = {}
    for g, gm in (("central", ~loc), ("local", loc)):
        dz, _ = est(agg(gm & periods["act (FY2560–68)"]), agg(gm & pre))
        G[g] = dz
    act = pd.DataFrame(rows)
    act.to_csv(REPORTS / "bunching_cluster_boot_agency.csv", index=False)
    d = (G["local"] - G["central"]) / T
    trend = (D["FY2568"] - D["FY2559"]) / T
    extra = {"central_act": (G["central"][0] / T, *np.percentile(G["central"][1:] / T, [2.5, 97.5])),
             "local_act": (G["local"][0] / T, *np.percentile(G["local"][1:] / T, [2.5, 97.5])),
             "local_minus_central": (d[0], *np.percentile(d[1:], [2.5, 97.5])),
             "change_2016_2025": (trend[0], *np.percentile(trend[1:], [2.5, 97.5]))}
    return act, extra, nA


def splitting_part(R, seed):
    B = pd.concat([pd.read_parquet(f) for f in sorted((PROCESSED / "_revision").glob("boot_25??.parquet"))],
                  ignore_index=True).fillna(0)
    B["agency"] = agency_of(B["unit"])
    cid, ags = pd.factorize(B["agency"])
    B["cid"] = cid
    nA = len(ags)
    act = (B.regime == "act") & B.fy.between(2561, 2568)
    pre = B.regime == "pre"

    def vec(mask, col):
        s = B.loc[mask].groupby("cid")[col].sum()
        v = np.zeros(nA); v[s.index.values] = s.values
        return v

    V = {}
    for lev in ("day", "month"):
        m = B.level == lev
        for per, pm in (("pre", pre), ("act", act)):
            for c in ("cl500", "sp500", "cl100", "sp100"):
                V[f"{per}_{c}_{lev}"] = vec(m & pm, c)
    mc = B.level == "contracts"
    for per, pm in (("pre", pre), ("act", act)):
        for c in ("n0520", "n23"):
            V[f"{per}_{c}"] = vec(mc & pm, c)
    M = np.vstack(list(V.values()))
    keys = list(V)
    rng = np.random.default_rng(seed + 1)
    Wt = np.vstack([np.ones(nA), rng.poisson(1.0, size=(R, nA))])
    S = pd.DataFrame(Wt @ M.T, columns=keys)
    o = pd.DataFrame(index=S.index)
    for lev in ("day", "month"):
        r_pre = S[f"pre_sp500_{lev}"] / S[f"pre_cl500_{lev}"]
        o[f"split500_pre_{lev}"] = r_pre
        o[f"split500_act_{lev}"] = S[f"act_sp500_{lev}"] / S[f"act_cl500_{lev}"]
        o[f"split100_pre_{lev}"] = S[f"pre_sp100_{lev}"] / S[f"pre_cl100_{lev}"]
        o[f"split100_act_{lev}"] = S[f"act_sp100_{lev}"] / S[f"act_cl100_{lev}"]
        o[f"excess_{lev}"] = S[f"act_sp500_{lev}"] - r_pre * S[f"act_cl500_{lev}"]
        o[f"alert_excess_share_{lev}"] = o[f"excess_{lev}"] / S[f"act_sp500_{lev}"]
    o["missing"] = S["pre_n0520"] / S["pre_n23"] * S["act_n23"] - S["act_n0520"]
    for lev in ("day", "month"):
        o[f"explained_{lev}"] = o[f"excess_{lev}"] / o["missing"]
    T_ = pd.DataFrame({"stat": o.columns, "estimate": o.iloc[0].values, "se": o.iloc[1:].std(ddof=1).values,
                       "ci_lo": o.iloc[1:].quantile(.025).values, "ci_hi": o.iloc[1:].quantile(.975).values})
    T_.to_csv(REPORTS / "revision_boot_agency.csv", index=False)
    log("[B] splitting done")
    return T_, nA


def main(R=500, seed=2026):
    A, X, nA = bunching_part(R, seed)
    S, nB = splitting_part(R, seed)
    old = pd.read_csv(REPORTS / "bunching_cluster_boot.csv").set_index("label")
    oldS = pd.read_csv(REPORTS / "revision_boot.csv").set_index("stat")
    L = [f"# Agency-clustered inference ({R} Poisson draws over agencies)", "",
         f"[A] difference-in-bunching at THB 500,000: {nA:,} agencies. Total interval = sampling (agency bootstrap) + shape "
         "error (normal deviate with the SD of placebo estimates at fake thresholds). Unit-clustered interval (v6) for comparison.", "",
         "| period | Δz/T | 95% CI, agency + shape | 95% CI, agency only | 95% CI, unit-clustered (v6) | excess contracts (SE) |",
         "|---|---|---|---|---|---|"]
    for r in A.itertuples():
        o = old.loc[r.label]
        L.append(f"| {r.label} | {r.dz_over_T:.3f} | [{r.ci95_lo_T:.3f}, {r.ci95_hi_T:.3f}] | "
                 f"[{r.ci95_lo_T_sampling:.3f}, {r.ci95_hi_T_sampling:.3f}] | [{o.ci95_lo_T:.3f}, {o.ci95_hi_T:.3f}] | "
                 f"{r.excess_B:,.0f} ({r.excess_se_cluster:,.0f}) |")
    L += ["", "| quantity (post-Act unless stated; sampling only) | estimate | 95% CI |", "|---|---|---|"]
    for k, (e, lo, hi) in X.items():
        L.append(f"| {k} | {e:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    L += ["", f"[B] splitting measures: {nB:,} agencies (unit-clustered v6 interval for comparison)", "",
          "| statistic | estimate | SE | 95% CI | 95% CI, unit-clustered (v6) |", "|---|---|---|---|---|"]
    for r in S.itertuples():
        big = abs(r.estimate) > 10
        f = (lambda x: f"{x:,.0f}") if big else (lambda x: f"{x:.3f}")
        o = oldS.loc[r.stat] if r.stat in oldS.index else None
        L.append(f"| {r.stat} | {f(r.estimate)} | {f(r.se)} | [{f(r.ci_lo)}, {f(r.ci_hi)}] | "
                 + (f"[{f(o.ci_lo)}, {f(o.ci_hi)}] |" if o is not None else "|"))
    out = REPORTS / "agency_bootstrap.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
