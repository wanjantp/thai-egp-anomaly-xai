#!/usr/bin/env python3
"""
Cost of discretion separately for central agencies and local governments (results review, GIQ v8).

After the Act, central agencies record the agreed price at or above the budget far more often than local
governments in the THB 0.1–0.5 million band (83% against 49%). If the fall in recorded discounts were only a
recording habit, it should appear mainly in central agencies. This script re-estimates the 2015-reform
difference-in-differences of revision_checks.did() by government level (agency prefix), with SEs clustered by
buying unit and, as a check, by agency.

Usage (Anaconda Prompt, env egp-xai; needs data/processed/_revision/did_int_2558–2561.parquet from
`python src\\revision_checks.py did`):  python src\\did_by_gov.py
Output: reports/did_by_gov.md (+ .csv).
"""
import sys

import pandas as pd

from bunching import REPORTS, _twfe, log
from revision_checks import OUT_DIR

LOCAL_PREFIX = ("องค์การบริหารส่วน", "เทศบาล")
LOCAL_EXACT = ("กรุงเทพมหานคร", "เมืองพัทยา")


def main():
    D = pd.concat([pd.read_parquet(OUT_DIR / f"did_int_{y}.parquet") for y in range(2558, 2562)], ignore_index=True)
    D = D[(D.ym >= "2014-10") & (D.ym <= "2017-08")].copy()
    D["post"] = (D.ym >= "2015-10").astype(float)
    D["agency"] = D.unit.str.split("|", n=1).str[0]
    D["gov"] = (D.agency.str.startswith(LOCAL_PREFIX) | D.agency.isin(LOCAL_EXACT)).map({True: "local", False: "central"})
    rows = []
    for gov, G in D.groupby("gov"):
        for ctrl in ("30k-100k", "2M-3M"):
            d = G[G.band.isin(["100k-500k", ctrl])].copy()
            d["treat"] = (d.band == "100k-500k").astype(float)
            d["tp"] = d.treat * d.post
            d["ub"] = d.unit + "|" + d.band
            out = {"gov": gov, "control": ctrl, "units": d.unit.nunique(), "contracts_treated": int(d.loc[d.treat == 1, "n"].sum())}
            for name, w, fn in (("first_stage", "n", lambda x: x.n_disc / x.n),
                                ("discount_vs_budget", "n_b", lambda x: x.sd_b / x.n_b),
                                ("share_at_or_above_budget", "n_b", lambda x: x.n_ge / x.n_b)):
                dd = d[d[w] > 0].copy()
                dd["yy"] = fn(dd)
                b, se = _twfe(dd, "yy", w, ["tp"], "ub", "ym", "unit")
                _, se_a = _twfe(dd, "yy", w, ["tp"], "ub", "ym", "agency")
                out[name], out[name + "_se"], out[name + "_se_agency"] = b[0], se[0], se_a[0]
            out["wald"] = out["discount_vs_budget"] / out["first_stage"]
            rows.append(out)
            log(f"{gov} vs {ctrl} done")
    R = pd.DataFrame(rows)
    R.to_csv(REPORTS / "did_by_gov.csv", index=False)
    L = ["# 2015-reform DiD by government level (treated THB 0.1–0.5M; Oct 2014 – Aug 2017; unit×band and month FE)", "",
         "SE clustered by buying unit; in brackets, by agency.", "",
         "| level | control | units | first stage | discount vs budget | per purchase moved | share at/above budget |",
         "|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        L.append(f"| {r.gov} | {r.control} | {r.units:,} | {r.first_stage:+.3f} ({r.first_stage_se:.3f}) | "
                 f"{r.discount_vs_budget:+.2%} ({r.discount_vs_budget_se:.2%}) [{r.discount_vs_budget_se_agency:.2%}] | {r.wald:+.2%} | "
                 f"{r.share_at_or_above_budget:+.3f} ({r.share_at_or_above_budget_se:.3f}) |")
    out = REPORTS / "did_by_gov.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
