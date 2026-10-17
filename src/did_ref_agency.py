#!/usr/bin/env python3
"""
Agency-clustered standard errors for every column of the 2015-reform discount DiD (Table 4; production audit, GIQ v18).

Re-runs the pre + transition specification of bunching.did_discount() (unit × band and month FE, cells weighted by
contracts, October 2014 – August 2017) and reports SEs clustered by buying unit and by agency for the first stage,
the discount against the reference price and the discount against the budget.

Usage (Anaconda Prompt, env egp-xai):  python src\\did_ref_agency.py
Output: reports/did_ref_agency.md
"""
import sys

import numpy as np
import pandas as pd

from bunching import DID_DIR, REPORTS, _twfe, log


def main():
    D = pd.concat([pd.read_parquet(f, columns=["unit", "ym", "band", "n", "n_disc", "n_r", "sd_r", "n_b", "sd_b"])
                   for f in sorted(DID_DIR.glob("*.parquet"))], ignore_index=True)
    D = D[(D.ym >= "2014-10") & (D.ym <= "2017-08")]
    D = D.groupby(["unit", "ym", "band"], as_index=False).sum()
    D["post"] = (D.ym >= "2015-10").astype(float)
    D["agency"] = D.unit.str.split("|", n=1).str[0]
    L = ["# Table 4 with agency-clustered SEs (Oct 2014 – Aug 2017; unit×band and month FE; weights = contracts)", "",
         "| control | outcome | estimate | SE unit | SE agency |", "|---|---|---|---|---|"]
    for ctrl in ("30k-100k", "2M-3M"):
        d = D[D.band.isin(["100k-500k", ctrl])].copy()
        d["treat"] = (d.band == "100k-500k").astype(float)
        d["tp"] = d.treat * d.post
        d["ub"] = d.unit + "|" + d.band
        for name, w, num in (("first stage", "n", "n_disc"), ("discount vs reference price", "n_r", "sd_r"),
                             ("discount vs budget", "n_b", "sd_b")):
            dd = d[d[w] > 0].copy()
            dd["yy"] = dd[num] / dd[w]
            b, se_u = _twfe(dd, "yy", w, ["tp"], "ub", "ym", "unit")
            _, se_a = _twfe(dd, "yy", w, ["tp"], "ub", "ym", "agency")
            scale = 1 if name == "first stage" else 100
            L.append(f"| {ctrl} | {name} | {b[0] * scale:+.3f} | {se_u[0] * scale:.3f} | {se_a[0] * scale:.3f} |")
            log(f"{ctrl} {name}: {b[0] * scale:+.3f} ({se_u[0] * scale:.3f}) [{se_a[0] * scale:.3f}]")
    L.append("\nDiscount rows in percentage points.")
    out = REPORTS / "did_ref_agency.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
