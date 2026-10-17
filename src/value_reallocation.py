#!/usr/bin/env python3
"""
Value reallocation across price bands in the balanced panel, central and local (results review, GIQ v8).

Balanced panel = buying units with contracts in every fiscal year 2015–2025 (bunching_unit_bands.parquet).
For each government level, each band's value in fiscal 2018–2025 is compared with the value it would have if the
band kept its fiscal 2015 share of the level's total; the table reports the yearly mean difference (THB billion).

Usage (Anaconda Prompt, env egp-xai):  python src\\value_reallocation.py
Output: reports/value_reallocation.md
"""
import sys

import numpy as np
import pandas as pd

from bunching import REPORTS, UNIT_BANDS, log

BANDS = {30000: "≤30k", 100000: "30–100k", 200000: "0.1–0.2M", 400000: "0.2–0.4M", 450000: "0.4–0.45M",
         500000: "0.45–0.5M", 1000000: "0.5–1M", 2000000: "1–2M", 3000000: "2–3M"}
GROUPS = {"below THB 100,000": (30000, 100000), "THB 0.1–0.5 million": (200000, 400000, 450000, 500000),
          "THB 0.5–2 million": (1000000, 2000000), "THB 2–3 million": (3000000,)}


def main():
    B = pd.read_parquet(UNIT_BANDS)
    B["unit"] = B.agency + "|" + B.sub_agency.fillna("")
    pres = B.groupby("unit").fiscal_year.nunique()
    b = B[B.unit.isin(pres[pres == pres.max()].index)]
    L = ["# Value reallocation in the balanced panel (THB billion a year, fiscal 2018–2025, against fiscal 2015 band shares)", "",
         "| level | " + " | ".join(GROUPS) + " |", "|---" * (len(GROUPS) + 1) + "|"]
    for gov, g in b.groupby("gov"):
        V = g.groupby(["fiscal_year", "band_hi"]).v.sum().unstack().fillna(0) / 1e9
        share15 = V.loc[2558] / V.loc[2558].sum()
        post = V.loc[2561:2568]
        diff = (post - np.outer(post.sum(axis=1), share15)).mean()
        L.append(f"| {gov} | " + " | ".join(f"{diff[list(c)].sum():+.1f}" for c in GROUPS.values()) + " |")
    out = REPORTS / "value_reallocation.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
