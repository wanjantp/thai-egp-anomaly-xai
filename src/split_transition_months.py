#!/usr/bin/env python3
"""
Monthly same-day split shares at the four thresholds of Figure 4, and the transition-period shares with the
October–November 2015 surge excluded (cross-reference audit, GIQ v20; supports the THB 1 million placebo text in 5.4).

Reads data/processed/_split_placebo_T (bunching.split_placebo_T_aggregate).
Usage (Anaconda Prompt, env egp-xai):  python src\\split_transition_months.py
Output: reports/split_transition_months.md
"""
import sys

import pandas as pd

from bunching import PLACEBO_T_DIR, REPORTS, log

K = {"100": "THB 100,000", "500": "THB 500,000", "2m": "THB 2 million", "1m": "THB 1 million (placebo)"}


def main():
    D = pd.concat([pd.read_parquet(f) for f in sorted(PLACEBO_T_DIR.glob("*.parquet"))], ignore_index=True)
    g = D.groupby("ym").sum(numeric_only=True)
    per = pd.Series(["pre" if m < "2015-10" else "transition" if m < "2017-09" else "act" for m in g.index], index=g.index)
    T = g[per == "transition"]
    variants = {"transition, all months": T,
                "transition without Oct–Nov 2015": T.drop(index=["2015-10", "2015-11"]),
                "transition without Oct 2015 – Mar 2016": T.drop(index=[m for m in T.index if "2015-10" <= m <= "2016-03"])}
    L = ["# Same-day split shares: transition surges", "",
         "| threshold | pre | " + " | ".join(variants) + " | act |", "|---" * (len(variants) + 3) + "|"]
    for k, name in K.items():
        sh = lambda x: x[f"s_{k}"].sum() / x[f"n_{k}"].sum()
        L.append(f"| {name} | {sh(g[per == 'pre']):.1%} | " + " | ".join(f"{sh(v):.1%}" for v in variants.values())
                 + f" | {sh(g[per == 'act']):.1%} |")
    L += ["", "## Monthly shares, June 2015 – September 2016", "", "| month | " + " | ".join(K.values()) + " |",
          "|---" * (len(K) + 1) + "|"]
    for m in g.loc["2015-06":"2016-09"].index:
        L.append(f"| {m} | " + " | ".join(f"{g.loc[m, f's_{k}'] / g.loc[m, f'n_{k}']:.1%}" for k in K) + " |")
    G = D[D.ym.isin(["2015-10", "2015-11"])].groupby(["ym", "gov"])[["n_1m", "s_1m", "n_2m", "s_2m"]].sum()
    L += ["", "## October–November 2015 by government level", "", "| month | level | THB 1M: split / purchases | THB 2M: split / purchases |", "|---|---|---|---|"]
    for (m, gov), r in G.iterrows():
        L.append(f"| {m} | {gov} | {r.s_1m:,} / {r.n_1m:,} | {r.s_2m:,} / {r.n_2m:,} |")
    out = REPORTS / "split_transition_months.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
