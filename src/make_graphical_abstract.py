#!/usr/bin/env python3
"""
Graphical abstract of the GIQ manuscript (Elsevier: at least 1328 x 531 px, readable at 13 x 5 cm).

Panel 1: yearly bunching response Δz/T at THB 500,000 (reports/worktype_mix.csv, group "all": the manuscript's
         estimator, agency-clustered 95% intervals).
Panel 2: share of THB 0.5–2 million (and 0.1–0.5 million) purchases executed as several sub-threshold contracts,
         pre-reform / transition / post-Act (manuscript Section 5.4 and Figure 4).
Panel 3: headline numbers from Sections 5.3, 5.5 and 5.6.
Run from src/:  python make_graphical_abstract.py   ->  figs/Graphical_abstract.{pdf,tif,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figs"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, GREY, RED = "#1f4e79", "#9a9a9a", "#b2182b"

# Section 5.4 / Figure 4: same-day split shares (pre-reform, transition, post-Act)
SPLIT = {"THB 500,000\n(new)": (12, 27, 42), "THB 100,000\n(abolished)": (33, 15, 12)}


def main():
    R = pd.read_csv(ROOT / "reports" / "worktype_mix.csv")
    R = R[R.group == "all"].sort_values("fiscal_year")
    fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], left=0.075, right=0.97, bottom=0.2, top=0.86, wspace=0.42)

    ax = fig.add_subplot(gs[0])
    ax.fill_between(R.fiscal_year, 100 * R.lo, 100 * R.hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(R.fiscal_year, 100 * R.dz_T, color=BLUE, lw=1.4, marker="o", ms=2.2)
    ax.set_ylim(0, 45)
    ax.set_xticks([2016, 2019, 2022, 2025])
    ax.set_ylabel("Excess mass (% of threshold)", fontsize=6.2)
    ax.set_title("1. Bunching below THB 500,000", fontsize=7, loc="left", fontweight="bold", x=-0.2)
    for y in (2016, 2025):
        v = 100 * R.loc[R.fiscal_year == y, "dz_T"].iloc[0]
        ax.annotate(f"{v:.0f}%", (y, v), textcoords="offset points", xytext=(6, -9) if y == 2016 else (0, 5),
                    ha="center", fontsize=6.5, color=BLUE)
    ax.text(0.98, 0.03, "Thailand, 45 million\ne-GP contracts", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.6, color="#444444")

    ax = fig.add_subplot(gs[1])
    w = 0.26
    for i, (lab, vals) in enumerate(SPLIT.items()):
        for j, (v, c) in enumerate(zip(vals, ("#c9c9c9", GREY, BLUE if i == 0 else RED))):
            ax.bar(i + (j - 1) * w, v, w * 0.92, color=c)
            ax.text(i + (j - 1) * w, v + 1, f"{v}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks([0, 1], list(SPLIT), fontsize=6.2)
    ax.set_ylim(0, 60)
    ax.set_ylabel("Purchases split (%)", fontsize=6.2)
    ax.set_title("2. Splitting follows the threshold", fontsize=7, loc="left", fontweight="bold", x=-0.2)
    ax.text(0.5, 0.99, "bars: before reform, transition,\nafter the 2017 Act", transform=ax.transAxes, ha="center", va="top",
            fontsize=5.6, color="#444444")
    for a in fig.axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(width=0.6, length=2.5, labelsize=6.2)

    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    ax.set_title("3. What it means", fontsize=7, loc="left", fontweight="bold")
    items = [("73%", "of contracts missing above the\nthreshold matched by same-day splits"),
             ("−6 to −9 pp", "recorded discount when a purchase\nleaves competition for discretion"),
             ("≈46,000", "purchases a year a check at\ncontract entry would flag")]
    for k, (big, small) in enumerate(items):
        y = 0.88 - 0.36 * k
        ax.text(0.0, y, big, fontsize=9.5, fontweight="bold", color=BLUE if k != 1 else RED, va="center",
                transform=ax.transAxes)
        ax.text(0.0, y - 0.155, small, fontsize=5.9, va="center", transform=ax.transAxes, linespacing=1.15)
    fig.text(0.5, 0.015, "Published records made threshold circumvention measurable, but did not prevent it.",
             ha="center", va="bottom", fontsize=6.5, style="italic")

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "Graphical_abstract.pdf")
    fig.savefig(OUT / "Graphical_abstract.png", dpi=600)
    fig.savefig(OUT / "Graphical_abstract.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    print("→", OUT / "Graphical_abstract.{pdf,png,tif}")


if __name__ == "__main__":
    main()
