"""Figures 1–5 of the GIQ manuscript (vector PDF + EPS, embedded fonts, no titles). Run from src/: python make_giq_figures.py
Figure 1 is computed from data/processed/bunching_bins.parquet, Figure 3 from reports/bunching_event_monthly_agency.csv;
Figures 2, 4 and 5 use the estimates reported in reports/bunching_cluster_boot_agency.csv, split_placebo_T and bunching_consolidation.md."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np, pandas as pd
from bunching import BINS, REPORTS, dib_estimate, _bins5k, sel
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9})
BLUE = "#1f5fbf"; GREY = "#8a8a8a"; LIGHT = "#d9d9d9"; RED = "#b2182b"
d = pd.read_parquet(str(BINS))
pr = d[d['var'] == 'price']
W, LO, HI, T = 5000, 100000, 2000000, 500000
ref = _bins5k(sel(pr, regime='pre'), W, LO, HI); grid = ref.index.values
yt = _bins5k(sel(pr, regime='act'), W, LO, HI).values
r = dib_estimate(yt, ref.values, grid, T, W, 200000, 400000, 50000, [1500000], 'const'); cf = r['_cf']
print('act dz/T', r['dz'] / T, 'excess', r['excess_B'])
m = grid <= 1_000_000
fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.bar(grid[m] / 1000 - 2.5, yt[m] / 1000, width=4.6, color="#4a7fd0", label="Contracts signed under the 2017 Act")
ax.plot(grid[m] / 1000 - 2.5, cf[m] / 1000, color=RED, lw=1.6, label="Counterfactual: fiscal 2015 distribution, scaled")
ax.axvspan(200, 400, color="#eeeeee", lw=0, zorder=0, label="Scaling band")
ax.axvspan(450, 500, color="#f4d3d7", lw=0, zorder=0, label="Bunching window")
ax.axvline(500, color="k", ls="--", lw=0.9)
ax.text(510, 280, "THB 500,000\nceiling", fontsize=8, va="top")
ax.set_xlim(100, 1000); ax.set_xlabel("Agreed contract value (thousand THB, 5,000 bins)")
ax.set_ylabel("Thousand contracts per bin")
ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.72))
fig.tight_layout(); fig.savefig("Figure_1.pdf"); fig.savefig("Figure_1.eps"); plt.close(fig)

_ea = REPORTS / 'bunching_event_monthly_agency.csv'                  # agency-clustered (agency_event.py)
E = pd.read_csv(str(_ea if _ea.exists() else REPORTS / 'bunching_event_monthly.csv'))
if "ci_lo" not in E:
    E["ci_lo"], E["ci_hi"] = E.dz_over_T - 1.96 * E.se, E.dz_over_T + 1.96 * E.se
fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True)
for ax, Tt, col, lab in ((axes[0], 500000, BLUE, "THB 500,000 (reference: fiscal 2015)"), (axes[1], 100000, GREY, "THB 100,000 (reference: post-Act months)")):
    e = E[E["T"] == Tt].copy(); e["t"] = pd.to_datetime(e["ym"])
    ax.fill_between(e["t"], e.ci_lo * 100, e.ci_hi * 100, color=("#c6d7f0" if col == BLUE else "#e3e3e3"), lw=0)
    ax.plot(e["t"], e.dz_over_T * 100, color=col, lw=1.2)
    s = e[e.ym.str.endswith("-09")]
    ax.plot(s["t"], s.dz_over_T * 100, "o", ms=3, color="k", label="September")
    ax.axhline(0, color="k", lw=0.6)
    for x, ls in (("2015-08-28", ":"), ("2015-10-01", "--"), ("2017-08-23", "--")):
        ax.axvline(pd.Timestamp(x), color="#555555", ls=ls, lw=0.9)
    ax.set_ylabel("Δz/T (%)"); ax.text(0.01, 0.96, lab, transform=ax.transAxes, fontsize=8.5, va="top", fontweight="bold")
    ax.grid(axis="y", color=LIGHT, lw=0.6); ax.set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
axes[1].xaxis.set_major_locator(mdates.YearLocator()); axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].set_xlabel("Month of contract signing")
fig.tight_layout(); fig.savefig("Figure_3.pdf"); fig.savefig("Figure_3.eps"); plt.close(fig)


# Figure 2: yearly bunching response with clustered 95% CI
fy = np.arange(2016, 2026)
cb = REPORTS / "bunching_cluster_boot_agency.csv"            # agency-clustered, sampling + shape error (agency_bootstrap.py)
cb = pd.read_csv(cb if cb.exists() else REPORTS / "bunching_cluster_boot.csv").set_index("label")
yr = [f"FY{y + 543}" for y in fy]
dz, lo, hi = (cb.loc[yr, c].values for c in ("dz_over_T", "ci95_lo_T", "ci95_hi_T"))
fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.errorbar(fy, dz*100, yerr=[(dz-lo)*100, (hi-dz)*100], fmt="o", color=BLUE, ecolor=BLUE, capsize=3, ms=5, lw=1.2)
ax.axvline(2017.5, color=GREY, ls="--", lw=0.9)
ax.text(2017.58, 49, "2017 Act in force (23 Aug 2017, last weeks of FY2017)", color="#555555", fontsize=8, va="top")
ax.annotate(f"{dz[0]*100:.0f}%", (fy[0], dz[0]*100), xytext=(8, -3), textcoords="offset points", fontsize=8.5, fontweight="bold")
ax.annotate(f"{dz[-1]*100:.0f}%", (fy[-1], dz[-1]*100), xytext=(-26, -3), textcoords="offset points", fontsize=8.5, fontweight="bold")
ax.set_ylim(0, 50); ax.set_yticks(range(0, 51, 10)); ax.set_yticklabels([f"{t}%" for t in range(0, 51, 10)])
ax.set_xticks(fy); ax.set_xlabel("Fiscal year (FY2016 = Oct 2015 – Sep 2016)")
ax.set_ylabel("Bunching response Δz/T")
ax.grid(axis="y", color=LIGHT, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("Figure_2.pdf"); fig.savefig("Figure_2.eps"); plt.close(fig)

# Figure 4: split shares at four thresholds by regime
panels = [("THB 100k", "ceiling removed 2015", [32.9, 15.1, 11.6]),
          ("THB 500k", "new ceiling 2015", [11.9, 27.0, 42.1]),
          ("THB 2M", "ceiling removed 2017", [16.8, 21.8, 11.0]),
          ("THB 1M", "never a ceiling (placebo)", [2.5, 5.7, 2.4])]
pers = ["Pre", "Trans.", "Act"]; shades = ["#c6d7f0", "#7fa6dc", BLUE]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 3.0), sharey=True)
for ax, (name, role, vals) in zip(axes, panels):
    bars = ax.bar(range(3), vals, color=shades, width=0.7, edgecolor="none")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=7.5, fontweight="bold" if i == 2 else "normal")
    ax.set_xticks(range(3)); ax.set_xticklabels(pers, fontsize=8)
    ax.set_title(f"{name}\n{role}", fontsize=8.5, loc="left")
    ax.set_ylim(0, 48); ax.spines["left"].set_visible(ax is axes[0])
    if ax is not axes[0]: ax.tick_params(left=False)
axes[0].set_ylabel("Share of purchases split (%)")
fig.tight_layout(); fig.savefig("Figure_4.pdf"); fig.savefig("Figure_4.eps"); plt.close(fig)

# Figure 5: excess split clusters vs missing contracts
missing = [9480,15178,30183,36135,39942,47816,46326,50198,52408,57797]
month = [13006,9917,22326,29735,42570,43948,41379,51423,54235,49655]
day = [13168,8330,16582,23309,33024,34347,31438,40556,43851,41836]
fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.plot(fy, np.array(missing)/1000, color=GREY, lw=2.2, marker="o", ms=3.5, label="Missing THB 0.5–2M contracts")
ax.plot(fy, np.array(month)/1000, color=BLUE, lw=1.6, ls="--", marker="s", ms=3.5, label="Excess split purchases, same month")
ax.plot(fy, np.array(day)/1000, color=BLUE, lw=1.8, marker="o", ms=3.5, label="Excess split purchases, same day")
ax.axvline(2017.5, color=GREY, ls="--", lw=0.9)
ax.text(2017.58, 63, "2017 Act in force (23 Aug 2017, last weeks of FY2017)", color="#555555", fontsize=8, va="top")
ax.set_ylim(0, 65); ax.set_xticks(fy); ax.set_xlabel("Fiscal year")
ax.set_ylabel("Thousand contracts or purchases per year")
ax.grid(axis="y", color=LIGHT, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("Figure_5.pdf"); fig.savefig("Figure_5.eps"); plt.close(fig)
print("done")
