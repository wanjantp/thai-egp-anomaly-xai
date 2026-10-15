import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9})
BLUE = "#1f5fbf"; GREY = "#8a8a8a"; LIGHT = "#d9d9d9"

# Figure 1: yearly bunching response with clustered 95% CI
fy = np.arange(2016, 2026)
dz = np.array([0.160,0.186,0.187,0.233,0.251,0.239,0.241,0.252,0.283,0.304])
lo = np.array([0.118,0.142,0.143,0.181,0.200,0.190,0.190,0.201,0.223,0.234])
hi = np.array([0.201,0.233,0.228,0.278,0.298,0.285,0.287,0.298,0.338,0.365])
fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.errorbar(fy, dz*100, yerr=[(dz-lo)*100, (hi-dz)*100], fmt="o", color=BLUE, ecolor=BLUE, capsize=3, ms=5, lw=1.2)
ax.axvline(2017.5, color=GREY, ls="--", lw=0.9)
ax.text(2017.58, 39, "2017 Act in force (23 Aug 2017)", color="#555555", fontsize=8, va="top")
ax.annotate(f"{dz[0]*100:.0f}%", (fy[0], dz[0]*100), xytext=(8, -3), textcoords="offset points", fontsize=8.5, fontweight="bold")
ax.annotate(f"{dz[-1]*100:.0f}%", (fy[-1], dz[-1]*100), xytext=(-26, -3), textcoords="offset points", fontsize=8.5, fontweight="bold")
ax.set_ylim(0, 40); ax.set_yticks(range(0, 41, 10)); ax.set_yticklabels([f"{t}%" for t in range(0, 41, 10)])
ax.set_xticks(fy); ax.set_xlabel("Fiscal year (FY2016 = Oct 2015 – Sep 2016)")
ax.set_ylabel("Bunching response Δz/T")
ax.grid(axis="y", color=LIGHT, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("Figure_1.pdf"); fig.savefig("Figure_1.eps"); plt.close(fig)

# Figure 2: split shares at four thresholds by regime
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
fig.tight_layout(); fig.savefig("Figure_2.pdf"); fig.savefig("Figure_2.eps"); plt.close(fig)

# Figure 3: excess split clusters vs missing contracts
missing = [9480,15178,30183,36135,39942,47816,46326,50198,52408,57797]
month = [13006,9917,22326,29735,42570,43948,41379,51423,54235,49655]
day = [13168,8330,16582,23309,33024,34347,31438,40556,43851,41836]
fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.plot(fy, np.array(missing)/1000, color=GREY, lw=2.2, marker="o", ms=3.5, label="Missing THB 0.5–2M contracts")
ax.plot(fy, np.array(month)/1000, color=BLUE, lw=1.6, ls="--", marker="s", ms=3.5, label="Excess split purchases, same month")
ax.plot(fy, np.array(day)/1000, color=BLUE, lw=1.8, marker="o", ms=3.5, label="Excess split purchases, same day")
ax.axvline(2017.5, color=GREY, ls=":", lw=0.9)
ax.text(2017.58, 63, "2017 Act in force", color="#555555", fontsize=8, va="top")
ax.set_ylim(0, 65); ax.set_xticks(fy); ax.set_xlabel("Fiscal year")
ax.set_ylabel("Thousand contracts or purchases per year")
ax.grid(axis="y", color=LIGHT, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("Figure_3.pdf"); fig.savefig("Figure_3.eps"); plt.close(fig)
print("done")
