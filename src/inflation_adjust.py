#!/usr/bin/env python3
"""
Inflation-adjusted difference-in-bunching at THB 500,000 (Table 4 and Section 6 of the GIQ manuscript).

The threshold is nominal. Each fiscal year's reference (the fiscal 2015 shape) is shifted by a price index
before scaling, for three deflators:
  CPI           World Bank WDI FP.CPI.TOTL.ZG, Thailand, annual % change (calendar years), accessed 16 Oct 2026
  CMI calendar  Trade Policy and Strategy Office (TPSO), Ministry of Commerce, Construction Materials Price Index,
                base 2015 = 100, calendar-year averages (index.tpso.go.th/materials-price-index, accessed 16 Oct 2026)
  CMI fiscal    the same index averaged over October-September; fiscal 2015 uses January-September 2015
                because the base-2015 series starts in January 2015
Index values are embedded below so the script needs no network access.

Shifting methods. The reference is held in THB 1,000 bins and has spikes at round values (e.g. THB 400,000 and
500,000) that reflect pricing conventions, not the price level, and that recur at the same round values in every
year. Method A: the excess of each bin at a multiple of THB 10,000 over the median of its neighbours stays where
it is, and only the remaining mass is shifted (bin midpoints times the index). Method B also holds in place the
reference's own bunching just below THB 500,000 (pilot agencies from February 2015). The "naive"
method shifts every bin label; it moves the round-value spikes across bin edges (a 0.1% shift already moves the
THB 500,000 spike above the threshold) and is reported for comparison only.

Usage (Anaconda Prompt, env egp-xai):  python src\\inflation_adjust.py
Output: reports/inflation_adjust.md (+ .csv). Reads data/processed/bunching_bins.parquet only.
"""
import sys

import numpy as np
import pandas as pd

from bunching import BINS, REPORTS, _bins5k, dib_estimate, log, sel

W, LO, HI, T, ZL = 5000, 100000, 2000000, 500000, 50000
CPI_INFL = {2016: 0.186235, 2017: 0.664415, 2018: 1.066835, 2019: 0.705525, 2020: -0.846088,
            2021: 1.231045, 2022: 6.075900, 2023: 1.229090, 2024: 0.398333, 2025: -0.133634}
CMI_MONTHLY = {   # overall index, January-December, base 2015 = 100
    2015: [102.5, 101.6, 101.2, 101.3, 101.1, 100.9, 100.2, 99.7, 99.0, 98.1, 97.4, 96.8],
    2016: [96.2, 96.5, 97.1, 98.7, 99.6, 98.2, 96.4, 96.6, 97.1, 96.6, 97.0, 97.6],
    2017: [98.6, 98.5, 98.6, 98.4, 97.4, 97.5, 98.0, 99.6, 101.2, 100.7, 100.5, 100.6],
    2018: [101.3, 101.3, 101.2, 101.0, 101.1, 101.9, 102.5, 102.9, 102.6, 101.9, 101.8, 101.0],
    2019: [101.0, 101.2, 101.5, 101.7, 101.4, 101.0, 100.4, 100.6, 99.8, 98.7, 98.7, 98.9],
    2020: [99.3, 99.1, 98.6, 97.7, 97.6, 98.1, 98.7, 98.4, 98.6, 98.9, 99.0, 99.8],
    2021: [103.0, 103.4, 103.9, 105.8, 108.0, 108.0, 106.4, 107.1, 107.1, 108.8, 109.4, 108.6],
    2022: [109.3, 110.4, 112.9, 115.1, 115.0, 114.0, 113.1, 112.7, 112.6, 112.7, 112.7, 112.5],
    2023: [113.1, 113.3, 113.6, 113.5, 113.3, 113.0, 112.5, 112.5, 112.6, 112.7, 112.4, 112.0],
    2024: [112.0, 112.0, 112.2, 112.2, 113.0, 113.3, 113.1, 113.3, 113.2, 113.1, 112.8, 112.4],
    2025: [112.3, 112.1, 112.8, 113.1, 113.3, 113.6, 113.6, 113.7, 113.6, 113.4, 113.0, 112.4],
}


def deflators():
    cpi = {2015: 1.0}
    for y in range(2016, 2026):
        cpi[y] = cpi[y - 1] * (1 + CPI_INFL[y] / 100)
    cal = {y: float(np.mean(v)) for y, v in CMI_MONTHLY.items()}
    cmi_cal = {y: cal[y] / cal[2015] for y in cal}
    s = pd.Series({(y, m + 1): v for y, vals in CMI_MONTHLY.items() for m, v in enumerate(vals)})
    fy = [y + 1 if m >= 10 else y for (y, m) in s.index]
    fyavg = s.groupby(fy).mean()
    cmi_fy = {y: fyavg[y] / fyavg[2015] for y in range(2015, 2026)}
    return {"nominal": lambda y: 1.0, "CPI": lambda y: cpi[y],
            "CMI calendar": lambda y: cmi_cal[y], "CMI fiscal": lambda y: cmi_fy[y]}


def main():
    d = pd.read_parquet(BINS)
    pr = d[d["var"] == "price"]
    ref1k = sel(pr, regime="pre")
    grid = _bins5k(ref1k, W, LO, HI).index.values
    below = (grid > T - ZL) & (grid <= T)

    lab_ = ref1k.index.values
    v = ref1k.values.astype(float)
    rnd = lab_ % 10000 == 0
    nb = pd.Series(v, index=lab_).where(~rnd).rolling(9, center=True, min_periods=2).median().values
    spike = np.nan_to_num(np.where(rnd, np.maximum(v - nb, 0), 0.0))
    # the reference's own bunching just below THB 500,000 (pilot agencies, Feb-Sep 2015): non-round excess in
    # THB 490,001-500,000 over a linear fit to THB 440,000-489,000
    sm = pd.Series(v - spike, index=lab_)
    fit = sm.loc[440000:489000]
    coef = np.polyfit(fit.index.values, fit.values, 1)
    near = (lab_ > 490000) & (lab_ <= 500000)
    own = np.where(near, np.maximum(sm.values - np.polyval(coef, lab_), 0), 0.0)
    held = {"A": spike, "B": spike + own}

    def shifted(f, method):
        if method == "naive":
            s = ref1k.copy()
            s.index = s.index.values * f
            return _bins5k(s, W, LO, HI).values
        k = pd.Series(held[method], index=lab_)
        s = pd.Series(v, index=lab_) - k
        s.index = (s.index.values - 500) * f + 500      # shift bin midpoints
        a = _bins5k(s, W, LO, HI)
        return (a + _bins5k(k[k > 0], W, LO, HI).reindex(a.index, fill_value=0)).values

    D = deflators()
    cols = [("nominal", "A")] + [(k, m) for k in ("CPI", "CMI calendar", "CMI fiscal") for m in ("A", "B")] + [("CPI", "naive")]
    name = lambda c: c[0] if c[0] == "nominal" else f"{c[0]} [{c[1]}]"
    rows, agg = [], {name(c): [0.0, 0.0] for c in cols}
    for fy in range(2559, 2569):
        ce = fy - 543
        yt = _bins5k(sel(pr, fiscal_year=fy, regime=["transition", "act"]), W, LO, HI).values
        out = {"fiscal_year": ce, **{"index " + k: D[k](ce) for k in ("CPI", "CMI calendar", "CMI fiscal")}}
        for c in cols:
            r = dib_estimate(yt, shifted(D[c[0]](ce), c[1]), grid, T, W, 200000, 400000, ZL, [1500000], "const")
            out[name(c)] = r["dz"] / T
            if ce >= 2018:
                agg[name(c)][0] += r["excess_B"]
                agg[name(c)][1] += r["_cf"][below].sum()
        rows.append(out)
    R = pd.DataFrame(rows)
    R.to_csv(REPORTS / "inflation_adjust.csv", index=False)
    pooled = {k: a[0] / (a[1] / (ZL / W) / W) / T for k, a in agg.items()}
    names = [name(c) for c in cols]
    L = ["# Inflation-adjusted bunching response at THB 500,000 (Δz/T)", "",
         "Reference = fiscal 2015 distribution shifted by each fiscal year's price index before scaling.",
         "[A] round-value spikes held in place; [B] also the reference's own bunching at THB 490,001–500,000 held in place;",
         "[naive] every bin label multiplied by the index (moves spikes across bin edges; for comparison only).",
         f"Reference's own near-threshold excess held in [B]: {own.sum():,.0f} contracts.", "",
         "| fiscal year | " + " | ".join(names) + " | index CPI | index CMI cal. | index CMI fisc. |",
         "|---" * (len(names) + 4) + "|"]
    for _, r in R.iterrows():
        L.append(f"| {int(r.fiscal_year)} | " + " | ".join(f"{r[k]:.3f}" for k in names) + " | "
                 + " | ".join(f"{r['index ' + k]:.3f}" for k in ("CPI", "CMI calendar", "CMI fiscal")) + " |")
    L += ["", "| pooled fiscal 2018–2025 | " + " | ".join(f"{pooled[k]:.3f}" for k in names) + " |",
          "|---" * (len(names) + 1) + "|"]
    out = REPORTS / "inflation_adjust.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
