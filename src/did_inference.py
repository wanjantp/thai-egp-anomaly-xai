#!/usr/bin/env python3
"""
Inference for the 2015-reform DiD on discounts that allows for band-level shocks (Reviewer-2, round 2, item 4).

The reform is one national change and treats one price band, so unit-clustered SEs ignore shocks common to a
band in a month. This script reports three checks on the discount vs budget (1 − agreed/budget):

  1. Event-study series: δ_m = treated − control gap in month m (unit×band and month FE, ref = Sep 2015).
     DiD from the series = mean(δ, Oct 2015 – Aug 2017) − mean(δ, Oct 2014 – Sep 2015), with Newey–West SE
     over the 35 monthly gaps (lags 3 and 6).
  2. Placebo break dates: the same before/after contrast at every fake break inside the pre-period
     (Jan–Jul 2015) and inside the transition (Jan 2016 – May 2017), each using only that sub-window.
     Randomization p = (1 + #|placebo| >= |actual|) / (1 + #placebos).
  3. Placebo bands at the real date: two pairs of bands whose method did not change in 2015
     (THB 0–30k vs 30k–100k, both discretionary; THB 0.5–2M vs 2–3M, both competitive).

  4. (v7, statistical review) Extended placebo set and HAC sensitivity, from the saved series alone
     (`python src\\did_inference.py extended`): placebo bands at every break date Jan 2015 – May 2017 added to the
     placebo dates, the rank of the actual estimate among all placebos, and Newey–West SEs with 3, 6 and 12 lags
     judged against t(33) critical values.

Usage (Anaconda Prompt, env egp-xai; needs the DiD cache from `python revision_checks.py did`):
    python src\\did_inference.py            (steps 1–4)
    python src\\did_inference.py extended   (step 4 only, from reports/did_inference_series.csv)
Output: reports/did_inference.md (+ reports/did_inference_series.csv). Reads data/processed only.
"""
import sys

import numpy as np
import pandas as pd

from bunching import REPORTS, _twfe, log
from revision_checks import OUT_DIR

REF = "2015-09"
START, POST, END = "2014-10", "2015-10", "2017-08"
TREAT = "100k-500k"


def load():
    files = [OUT_DIR / f"did_int_{y}.parquet" for y in range(2558, 2562)]
    if not all(f.exists() for f in files):
        log("DiD cache missing → building it with revision_checks.did()")
        import revision_checks
        revision_checks.did()
    D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    D = D[(D.ym >= START) & (D.ym <= END) & (D.n_b > 0)].copy()
    D["y"] = D.sd_b / D.n_b
    return D


def newey_west(y, x, lags):
    """OLS of y on [1, x] with Newey–West (Bartlett) SE; returns slope and SE."""
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = (y - X @ b)[:, None] * X
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return b[1], float(np.sqrt(V[1, 1]))


def series(D, treat, ctrl):
    """Monthly treated–control gaps δ_m from a TWFE event study (ref month = Sep 2015 → δ = 0)."""
    d = D[D.band.isin([treat, ctrl])].copy()
    d["ub"] = d.unit + "|" + d.band
    tr = (d.band == treat).astype(float)
    months = sorted(d.ym.unique())
    cols = []
    for m in months:
        if m == REF:
            continue
        c = "e_" + m
        d[c] = tr * (d.ym == m)
        cols.append(c)
    b, se = _twfe(d, "y", "n_b", cols, "ub", "ym", "unit")
    s = pd.DataFrame({"ym": [c[2:] for c in cols], "delta": b, "se_unit": se})
    s = pd.concat([s, pd.DataFrame({"ym": [REF], "delta": [0.0], "se_unit": [0.0]})]).sort_values("ym")
    # pooled TWFE DiD with unit-clustered SE, for comparison
    d["tp"] = tr * (d.ym >= POST)
    bp, sp = _twfe(d, "y", "n_b", ["tp"], "ub", "ym", "unit")
    return s.reset_index(drop=True), bp[0], sp[0]


def contrast(s, lo, brk, hi):
    w = s[(s.ym >= lo) & (s.ym <= hi)]
    return w.loc[w.ym >= brk, "delta"].mean() - w.loc[w.ym < brk, "delta"].mean()


def placebo_dates(s):
    out = []
    for brk in pd.period_range("2015-01", "2015-07", freq="M").astype(str):
        out.append(("pre", brk, contrast(s, START, brk, REF)))
    for brk in pd.period_range("2016-01", "2017-05", freq="M").astype(str):
        out.append(("transition", brk, contrast(s, POST, brk, END)))
    return pd.DataFrame(out, columns=["window", "break", "beta"])


def main():
    D = load()
    log(f"rows {len(D):,}, units {D.unit.nunique():,}")
    rows, keep = [], []
    pairs = [("main", TREAT, "30k-100k"), ("main", TREAT, "2M-3M"),
             ("placebo band", "0-30k", "30k-100k"), ("placebo band", "500k-2M", "2M-3M")]
    for kind, t, c in pairs:
        s, b_twfe, se_twfe = series(D, t, c)
        s["treat"], s["control"] = t, c
        keep.append(s)
        post = (s.ym >= POST).astype(float).values
        b3, se3 = newey_west(s.delta.values, post, 3)
        b6, se6 = newey_west(s.delta.values, post, 6)
        r = {"kind": kind, "treat": t, "control": c, "twfe": b_twfe, "se_unit": se_twfe,
             "series": b3, "se_nw3": se3, "se_nw6": se6}
        if kind == "main":
            P = placebo_dates(s)
            r["n_placebo"] = len(P)
            r["placebo_max_abs"] = P.beta.abs().max()
            r["placebo_sd"] = P.beta.std()
            r["p_rand"] = (1 + (P.beta.abs() >= abs(b3)).sum()) / (1 + len(P))
            P.assign(treat=t, control=c).to_csv(REPORTS / f"did_inference_placebo_{c}.csv", index=False)
        rows.append(r)
        log(f"{kind} {t} vs {c}: twfe {b_twfe:+.4f} ({se_twfe:.4f}), series {b3:+.4f} NW3 {se3:.4f} NW6 {se6:.4f}")
    pd.concat(keep).to_csv(REPORTS / "did_inference_series.csv", index=False)
    R = pd.DataFrame(rows)

    L = ["# DiD inference with band-level shocks (Reviewer-2 round 2, item 4)", "",
         "Outcome: discount vs budget (1 − agreed/budget), unit×band cells by month, Oct 2014 – Aug 2017, weights = contracts.",
         "TWFE = pooled DiD with unit-clustered SE (as in the manuscript). Series = mean monthly gap after Oct 2015 minus before,",
         "from a TWFE event study (ref = Sep 2015), with Newey–West SE over the 35 monthly gaps.", "",
         "## 1) Main estimates", "",
         "| treated vs control | TWFE (unit SE) | series DiD | NW SE (3 lags) | NW SE (6 lags) | t (NW 6) |",
         "|---|---|---|---|---|---|"]
    for r in R[R.kind == "main"].itertuples():
        L.append(f"| {r.treat} vs {r.control} | {r.twfe:+.2%} ({r.se_unit:.2%}) | {r.series:+.2%} | "
                 f"{r.se_nw3:.2%} | {r.se_nw6:.2%} | {r.series / r.se_nw6:+.1f} |")
    L += ["", "## 2) Placebo break dates (inside the pre-period and inside the transition)", "",
          "| treated vs control | actual (series) | placebos | SD of placebos | max |placebo| | randomization p |",
          "|---|---|---|---|---|---|"]
    for r in R[R.kind == "main"].itertuples():
        L.append(f"| {r.treat} vs {r.control} | {r.series:+.2%} | {int(r.n_placebo)} | {r.placebo_sd:.2%} | "
                 f"{r.placebo_max_abs:.2%} | {r.p_rand:.3f} |")
    L += ["", "## 3) Placebo bands at the real date (method unchanged in 2015)", "",
          "| pair | TWFE (unit SE) | series DiD | NW SE (6 lags) |", "|---|---|---|---|"]
    for r in R[R.kind == "placebo band"].itertuples():
        L.append(f"| {r.treat} vs {r.control} | {r.twfe:+.2%} ({r.se_unit:.2%}) | {r.series:+.2%} | {r.se_nw6:.2%} |")
    L += ["", "Note: 0.5–2M vs 2–3M is not a clean placebo — tenders above THB 2M moved to e-auctions in 2015 and "
          "purchases left the 0.5–2M band by splitting; 0–30k vs 30k–100k changed in composition when purchases moved up."]
    out = REPORTS / "did_inference.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


def extended(S=None):
    from scipy import stats
    S = pd.read_csv(REPORTS / "did_inference_series.csv") if S is None else S
    breaks = pd.period_range("2015-01", "2017-05", freq="M").astype(str)
    band_pl = []
    for (t, c), s in S.groupby(["treat", "control"]):
        if t != TREAT:
            band_pl += [(f"{t} vs {c}", b, contrast(s, START, b, END)) for b in breaks]
    band_pl = pd.DataFrame(band_pl, columns=["pair", "break", "beta"])
    L = ["# DiD inference, extended placebo set and HAC sensitivity (statistical review, v7)", "",
         "Placebos: (a) fake break dates for the treated pair (Jan–Jul 2015 within the pre-period, Jan 2016 – May 2017 "
         "within the transition); (b) two pairs of bands whose method did not change, at every break date Jan 2015 – "
         f"May 2017 over the full window ({len(band_pl)} contrasts). p = (1 + #|placebo| ≥ |actual|) / (1 + #placebos).", "",
         "| treated vs control | actual | placebos | rank of actual | max |placebo| | p | NW SE 3 / 6 / 12 lags | two-sided p, t(33), 12 lags |",
         "|---|---|---|---|---|---|---|---|"]
    for (t, c), s in S.groupby(["treat", "control"]):
        if t != TREAT:
            continue
        s = s.sort_values("ym")
        post = (s.ym >= POST).astype(float).values
        b, _ = newey_west(s.delta.values, post, 3)
        ses = [newey_west(s.delta.values, post, l)[1] for l in (3, 6, 12)]
        P = pd.concat([placebo_dates(s).beta, band_pl.beta], ignore_index=True)
        k = int((P.abs() >= abs(b)).sum())
        p = (1 + k) / (1 + len(P))
        pt = 2 * stats.t.sf(abs(b) / ses[2], df=len(s) - 2)
        L.append(f"| {t} vs {c} | {b:+.2%} | {len(P)} | {k + 1} of {len(P) + 1} | {P.abs().max():.2%} | {p:.3f} | "
                 + " / ".join(f"{x:.2%}" for x in ses) + f" | {pt:.1e} |")
    L += ["", "Band placebos, largest contrasts:", "", "| pair | break | contrast |", "|---|---|---|"]
    for r in band_pl.reindex(band_pl.beta.abs().sort_values(ascending=False).index).head(5).itertuples():
        L.append(f"| {r.pair} | {r._2} | {r.beta:+.2%} |")
    out = REPORTS / "did_inference_extended.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["extended"]:
        extended()
    else:
        main()
        extended()
