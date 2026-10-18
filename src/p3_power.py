"""Paper 3, design-stage power check: can unit-level heterogeneity of the discretion effect be detected?

Implements the locked plan in the project doc claude/p3-power-plan.md (19 Oct 2026). Uses only sample sizes and
sampling noise of unit-level DiD estimates (no CATE, no variance of true effects, no covariates).

  python src/p3_power.py          -> data/processed/p3/cells_YYYY.parquet (cache), reports/p3_power.md / .json

Cells (definitions as in revision_checks.did): unit x band x month, discount d = 1 - agreed/budget when
budget > 0 and agreed/budget in [0.3, 1.5]; with n, sum d, sum d^2. Unit DiD: treated 100k-500k minus control
(30k-100k main; 2M-3M secondary), post minus pre (pre Oct 2014 - Sep 2015; post Oct-Nov 2015 main,
Oct 2015 - Aug 2017 alt). Noise v_u = sum over the 4 cells of s^2/n. Eligible: n >= m in each cell.
MDE (top vs bottom quintile GATES) = 2.8 * sqrt(2 * mean(v_u) / (N/5)).
Decision: GO if N >= 2,000 and MDE <= 3.0 pp in the main window; GO-conditional if only the alt window passes;
STOP/pivot otherwise.
"""
import json
import time

import numpy as np
import pandas as pd

from bunching import GOV_SQL, MAX_PRICE, PROCESSED, REPORTS, connect, log
from revision_checks import DEDUP, src

OUT = PROCESSED / "p3"
PRE = ("2014-10", "2015-09")
POSTS = {"main (Oct–Nov 2015)": ("2015-10", "2015-11"), "alt (Oct 2015 – Aug 2017)": ("2015-10", "2017-08")}
TREAT, CONTROLS = "100k-500k", ("30k-100k", "2M-3M")
N_MIN, MDE_MAX, REF_ATE = 2000, 0.030, 0.06
BAND = """CASE WHEN agreed_price <= 30000 THEN '0-30k' WHEN agreed_price <= 100000 THEN '30k-100k'
               WHEN agreed_price <= 500000 THEN '100k-500k' WHEN agreed_price <= 2000000 THEN '500k-2M' ELSE '2M-3M' END"""


def cells():
    OUT.mkdir(parents=True, exist_ok=True)
    con = connect()
    parts = []
    for y in range(2558, 2562):
        f = OUT / f"cells_{y}.parquet"
        if not f.exists():
            t0 = time.time()
            con.execute(f"""COPY (
                WITH c AS (SELECT {DEDUP} agency || '|' || coalesce(sub_agency, '') AS unit, {GOV_SQL} AS gov,
                                  strftime(sign_date, '%Y-%m') AS ym, agreed_price, budget
                           FROM read_parquet('{src(y)}')
                           WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE}),
                     d AS (SELECT *, {BAND} AS band, 1 - agreed_price / budget AS disc FROM c
                           WHERE budget > 0 AND agreed_price / budget BETWEEN 0.3 AND 1.5)
                SELECT unit, any_value(gov) AS gov, ym, band, count(*)::INT n, sum(disc) s1, sum(disc * disc) s2
                FROM d GROUP BY unit, ym, band) TO '{f.as_posix()}' (FORMAT parquet)""")
            log(f"p3 cells {y}: {time.time() - t0:.0f}s")
        parts.append(pd.read_parquet(f))
    D = pd.concat(parts, ignore_index=True)
    return D[(D.ym >= PRE[0]) & (D.ym <= "2017-08")]


def unit_did(D, ctrl, post, m):
    d = D[D.band.isin([TREAT, ctrl])].copy()
    d["per"] = np.where((d.ym >= PRE[0]) & (d.ym <= PRE[1]), "pre",
                        np.where((d.ym >= post[0]) & (d.ym <= post[1]), "post", None))
    d = d[d.per.notna()]
    g = d.groupby(["unit", "band", "per"], as_index=False)[["n", "s1", "s2"]].sum()
    g["mean"] = g.s1 / g.n
    g["var_mean"] = np.where(g.n > 1, (g.s2 - g.n * g["mean"] ** 2) / (g.n - 1), np.nan) / g.n
    w = g.pivot_table(index="unit", columns=["band", "per"], values=["n", "mean", "var_mean"])
    keys = [(TREAT, "post"), (TREAT, "pre"), (ctrl, "post"), (ctrl, "pre")]
    if not all(("n",) + k in w.columns for k in keys):
        return pd.DataFrame(columns=["unit", "tau", "v"])
    ok = np.logical_and.reduce([w[("n",) + k].fillna(0) >= m for k in keys])
    w = w[ok]
    tau = (w[("mean", TREAT, "post")] - w[("mean", TREAT, "pre")]) - (w[("mean", ctrl, "post")] - w[("mean", ctrl, "pre")])
    v = sum(w[("var_mean",) + k].fillna(0) for k in keys)
    return pd.DataFrame({"unit": w.index, "tau": tau.values, "v": v.values})


def mde(v, n):
    return 2.8 * np.sqrt(2 * np.mean(v) / (n / 5)) if n >= 5 else np.nan


def main():
    t0 = time.time()
    D = cells()
    gov = D.groupby("unit").gov.first()
    rows = []
    for ctrl in CONTROLS:
        for pname, post in POSTS.items():
            for m in (1, 3, 5):
                U = unit_did(D, ctrl, post, m)
                n = len(U)
                r = {"control": ctrl, "window": pname, "m": m, "N": n, "MDE": mde(U.v.values, n) if n else np.nan,
                     "median_noise_sd": float(np.sqrt(U.v.median())) if n else np.nan}
                for g in ("central", "local"):
                    Ug = U[U.unit.map(gov) == g]
                    r[f"N_{g}"], r[f"MDE_{g}"] = len(Ug), mde(Ug.v.values, len(Ug)) if len(Ug) else np.nan
                rows.append(r)
    R = pd.DataFrame(rows)

    def row(ctrl, w, m=3):
        return R[(R.control == ctrl) & (R.window == w) & (R.m == m)].iloc[0]

    main_r, alt_r = row(CONTROLS[0], list(POSTS)[0]), row(CONTROLS[0], list(POSTS)[1])
    passes = lambda r: bool(r.N >= N_MIN and r.MDE <= MDE_MAX)
    verdict = "GO" if passes(main_r) else ("GO-conditional (alt window main)" if passes(alt_r) else "STOP / pivot")
    res = {"verdict": verdict, "main": main_r.to_dict(), "alt": alt_r.to_dict(), "seconds": round(time.time() - t0)}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "p3_power.json").write_text(json.dumps({"decision": res, "table": R.to_dict("records")}, indent=1, default=float),
                                           encoding="utf-8")
    T_ = R.copy()
    for c in ("MDE", "MDE_central", "MDE_local", "median_noise_sd"):
        T_[c] = (T_[c] * 100).map(lambda x: "–" if pd.isna(x) else f"{x:.2f} pp")
    for c in ("N", "N_central", "N_local"):
        T_[c] = T_[c].map(lambda x: f"{int(x):,}")
    L = [f"# Paper 3 design-stage power check — **{verdict}**", "",
         "Locked plan: project doc `claude/p3-power-plan.md` (19 Oct 2026). Sampling noise only; no CATE estimated.", "",
         f"Rule: N ≥ {N_MIN:,} eligible units and MDE (top vs bottom quintile, 80% power) ≤ {MDE_MAX * 100:.1f} pp "
         f"(half of the reference |ATE| {REF_ATE * 100:.0f} pp), main window, control 30k–100k, m = 3.", "",
         f"- Main window: N = {int(main_r.N):,}, MDE = {main_r.MDE * 100:.2f} pp → {'pass' if passes(main_r) else 'fail'}",
         f"- Alt window: N = {int(alt_r.N):,}, MDE = {alt_r.MDE * 100:.2f} pp → {'pass' if passes(alt_r) else 'fail'}", "",
         "## All specifications (m = minimum discounts per cell)", "", T_.to_markdown(index=False), "",
         f"Run time {res['seconds']} s."]
    (REPORTS / "p3_power.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:9]), flush=True)


if __name__ == "__main__":
    main()
