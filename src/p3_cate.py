"""Paper 3, steps 3-4 of the locked pre-analysis plan (project doc claude/p3-pre-analysis-plan.md): heterogeneity of the
effect of discretion on recorded discounts across buying units.

  python src/p3_cate.py --permute     pipeline test: covariate rows shuffled across units within government level
  python src/p3_cate.py               the real run (once)
  options: --splits N (default 100), --quick (few trees / splits; code test only, never reported)

Inputs from src/p3_panel.py (data/processed/p3/): tau.parquet, X.parquet, x_meta.json, build/cells_c_*.parquet.
E1 (primary) honest regression forest on the unit DiD pseudo-outcome tau_u (econml.grf.RegressionForest); E2 causal
forest with fixed effects (causalfe CFFEForest, sped up by p3_fastcffe) on unit x band x month cells; E3 linear BLP.
Tests: repeated sample splitting by agency (BLP, GATES, CLAN, AUTOC; median estimates, CIs of per-split 1-alpha/2
intervals, p = 2 x median p; Chernozhukov, Demirer, Duflo & Fernandez-Val 2025). Reports aggregates only.
Outputs: reports/p3_cate[_permuted].md / .json; data/processed/p3/scores[_permuted].parquet (internal).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from bunching import PROCESSED, REPORTS, log

P3 = PROCESSED / "p3"
SEED = 20261019
ALPHA = 0.05
NJ = int(os.environ.get("EGPQ_THREADS", os.cpu_count() or 4))
CAT = ("agency_cat", "region")
E1_KW = dict(n_estimators=2000, min_samples_leaf=20, honest=True)
E2_TREES = {"main": 500, "alt": 200}
PRE = ("2014-10", "2015-09")
POST = {"main": ("2015-10", "2015-11"), "alt": ("2015-10", "2017-08"), "alt_late": ("2015-12", "2017-08")}


# ---------------------------------------------------------------- data
def load():
    T = pd.read_parquet(P3 / "tau.parquet")
    X = pd.read_parquet(P3 / "X.parquet").set_index("unit")
    meta = json.loads((P3 / "x_meta.json").read_text(encoding="utf-8"))
    cov = [c for c in meta["covariates"] if c not in meta["mechanical_drops"]]
    return T, X, cov


def permute_rows(X, cov, seed):
    """Pipeline test: shuffle covariate rows across units within government level (local kept)."""
    Xp = X.copy()
    rng = np.random.default_rng(seed)
    feat = [c for c in cov if c != "local"]
    for g in (0, 1):
        idx = X.index[X.local == g]
        Xp.loc[idx, feat] = X.loc[rng.permutation(idx), feat].values
    return Xp


def design(X, units, cov):
    """Raw imputed matrix (forests, trees) and standardised matrix (E3) for the given units."""
    D = X.reindex(units)
    num = [c for c in cov if c not in CAT]
    out = {}
    for c in num:
        x = D[c].astype(float)
        if x.isna().any():
            out[f"{c}_missing"] = x.isna().astype(float)
            x = x.fillna(x.median())
        out[c] = x
    R = pd.DataFrame(out, index=units)
    for c in (c for c in CAT if c in cov):
        dm = pd.get_dummies(D[c].fillna("unmapped"), prefix=c, dtype=float)
        R = R.join(dm)
    R = R.loc[:, R.std() > 0]
    Z = (R - R.mean()) / R.std()
    # E3 reference categories: drop the largest dummy of each categorical
    for c in (c for c in CAT if c in cov):
        cols = [k for k in Z.columns if k.startswith(c + "_")]
        if cols:
            Z = Z.drop(columns=R[cols].sum().idxmax())
    return R, Z


# ---------------------------------------------------------------- inference helpers
def ols_cl(y, M, g, w=None):
    """OLS / WLS with agency-clustered sandwich (small-sample corrected). Returns beta, V."""
    y, M = np.asarray(y, float), np.asarray(M, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    Mw = M * w[:, None]
    A = np.linalg.pinv(M.T @ Mw)
    b = A @ (Mw.T @ y)
    e = y - M @ b
    _, gi = np.unique(np.asarray(g), return_inverse=True)
    S = np.zeros((gi.max() + 1, M.shape[1]))
    np.add.at(S, gi, Mw * e[:, None])
    G, n, k = S.shape[0], len(y), M.shape[1]
    c = G / (G - 1) * (n - 1) / max(n - k, 1) if G > 1 else 1.0
    return b, c * A @ (S.T @ S) @ A


def pz(est, se):
    return float(2 * stats.norm.sf(abs(est / se))) if se > 0 else np.nan


Z_SPLIT = stats.norm.ppf(1 - ALPHA / 4)       # per-split interval at level 1 - alpha/2


def e1(seed):
    from econml.grf import RegressionForest
    return RegressionForest(**E1_KW, random_state=seed, n_jobs=NJ)


def autoc(tau, s):
    o = np.argsort(-s, kind="stable")
    t = tau[o]
    toc = np.cumsum(t) / np.arange(1, len(t) + 1) - t.mean()
    return float(toc.mean())


def split_stats(tau, s, g, w, R, fs, rng):
    """BLP, GATES, CLAN, AUTOC on the main half, given the out-of-sample score s."""
    r = {}
    if np.std(s) < 1e-12:                    # forest predicted a constant (tiny samples): no test possible this split
        nan = float("nan")
        r.update(blp_b1=float(np.mean(tau)), blp_b2=nan, blp_se2=nan, gates_diff=nan, gates_diff_se=nan, autoc=nan,
                 autoc_se=nan, clan={}, **{f"gates{k}": nan for k in range(1, 6)}, **{f"gates{k}_se": nan for k in range(1, 6)})
        return r
    b, V = ols_cl(tau, np.column_stack([np.ones(len(s)), s - s.mean()]), g, w)
    r["blp_b1"], r["blp_b2"], r["blp_se2"] = b[0], b[1], np.sqrt(V[1, 1])
    q = pd.qcut(pd.Series(s).rank(method="first"), 5, labels=False).to_numpy()
    Q = np.column_stack([(q == k).astype(float) for k in range(5)])
    b, V = ols_cl(tau, Q, g, w)
    for k in range(5):
        r[f"gates{k + 1}"], r[f"gates{k + 1}_se"] = b[k], np.sqrt(V[k, k])
    r["gates_diff"] = b[4] - b[0]
    r["gates_diff_se"] = np.sqrt(V[4, 4] + V[0, 0] - 2 * V[0, 4])
    clan = {}
    sel = (q == 0) | (q == 4)
    for c in list(R.columns) + ["first_stage"]:
        x = (fs if c == "first_stage" else R[c].to_numpy())[sel]
        ok = ~np.isnan(x)
        hi = (q[sel] == 4)[ok].astype(float)
        if ok.sum() < 10 or hi.min() == hi.max() or np.std(x[ok]) < 1e-12:
            continue
        bb, VV = ols_cl(x[ok], np.column_stack([np.ones(ok.sum()), hi]), g[sel][ok])
        clan[c] = (bb[0], bb[0] + bb[1], bb[1], np.sqrt(VV[1, 1]))
    r["clan"] = clan
    a = autoc(tau, s)
    ag = np.unique(g)
    boots = []
    idx_by = {k: np.nonzero(g == k)[0] for k in ag}
    for _ in range(200):
        pick = np.concatenate([idx_by[k] for k in rng.choice(ag, len(ag))])
        boots.append(autoc(tau[pick], s[pick]))
    r["autoc"], r["autoc_se"] = a, float(np.std(boots, ddof=1))
    return r


def summarise(per):
    """Median over splits; CI = medians of per-split (1 - alpha/2) bounds; p = min(1, 2 x median p)."""
    out = {}
    for est, se in (("blp_b2", "blp_se2"), ("gates_diff", "gates_diff_se"), ("autoc", "autoc_se")) + \
            tuple((f"gates{k}", f"gates{k}_se") for k in range(1, 6)):
        e, s = np.array([p[est] for p in per]), np.array([p[se] for p in per])
        pv = np.array([pz(a, b) if np.isfinite(a) and np.isfinite(b) else np.nan for a, b in zip(e, s)])
        if np.isnan(e).all():
            out[est] = {"est": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
            continue
        out[est] = {"est": float(np.nanmedian(e)), "lo": float(np.nanmedian(e - Z_SPLIT * s)),
                    "hi": float(np.nanmedian(e + Z_SPLIT * s)), "p": float(min(1, 2 * np.nanmedian(pv)))}
    out["valid_splits"] = int(sum(np.isfinite(p["blp_b2"]) for p in per))
    out["blp_b1"] = float(np.median([p["blp_b1"] for p in per]))
    keys = sorted(set().union(*[p["clan"] for p in per]))
    out["clan"] = {k: {"bottom": float(np.median([p["clan"][k][0] for p in per if k in p["clan"]])),
                       "top": float(np.median([p["clan"][k][1] for p in per if k in p["clan"]])),
                       "diff": float(np.median([p["clan"][k][2] for p in per if k in p["clan"]])),
                       "p": float(min(1, 2 * np.median([pz(p["clan"][k][2], p["clan"][k][3]) for p in per if k in p["clan"]])))}
                   for k in keys}
    out["splits"] = len(per)
    return out


def cddf(tau, g, R, fs, w, n_splits, seed):
    """Repeated sample splitting by agency: E1 trained on the auxiliary half, tests on the main half."""
    ag = np.unique(g)
    per = []
    for s in range(n_splits):
        rng = np.random.default_rng(seed + s)
        aux_ag = set(rng.choice(ag, len(ag) // 2, replace=False))
        A = np.array([x in aux_ag for x in g])
        m = e1(seed + s).fit(R.to_numpy()[A], tau[A])
        sc = m.predict(R.to_numpy()[~A]).ravel()
        per.append(split_stats(tau[~A], sc, g[~A], None if w is None else w[~A], R.iloc[~A], fs[~A], rng))
    return summarise(per)


def crossfit_e1(tau, g, R, seed, k=5):
    from sklearn.model_selection import GroupKFold
    s = np.full(len(tau), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(R, tau, g):
        s[te] = e1(seed).fit(R.to_numpy()[tr], tau[tr]).predict(R.to_numpy()[te]).ravel()
    return s


def gates_once(tau, s, g):
    q = pd.qcut(pd.Series(s).rank(method="first"), 5, labels=False).to_numpy()
    b, V = ols_cl(tau, np.column_stack([(q == k).astype(float) for k in range(5)]), g)
    d, sd = b[4] - b[0], np.sqrt(V[4, 4] + V[0, 0] - 2 * V[0, 4])
    return {"gates": [float(x) for x in b], "se": [float(np.sqrt(V[k, k])) for k in range(5)],
            "diff": float(d), "diff_se": float(sd), "p": pz(d, sd), "q": q}


# ---------------------------------------------------------------- E2 (CFFE on monthly cells)
def month_cells(units, window):
    C = pd.concat([pd.read_parquet(f) for f in sorted((P3 / "build").glob("cells_c_*.parquet"))], ignore_index=True)
    C = C[C.unit.isin(set(units)) & C.band.isin(["100k-500k", "30k-100k"])]
    post = POST[window]
    C = C[((C.ym >= PRE[0]) & (C.ym <= PRE[1])) | ((C.ym >= post[0]) & (C.ym <= post[1]))]
    C = C.groupby(["unit", "band", "ym"], as_index=False)[["n", "s1"]].sum()
    C = C[C.n > 0]
    C["y"] = C.s1 / C.n
    C["d"] = ((C.band == "100k-500k") & (C.ym >= post[0])).astype(float)
    return C


def crossfit_e2(units, g, R, window, seed, k=5, quick=False):
    import p3_fastcffe
    from causalfe import CFFEForest
    from sklearn.model_selection import GroupKFold
    p3_fastcffe.install()
    C = month_cells(units, window)
    pos = pd.Series(np.arange(len(units)), index=units)
    ui = pos.reindex(C.unit).to_numpy()
    Xc = R.to_numpy()[ui]
    ent = (C.unit + "|" + C.band).to_numpy()
    leaf = int(20 * len(C) / len(units))
    s = np.full(len(units), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(R, np.zeros(len(units)), g):
        rows = np.isin(ui, tr)
        f = CFFEForest(n_trees=8 if quick else E2_TREES[window], max_depth=5, min_leaf=leaf, seed=seed)
        f.fit(Xc[rows], C.y.to_numpy()[rows], C.d.to_numpy()[rows], ent[rows], C.ym.to_numpy()[rows])
        s[te] = f.predict(R.to_numpy()[te])
    return s, {"rows": int(len(C)), "min_leaf_rows": leaf, "trees": 8 if quick else E2_TREES[window]}


# ---------------------------------------------------------------- diagnostics
def pretrend(units, g, s):
    """Per-unit slope of the monthly treated-minus-control discount gap in the pre period, by score quintile."""
    C = month_cells(units, "main")
    C = C[C.ym <= PRE[1]]
    W = C.pivot_table(index=["unit", "ym"], columns="band", values="y").dropna().reset_index()
    W["gap"] = W["100k-500k"] - W["30k-100k"]
    W["t"] = (pd.to_datetime(W.ym) - pd.Timestamp("2014-10-01")).dt.days / 30.44
    sl = W.groupby("unit").filter(lambda d: len(d) >= 3).groupby("unit").apply(
        lambda d: np.polyfit(d.t, d.gap, 1)[0], include_groups=False)
    q = gates_once(np.zeros(len(units)), s, g)["q"]
    d = pd.DataFrame({"slope": sl}).join(pd.DataFrame({"q": q, "g": g}, index=units), how="inner")
    Q = np.column_stack([(d.q == k).astype(float) for k in range(5)])
    b, V = ols_cl(d.slope, Q, d.g)
    Rm = np.zeros((4, 5))
    for i in range(4):
        Rm[i, 0], Rm[i, i + 1] = -1, 1
    rb = Rm @ b
    W_ = float(rb @ np.linalg.pinv(Rm @ V @ Rm.T) @ rb)
    return {"n_units": int(len(d)), "slope_by_quintile_pp_per_month": [float(x * 100) for x in b],
            "wald": W_, "p": float(stats.chi2.sf(W_, 4))}


# ---------------------------------------------------------------- main
def sample(T, X, spec, window, m, extra=None):
    d = T[(T.spec == spec) & (T.window == window) & (T.m == m)]
    d = d[d.unit.isin(X.index)]
    if extra is not None:
        d = d[extra(d)]
    d = d.sort_values("unit").reset_index(drop=True)
    return d, X.loc[d.unit, "agency"].to_numpy()


def holm(ps):
    o = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        E1_KW["n_estimators"], a.splits = 100, min(a.splits, 4)
    t0 = time.time()
    T, X, cov = load()
    if a.permute:
        X = permute_rows(X, cov, SEED + 999)
    tag = "_permuted" if a.permute else ""
    res, scores = {"permuted": a.permute, "quick": a.quick, "covariates": cov, "E1": E1_KW, "splits": a.splits}, []

    def run(name, spec, window, m, extra=None, weighted=False):
        d, g = sample(T, X, spec, window, m, extra)
        R, _ = design(X, d.unit.tolist(), cov)
        tau = d.tau.to_numpy()
        w = None
        if weighted:
            s2 = max(np.var(tau) - d.v.mean(), 1e-6)
            w = 1 / (d.v.to_numpy() + s2)
        t1 = time.time()
        r = cddf(tau, g, R, d.fs.to_numpy() if "fs" in d else np.full(len(d), np.nan), w, a.splits, SEED)
        r.update(N=int(len(d)), N_agencies=int(len(np.unique(g))), mean_tau=float(tau.mean()), seconds=round(time.time() - t1))
        res[name] = r
        log(f"p3_cate {name}: N={len(d)} beta2={r['blp_b2']['est']:.3f} (p={r['blp_b2']['p']:.3g}) "
            f"gates diff={r['gates_diff']['est'] * 100:.2f} pp (p={r['gates_diff']['p']:.3g}) {r['seconds']}s")
        return d, g, R

    # confirmatory: H1-H2 (main), H3 (alt)
    dm, gm, Rm = run("main", "O1", "main", 3)
    da, ga, Ra = run("alt", "O1", "alt", 3)
    ph = holm([res["main"]["blp_b2"]["p"], res["main"]["gates_diff"]["p"]])
    res["H"] = {"H1_p": res["main"]["blp_b2"]["p"], "H2_p": res["main"]["gates_diff"]["p"], "H1_holm": ph[0], "H2_holm": ph[1],
                "H3_blp_p": res["alt"]["blp_b2"]["p"], "H3_gates_p": res["alt"]["gates_diff"]["p"],
                "H3_holm": holm([res["alt"]["blp_b2"]["p"], res["alt"]["gates_diff"]["p"]])}

    # cross-fitted scores, E2 agreement, E3, diagnostics
    for w, d, g, R in (("main", dm, gm, Rm), ("alt", da, ga, Ra)):
        s1 = crossfit_e1(d.tau.to_numpy(), g, R, SEED)
        t1 = time.time()
        s2, info = crossfit_e2(d.unit.tolist(), g, R, w, SEED, quick=a.quick)
        g1, g2 = gates_once(d.tau.to_numpy(), s1, g), gates_once(d.tau.to_numpy(), s2, g)
        _, Z = design(X, d.unit.tolist(), cov)
        b, V = ols_cl(d.tau.to_numpy(), np.column_stack([np.ones(len(d)), Z.to_numpy()]), g)
        se = np.sqrt(np.diag(V))
        res[f"agreement_{w}"] = {"spearman_E1_E2": float(stats.spearmanr(s1, s2).statistic), "E2": info,
                                 "E2_seconds": round(time.time() - t1),
                                 "E1_crossfit_gates": {k: v for k, v in g1.items() if k != "q"},
                                 "E2_crossfit_gates": {k: v for k, v in g2.items() if k != "q"}}
        res[f"E3_{w}"] = {"intercept": float(b[0]), "coef": {c: {"b": float(b[i + 1]), "se": float(se[i + 1]),
                                                                  "p": pz(b[i + 1], se[i + 1])} for i, c in enumerate(Z.columns)}}
        scores.append(pd.DataFrame({"unit": d.unit, "window": w, "s_e1": s1, "s_e2": s2}))
        if w == "main":
            res["pretrend_main"] = pretrend(d.unit.tolist(), g, s1)
    log("p3_cate: cross-fit, E2, E3, pre-trend done")

    # H3 cross-window check: score trained on all main units, GATES on alt_late units
    dl, gl = sample(T, X, "O1", "alt_late", 3)
    Rl, _ = design(X, dl.unit.tolist(), cov)
    cols = [c for c in Rm.columns if c in Rl.columns]
    m_ = e1(SEED).fit(Rm[cols].to_numpy(), dm.tau.to_numpy())
    gx = gates_once(dl.tau.to_numpy(), m_.predict(Rl[cols].to_numpy()).ravel(), gl)
    res["cross_window"] = {k: v for k, v in gx.items() if k != "q"} | {
        "N": int(len(dl)), "overlap_with_main": int(dl.unit.isin(set(dm.unit)).sum()), "features_used": len(cols)}

    # robustness R1-R8 (E1, same procedure)
    zero = X.zero_discount_share
    run("R1_purchase", "O1P", "main", 3)
    run("R2_m1", "O1", "main", 1)
    run("R2_m5", "O1", "main", 5)
    run("R3_weighted", "O1", "main", 3, weighted=True)
    run("R4_pre_to_jul", "R4", "main", 3)
    run("R5_no_stimulus", "R5", "main", 3)
    run("R7_drop_zero_discount", "O1", "main", 3, extra=lambda d: ~(d.unit.map(zero).fillna(0) >= 0.9).to_numpy())
    run("R8_alt_local", "O1", "alt", 3, extra=lambda d: (d.unit.map(X.local) == 1).to_numpy())
    r6 = {}
    for w in ("main", "alt"):
        d, g = sample(T, X, "R6", w, 3)
        loc = d.unit.map(X.local).to_numpy()
        b, V = ols_cl(d.tau.to_numpy(), np.column_stack([1 - loc, loc]), g)
        r6[w] = {"N": int(len(d)), "central": float(b[0]), "local": float(b[1]), "diff": float(b[1] - b[0]),
                 "diff_se": float(np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1]))}
        r6[w]["p"] = pz(r6[w]["diff"], r6[w]["diff_se"])
    res["R6_group_2M3M"] = r6

    res["seconds"] = round(time.time() - t0)
    pd.concat(scores).to_parquet(P3 / f"scores{tag}.parquet", index=False)
    (REPORTS / f"p3_cate{tag}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (REPORTS / f"p3_cate{tag}.md").write_text(report(res), encoding="utf-8")
    print(report(res).split("## Robustness")[0], flush=True)


def fmt(x, pp=True):
    return f"{x * 100:.2f}" if pp else f"{x:.3f}"


def report(res):
    t = "PERMUTED PIPELINE TEST — covariates shuffled across units; not a result" if res["permuted"] else "real run"
    if res["quick"]:
        t += " — QUICK CODE TEST, not reportable"
    L = [f"# Paper 3 heterogeneity (E1 primary) — {t}", "",
         "Plan: project doc `claude/p3-pre-analysis-plan.md` (locked 19 Oct 2026). Effects in pp of budget (recorded discount); "
         f"medians over {res['splits']} agency splits; CI = median of per-split {100 * (1 - ALPHA / 2):.1f}% bounds; p = 2 × median p.", "",
         "## Confirmatory tests", "", "| Test | Estimate | CI | p | Holm p |", "|---|---|---|---|---|"]
    m, al, H = res["main"], res["alt"], res["H"]
    L += [f"| H1 BLP β₂ (main) | {fmt(m['blp_b2']['est'], 0)} | [{fmt(m['blp_b2']['lo'], 0)}, {fmt(m['blp_b2']['hi'], 0)}] | {H['H1_p']:.3g} | {H['H1_holm']:.3g} |",
          f"| H2 GATES top − bottom (main) | {fmt(m['gates_diff']['est'])} pp | [{fmt(m['gates_diff']['lo'])}, {fmt(m['gates_diff']['hi'])}] | {H['H2_p']:.3g} | {H['H2_holm']:.3g} |",
          f"| H3 BLP β₂ (alt) | {fmt(al['blp_b2']['est'], 0)} | [{fmt(al['blp_b2']['lo'], 0)}, {fmt(al['blp_b2']['hi'], 0)}] | {H['H3_blp_p']:.3g} | {H['H3_holm'][0]:.3g} |",
          f"| H3 GATES top − bottom (alt) | {fmt(al['gates_diff']['est'])} pp | [{fmt(al['gates_diff']['lo'])}, {fmt(al['gates_diff']['hi'])}] | {H['H3_gates_p']:.3g} | {H['H3_holm'][1]:.3g} |", ""]
    for w in ("main", "alt"):
        r = res[w]
        L += [f"### {w}: N = {r['N']:,} units, {r['N_agencies']:,} agencies; mean τ̂ = {fmt(r['mean_tau'])} pp; BLP β₁ = {fmt(r['blp_b1'])} pp; "
              f"AUTOC = {fmt(r['autoc']['est'])} pp (p {r['autoc']['p']:.3g})", "",
              "| GATES quintile (score low → high) | " + " | ".join(f"G{k}" for k in range(1, 6)) + " |", "|---" * 6 + "|",
              "| estimate (pp) | " + " | ".join(fmt(r[f'gates{k}']['est']) for k in range(1, 6)) + " |",
              "| CI | " + " | ".join(f"[{fmt(r[f'gates{k}']['lo'])}, {fmt(r[f'gates{k}']['hi'])}]" for k in range(1, 6)) + " |", "",
              "CLAN (bottom vs top quintile; sorted by |diff| / scale, top 12):", "", "| covariate | bottom | top | diff | p |", "|---|---|---|---|---|"]
        cl = sorted(r["clan"].items(), key=lambda kv: kv[1]["p"])[:12]
        L += [f"| {k} | {v['bottom']:.3f} | {v['top']:.3f} | {v['diff']:+.3f} | {v['p']:.3g} |" for k, v in cl] + [""]
    L += ["## Estimator agreement (5-fold agency cross-fit, full sample)", ""]
    for w in ("main", "alt"):
        g = res[f"agreement_{w}"]
        L += [f"- {w}: Spearman(E1, E2) = {g['spearman_E1_E2']:.3f}; cross-fitted GATES diff E1 {fmt(g['E1_crossfit_gates']['diff'])} pp "
              f"(p {g['E1_crossfit_gates']['p']:.3g}), E2 {fmt(g['E2_crossfit_gates']['diff'])} pp (p {g['E2_crossfit_gates']['p']:.3g}); "
              f"E2 {g['E2']['trees']} trees, {g['E2']['rows']:,} monthly cells, min leaf {g['E2']['min_leaf_rows']} rows, {g['E2_seconds']} s"]
    pt = res["pretrend_main"]
    L += ["", f"Pre-trend diagnostic (main, {pt['n_units']:,} units): slope of the treated − control gap by E1 quintile (pp/month) "
          + ", ".join(f"{x:.3f}" for x in pt["slope_by_quintile_pp_per_month"]) + f"; Wald χ²(4) = {pt['wald']:.2f}, p = {pt['p']:.3g}.",
          f"Cross-window check (score from main, GATES on alt_late, N = {res['cross_window']['N']:,}, {res['cross_window']['overlap_with_main']:,} "
          f"also in main): top − bottom {fmt(res['cross_window']['diff'])} pp (p {res['cross_window']['p']:.3g}).", ""]
    L += ["## E3 linear BLP (main; standardised covariates; agency-clustered; 10 smallest p)", "", "| covariate | b (pp per SD) | se | p |", "|---|---|---|---|"]
    co = sorted(res["E3_main"]["coef"].items(), key=lambda kv: kv[1]["p"])[:10]
    L += [f"| {k} | {fmt(v['b'])} | {fmt(v['se'])} | {v['p']:.3g} |" for k, v in co] + [""]
    L += ["## Robustness (E1, same procedure; none decides H1–H4)", "", "| spec | N | valid splits | β₂ | p | GATES diff (pp) | p |", "|---|---|---|---|---|---|---|"]
    for k in ("R1_purchase", "R2_m1", "R2_m5", "R3_weighted", "R4_pre_to_jul", "R5_no_stimulus", "R7_drop_zero_discount", "R8_alt_local"):
        r = res[k]
        L.append(f"| {k} | {r['N']:,} | {r['valid_splits']} | {fmt(r['blp_b2']['est'], 0)} | {r['blp_b2']['p']:.3g} | {fmt(r['gates_diff']['est'])} | {r['gates_diff']['p']:.3g} |")
    L += ["", "R6 (control 2M–3M, group level): " + "; ".join(
        f"{w}: N = {v['N']:,}, central {fmt(v['central'])} pp, local {fmt(v['local'])} pp, diff {fmt(v['diff'])} (p {v['p']:.3g})"
        for w, v in res["R6_group_2M3M"].items()), "", f"Run time {res['seconds']} s."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
