"""Paper 3, step 4 of the locked pre-analysis plan (project doc claude/p3-pre-analysis-plan.md, section 7): who should keep
discretion up to 500k (D = 1, the post-reform status quo) and who should revert to 100k (D = 0)?

  python src/p3_policy.py --permute    pipeline test (covariate rows shuffled within government level)
  python src/p3_policy.py              real run   [--quick: code test only]

Reward of reverting unit u: Gamma_u(c) = -tau_u * B_u - c * N_u (THB per year), with tau_u the unit DiD effect of
discretion on the recorded discount, B_u its treated-band budget per year, N_u its treated-band purchases per year
(both Oct 2014 - Sep 2015) and c the admin cost per extra competitive procedure.
Policy value of an assignment pi (1 = revert): V(pi) = mean_u pi_u Gamma_u (THB / unit / year, relative to all-500k).
Rules: P0 all 500k; P1 all 100k; P2 policy tree depth 2 (primary) and 3; P3 local -> 100k; P4 revert the top-k% units
by E1 score (k chosen in the training folds). Learned rules are cross-fitted by agency (5 folds); CIs by agency-cluster
bootstrap of the out-of-fold assignments (1,000). H4: V(P2 depth 2) - max(V(P0), V(P1)) > 0 at c = 10,000.
Outputs: reports/p3_policy[_permuted].md / .json (aggregates only; tree leaves reported only with >= 10 units).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from bunching import REPORTS, log
from p3_cate import NJ, SEED, design, e1, load, permute_rows, pz, sample

COSTS = (0, 5_000, 10_000, 20_000, 50_000)
C_REF = 10_000
KGRID = np.arange(0, 101, 5)
LEAF = 20
B = 1000


def tree(depth, seed):
    from econml.policy import PolicyTree
    return PolicyTree(max_depth=depth, min_samples_leaf=LEAF, honest=False, random_state=seed)


def fit_rules(R, tau, gam, loc, g, seed, quick):
    """Out-of-fold assignments of every rule."""
    n = len(tau)
    A = {k: np.zeros(n) for k in ("P0", "P1", "P2_d2", "P2_d3", "P3", "P4")}
    A["P1"][:] = 1
    A["P3"] = loc.astype(float)
    Xr = R.to_numpy()
    kchosen = []
    for tr, te in GroupKFold(n_splits=5).split(Xr, tau, g):
        y = np.column_stack([np.zeros(len(tr)), gam[tr]])
        for d in (2, 3):
            A[f"P2_d{d}"][te] = tree(d, seed).fit(Xr[tr], y).predict(Xr[te])
        m = e1(seed)
        if quick:
            m.set_params(n_estimators=100)
        m.fit(Xr[tr], tau[tr])
        oob, s_te = m.oob_predict(Xr[tr]).ravel(), m.predict(Xr[te]).ravel()
        best, kb = -np.inf, 0
        for k in KGRID:           # revert the k% most negative predicted effects
            cut = np.percentile(oob, k) if k > 0 else -np.inf
            v = np.mean((oob <= cut) * gam[tr]) if k > 0 else 0.0
            if v > best + 1e-9:
                best, kb = v, k
        kchosen.append(int(kb))
        cut = np.percentile(oob, kb) if kb > 0 else -np.inf
        A["P4"][te] = (s_te <= cut) if kb > 0 else 0.0
    return A, kchosen


def values(A, gam, Bu, g, rng):
    ag = np.unique(g)
    idx = {k: np.nonzero(g == k)[0] for k in ag}
    picks = [np.concatenate([idx[k] for k in rng.choice(ag, len(ag))]) for _ in range(B)]
    out = {}
    for k, pi in A.items():
        v = np.mean(pi * gam)
        bs = np.array([np.mean(pi[p] * gam[p]) for p in picks])
        out[k] = {"V": float(v), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                  "share_revert": float(pi.mean()), "total_mTHB": float(np.sum(pi * gam) / 1e6),
                  "pct_of_treated_budget": float(100 * np.sum(pi * gam) / np.sum(Bu))}
    d = np.mean(A["P2_d2"] * gam) - max(0.0, np.mean(gam))
    bs = np.array([np.mean(A["P2_d2"][p] * gam[p]) - max(0.0, np.mean(gam[p])) for p in picks])
    out["H4"] = {"diff": float(d), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                 "se": float(bs.std(ddof=1)), "p": pz(d, bs.std(ddof=1))}
    return out


def describe(R, tau, gam, Bu, depth, seed):
    """Full-sample tree at the reference cost: splits and leaves (>= 10 units)."""
    t = tree(depth, seed).fit(R.to_numpy(), np.column_stack([np.zeros(len(tau)), gam]))
    tr = t.tree_model_.tree_
    leaf = t.apply(R.to_numpy()) if hasattr(t, "apply") else None
    act = t.predict(R.to_numpy())
    nodes = []

    def walk(i, path):
        if tr.children_left[i] < 0:
            m = leaf == i
            nodes.append({"rule": " and ".join(path) or "(all)", "units": int(m.sum()),
                          "action": "revert to 100k" if act[m].mean() > 0.5 else "keep 500k",
                          "mean_tau_pp": float(100 * tau[m].mean()) if m.sum() >= 10 else None,
                          "mean_gamma_THB": float(gam[m].mean()) if m.sum() >= 10 else None})
            return
        f, th = R.columns[tr.feature[i]], tr.threshold[i]
        walk(tr.children_left[i], path + [f"{f} ≤ {th:.4g}"])
        walk(tr.children_right[i], path + [f"{f} > {th:.4g}"])
    walk(0, [])
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    T, X, cov = load()
    if a.permute:
        X = permute_rows(X, cov, SEED + 999)
    tag = "_permuted" if a.permute else ""
    res = {"permuted": a.permute, "quick": a.quick, "costs": COSTS, "c_ref": C_REF, "leaf_min_units": LEAF}
    for w in ("main", "alt"):
        d, g = sample(T, X, "O1", w, 3)
        d = d[d.unit.map(X.B_u).notna() & d.unit.map(X.N_u).notna()].reset_index(drop=True)
        g = X.loc[d.unit, "agency"].to_numpy()
        R, _ = design(X, d.unit.tolist(), cov)
        tau = d.tau.to_numpy()
        Bu, Nu = d.unit.map(X.B_u).to_numpy(float), d.unit.map(X.N_u).to_numpy(float)
        loc = d.unit.map(X.local).fillna(0).to_numpy()
        rw = {"N": int(len(d)), "sum_B_mTHB": float(Bu.sum() / 1e6), "sum_N": float(Nu.sum()),
              "c_star_uniform": float(np.mean(-tau * Bu) / np.mean(Nu)), "by_cost": {}}
        for c in COSTS:
            gam = -tau * Bu - c * Nu
            A, kc = fit_rules(R, tau, gam, loc, g, SEED, a.quick)
            v = values(A, gam, Bu, g, np.random.default_rng(SEED + c))
            v["P4_k_chosen"] = kc
            rw["by_cost"][str(c)] = v
            log(f"p3_policy {w} c={c}: V(P1)={v['P1']['V']:.0f} V(P2)={v['P2_d2']['V']:.0f} H4 diff={v['H4']['diff']:.0f} "
                f"(p {v['H4']['p']:.3g})")
        gam = -tau * Bu - C_REF * Nu
        rw["tree_d2"] = describe(R, tau, gam, Bu, 2, SEED)
        res[w] = rw
    res["seconds"] = round(time.time() - t0)
    (REPORTS / f"p3_policy{tag}.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    (REPORTS / f"p3_policy{tag}.md").write_text(report(res), encoding="utf-8")
    print(report(res), flush=True)


def report(res):
    t = "PERMUTED PIPELINE TEST — not a result" if res["permuted"] else "real run"
    if res["quick"]:
        t += " — QUICK CODE TEST, not reportable"
    L = [f"# Paper 3 policy learning — {t}", "", "Plan: project doc `claude/p3-pre-analysis-plan.md` §7. Values in THB per unit "
         "per year relative to keeping 500k everywhere (P0); out-of-fold (5 agency folds); 95% agency-bootstrap intervals.", ""]
    for w in ("main", "alt"):
        r = res[w]
        h = r["by_cost"][str(res["c_ref"])]["H4"]
        L += [f"## {w}: N = {r['N']:,} units; treated-band budget {r['sum_B_mTHB']:,.0f} m THB/yr; {r['sum_N']:,.0f} purchases/yr",
              f"Break-even cost for the uniform rules (P1 = P0): c* = {r['c_star_uniform']:,.0f} THB per procedure.",
              f"**H4 (c = {res['c_ref']:,}):** V(P2 depth 2) − max(V(P0), V(P1)) = {h['diff']:,.0f} THB/unit/yr "
              f"[{h['lo']:,.0f}, {h['hi']:,.0f}], p = {h['p']:.3g}", "",
              "| c (THB) | " + " | ".join(("P1 all 100k", "P2 tree d2", "P2 tree d3", "P3 local→100k", "P4 top-k%")) + " | P2 d2 share reverting | P4 k (folds) |",
              "|---" * 8 + "|"]
        for c in res["costs"]:
            v = r["by_cost"][str(c)]
            cells = [f"{v[k]['V']:,.0f} [{v[k]['lo']:,.0f}, {v[k]['hi']:,.0f}]" for k in ("P1", "P2_d2", "P2_d3", "P3", "P4")]
            L.append(f"| {c:,} | " + " | ".join(cells) + f" | {v['P2_d2']['share_revert']:.0%} | {v['P4_k_chosen']} |")
        L += ["", f"Depth-2 tree on the full {w} sample at c = {res['c_ref']:,} (descriptive; leaves with < 10 units not described):", "",
              "| leaf rule | units | action | mean τ̂ (pp) | mean Γ (THB) |", "|---|---|---|---|---|"]
        for n in r["tree_d2"]:
            mt = "–" if n["mean_tau_pp"] is None else f"{n['mean_tau_pp']:.2f}"
            mg = "–" if n["mean_gamma_THB"] is None else f"{n['mean_gamma_THB']:,.0f}"
            L.append(f"| {n['rule']} | {n['units']:,} | {n['action']} | {mt} | {mg} |")
        L.append("")
    L.append(f"Run time {res['seconds']} s.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
