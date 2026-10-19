"""Faster, numerically equivalent replacements for two pure-Python loops in causalfe 0.3.2 (CFFE, Gavrilova et al.).

causalfe's split search tries every threshold of every feature with a fresh boolean mask (O(p n^2) per node) and its
fixed-effects demeaning loops over every group with a mask (O(G n) per sweep). Both are too slow for 2,000-10,000
units x 4 cells. The functions below compute the same quantities with sorting / cumulative sums and bincount:
same thresholds (midpoints of sorted unique values), same score (nL nR / n^2 (tauL - tauR)^2), same tie-breaking
(first maximum by feature, then threshold), same iterative two-way demeaning. Results agree up to floating-point
summation order; `check()` compares them with the originals on random data.

    import p3_fastcffe; p3_fastcffe.install()     # before fitting causalfe.CFFEForest
"""
import numpy as np


def fe_residualize(Y, D, unit, time, iters=5):
    Y = np.asarray(Y, dtype=float).copy()
    D = np.asarray(D, dtype=float).copy()
    _, iu = np.unique(np.asarray(unit), return_inverse=True)
    _, it = np.unique(np.asarray(time), return_inverse=True)
    cu, ct = np.bincount(iu), np.bincount(it)
    for _ in range(iters):
        for idx, cnt in ((iu, cu), (it, ct)):
            Y -= (np.bincount(idx, Y, minlength=len(cnt)) / cnt)[idx]
            D -= (np.bincount(idx, D, minlength=len(cnt)) / cnt)[idx]
    return Y, D


TOL = 1e-12   # relative; absorbs summation-order noise so exact ties are broken as in the original


def _tau(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b < 1e-10, 0.0, a / b)


def find_best_split(X, Y_tilde, D_tilde, min_leaf):
    n, p = X.shape
    a_all, b_all = D_tilde * Y_tilde, D_tilde ** 2
    best_score, best_feature, best_threshold = -1.0, -1, 0.0
    for j in range(p):
        x = X[:, j]
        o = np.argsort(x, kind="stable")
        xs = x[o]
        cut = np.nonzero(xs[:-1] != xs[1:])[0]          # left = rows 0..i
        if len(cut) == 0:
            continue
        nL = cut + 1
        nR = n - nL
        ok = (nL >= min_leaf) & (nR >= min_leaf)
        if not ok.any():
            continue
        ca, cb = np.cumsum(a_all[o]), np.cumsum(b_all[o])
        AL, BL = ca[cut], cb[cut]
        AR, BR = ca[-1] - AL, cb[-1] - BL
        score = (nL * nR / (n * n)) * (_tau(AL, BL) - _tau(AR, BR)) ** 2
        score = np.where(ok, score, -np.inf)
        top = score.max()
        k = int(np.argmax(score >= top - TOL * abs(top)))   # first (near-)maximum, as the original's strict ">" scan
        if score[k] > best_score + TOL * abs(best_score):   # exact ties across features keep the earlier feature
            best_score, best_feature = float(score[k]), j
            best_threshold = (xs[cut[k]] + xs[cut[k] + 1]) / 2
    return best_feature, best_threshold, best_score


def fit(self, X, Y, D, unit, time):
    """CFFEForest.fit (causalfe 0.3.2) with the cluster-robust ATE variance computed by bincount instead of a loop
    over clusters with masks; tree building, ATE and recentering are unchanged."""
    import causalfe.forest.cffe_forest as cf
    X, Y, D = np.asarray(X), np.asarray(Y), np.asarray(D, dtype=float)
    unit, time = np.asarray(unit), np.asarray(time)
    self._rng = np.random.default_rng(self.seed)
    self.trees = []
    self._n_samples, self._n_features = X.shape
    unique_units, ui = np.unique(unit, return_inverse=True)
    self._n_units = len(unique_units)
    n_subsample = max(1, int(self._n_units * self.subsample_ratio))
    for _ in range(self.n_trees):
        sampled = self._rng.choice(unique_units, size=n_subsample, replace=True)
        idx = np.where(np.isin(unit, sampled))[0]
        tree_seed = int(self._rng.integers(0, 2**31 - 1))
        t = cf.CFFETree(max_depth=self.max_depth, min_leaf=self.min_leaf, honest=self.honest, seed=tree_seed)
        t.fit(X[idx], Y[idx], D[idx], unit[idx], time[idx])
        self.trees.append(t)
    self._is_fitted = True
    Yt, Dt = fe_residualize(Y, D, unit, time, iters=10)
    self._ate = float(cf.estimate_tau(Yt, Dt))
    resid = Yt - self._ate * Dt
    denom = float(np.sum(Dt ** 2))
    G, n, k = self._n_units, len(Y), 1
    if denom > 1e-12:
        sg = np.bincount(ui, Dt * resid, minlength=G)
        corr = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else 1.0
        self._ate_var = corr * float(np.sum(sg * sg)) / denom ** 2
        self._ate_se = float(np.sqrt(self._ate_var))
    else:
        self._ate_var = self._ate_se = np.nan
    self._n_clusters = G
    self._train_X, self._train_D = X, D
    self._recenter_offset = (self._ate - self._raw_predict(X).mean()) if self.recenter else 0.0
    return self


def install():
    import causalfe.forest.cffe_forest as cf
    import causalfe.forest.tree as tr
    tr.fe_residualize = fe_residualize
    tr.find_best_split = find_best_split
    cf.fe_residualize = fe_residualize
    cf.CFFEForest.fit = fit


def check(seed=0):
    """Compare with the original causalfe functions on random data (returns the largest differences)."""
    import causalfe.forest.residuals as res
    import causalfe.forest.splitting as sp
    rng = np.random.default_rng(seed)
    U = 150
    unit = np.repeat(np.arange(U), 4) * 2 + np.tile([0, 0, 1, 1], U)
    time = np.repeat(np.arange(U), 4) * 2 + np.tile([0, 1, 0, 1], U)
    Y, D = rng.normal(size=4 * U), np.tile([0.0, 1.0, 0.0, 0.0], U)
    X = np.round(rng.normal(size=(U, 6)), 1)[np.repeat(np.arange(U), 4)]
    y1, d1 = res.fe_residualize(Y, D, unit, time)
    y2, d2 = fe_residualize(Y, D, unit, time)
    s1, s2 = sp.find_best_split(X, y1, d1, 20), find_best_split(X, y1, d1, 20)
    return {"resid_y": float(np.abs(y1 - y2).max()), "resid_d": float(np.abs(d1 - d2).max()),
            "split_orig": s1, "split_fast": s2}


if __name__ == "__main__":
    print(check())
