"""
เทียบโมเดลตรวจจับความผิดปกติกับ Isolation Forest บนเมทริกซ์ฟีเจอร์ชุดเดียวกัน (data/processed/model_{year}_le500k/X.npy)

- ECOD (Li et al. 2022): ความน่าจะเป็นหางของแต่ละฟีเจอร์จาก empirical CDF → รวม −log; อธิบายได้ในตัว (คะแนนแยกตามฟีเจอร์)
- HBOS (Goldstein & Dengel 2012): histogram รายฟีเจอร์ → รวม −log density
- ENS: ค่าเฉลี่ยอันดับ (rank) ของ IF และ ECOD
ประเมินด้วย injection test ชุดเดียวกับ model_iforest.py, การซ้อนทับ top1% กับ IF, Spearman, และฟีเจอร์ที่อธิบาย top1% ของ ECOD

  python src\\model_compare.py --year 2568
  python src\\model_compare.py --year 2568 --test      (ทดสอบเร็ว: ใช้ 300k แถวแรก)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model_iforest import FNAMES, PROCESSED, REPORTS, SCOPE

FIT_SAMPLE = 1_000_000
CHUNK = 500_000
HBOS_BINS = 50
EPS = 1e-6


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class ECOD:
    def fit(self, Xf):
        self.sorted = [np.sort(Xf[:, j]) for j in range(Xf.shape[1])]
        self.n = Xf.shape[0]
        sk = ((Xf - Xf.mean(0)) ** 3).mean(0) / (Xf.std(0) ** 3 + 1e-12)
        self.skew_pos = sk > 0
        return self

    def feature_scores(self, X):
        """คืน (left, right) −log tail prob ต่อฟีเจอร์"""
        L = np.empty(X.shape, dtype=np.float32)
        R = np.empty(X.shape, dtype=np.float32)
        for j, s in enumerate(self.sorted):
            le = np.searchsorted(s, X[:, j], side="right") / self.n           # P(X <= x)
            ge = 1 - np.searchsorted(s, X[:, j], side="left") / self.n        # P(X >= x)
            L[:, j] = -np.log(np.clip(le, EPS, 1))
            R[:, j] = -np.log(np.clip(ge, EPS, 1))
        return L, R

    def score(self, X):
        L, R = self.feature_scores(X)
        A = np.where(self.skew_pos[None, :], R, L)
        return np.maximum.reduce([L.sum(1), R.sum(1), A.sum(1)])

    def explain(self, X):
        L, R = self.feature_scores(X)
        return np.maximum(L, R)  # ความผิดปกติรายฟีเจอร์ (หางที่สุดขั้วกว่า)


class HBOS:
    def fit(self, Xf):
        self.edges, self.dens = [], []
        for j in range(Xf.shape[1]):
            x = Xf[:, j]
            lo, hi = float(x.min()), float(x.max())
            if hi <= lo:
                hi = lo + 1.0
            e = np.linspace(lo, hi, HBOS_BINS + 1)
            h, _ = np.histogram(x, bins=e)
            d = h / h.max()
            self.edges.append(e); self.dens.append(np.clip(d, EPS, 1))
        return self

    def score(self, X):
        s = np.zeros(len(X), dtype=np.float64)
        for j, (e, d) in enumerate(zip(self.edges, self.dens)):
            idx = np.clip(np.searchsorted(e, X[:, j], side="right") - 1, 0, HBOS_BINS - 1)
            out = (X[:, j] < e[0]) | (X[:, j] > e[-1])
            s += np.where(out, -np.log(EPS), -np.log(d[idx]))
        return s


def rank01(a):
    r = np.empty(len(a), dtype=np.float64)
    r[np.argsort(a, kind="stable")] = np.arange(len(a))
    return r / max(len(a) - 1, 1)


SCEN = {
    "ซอยสัญญาใต้เพดาน": {"gap_to_500k": 0.004, "log_split_n_day": np.log1p(6), "log_split_sum_day": np.log1p(2_990_000),
                         "log_split_n_month": np.log1p(8), "is_round_1000": 0},
    "ราคาสูงกว่ากลุ่ม+ไม่มีส่วนลด": {"price_z_peer": 4.0, "price_eq_ref": 1, "ratio_price_ref": 1.0, "ratio_price_budget": 1.0,
                                      "is_round_1000": 1},
    "ผูกขาดผู้ขายในหน่วยงาน": {"pair_value_share": 0.9, "unit_hhi": 0.82, "log_pair_n": np.log1p(150),
                              "log_vendor_units": np.log1p(1), "log_vendor_prov": np.log1p(1)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2568")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    d = PROCESSED / f"model_{a.year}_{SCOPE}"
    X = np.load(d / "X.npy", mmap_mode="r")
    n = min(len(X), 300_000) if a.test else len(X)
    sc_if = np.load(d / "score_seed0.npy")[:n]
    t0 = time.time()
    rng = np.random.default_rng(11)
    fit_rows = np.sort(rng.choice(n, size=min(FIT_SAMPLE, n), replace=False))
    Xf = np.asarray(X[fit_rows], dtype=np.float64)
    ecod, hbos = ECOD().fit(Xf), HBOS().fit(Xf)
    del Xf
    log(f"fit {time.time() - t0:.0f}s")
    sc_e = np.empty(n); sc_h = np.empty(n)
    for lo in range(0, n, CHUNK):
        blk = np.asarray(X[lo:min(lo + CHUNK, n)], dtype=np.float64)
        sc_e[lo:lo + len(blk)] = ecod.score(blk); sc_h[lo:lo + len(blk)] = hbos.score(blk)
    log(f"score {time.time() - t0:.0f}s")
    r_if, r_e, r_h = rank01(sc_if), rank01(sc_e), rank01(sc_h)
    sc_ens = (r_if + r_e) / 2
    scores = {"IF": sc_if, "ECOD": sc_e, "HBOS": sc_h, "ENS(IF+ECOD)": sc_ens}
    ranks = {"IF": r_if, "ECOD": r_e, "HBOS": r_h, "ENS(IF+ECOD)": rank01(sc_ens)}
    k1 = int(0.01 * n)
    tops = {m: set(np.argsort(-s)[:k1]) for m, s in scores.items()}
    res = {"year": a.year, "n": n, "test": a.test, "overlap_top1pct_jaccard": {}, "spearman": {}, "injection": {}}
    ms = list(scores)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            A, B = tops[ms[i]], tops[ms[j]]
            res["overlap_top1pct_jaccard"][f"{ms[i]}~{ms[j]}"] = len(A & B) / len(A | B)
            res["spearman"][f"{ms[i]}~{ms[j]}"] = float(np.corrcoef(ranks[ms[i]], ranks[ms[j]])[0, 1])
    # injection: แถวฐานคือแถวปกติของ IF (score ต่ำกว่า median) เหมือน model_iforest.py
    import joblib
    m_if = joblib.load(d / "iforest_seed0.joblib")
    base_pool = np.where(sc_if < np.median(sc_if))[0]
    jidx = {f: i for i, f in enumerate(FNAMES)}
    sorted_s = {m: np.sort(s) for m, s in scores.items() if m != "ENS(IF+ECOD)"}
    for name, mods in SCEN.items():
        rows = np.sort(np.random.default_rng(7).choice(base_pool, 500, replace=False))
        Xi = np.asarray(X[rows], dtype=np.float64).copy()
        for f, v in mods.items():
            Xi[:, jidx[f]] = v
        si = {"IF": -m_if.score_samples(Xi.astype(np.float32)), "ECOD": ecod.score(Xi), "HBOS": hbos.score(Xi)}
        pct = {m: np.searchsorted(sorted_s[m], si[m]) / n for m in si}
        pct["ENS(IF+ECOD)"] = (pct["IF"] + pct["ECOD"]) / 2
        ens_sorted = np.sort(ranks["ENS(IF+ECOD)"])  # เทียบ ENS ด้วยเปอร์เซ็นไทล์เฉลี่ย (ประมาณ)
        res["injection"][name] = {}
        for m, p in pct.items():
            res["injection"][name][m] = {"recall_top1pct": float((p >= 0.99).mean()), "recall_top5pct": float((p >= 0.95).mean()),
                                         "median_percentile": float(np.median(p))}
    # อธิบาย ECOD: ฟีเจอร์ที่สุดขั้วที่สุดของแถวใน top1%
    top_e = np.sort(np.argsort(-sc_e)[:k1])
    ex = ecod.explain(np.asarray(X[top_e], dtype=np.float64))
    top3 = np.argsort(-ex, axis=1)[:, :3]
    cnt = pd.Series(np.array(FNAMES)[top3.ravel()]).value_counts() / len(top_e)
    res["ecod_top1pct_driver_share"] = cnt.round(4).to_dict()
    tag = "_test" if a.test else ""
    out_json = REPORTS / f"model_compare_{a.year}{tag}.json"
    json.dump(res, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not a.test:
        np.save(d / "score_ecod.npy", sc_e.astype(np.float32)); np.save(d / "score_hbos.npy", sc_h.astype(np.float32))
    # รายงาน
    L = [f"# เทียบโมเดล ปีงบ {a.year} (scope {SCOPE}, n = {n:,}{', ทดสอบ' if a.test else ''})", "",
         "## Injection test (500 แถวต่อสถานการณ์)", "", "| สถานการณ์ | โมเดล | recall@top1% | recall@top5% | median percentile |", "|---|---|---|---|---|"]
    for name, rr in res["injection"].items():
        for m, v in rr.items():
            L.append(f"| {name} | {m} | {v['recall_top1pct']:.2f} | {v['recall_top5pct']:.2f} | {v['median_percentile']:.3f} |")
    L += ["", "## ความสอดคล้องระหว่างโมเดล", "", "| คู่ | Jaccard top1% | Spearman |", "|---|---|---|"]
    for k in res["spearman"]:
        L.append(f"| {k} | {res['overlap_top1pct_jaccard'][k]:.3f} | {res['spearman'][k]:.3f} |")
    L += ["", "## ECOD: ฟีเจอร์ที่ติด 3 อันดับสุดขั้วของแถวใน top1% (สัดส่วนแถว)", ""]
    L += [f"- {f}: {v:.0%}" for f, v in list(res["ecod_top1pct_driver_share"].items())[:10]]
    (REPORTS / f"model_compare_{a.year}{tag}.md").write_text("\n".join(L), encoding="utf-8")
    log(f"→ {out_json.name} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
