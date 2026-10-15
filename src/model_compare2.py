"""
เทียบโมเดลรอบ 2: Autoencoder (MLP) และ LOF (novelty) + ensemble แบบ max-rank เทียบกับ IF / ECOD / HBOS
ใช้ X.npy เดียวกับ model_iforest.py; injection test ชุดเดียวกับ model_compare.py

  python src\\model_compare2.py --year 2568
  python src\\model_compare2.py --year 2568 --test     (ทดสอบเร็ว: 200k แถวแรก, ตัวอย่างฝึกเล็ก)
LOF ใช้เวลานาน (ค้นหาเพื่อนบ้านทุกแถว) — เหมาะรันข้ามคืน
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from model_iforest import FNAMES, PROCESSED, REPORTS, SCOPE
from model_compare import ECOD, HBOS, SCEN, rank01

CHUNK = 200_000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2568")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--lof-fit", type=int, default=100_000)
    ap.add_argument("--ae-fit", type=int, default=400_000)
    a = ap.parse_args()
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import RobustScaler

    d = PROCESSED / f"model_{a.year}_{SCOPE}"
    X = np.load(d / "X.npy", mmap_mode="r")
    n = min(len(X), 200_000) if a.test else len(X)
    lof_fit = 20_000 if a.test else a.lof_fit
    ae_fit = 50_000 if a.test else a.ae_fit
    t0 = time.time()
    rng = np.random.default_rng(23)
    sc_if = np.load(d / "score_seed0.npy")[:n]
    # ECOD/HBOS: ใช้คะแนนที่บันทึกไว้ถ้ามี ไม่งั้นคำนวณใหม่
    if not a.test and (d / "score_ecod.npy").exists():
        sc_e = np.load(d / "score_ecod.npy").astype(float); sc_h = np.load(d / "score_hbos.npy").astype(float)
        fit_rows = np.sort(np.random.default_rng(11).choice(n, size=min(1_000_000, n), replace=False))
        Xf = np.asarray(X[fit_rows], dtype=np.float64); ecod, hbos = ECOD().fit(Xf), HBOS().fit(Xf); del Xf
    else:
        fit_rows = np.sort(np.random.default_rng(11).choice(n, size=min(1_000_000, n), replace=False))
        Xf = np.asarray(X[fit_rows], dtype=np.float64); ecod, hbos = ECOD().fit(Xf), HBOS().fit(Xf); del Xf
        sc_e = np.concatenate([ecod.score(np.asarray(X[i:min(i + CHUNK, n)], dtype=np.float64)) for i in range(0, n, CHUNK)])
        sc_h = np.concatenate([hbos.score(np.asarray(X[i:min(i + CHUNK, n)], dtype=np.float64)) for i in range(0, n, CHUNK)])
    # scaler (robust: ฟีเจอร์มีหางยาว)
    sub = np.asarray(X[np.sort(rng.choice(n, size=min(ae_fit, n), replace=False))], dtype=np.float64)
    scaler = RobustScaler(quantile_range=(5, 95)).fit(sub)
    clipz = lambda Z: np.clip(Z, -20, 20)
    # ---- Autoencoder
    ae = MLPRegressor(hidden_layer_sizes=(32, 8, 32), activation="relu", max_iter=60, batch_size=1024,
                      early_stopping=True, n_iter_no_change=5, random_state=0)
    Zs = clipz(scaler.transform(sub))
    ae.fit(Zs, Zs)
    log(f"AE fit {time.time() - t0:.0f}s, iters {ae.n_iter_}")
    def ae_score(B):
        Z = clipz(scaler.transform(B)); return ((ae.predict(Z) - Z) ** 2).mean(1)
    sc_ae = np.concatenate([ae_score(np.asarray(X[i:min(i + CHUNK, n)], dtype=np.float64)) for i in range(0, n, CHUNK)])
    log(f"AE score {time.time() - t0:.0f}s")
    # ---- LOF (novelty): fit บนตัวอย่าง แล้วให้คะแนนทุกแถว
    lsub = clipz(scaler.transform(np.asarray(X[np.sort(rng.choice(n, size=min(lof_fit, n), replace=False))], dtype=np.float64)))
    lof = LocalOutlierFactor(n_neighbors=50, novelty=True, n_jobs=-1).fit(lsub)
    sc_lof = np.empty(n)
    for i in range(0, n, CHUNK):
        sc_lof[i:min(i + CHUNK, n)] = -lof.score_samples(clipz(scaler.transform(np.asarray(X[i:min(i + CHUNK, n)], dtype=np.float64))))
        log(f"LOF {min(i + CHUNK, n):,}/{n:,} ({time.time() - t0:.0f}s)")
    # ---- ensembles
    R = {"IF": rank01(sc_if), "ECOD": rank01(sc_e), "HBOS": rank01(sc_h), "AE": rank01(sc_ae), "LOF": rank01(sc_lof)}
    R["MAX(IF,ECOD,HBOS)"] = np.maximum.reduce([R["IF"], R["ECOD"], R["HBOS"]])
    R["MAX(all5)"] = np.maximum.reduce([R["IF"], R["ECOD"], R["HBOS"], R["AE"], R["LOF"]])
    raw = {"IF": sc_if, "ECOD": sc_e, "HBOS": sc_h, "AE": sc_ae, "LOF": sc_lof}
    k1 = int(0.01 * n)
    tops = {m: set(np.argsort(-r, kind="stable")[:k1]) for m, r in R.items()}
    res = {"year": a.year, "n": n, "test": a.test, "jaccard_top1pct": {}, "injection": {}}
    ms = list(R)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            A_, B_ = tops[ms[i]], tops[ms[j]]
            res["jaccard_top1pct"][f"{ms[i]}~{ms[j]}"] = len(A_ & B_) / len(A_ | B_)
    # injection: เปอร์เซ็นไทล์ของแถวที่แทรก เทียบการกระจายคะแนนจริงของแต่ละโมเดล
    m_if = joblib.load(d / "iforest_seed0.joblib")
    base_pool = np.where(sc_if < np.median(sc_if))[0]
    jidx = {f: i for i, f in enumerate(FNAMES)}
    srt = {m: np.sort(s) for m, s in raw.items()}
    for name, mods in SCEN.items():
        rows = np.sort(np.random.default_rng(7).choice(base_pool, 500, replace=False))
        Xi = np.asarray(X[rows], dtype=np.float64).copy()
        for f, v in mods.items():
            Xi[:, jidx[f]] = v
        si = {"IF": -m_if.score_samples(Xi.astype(np.float32)), "ECOD": ecod.score(Xi), "HBOS": hbos.score(Xi), "AE": ae_score(Xi),
              "LOF": -lof.score_samples(clipz(scaler.transform(Xi)))}
        pct = {m: np.searchsorted(srt[m], si[m]) / n for m in si}
        pct["MAX(IF,ECOD,HBOS)"] = np.maximum.reduce([pct["IF"], pct["ECOD"], pct["HBOS"]])
        pct["MAX(all5)"] = np.maximum.reduce([pct[m] for m in raw])
        # max-rank ยกระดับทุกแถว → เกณฑ์ top1% ของ ensemble ต้องคำนวณจากการกระจายของ ensemble จริง
        thr1 = {m: np.quantile(R[m], 0.99) for m in R}; thr5 = {m: np.quantile(R[m], 0.95) for m in R}
        res["injection"][name] = {m: {"recall_top1pct": float((p >= thr1[m]).mean()), "recall_top5pct": float((p >= thr5[m]).mean()),
                                      "median_percentile": float(np.median(p))} for m, p in pct.items()}
    # การประเมินด้วยข้อมูลจริง: lift ของ red flag ตามกฎ (ids.parquet) ใน top1% ของแต่ละโมเดล
    ids = pd.read_parquet(d / "ids.parquet")
    rcols = [c for c in ids.columns if c.startswith("r_")]
    ids = ids.iloc[:n]
    base = ids[rcols].mean()
    res["rule_lift_top1pct"] = {}
    for m, r in R.items():
        top = np.argsort(-r, kind="stable")[:k1]
        res["rule_lift_top1pct"][m] = {c: float(ids[c].values[top].mean() / max(base[c], 1e-9)) for c in rcols}
    res["rule_base_rate"] = base.round(4).to_dict()
    tag = "_test" if a.test else ""
    json.dump(res, open(REPORTS / f"model_compare2_{a.year}{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not a.test:
        np.save(d / "score_ae.npy", sc_ae.astype(np.float32)); np.save(d / "score_lof.npy", sc_lof.astype(np.float32))
    L = [f"# เทียบโมเดลรอบ 2 ปีงบ {a.year} (n = {n:,}{', ทดสอบ' if a.test else ''}; LOF fit {lof_fit:,}, AE fit {ae_fit:,})", "",
         "| สถานการณ์ | โมเดล | recall@top1% | recall@top5% | median pct |", "|---|---|---|---|---|"]
    for name, rr in res["injection"].items():
        for m, v in rr.items():
            L.append(f"| {name} | {m} | {v['recall_top1pct']:.2f} | {v['recall_top5pct']:.2f} | {v['median_percentile']:.3f} |")
    L += ["", "## Lift ของ red flag จริง (ตามกฎ) ใน top1% ของแต่ละโมเดล (เท่าของอัตราพื้นฐาน)", "",
          "| โมเดล | " + " | ".join(rcols) + " |", "|---" * (len(rcols) + 1) + "|"]
    for m, v in res["rule_lift_top1pct"].items():
        L.append(f"| {m} | " + " | ".join(f"{v[c]:.1f}" for c in rcols) + " |")
    L.append("| อัตราพื้นฐาน | " + " | ".join(f"{base[c]:.3f}" for c in rcols) + " |")
    L += ["", "หมายเหตุ: injection แก้ฟีเจอร์บางตัวโดยไม่แก้ตัวที่สัมพันธ์กัน → โมเดลที่ดูการกระจายร่วม (AE, LOF) จับความไม่สอดคล้องได้ง่าย จึงได้ recall สูงเกินจริง; ใช้ lift ของ red flag จริงประกอบ"]
    L += ["", "| คู่ | Jaccard top1% |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in res["jaccard_top1pct"].items()]
    (REPORTS / f"model_compare2_{a.year}{tag}.md").write_text("\n".join(L), encoding="utf-8")
    log(f"→ model_compare2_{a.year}{tag} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
