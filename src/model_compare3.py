"""
model_compare3.py — เทียบโมเดลด้วย injection แบบสอดคล้อง (consistent / data-level injection)

ปัญหาของรอบ 1–2: injection เขียนทับฟีเจอร์บางตัว (เช่น gap_to_500k = 0.004) โดยไม่แก้ฟีเจอร์ที่คำนวณจากข้อมูลเดียวกัน
(price_z_peer, log_pair_n, unit_hhi, vendor_log_value ...) → แถวที่ฉีดมีฟีเจอร์ขัดกันเอง → AE/LOF จับ "ความไม่สอดคล้อง"
ไม่ใช่พฤติกรรม (lift ของ red flag การแบ่งจริงใน top1% ของ AE/LOF ต่ำเพียง 1.5–5.5 เท่า)

รอบนี้แก้ที่ระดับ "สัญญา" แล้วคำนวณฟีเจอร์ทุกตัวที่ได้รับผลใหม่ตามนิยามใน features.py
(peer z, หน่วยงาน, คู่ผู้ขาย–หน่วยงาน, วัน/เดือนเดียวกัน, ผู้ขาย — จาก features_YYYY.parquet)
  ซอย 2 ส่วน          : ซื้อมูลค่ารวม T ~ U(520k, 1M) จากผู้ขายเดิม วันเดียวกัน แบ่งเป็น 2 สัญญา (กรณีที่พบบ่อยที่สุด)
  ซอยหลายส่วนชิดเพดาน : T ~ U(1.5M, 3M) แบ่งเป็น k = ceil(T / 499,999) สัญญา (4–7)
  ราคาสูง+ไม่มีส่วนลด  : ราคา × U(1.5, 3) (≤ 499,000), ราคากลาง = งบประมาณ = ราคา (แถวฐานราคา ≤ 250k)
  ผูกขาดผู้ขาย         : ผู้ขายได้สัญญาที่เดิมเป็นของผู้ขายรายอื่นในหน่วยงานเดียวกัน จนส่วนแบ่งมูลค่า = U(0.6, 0.9)
                        (มูลค่า/จำนวนสัญญาของหน่วยงานคงเดิม; กระจายทั้งปี; หน่วยงาน 20–1,000 สัญญา, ส่วนแบ่งเดิม < 0.3)
  แถวฐาน: สัญญาจริงที่ไม่มี red flag ตามกฎเลย (rule_score = 0) — ไม่ขึ้นกับโมเดลใด
รัน injection แบบเดิม (เขียนทับฟีเจอร์, แถวฐานชุดเดียวกับรอบ 2) คู่กันเพื่อเทียบ
AE: max_iter 300 (รอบ 2 ชน 60 ทุกปี) → score_ae300.npy; LOF: ใช้คะแนนรอบ 2 (สุ่มตัวอย่างชุดเดิม ตรวจว่าตรงกันก่อนใช้)

  python src\\model_compare3.py --year 2568
  python src\\model_compare3.py --year 2568 --test      (ทดสอบเร็ว: 200k แถวแรก)
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from model_iforest import FNAMES, PROCESSED, REPORTS, SCOPE
from model_compare import ECOD, HBOS, SCEN, rank01

CHUNK = 200_000
CEIL = 500_000
N_CAND = 30_000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_base(year, sel: pd.DataFrame, mem="2GB"):
    """สถิติที่ต้องใช้คำนวณฟีเจอร์ใหม่ ของแถวฐานแต่ละแถว (ตามนิยามใน features.py)"""
    src = str(PROCESSED / f"features_{year}.parquet").replace("\\", "/")
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{(Path(tempfile.gettempdir()) / 'duckdb_egp').as_posix()}'")
    con.execute(f"SET threads=2; SET memory_limit='{mem}'")
    con.execute(f"""CREATE VIEW F AS SELECT project_id, unit_key, entity_key, project_type, method_cat, is_medical, agreed_price
                    FROM read_parquet('{src}')""")
    s = pd.DataFrame({"i": np.arange(len(sel)), "project_id": sel["project_id"].astype(str).values,
                      "unit_key": sel["unit_key"].values, "entity_key": sel["entity_key"].values})
    con.register("s", s)
    con.execute("CREATE TEMP TABLE uk AS SELECT DISTINCT unit_key FROM s")
    st = con.execute(f"""
    WITH sub AS (SELECT * FROM F WHERE unit_key IN (SELECT unit_key FROM uk)),
    u AS (SELECT unit_key, count(*) n_unit, sum(agreed_price) uv,
                 avg(CASE WHEN method_cat = 'specific' AND agreed_price BETWEEN 450000 AND {CEIL} THEN 1.0 ELSE 0 END) sh_near
          FROM sub GROUP BY 1),
    v AS (SELECT unit_key, entity_key, sum(agreed_price) v FROM sub WHERE NOT is_medical AND entity_key IS NOT NULL GROUP BY 1, 2),
    nm AS (SELECT unit_key, sum(v) tv_nm, sum(v * v) s2 FROM v GROUP BY 1),
    p AS (SELECT unit_key, entity_key, count(*) pn, sum(agreed_price) pv FROM sub WHERE entity_key IS NOT NULL GROUP BY 1, 2),
    pt AS (SELECT s.i, any_value(sub.project_type) project_type FROM s JOIN sub
           ON sub.project_id::VARCHAR = s.project_id AND sub.unit_key = s.unit_key GROUP BY 1)
    SELECT s.i, u.n_unit, u.uv, u.sh_near, nm.tv_nm, nm.s2, coalesce(v.v, 0) v_nm, p.pn, p.pv, pt.project_type
    FROM s LEFT JOIN u ON u.unit_key = s.unit_key LEFT JOIN nm ON nm.unit_key = s.unit_key
    LEFT JOIN v ON v.unit_key = s.unit_key AND v.entity_key = s.entity_key
    LEFT JOIN p ON p.unit_key = s.unit_key AND p.entity_key = s.entity_key
    LEFT JOIN pt ON pt.i = s.i ORDER BY s.i""").df()
    peer = con.execute("""
    WITH x AS (SELECT project_type, log10(agreed_price) lp FROM F WHERE method_cat = 'specific' AND agreed_price > 0),
    m AS (SELECT project_type, median(lp) med FROM x GROUP BY 1)
    SELECT m.project_type, m.med, median(abs(x.lp - m.med)) mad
    FROM x JOIN m ON x.project_type IS NOT DISTINCT FROM m.project_type GROUP BY 1, 2""").df()
    con.close()
    st = st.merge(peer, on="project_type", how="left")
    return st.sort_values("i").reset_index(drop=True)


def rebuild(B, jidx, med_imp, p, k_add=0, dv=0.0, day_add=0, day_dv=0.0, month_add=0, near_add=0.0,
            ratio_ref=None, ratio_budget=None, eq_ref=None, reassign=False):
    """คำนวณฟีเจอร์ของสัญญาที่ถูกแก้ใหม่ทั้งหมดตามนิยามใน features.py / model_iforest.FEATURES
    reassign=True: มูลค่า dv ย้ายจากผู้ขายรายอื่นในหน่วยงานเดียวกันมาที่ผู้ขายนี้ (มูลค่า/จำนวนสัญญาของหน่วยงานคงเดิม,
                   ผู้ขายรายอื่นลดลงตามสัดส่วน) แทนการเพิ่มสัญญาใหม่"""
    Xn = B["X"].copy()
    def setf(f, v):
        Xn[:, jidx[f]] = v
    with np.errstate(divide="ignore", invalid="ignore"):
        setf("price_z_peer", (np.log10(p) - B["med"]) / (1.4826 * B["mad"]))
        setf("gap_to_500k", np.clip((CEIL - p) / CEIL, -5, 1))
        if ratio_ref is not None:
            setf("ratio_price_ref", ratio_ref)
        if ratio_budget is not None:
            setf("ratio_price_budget", np.minimum(ratio_budget, 10))
        if eq_ref is not None:
            setf("price_eq_ref", eq_ref)
        setf("is_round_1000", (np.round(p) % 1000 == 0).astype(float))
        n2 = B["n_unit"] + (0 if reassign else k_add)
        setf("log_unit_n", np.log1p(n2))
        setf("unit_share_near500k", (B["sh_near"] * B["n_unit"] + near_add) / n2)
        if reassign:
            setf("pair_value_share", (B["pv"] + dv) / B["uv"])
            v2 = B["v_nm"] + dv
            o, o2 = B["tv_nm"] - B["v_nm"], np.maximum(B["tv_nm"] - v2, 0)
            setf("unit_hhi", (v2 ** 2 + (B["s2"] - B["v_nm"] ** 2) * np.where(o > 0, (o2 / o) ** 2, 0)) / B["tv_nm"] ** 2)
        else:
            setf("pair_value_share", (B["pv"] + dv) / (B["uv"] + dv))
            v2, tv2 = B["v_nm"] + dv, B["tv_nm"] + dv
            setf("unit_hhi", (B["s2"] - B["v_nm"] ** 2 + v2 ** 2) / tv2 ** 2)
        setf("log_pair_n", np.log1p(B["pn"] + k_add))
        setf("log_split_n_day", np.log1p(B["sd_n"] + day_add))
        setf("log_split_sum_day", np.log1p(B["sd_sum"] + day_dv))
        setf("log_split_n_month", np.log1p(B["sm_n"] + month_add))
        setf("vendor_log_value", np.log10(1 + B["vval"] + dv))
    bad = ~np.isfinite(Xn)
    Xn[bad] = np.take(med_imp, np.where(bad)[1])
    return Xn


def take(B, idx):
    return {k: v[idx] for k, v in B.items()}


def near(p):
    return ((p >= 450_000) & (p <= CEIL)).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2568")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--lof-fit", type=int, default=100_000)
    ap.add_argument("--ae-fit", type=int, default=400_000)
    ap.add_argument("--ae-iter", type=int, default=300)
    ap.add_argument("--n-inj", type=int, default=500)
    a = ap.parse_args()
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import RobustScaler

    d = PROCESSED / f"model_{a.year}_{SCOPE}"
    X = np.load(d / "X.npy", mmap_mode="r")
    n = min(len(X), 200_000) if a.test else len(X)
    lof_fit = 20_000 if a.test else a.lof_fit
    ae_fit = 50_000 if a.test else a.ae_fit
    N = a.n_inj
    t0 = time.time()
    jidx = {f: i for i, f in enumerate(FNAMES)}
    med_imp = np.array(json.load(open(d / "prep.json", encoding="utf-8"))["impute_median"], dtype=np.float64)
    cols_ = ["project_id", "unit_key", "entity_key", "agreed_price", "sign_date", "rule_score"]
    if a.test:   # อ่านเฉพาะ n แถวแรก (ประหยัดหน่วยความจำ)
        import pyarrow as pa
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(d / "ids.parquet")
        allc = cols_ + [c for c in pf.schema_arrow.names if c.startswith("r_")]
        bs, got = [], 0
        for bt in pf.iter_batches(batch_size=100_000, columns=allc):
            bs.append(bt); got += bt.num_rows
            if got >= n:
                break
        ids = pa.Table.from_batches(bs).slice(0, n).to_pandas()
    else:
        ids = pd.read_parquet(d / "ids.parquet")
        ids = ids[cols_ + [c for c in ids.columns if c.startswith("r_")]].iloc[:n]
    rcols = [c for c in ids.columns if c.startswith("r_")]

    # ---------------- แถวฐาน + สถิติสำหรับคำนวณฟีเจอร์ใหม่
    ok = ((ids["rule_score"].values == 0) & ids["entity_key"].notna().values & ids["sign_date"].notna().values
          & (ids["agreed_price"].values > 0))
    pool = np.where(ok)[0]
    cand = np.sort(np.random.default_rng(5).choice(pool, size=min(N_CAND, len(pool)), replace=False))
    st = build_base(a.year, ids.iloc[cand], mem="700MB" if a.test else "2GB")
    Xc = np.asarray(X[cand], dtype=np.float64)
    B = {"row": cand, "X": Xc, "p0": ids["agreed_price"].values[cand].astype(float)}
    for c in ["n_unit", "uv", "sh_near", "tv_nm", "s2", "v_nm", "pn", "pv", "med", "mad"]:
        B[c] = st[c].values.astype(float)
    B["sd_n"] = np.expm1(Xc[:, jidx["log_split_n_day"]]); B["sd_sum"] = np.expm1(Xc[:, jidx["log_split_sum_day"]])
    B["sm_n"] = np.expm1(Xc[:, jidx["log_split_n_month"]]); B["vval"] = 10 ** Xc[:, jidx["vendor_log_value"]] - 1
    good = np.all(np.isfinite(np.column_stack([B[c] for c in ["n_unit", "uv", "tv_nm", "s2", "pn", "pv", "med", "mad"]])), 1) & (B["mad"] > 0)
    B = take(B, np.where(good)[0])
    # ตรวจ: สร้างฟีเจอร์ของแถวฐานใหม่โดยไม่แก้อะไร ต้องได้ค่าเดิมใน X
    Xr = rebuild(B, jidx, med_imp, B["p0"])
    diag = {"n_candidates": int(len(cand)), "n_valid": int(good.sum())}
    for f in ["price_z_peer", "gap_to_500k", "unit_hhi", "unit_share_near500k", "pair_value_share", "log_unit_n", "log_pair_n",
              "log_split_sum_day", "vendor_log_value", "is_round_1000"]:
        e = np.abs(Xr[:, jidx[f]] - B["X"][:, jidx[f]])
        diag[f"absdiff_{f}"] = {"median": float(np.median(e)), "p99": float(np.quantile(e, 0.99))}
    log(f"base rows {len(B['p0']):,} ({time.time() - t0:.0f}s); recompute check p99 = "
        + ", ".join(f"{k[8:]}:{v['p99']:.1e}" for k, v in diag.items() if k.startswith("absdiff")))

    # ---------------- สถานการณ์แบบสอดคล้อง
    INJ, meta = {}, {}
    m_all = len(B["p0"])
    # 1) ซอย 2 ส่วน
    r = np.random.default_rng(101); idx = np.sort(r.choice(m_all, N, replace=False)); b = take(B, idx)
    T = r.uniform(520_000, 1_000_000, N); k = np.full(N, 2.0)
    rnd = b["X"][:, jidx["is_round_1000"]] > 0.5
    p = np.where(rnd, np.floor(T / k / 1000) * 1000, np.floor(T / k)); dv = k * p - b["p0"]
    INJ["ซอย 2 ส่วน"] = rebuild(b, jidx, med_imp, p, k_add=k - 1, dv=dv, day_add=k - 1, day_dv=dv, month_add=k - 1,
                                near_add=k * near(p) - near(b["p0"]))
    meta["ซอย 2 ส่วน"] = {"price_median": float(np.median(p)), "k_median": 2, "split_flag_share": float(np.mean((b["sd_n"] + k - 1 >= 2) & (b["sd_sum"] + dv > CEIL)))}
    # 2) ซอยหลายส่วนชิดเพดาน
    r = np.random.default_rng(102); idx = np.sort(r.choice(m_all, N, replace=False)); b = take(B, idx)
    T = r.uniform(1_500_000, 3_000_000, N); k = np.ceil(T / 499_999)
    rnd = b["X"][:, jidx["is_round_1000"]] > 0.5
    p = np.where(rnd, np.floor(T / k / 1000) * 1000, np.floor(T / k)); dv = k * p - b["p0"]
    INJ["ซอยหลายส่วนชิดเพดาน"] = rebuild(b, jidx, med_imp, p, k_add=k - 1, dv=dv, day_add=k - 1, day_dv=dv, month_add=k - 1,
                                         near_add=k * near(p) - near(b["p0"]))
    meta["ซอยหลายส่วนชิดเพดาน"] = {"price_median": float(np.median(p)), "k_median": float(np.median(k)),
                                   "near_500k_share": float(near(p).mean()),
                                   "split_flag_share": float(np.mean((b["sd_n"] + k - 1 >= 2) & (b["sd_sum"] + dv > CEIL)))}
    # 3) ราคาสูง + ไม่มีส่วนลด
    el = np.where(B["p0"] <= 250_000)[0]
    r = np.random.default_rng(103); idx = np.sort(r.choice(el, min(N, len(el)), replace=False)); b = take(B, idx)
    mult = r.uniform(1.5, 3.0, len(idx)); p = np.minimum(np.round(b["p0"] * mult, -2), 499_000); dv = p - b["p0"]
    one = np.ones(len(idx))
    INJ["ราคาสูง+ไม่มีส่วนลด"] = rebuild(b, jidx, med_imp, p, dv=dv, day_dv=dv, near_add=near(p) - near(b["p0"]),
                                       ratio_ref=one, ratio_budget=one, eq_ref=one)
    meta["ราคาสูง+ไม่มีส่วนลด"] = {"mult_median": float(np.median(p / b["p0"])),
                                   "dz_median": float(np.median(np.log10(p / b["p0"]) / (1.4826 * b["mad"])))}
    # 4) ผูกขาดผู้ขาย
    share0 = B["pv"] / B["uv"]
    el = np.where((B["n_unit"] >= 20) & (B["n_unit"] <= 1000) & (share0 < 0.3))[0]
    r = np.random.default_rng(104); idx = np.sort(r.choice(el, min(N, len(el)), replace=False)); b = take(B, idx)
    # ผู้ขายนี้ได้สัญญาที่เดิมเป็นของผู้ขายรายอื่น จนส่วนแบ่งมูลค่า = s (จำนวน = มูลค่าที่ย้าย / มูลค่าเฉลี่ยต่อสัญญาของหน่วยงาน)
    s = r.uniform(0.6, 0.9, len(idx))
    dv = np.minimum(s * b["uv"] - b["pv"], np.maximum(b["tv_nm"] - b["v_nm"], 0))
    kk = np.minimum(np.maximum(1, np.round(dv / (b["uv"] / b["n_unit"]))), b["n_unit"] - b["pn"])
    INJ["ผูกขาดผู้ขาย"] = rebuild(b, jidx, med_imp, b["p0"], k_add=kk, dv=dv, month_add=np.round(kk / 12), reassign=True)
    meta["ผูกขาดผู้ขาย"] = {"k_median": float(np.median(kk)), "share_target_median": float(np.median((b["pv"] + dv) / b["uv"])),
                           "n_unit_median": float(np.median(b["n_unit"]))}
    log(f"consistent injection built ({time.time() - t0:.0f}s)")
    import os
    if os.environ.get("C3_DUMP"):
        np.savez(os.environ["C3_DUMP"], base=B["X"], **{f"inj{i}": v for i, v in enumerate(INJ.values())})

    # ---------------- โมเดล
    sc_if = np.load(d / "score_seed0.npy")[:n]
    m_if = joblib.load(d / "iforest_seed0.joblib")
    fit_rows = np.sort(np.random.default_rng(11).choice(n, size=min(1_000_000, n), replace=False))
    Xf = np.asarray(X[fit_rows], dtype=np.float64); ecod, hbos = ECOD().fit(Xf), HBOS().fit(Xf); del Xf
    if not a.test and (d / "score_ecod.npy").exists():
        sc_e = np.load(d / "score_ecod.npy").astype(float); sc_h = np.load(d / "score_hbos.npy").astype(float)
    else:
        sc_e = np.concatenate([ecod.score(np.asarray(X[i:min(i + CHUNK, n)], dtype=np.float64)) for i in range(0, n, CHUNK)])
        sc_h = np.concatenate([hbos.score(np.asarray(X[i:min(i + CHUNK, n)], dtype=np.float64)) for i in range(0, n, CHUNK)])
    log(f"IF/ECOD/HBOS ready ({time.time() - t0:.0f}s)")
    rng = np.random.default_rng(23)   # ลำดับการสุ่มเหมือน model_compare2.py → scaler/LOF ชุดเดิม
    sub = np.asarray(X[np.sort(rng.choice(n, size=min(ae_fit, n), replace=False))], dtype=np.float64)
    scaler = RobustScaler(quantile_range=(5, 95)).fit(sub)
    clipz = lambda Z: np.clip(Z, -20, 20)
    ae = MLPRegressor(hidden_layer_sizes=(32, 8, 32), activation="relu", max_iter=a.ae_iter, batch_size=1024,
                      early_stopping=True, n_iter_no_change=5, random_state=0)
    Zs = clipz(scaler.transform(sub)); ae.fit(Zs, Zs); del Zs, sub
    ae_info = {"max_iter": a.ae_iter, "n_iter": int(ae.n_iter_), "converged": bool(ae.n_iter_ < a.ae_iter),
               "best_val_r2": float(ae.best_validation_score_) if ae.best_validation_score_ is not None else None}
    log(f"AE fit {time.time() - t0:.0f}s, iters {ae.n_iter_}/{a.ae_iter}")
    def ae_score(Bk):
        Z = clipz(scaler.transform(Bk)); return ((ae.predict(Z) - Z) ** 2).mean(1)
    sc_ae = np.concatenate([ae_score(np.asarray(X[i:min(i + CHUNK, n)], dtype=np.float64)) for i in range(0, n, CHUNK)])
    log(f"AE score {time.time() - t0:.0f}s")
    lsub = clipz(scaler.transform(np.asarray(X[np.sort(rng.choice(n, size=min(lof_fit, n), replace=False))], dtype=np.float64)))
    lof = LocalOutlierFactor(n_neighbors=50, novelty=True, n_jobs=-1).fit(lsub)
    lof_s = lambda Bk: -lof.score_samples(clipz(scaler.transform(Bk)))
    lof_reused = False
    if not a.test and (d / "score_lof.npy").exists():
        old = np.load(d / "score_lof.npy")
        chk = np.sort(np.random.default_rng(1).choice(n, 2000, replace=False))
        if np.allclose(lof_s(np.asarray(X[chk], dtype=np.float64)), old[chk], rtol=1e-4, atol=1e-5):
            sc_lof = old.astype(float); lof_reused = True
    if not lof_reused:
        sc_lof = np.empty(n)
        for i in range(0, n, CHUNK):
            sc_lof[i:min(i + CHUNK, n)] = lof_s(np.asarray(X[i:min(i + CHUNK, n)], dtype=np.float64))
            log(f"LOF {min(i + CHUNK, n):,}/{n:,} ({time.time() - t0:.0f}s)")
    log(f"LOF ready (reused={lof_reused}) {time.time() - t0:.0f}s")

    raw = {"IF": sc_if, "ECOD": sc_e, "HBOS": sc_h, "AE": sc_ae, "LOF": sc_lof}
    R = {m: rank01(s_) for m, s_ in raw.items()}
    R["MAX(IF,ECOD,HBOS)"] = np.maximum.reduce([R["IF"], R["ECOD"], R["HBOS"]])
    R["MAX(all5)"] = np.maximum.reduce([R[m] for m in raw])
    srt = {m: np.sort(s_) for m, s_ in raw.items()}
    thr1 = {m: np.quantile(R[m], 0.99) for m in R}; thr5 = {m: np.quantile(R[m], 0.95) for m in R}

    def evaluate(Xi):
        si = {"IF": -m_if.score_samples(Xi.astype(np.float32)), "ECOD": ecod.score(Xi), "HBOS": hbos.score(Xi),
              "AE": ae_score(Xi), "LOF": lof_s(Xi)}
        pct = {m: np.searchsorted(srt[m], si[m]) / n for m in si}
        pct["MAX(IF,ECOD,HBOS)"] = np.maximum.reduce([pct["IF"], pct["ECOD"], pct["HBOS"]])
        pct["MAX(all5)"] = np.maximum.reduce([pct[m] for m in raw])
        return {m: {"recall_top1pct": float((p_ >= thr1[m]).mean()), "recall_top5pct": float((p_ >= thr5[m]).mean()),
                    "median_percentile": float(np.median(p_))} for m, p_ in pct.items()}

    res = {"year": a.year, "n": n, "test": a.test, "ae": ae_info, "lof_reused": lof_reused, "diag": diag, "scenario_meta": meta,
           "inj_consistent": {k_: evaluate(v) for k_, v in INJ.items()}, "inj_naive": {}}
    # injection แบบเดิม (เขียนทับฟีเจอร์) — แถวฐานชุดเดียวกับรอบ 2
    base_pool = np.where(sc_if < np.median(sc_if))[0]
    for name, mods in SCEN.items():
        rows = np.sort(np.random.default_rng(7).choice(base_pool, 500, replace=False))
        Xi = np.asarray(X[rows], dtype=np.float64).copy()
        for f, v in mods.items():
            Xi[:, jidx[f]] = v
        res["inj_naive"][name] = evaluate(Xi)
    # lift ของ red flag จริงใน top1% + ความคงที่ของ AE
    k1 = int(0.01 * n)
    base = ids[rcols].mean()
    res["rule_lift_top1pct"] = {}
    tops = {}
    for m, rr in R.items():
        top = np.argsort(-rr, kind="stable")[:k1]; tops[m] = set(top)
        res["rule_lift_top1pct"][m] = {c: float(ids[c].values[top].mean() / max(base[c], 1e-9)) for c in rcols}
    res["rule_base_rate"] = base.round(4).to_dict()
    res["jaccard_top1pct"] = {f"AE~{m}": len(tops["AE"] & tops[m]) / len(tops["AE"] | tops[m]) for m in ["IF", "ECOD", "HBOS", "LOF"]}
    if not a.test and (d / "score_ae.npy").exists():
        a60 = set(np.argsort(-np.load(d / "score_ae.npy").astype(float), kind="stable")[:k1])
        res["jaccard_top1pct"]["AE~AE(iter60, รอบ 2)"] = len(tops["AE"] & a60) / len(tops["AE"] | a60)
    tag = "_test" if a.test else ""
    json.dump(res, open(REPORTS / f"model_compare3_{a.year}{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not a.test:
        np.save(d / "score_ae300.npy", sc_ae.astype(np.float32))

    # ---------------- รายงาน
    L = [f"# เทียบโมเดลรอบ 3: injection แบบสอดคล้อง ปีงบ {a.year} (n = {n:,}{', ทดสอบ' if a.test else ''})", "",
         f"- AE: iters {ae_info['n_iter']}/{ae_info['max_iter']} (converged = {ae_info['converged']}); LOF ใช้คะแนนรอบ 2: {lof_reused}",
         f"- แถวฐาน: สัญญาที่ rule_score = 0, ใช้ได้ {diag['n_valid']:,}/{diag['n_candidates']:,}; ตรวจการคำนวณฟีเจอร์ใหม่ (p99 |ต่าง|): "
         + ", ".join(f"{k[8:]} {v['p99']:.1e}" for k, v in diag.items() if k.startswith("absdiff")), "",
         "## Injection แบบสอดคล้อง (แก้ที่สัญญา แล้วคำนวณฟีเจอร์ใหม่ทั้งหมด)", "",
         "| สถานการณ์ | โมเดล | recall@top1% | recall@top5% | median pct |", "|---|---|---|---|---|"]
    for name, rr in res["inj_consistent"].items():
        for m, v in rr.items():
            L.append(f"| {name} | {m} | {v['recall_top1pct']:.2f} | {v['recall_top5pct']:.2f} | {v['median_percentile']:.3f} |")
    L += ["", "รายละเอียดสถานการณ์: " + json.dumps(meta, ensure_ascii=False), "",
          "## Injection แบบเดิม (เขียนทับฟีเจอร์) — เพื่อเทียบ", "",
          "| สถานการณ์ | โมเดล | recall@top1% | recall@top5% | median pct |", "|---|---|---|---|---|"]
    for name, rr in res["inj_naive"].items():
        for m, v in rr.items():
            L.append(f"| {name} | {m} | {v['recall_top1pct']:.2f} | {v['recall_top5pct']:.2f} | {v['median_percentile']:.3f} |")
    L += ["", "## Lift ของ red flag จริงใน top1% (AE = iter 300)", "", "| โมเดล | " + " | ".join(rcols) + " |", "|---" * (len(rcols) + 1) + "|"]
    for m, v in res["rule_lift_top1pct"].items():
        L.append(f"| {m} | " + " | ".join(f"{v[c]:.1f}" for c in rcols) + " |")
    L.append("| อัตราพื้นฐาน | " + " | ".join(f"{base[c]:.3f}" for c in rcols) + " |")
    L += ["", "| คู่ | Jaccard top1% |", "|---|---|"] + [f"| {k_} | {v:.3f} |" for k_, v in res["jaccard_top1pct"].items()]
    (REPORTS / f"model_compare3_{a.year}{tag}.md").write_text("\n".join(L), encoding="utf-8")
    log(f"→ model_compare3_{a.year}{tag} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
