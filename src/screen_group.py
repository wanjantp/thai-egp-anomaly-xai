"""
screen_group.py — ตัวคัดกรองที่รู้เรื่องเพดาน (ceiling-aware screener) ระดับ "การซื้อ" เทียบกับ anomaly detection รายสัญญา

แนวคิด: unsupervised รายสัญญา (IF/ECOD/HBOS/AE/LOF) จับการซอย 2 ส่วนไม่ได้ (model_compare3: recall@top1% ≤ 0.04)
เพราะแต่ละสัญญาที่ถูกแบ่งดูปกติ → เปลี่ยน "หน่วยวิเคราะห์" จากสัญญาเป็นกลุ่มการซื้อ แล้ววัดยอดรวมเทียบเพดาน
  กลุ่มวัน   D = หน่วยงาน × ผู้ขาย × วันลงนาม   (เฉพาะสัญญาใน scope le500k)
  กลุ่มเดือน M = หน่วยงาน × ผู้ขาย × เดือนลงนาม
ฟีเจอร์กลุ่ม (ทุกตัวอ้างอิงเพดาน 500k):
  d_log_n, d_sum_ratio = log2(ยอดรวม/500k), d_max_ratio = ราคาสูงสุด/500k, d_balance = ต่ำสุด/สูงสุด,
  d_near_share (450k–500k), d_eqref_share (ราคา = ราคากลาง), m_log_n, m_sum_ratio, m_near_share
ตัวคัดกรอง (ทุกตัวให้คะแนนรายสัญญา แล้วจัดอันดับ top-k% ของสัญญา):
  C-HBOS     HBOS รายสัญญา (ดีที่สุดในรอบ 3) — baseline
  G-HBOS     HBOS บนฟีเจอร์กลุ่ม (fit บนกลุ่มวัน) → ทุกสัญญาในกลุ่มได้คะแนนเดียวกัน   [เปลี่ยนตัวแทนข้อมูล]
  G-GATE     ประตูความรู้: (ยอดวัน > 500k & ≥2 สัญญา) + (ยอดเดือน > 500k & ≥2 สัญญา) + อันดับ G-HBOS (0–1) [ความรู้ + จัดอันดับ]
  COMBO      max-rank(G-GATE, C-HBOS) — ให้ครอบคลุมสถานการณ์ราคา/ผูกขาดด้วย
  (IF, ECOD รายสัญญาเพื่อเทียบ)
ประเมิน: injection แบบสอดคล้อง (แถวฐาน = สัญญาจริง rule_score = 0, คำนวณฟีเจอร์ใหม่ทั้งหมด) — การซื้อหนึ่งครั้ง "ถูกพบ"
  ถ้ามีสัญญาส่วนใดอยู่ใน top-k% (ยุติธรรมกับโมเดลรายสัญญา: ทุกส่วนถูกให้คะแนน)
  ซอย 2 ส่วน (เท่ากัน / ไม่เท่ากัน), ซอย 4–7 ส่วน (ไม่เท่ากัน), ซอย 2 ส่วนต่างวันในเดือนเดียวกัน (หลบกลุ่มวัน),
  ราคาสูง+ไม่มีส่วนลด, ผูกขาดผู้ขาย (จาก model_compare3)
  + lift ของ red flag จริงใน top1% และขนาดงบตรวจที่ต้องใช้ (recall 0.8)

  python src\\screen_group.py --year 2568
"""
import argparse
import json
import time

import duckdb
import joblib
import numpy as np
import pandas as pd

from model_iforest import FNAMES, PROCESSED, REPORTS, SCOPE
from model_compare import ECOD, HBOS, rank01
from model_compare3 import CEIL, build_base, near, rebuild, take

CHUNK = 400_000
N_INJ = 500
GF = ["d_log_n", "d_sum_ratio", "d_max_ratio", "d_balance", "d_near_share", "d_eqref_share", "m_log_n", "m_sum_ratio", "m_near_share"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gfeat(n, s, mx, mn, nnear, neq, mn_, ms, mnear):
    """ฟีเจอร์กลุ่มจากสถิติรวม (ใช้ทั้งข้อมูลจริงและกลุ่มที่ฉีด → นิยามเดียวกัน)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack([
            np.log1p(n), np.clip(np.log2(np.maximum(s, 1) / CEIL), -8, 4), mx / CEIL, np.where(mx > 0, mn / mx, 1.0),
            nnear / n, neq / n, np.log1p(mn_), np.clip(np.log2(np.maximum(ms, 1) / CEIL), -8, 4), mnear / mn_]).astype(np.float64)


def split_parts(T, k, rnd, rng, equal):
    """แบ่งยอด T เป็น k ส่วน แต่ละส่วน ≤ 499,999 (ไม่เท่ากัน: Dirichlet(3) แล้วตัดส่วนที่เกินไปให้ส่วนอื่น)"""
    P = []
    for t, kk, r in zip(T, k.astype(int), rnd):
        if equal:
            w = np.full(kk, 1.0 / kk)
        else:
            for _ in range(200):
                w = rng.dirichlet(np.full(kk, 3.0))
                if (t * w).max() <= 499_999:
                    break
            else:
                w = np.full(kk, 1.0 / kk)
        p = t * w
        p = np.floor(p / 1000) * 1000 if r else np.floor(p)
        P.append(p)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2568")
    a = ap.parse_args()
    t0 = time.time()
    d = PROCESSED / f"model_{a.year}_{SCOPE}"
    X = np.load(d / "X.npy", mmap_mode="r")
    n = len(X)
    jidx = {f: i for i, f in enumerate(FNAMES)}
    med_imp = np.array(json.load(open(d / "prep.json", encoding="utf-8"))["impute_median"], dtype=np.float64)

    # ---------------- กลุ่มจริง (gid = กลุ่มวัน, mid = กลุ่มเดือน; รวมด้วย numpy)
    import tempfile
    from pathlib import Path
    con = duckdb.connect()
    con.execute(f"SET threads=2; SET memory_limit='2GB'; SET preserve_insertion_order=false; "
                f"SET temp_directory='{(Path(tempfile.gettempdir()) / 'duckdb_egp').as_posix()}'")
    p = str(d / "ids.parquet").replace("\\", "/")
    q = con.execute(f"""WITH c AS (SELECT rid, unit_key, coalesce(entity_key, 'r' || rid::VARCHAR) ek,
                          coalesce(sign_date, DATE '1900-01-01' + (rid % 1000000)::INT) sd FROM read_parquet('{p}'))
        SELECT rid, (dense_rank() OVER (ORDER BY unit_key, ek, sd)) - 1 gid,
                    (dense_rank() OVER (ORDER BY unit_key, ek, date_trunc('month', sd))) - 1 mid FROM c ORDER BY rid""").fetchnumpy()
    con.close()
    gid = q["gid"].astype(np.int64); mid = q["mid"].astype(np.int64); del q
    prices = pd.read_parquet(d / "ids.parquet", columns=["agreed_price"])["agreed_price"].values.astype(float)
    eq = np.asarray(X[:, jidx["price_eq_ref"]], dtype=np.float32)
    nr = ((prices >= 450_000) & (prices <= CEIL)).astype(float)
    ng = int(gid.max()) + 1; nm = int(mid.max()) + 1
    bc = lambda idx, w, m: np.bincount(idx, weights=w, minlength=m)
    gn = bc(gid, None, ng); gs = bc(gid, prices, ng); gnear = bc(gid, nr, ng); geq = bc(gid, (eq > 0.5).astype(float), ng)
    gmx = np.full(ng, -np.inf); np.maximum.at(gmx, gid, prices); gmn = np.full(ng, np.inf); np.minimum.at(gmn, gid, prices)
    mn_ = bc(mid, None, nm); ms = bc(mid, prices, nm); mnear = bc(mid, nr, nm)
    g2m = np.zeros(ng, np.int64); g2m[gid] = mid
    G = pd.DataFrame({"n": gn, "s": gs, "mx": gmx, "mn": gmn, "nnear": gnear, "neq": geq,
                      "mn_": mn_[g2m], "ms": ms[g2m], "mnear": mnear[g2m]})
    Gf = gfeat(*[G[c].values.astype(float) for c in ["n", "s", "mx", "mn", "nnear", "neq", "mn_", "ms", "mnear"]])
    ng = len(G)
    gate_g = (G["n"].values >= 2) & (G["s"].values > CEIL)
    gatem_g = (G["mn_"].values >= 2) & (G["ms"].values > CEIL)
    log(f"groups: {ng:,} day-groups; contracts in day-split {gate_g[gid].mean():.3f}, month-split {gatem_g[gid].mean():.3f} ({time.time()-t0:.0f}s)")

    # ---------------- คะแนนจริง
    hg = HBOS().fit(Gf)
    sg_g = hg.score(Gf)                        # ต่อกลุ่ม
    sg = sg_g[gid]                              # ต่อสัญญา
    # percentile แบบ mid-rank (คะแนนซ้ำกันได้ค่าเดียวกัน) — ใช้นิยามเดียวกันกับข้อมูลจริงและแถวที่ฉีด
    pr = lambda srt_, x: (np.searchsorted(srt_, x, "left") + np.searchsorted(srt_, x, "right")) / (2.0 * len(srt_))
    srt = {"IF": np.sort(np.load(d / "score_seed0.npy").astype(float)), "ECOD": np.sort(np.load(d / "score_ecod.npy").astype(float)),
           "C-HBOS": np.sort(np.load(d / "score_hbos.npy").astype(float)), "G-HBOS": np.sort(sg)}
    raw = {"IF": np.load(d / "score_seed0.npy").astype(float), "ECOD": np.load(d / "score_ecod.npy").astype(float),
           "C-HBOS": np.load(d / "score_hbos.npy").astype(float), "G-HBOS": sg}
    R = {m: pr(srt[m], v) for m, v in raw.items()}
    gate_c = gate_g[gid].astype(float) + gatem_g[gid].astype(float)
    R["G-GATE"] = gate_c + R["G-HBOS"] * 0.999          # ประตู (0/1/2) แล้วจัดอันดับในชั้นด้วย G-HBOS
    srt["G-GATE"] = np.sort(R["G-GATE"])
    R["COMBO"] = np.maximum(pr(srt["G-GATE"], R["G-GATE"]), R["C-HBOS"])
    srt["COMBO"] = np.sort(R["COMBO"])
    THR = {m: (np.quantile(v, 0.99), np.quantile(v, 0.95)) for m, v in R.items()}
    # สัดส่วนสัญญาที่ต้องตรวจจนถึงค่า v (IF/ECOD/HBOS/G-HBOS อยู่ในสเกล percentile แล้ว; G-GATE/COMBO ต้องแปลง)
    topshare = lambda m, v: 1 - (pr(srt[m], np.asarray(v, float)) if m in ("G-GATE", "COMBO") else np.asarray(v, float))
    sc = R
    log(f"scores ready ({time.time()-t0:.0f}s)")

    # ---------------- โมเดลรายสัญญาสำหรับแถวที่ฉีด
    fit_rows = np.sort(np.random.default_rng(11).choice(n, size=min(1_000_000, n), replace=False))
    Xf = np.asarray(X[fit_rows], dtype=np.float64); ecod, hbos = ECOD().fit(Xf), HBOS().fit(Xf); del Xf
    m_if = joblib.load(d / "iforest_seed0.joblib")
    # ตรวจ: HBOS ที่ fit ใหม่ให้คะแนนตรงกับที่บันทึกไว้
    chk = np.sort(np.random.default_rng(1).choice(n, 2000, replace=False))
    hb_ok = bool(np.allclose(hbos.score(np.asarray(X[chk], dtype=np.float64)), raw["C-HBOS"][chk], rtol=1e-4, atol=1e-4))
    log(f"thresholds top1%: " + ", ".join(f"{m} {v[0]:.4f}" for m, v in THR.items())
        + f"; share gate=2 {np.mean(gate_c == 2):.3f}, gate=1 {np.mean(gate_c == 1):.3f}; G-HBOS pct among gate=2: median {np.median(R['G-HBOS'][gate_c == 2]):.4f}")
    log(f"contract models ready; HBOS refit matches saved = {hb_ok} ({time.time()-t0:.0f}s)")

    def pct_contract(Xi):
        Xi = Xi.astype(np.float32).astype(np.float64)
        s = {"IF": -m_if.score_samples(Xi.astype(np.float32)), "ECOD": ecod.score(Xi), "C-HBOS": hbos.score(Xi)}
        return {m: pr(srt[m], v) for m, v in s.items()}

    # ---------------- แถวฐาน (เหมือน model_compare3)
    ids = pd.read_parquet(d / "ids.parquet", columns=["project_id", "unit_key", "entity_key", "agreed_price", "sign_date", "rule_score"])
    ok = (ids["rule_score"].values == 0) & ids["entity_key"].notna().values & ids["sign_date"].notna().values & (ids["agreed_price"].values > 0)
    pool = np.where(ok)[0]
    cand = np.sort(np.random.default_rng(5).choice(pool, size=min(30_000, len(pool)), replace=False))
    cache = d / "_screen_base.parquet"
    if cache.exists():
        st = pd.read_parquet(cache)
    else:
        st = build_base(a.year, ids.iloc[cand]); st.to_parquet(cache)
    del ids
    log(f"base stats ready ({time.time()-t0:.0f}s)")
    Xc = np.asarray(X[cand], dtype=np.float64)
    B = {"row": cand, "X": Xc, "p0": prices[cand].astype(float), "gid": gid[cand]}
    for c in ["n_unit", "uv", "sh_near", "tv_nm", "s2", "v_nm", "pn", "pv", "med", "mad"]:
        B[c] = st[c].values.astype(float)
    B["sd_n"] = np.expm1(Xc[:, jidx["log_split_n_day"]]); B["sd_sum"] = np.expm1(Xc[:, jidx["log_split_sum_day"]])
    B["sm_n"] = np.expm1(Xc[:, jidx["log_split_n_month"]]); B["vval"] = 10 ** Xc[:, jidx["vendor_log_value"]] - 1
    good = np.all(np.isfinite(np.column_stack([B[c] for c in ["n_unit", "uv", "tv_nm", "s2", "pn", "pv", "med", "mad"]])), 1) & (B["mad"] > 0)
    B = take(B, np.where(good)[0])
    m_all = len(B["p0"])
    # สมาชิกอื่นของกลุ่มวันของแถวฐาน (สำหรับ min/max ใหม่)
    order = np.argsort(gid, kind="stable"); gstart = np.searchsorted(gid[order], np.arange(ng)); gend = np.searchsorted(gid[order], np.arange(ng), side="right")
    eqv = eq > 0.5

    def others(row, g):
        mem = order[gstart[g]:gend[g]]; mem = mem[mem != row]
        return prices[mem].astype(float), eqv[mem]

    def group_after(b_i, parts_day, parts_month_extra=(), eq_parts=None, extra_month_n=0.0, extra_month_s=0.0):
        """สถิติกลุ่มวัน/เดือนหลังแทนสัญญาฐานด้วย parts_day (วันเดียวกัน) + parts_month_extra (วันอื่นในเดือนเดียวกัน)"""
        g = B["gid"][b_i]; row = B["row"][b_i]
        op, oe = others(row, g)
        pd_ = np.asarray(parts_day, float); allp = np.concatenate([op, pd_])
        e_parts = np.zeros(len(pd_), bool) if eq_parts is None else eq_parts
        nnear = ((allp >= 450_000) & (allp <= CEIL)).sum(); neq = oe.sum() + e_parts.sum()
        p0 = B["p0"][b_i]; pe = np.asarray(parts_month_extra, float)
        mn_ = G["mn_"].values[g] - 1 + len(pd_) + len(pe) + extra_month_n
        ms = G["ms"].values[g] - p0 + pd_.sum() + pe.sum() + extra_month_s
        mnear = G["mnear"].values[g] - near(np.array([p0]))[0] + near(pd_).sum() + near(pe).sum()
        return [len(allp), allp.sum(), allp.max(), allp.min(), nnear, neq, mn_, ms, mnear]

    def eval_purchase(day_groups, contract_X_list):
        """day_groups: list ของสถิติกลุ่ม (วันเดียวกัน 1 กลุ่ม หรือหลายกลุ่มถ้าแยกวัน); contract_X_list: X ของทุกส่วน"""
        Gi = gfeat(*[np.array([g[j] for g in day_groups], float) for j in range(9)])
        s_g = hg.score(Gi)
        gate_i = np.array([int(g[0] >= 2 and g[1] > CEIL) + int(g[6] >= 2 and g[7] > CEIL) for g in day_groups], float)
        out = {"G-HBOS": pr(srt["G-HBOS"], s_g)}
        out["G-GATE"] = gate_i + out["G-HBOS"] * 0.999
        pc = pct_contract(np.vstack(contract_X_list))
        for m, v in pc.items():
            out[m] = v
        # COMBO: rank ของ G-GATE (ต่อกลุ่ม) กับ C-HBOS (ต่อสัญญา) → แต่ละสัญญาได้ max
        return out

    rng_all = np.random.default_rng(2024)
    res_inj = {}
    meta = {}

    def run_split(name, Tlo, Thi, kfun, equal, spread_days=False, seed=0):
        r = np.random.default_rng(seed)
        idx = np.sort(r.choice(m_all, N_INJ, replace=False))
        T = r.uniform(Tlo, Thi, N_INJ); k = kfun(T)
        rnd = B["X"][idx, jidx["is_round_1000"]] > 0.5
        P = split_parts(T, k, rnd, r, equal)
        hit = {m: {"1": 0, "5": 0} for m in ["IF", "ECOD", "C-HBOS", "G-HBOS", "G-GATE", "COMBO"]}
        need = {m: [] for m in hit}
        for j, b_i in enumerate(idx):
            if j % 100 == 0:
                log(f"  {name} {j}/{N_INJ}")
            parts = P[j]; kk = len(parts); b = take(B, np.array([b_i]))
            if not spread_days:
                gstats = [group_after(b_i, parts)]
                gmap = np.zeros(kk, int)
                dvd = parts.sum() - b["p0"][0]
                Xs = [rebuild(b, jidx, med_imp, np.array([pp]), k_add=kk - 1, dv=dvd, day_add=kk - 1, day_dv=dvd, month_add=kk - 1,
                              near_add=near(parts).sum() - near(b["p0"])[0]) for pp in parts]
            else:
                # ส่วนแรกวันเดิม, ส่วนที่เหลือคนละวัน (กลุ่มวันใหม่ ไม่มีสัญญาอื่น) ในเดือนเดียวกัน
                g0 = group_after(b_i, parts[:1], parts_month_extra=parts[1:])
                gstats = [g0]
                for pp in parts[1:]:
                    gstats.append([1, pp, pp, pp, int(450_000 <= pp <= CEIL), 0, g0[6], g0[7], g0[8]])
                gmap = np.arange(kk)
                dvd = parts.sum() - b["p0"][0]
                Xs = [rebuild(b, jidx, med_imp, np.array([parts[0]]), k_add=kk - 1, dv=dvd, day_add=0, day_dv=parts[0] - b["p0"][0],
                              month_add=kk - 1, near_add=near(parts).sum() - near(b["p0"])[0])]
                for pp in parts[1:]:
                    bb = {kq: (v.copy() if isinstance(v, np.ndarray) else v) for kq, v in b.items()}
                    bb["sd_n"] = np.array([0.0]); bb["sd_sum"] = np.array([0.0])
                    Xs.append(rebuild(bb, jidx, med_imp, np.array([pp]), k_add=kk - 1, dv=dvd, day_add=1, day_dv=pp,
                                      month_add=kk - 1, near_add=near(parts).sum() - near(b["p0"])[0]))
            out = eval_purchase(gstats, Xs)
            # ต่อสัญญา: กลุ่ม → map ไปที่ส่วน
            pc = {m: out[m] for m in ["IF", "ECOD", "C-HBOS"]}
            pg = {"G-HBOS": out["G-HBOS"][gmap], "G-GATE": out["G-GATE"][gmap]}
            # COMBO: rank ของ G-GATE และ C-HBOS เปลี่ยนเป็น percentile แล้วเทียบกับการกระจาย COMBO จริง
            combo = np.maximum(pr(srt["G-GATE"], pg["G-GATE"]), pc["C-HBOS"])
            allp = {**pc, **pg, "COMBO": combo}
            for m, v in allp.items():
                best = float(np.max(v))
                hit[m]["1"] += best >= THR[m][0]; hit[m]["5"] += best >= THR[m][1]
                need[m].append(float(topshare(m, best)))
        res_inj[name] = {m: {"recall_top1pct": hit[m]["1"] / N_INJ, "recall_top5pct": hit[m]["5"] / N_INJ,
                             "budget_for_recall80": float(np.quantile(need[m], 0.8))} for m in hit}
        meta[name] = {"k_median": float(np.median([len(x) for x in P])), "T_median": float(np.median(T)),
                      "min_part_over_max_median": float(np.median([x.min() / x.max() for x in P]))}
        log(f"{name}: " + ", ".join(f"{m} {v['recall_top1pct']:.2f}/{v['recall_top5pct']:.2f}" for m, v in res_inj[name].items()) + f" ({time.time()-t0:.0f}s)")

    two = lambda T: np.full(len(T), 2)
    run_split("ซอย 2 ส่วน (เท่ากัน)", 520_000, 1_000_000, two, True, seed=101)
    run_split("ซอย 2 ส่วน (ไม่เท่ากัน)", 520_000, 1_000_000, two, False, seed=111)
    run_split("ซอย 4–7 ส่วน (ไม่เท่ากัน)", 1_500_000, 3_000_000, lambda T: np.ceil(T / 499_999) + 0, False, seed=102)
    run_split("ซอย 2 ส่วน ต่างวันในเดือนเดียวกัน", 520_000, 1_000_000, two, False, spread_days=True, seed=112)

    # ราคาสูง+ไม่มีส่วนลด / ผูกขาด (ตามนิยาม model_compare3) — ต่อสัญญาเดียว
    def run_single(name, idx, Xi, gstats):
        out = eval_purchase(gstats, [Xi])
        allp = {m: out[m] for m in ["IF", "ECOD", "C-HBOS", "G-HBOS", "G-GATE"]}
        allp["COMBO"] = np.maximum(pr(srt["G-GATE"], out["G-GATE"]), out["C-HBOS"])
        return allp
    for name in ["ราคาสูง+ไม่มีส่วนลด", "ผูกขาดผู้ขาย"]:
        r = np.random.default_rng(103 if name.startswith("ราคา") else 104)
        if name.startswith("ราคา"):
            el = np.where(B["p0"] <= 250_000)[0]
        else:
            share0 = B["pv"] / B["uv"]; el = np.where((B["n_unit"] >= 20) & (B["n_unit"] <= 1000) & (share0 < 0.3))[0]
        idx = np.sort(r.choice(el, min(N_INJ, len(el)), replace=False))
        acc = {m: [] for m in ["IF", "ECOD", "C-HBOS", "G-HBOS", "G-GATE", "COMBO"]}
        for b_i in idx:
            b = take(B, np.array([b_i]))
            if name.startswith("ราคา"):
                pp = min(round(b["p0"][0] * r.uniform(1.5, 3.0), -2), 499_000); dv = pp - b["p0"][0]; one = np.ones(1)
                Xi = rebuild(b, jidx, med_imp, np.array([pp]), dv=dv, day_dv=dv, near_add=near(np.array([pp]))[0] - near(b["p0"])[0],
                             ratio_ref=one, ratio_budget=one, eq_ref=one)
                gs = [group_after(b_i, [pp], eq_parts=np.array([True]))]
            else:
                s = r.uniform(0.6, 0.9); dv = min(s * b["uv"][0] - b["pv"][0], max(b["tv_nm"][0] - b["v_nm"][0], 0))
                kk = min(max(1, round(dv / (b["uv"][0] / b["n_unit"][0]))), b["n_unit"][0] - b["pn"][0])
                Xi = rebuild(b, jidx, med_imp, b["p0"], k_add=np.array([kk]), dv=np.array([dv]), month_add=np.array([round(kk / 12)]), reassign=True)
                gs = [group_after(b_i, [b["p0"][0]], extra_month_n=round(kk / 12), extra_month_s=dv / 12)]
            for m, v in run_single(name, b_i, Xi, gs).items():
                acc[m].append(float(v[0]))
        res_inj[name] = {m: {"recall_top1pct": float(np.mean(np.array(v) >= THR[m][0])), "recall_top5pct": float(np.mean(np.array(v) >= THR[m][1])),
                             "budget_for_recall80": float(np.quantile(topshare(m, v), 0.8))} for m, v in acc.items()}
        log(f"{name}: " + ", ".join(f"{m} {v['recall_top1pct']:.2f}/{v['recall_top5pct']:.2f}" for m, v in res_inj[name].items()))

    # ---------------- lift ของ red flag จริงใน top1%
    import pyarrow.parquet as pq
    rcols = [c for c in pq.ParquetFile(d / "ids.parquet").schema_arrow.names if c.startswith("r_")]
    rf = pd.read_parquet(d / "ids.parquet", columns=rcols)
    base = rf.mean()
    k1 = int(0.01 * n)
    lift, tops = {}, {}
    for m, rr in R.items():
        top = np.argsort(-rr, kind="stable")[:k1]; tops[m] = set(top.tolist())
        lift[m] = {c: float(rf[c].values[top].mean() / max(base[c], 1e-9)) for c in rcols}
    jac = {f"{x}~{y}": len(tops[x] & tops[y]) / len(tops[x] | tops[y]) for x, y in [("G-GATE", "C-HBOS"), ("G-HBOS", "C-HBOS"), ("COMBO", "C-HBOS"), ("G-GATE", "G-HBOS")]}
    # กลุ่มที่ G-GATE จัดไว้ top 1%: ลักษณะ
    topg = np.argsort(-R["G-GATE"], kind="stable")[:k1]
    gg = np.unique(gid[topg])
    prof = {"n_groups": int(len(gg)), "median_n": float(np.median(G["n"].values[gg])), "median_sum": float(np.median(G["s"].values[gg])),
            "share_day_split": float(gate_g[gg].mean()), "share_eqref_all": float((G["neq"].values[gg] == G["n"].values[gg]).mean())}
    res = {"year": a.year, "n": n, "n_day_groups": ng, "hbos_refit_ok": hb_ok,
           "base_share_day_split": float(gate_g[gid].mean()), "base_share_month_split": float(gatem_g[gid].mean()),
           "injection": res_inj, "scenario_meta": meta, "rule_lift_top1pct": lift, "rule_base_rate": base.round(4).to_dict(),
           "jaccard_top1pct": jac, "gate_top1pct_profile": prof}
    json.dump(res, open(REPORTS / f"screen_group_{a.year}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.save(d / "score_group_gate.npy", R["G-GATE"].astype(np.float32)); np.save(d / "score_group_hbos.npy", sg.astype(np.float32))

    L = [f"# ตัวคัดกรองระดับการซื้อ (ceiling-aware) เทียบ anomaly detection รายสัญญา — ปีงบ {a.year} (n = {n:,})", "",
         f"- กลุ่มวัน (หน่วยงาน×ผู้ขาย×วัน) {ng:,} กลุ่ม; สัญญาในกลุ่มที่ยอดวัน > 500k {res['base_share_day_split']:.1%}, ยอดเดือน > 500k {res['base_share_month_split']:.1%}",
         "- การซื้อ \"ถูกพบ\" ถ้ามีสัญญาส่วนใดอยู่ใน top-k% ของสัญญาทั้งหมด; budget80 = สัดส่วนสัญญาที่ต้องตรวจเพื่อให้ recall = 0.8", "",
         "| สถานการณ์ | ตัวคัดกรอง | recall@top1% | recall@top5% | budget80 |", "|---|---|---|---|---|"]
    for sc_, rr in res_inj.items():
        for m, v in rr.items():
            L.append(f"| {sc_} | {m} | {v['recall_top1pct']:.2f} | {v['recall_top5pct']:.2f} | {v['budget_for_recall80']:.1%} |")
    L += ["", "รายละเอียดสถานการณ์ซอย: " + json.dumps(meta, ensure_ascii=False), "",
          "## Lift ของ red flag จริงใน top1%", "", "| ตัวคัดกรอง | " + " | ".join(c[2:] for c in rcols) + " |", "|---" * (len(rcols) + 1) + "|"]
    for m, v in lift.items():
        L.append(f"| {m} | " + " | ".join(f"{v[c]:.1f}" for c in rcols) + " |")
    L.append("| อัตราพื้นฐาน | " + " | ".join(f"{base[c]:.3f}" for c in rcols) + " |")
    L += ["", "Jaccard top1%: " + ", ".join(f"{k} {v:.2f}" for k, v in jac.items()), "", "กลุ่มใน top1% ของ G-GATE: " + json.dumps(prof, ensure_ascii=False)]
    (REPORTS / f"screen_group_{a.year}.md").write_text("\n".join(L), encoding="utf-8")
    log(f"→ screen_group_{a.year} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
