"""
model_iforest.py — Baseline (rule score) + Isolation Forest + SHAP + การประเมินเบื้องต้นแบบไม่มี label

ขอบเขตเริ่มต้น (--scope le500k): ปีงบ 2568, สัญญาวิธีเฉพาะเจาะจง ราคา ≤ 500,000 ที่ไม่ใช่หมวดการแพทย์
  และผู้ขายไม่ใช่หน่วยงานรัฐ/รัฐวิสาหกิจ/สหกรณ์ (STATE_VENDOR_KEYS, เลขภาษี 099…) — รอบที่สอง top 20 เป็น ปตท./NT
  เหตุผล: สัญญาเฉพาะเจาะจงที่ > 500k ส่วนใหญ่เป็นข้อยกเว้นตามกฎหมาย (ซื้อจากหน่วยงานรัฐ/รัฐวิสาหกิจ เช่น ปตท.)
  ถ้ารวมไว้ ฟีเจอร์ "เกินเพดาน" จะครอบงำโมเดล (รอบแรก top 20 เป็น ปตท. ทั้งหมด) → แยกวิเคราะห์ต่างหาก
  --scope all = ไม่จำกัดราคา
ขั้นตอน (แต่ละขั้นบันทึกผลลง data/processed/model_<tag>/ → รันซ้ำจะทำต่อจากขั้นที่ค้าง)
  prep    ดึงฟีเจอร์ → แปลง (log1p, impute median) → X.npy (float32) + ids.parquet + rule_score
  fit     Isolation Forest 3 seed (fit บนตัวอย่าง) → iforest_seed{s}.joblib
  score   anomaly score = -score_samples() ทุกแถว ทีละ chunk → score_seed{s}_{i}.npy
  shap    TreeSHAP บน anomaly score: top-N + สุ่ม N (ทุก seed, แถวชุดเดียวกัน) → shap_seed{s}.npy
  inject  ทดสอบใส่ความผิดปกติสังเคราะห์ 3 แบบ → recall@top1%
  report  scores_<tag>.parquet + reports/model_<tag>.md + reports/figures/*.png

ใช้งาน
  python src\\model_iforest.py                       # ทุกขั้น (ถ้ารันนาน สามารถหยุดแล้วรันใหม่ได้)
  python src\\model_iforest.py --year 2567 --stage score
หมายเหตุ: คะแนนสูง = "แตกต่างจากสัญญาส่วนใหญ่" ไม่ได้แปลว่าทุจริต
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

SEEDS = [0, 1, 2]
N_ESTIMATORS = 300
MAX_SAMPLES = 512
FIT_SAMPLE = 300_000
CHUNK = 400_000
SHAP_TOP = 600
SHAP_RAND = 600

# (ชื่อฟีเจอร์ในโมเดล, นิพจน์ SQL, คำอธิบายภาษาไทยสำหรับรายงาน)
FEATURES = [
    ("price_z_peer",        "price_z_peer",                                            "ราคาเทียบสัญญาประเภท/วิธีเดียวกัน (robust z)"),
    ("gap_to_500k",         "greatest(least((500000 - agreed_price) / 500000.0, 1), -5)", "ระยะห่างจากเพดาน 500k (ติดลบ = เกินเพดาน)"),
    ("ratio_price_ref",     "ratio_price_ref",                                         "ราคาตกลง / ราคากลาง"),
    ("ratio_price_budget",  "least(ratio_price_budget, 10)",                           "ราคาตกลง / งบประมาณ"),
    ("price_eq_ref",        "price_eq_ref",                                            "ราคาตกลงเท่าราคากลางพอดี"),
    ("is_round_1000",       "is_round_1000",                                           "ราคาลงตัวหลักพัน"),
    ("log_duration",        "ln(1 + greatest(duration_days, 0))",                      "ระยะสัญญา (log วัน)"),
    ("days_to_fy_end",      "days_to_fy_end",                                          "จำนวนวันถึงสิ้นปีงบ"),
    ("unit_hhi",            "unit_hhi",                                                "ความกระจุกตัวผู้ขายของหน่วยงาน (HHI)"),
    ("unit_share_near500k", "unit_share_near500k",                                     "สัดส่วนสัญญาใกล้เพดานของหน่วยงาน"),
    ("log_unit_n",          "ln(1 + unit_n_contracts)",                                "ขนาดหน่วยงาน (log จำนวนสัญญา)"),
    ("pair_value_share",    "pair_value_share",                                        "ส่วนแบ่งมูลค่าของผู้ขายนี้ในหน่วยงาน"),
    ("log_pair_n",          "ln(1 + pair_n_contracts)",                                "จำนวนสัญญาผู้ขายนี้กับหน่วยงานนี้ (log)"),
    ("log_split_n_day",     "ln(1 + split_n_sameday)",                                 "จำนวนสัญญาผู้ขาย/หน่วย/วันเดียวกัน (log)"),
    ("log_split_sum_day",   "ln(1 + split_sum_sameday)",                               "มูลค่ารวมผู้ขาย/หน่วย/วันเดียวกัน (log)"),
    ("log_split_n_month",   "ln(1 + split_n_month)",                                   "จำนวนสัญญาผู้ขาย/หน่วย/เดือนเดียวกัน (log)"),
    ("log_vendor_units",    "ln(1 + vendor_n_units)",                                  "จำนวนหน่วยงานที่ผู้ขายได้งาน (log)"),
    ("log_vendor_prov",     "ln(1 + vendor_n_provinces)",                              "จำนวนจังหวัดที่ผู้ขายได้งาน (log)"),
    ("vendor_log_value",    "vendor_log_value",                                        "มูลค่ารวมของผู้ขายทั้งปี (log10)"),
    ("winner_is_company",   "CASE WHEN winner_type IN ('company','partnership') THEN 1 ELSE 0 END", "ผู้ชนะเป็นนิติบุคคล"),
]
FNAMES = [f[0] for f in FEATURES]
FDESC = {f[0]: f[2] for f in FEATURES}

# Baseline: นับ flag ที่ตีความได้ (น้ำหนักเท่ากัน)
RULES = [
    ("near_500k",            "near_500k = 1"),
    ("split_flag",           "split_flag = 1"),
    ("split_flag_month",     "split_flag_month = 1"),
    ("price_gt_ref",         "coalesce(price_gt_ref, 0) = 1"),
    ("price_z_peer_gt3",     "price_z_peer > 3"),
    ("vendor_capture",       "coalesce(pair_value_share, 0) > 0.5 AND unit_n_contracts >= 20"),
    ("unit_hhi_gt05",        "coalesce(unit_hhi, 0) > 0.5"),
    ("september",            "is_sep = 1"),
]


# ผู้ขายที่เป็นหน่วยงานของรัฐ/รัฐวิสาหกิจ/สหกรณ์ — ระเบียบให้ซื้อด้วยวิธีเฉพาะเจาะจงได้โดยตรง (ข้อยกเว้น ไม่ใช่สัญญาณผิดปกติ)
# = เลขผู้เสียภาษีขึ้นต้น 099 (หน่วยงานรัฐ/องค์การ/สหกรณ์) หรือ winner_type = government หรือรัฐวิสาหกิจรูปบริษัทในรายการนี้
# (รายการสร้างจากผู้ขายที่ได้งาน ≥40 จังหวัดในปี 2568 — ยังไม่ครบทุกแห่ง ควรเทียบรายชื่อรัฐวิสาหกิจของ สคร.)
STATE_VENDOR_KEYS = ["ปตท", "ปตทน้ำมันและการค้าปลีก", "โทรคมนาคมแห่งชาติ", "ไปรษณีย์ไทย", "ทีโอที", "กสทโทรคมนาคม",
                     "อสมท", "ท่าอากาศยานไทย", "ธนาคารกรุงไทย", "ขนส่ง", "ธนารักษ์พัฒนาสินทรัพย์", "อินโนสเปซประเทศไทย"]
NOT_STATE = ("NOT (coalesce(tax_id, '') LIKE '099%' OR winner_type = 'government' OR entity_key IN ("
             + ", ".join(f"'{k}'" for k in STATE_VENDOR_KEYS) + "))")
SCOPES = {
    "le500k": f"method_cat = 'specific' AND NOT is_medical AND agreed_price <= 500000 AND {NOT_STATE}",
    "all":    f"method_cat = 'specific' AND NOT is_medical AND {NOT_STATE}",
}
SCOPE = "le500k"


def tagdir(year: str) -> Path:
    d = PROCESSED / f"model_{year}_{SCOPE}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ prep
def stage_prep(year, d: Path):
    """สตรีมจาก duckdb: ids → parquet โดยตรง, ฟีเจอร์ → np.float32 ที่จองไว้ (ไม่ผ่าน pandas เพื่อประหยัดหน่วยความจำ)"""
    if (d / "X.npy").exists():
        return
    src = str(PROCESSED / f"features_{year}.parquet").replace("\\", "/")
    con = duckdb.connect()
    # temp ของ duckdb ไปที่โฟลเดอร์ชั่วคราวของระบบ (ค่าเริ่มต้นคือ .tmp ในโฟลเดอร์ปัจจุบัน ซึ่งอาจช้ามากถ้าเป็น network/synced drive)
    tmpdir = Path(tempfile.gettempdir()) / "duckdb_egp"
    con.execute(f"SET temp_directory='{tmpdir.as_posix()}'")
    con.execute("SET threads=2; SET memory_limit='1500MB'; SET preserve_insertion_order=true")
    feat_sql = ", ".join(f"({e})::FLOAT AS {n}" for n, e, _ in FEATURES)
    rule_sql = ", ".join(f"(CASE WHEN {e} THEN 1 ELSE 0 END)::TINYINT AS r_{n}" for n, e in RULES)
    rule_sum = " + ".join(f"r_{n}" for n, _ in RULES)
    con.execute(f"""CREATE TABLE sel AS
        SELECT row_number() OVER () - 1 AS rid, project_id, contract_no, unit_key, entity_key, winner_type, agreed_price, sign_date,
               {feat_sql}, {rule_sql}
        FROM read_parquet('{src}') WHERE {SCOPES[SCOPE]}""")
    n = con.execute("SELECT count(*) FROM sel").fetchone()[0]
    log(f"prep: {n:,} rows")
    ids_path = str(d / "ids.parquet").replace("\\", "/")
    con.execute(f"""COPY (SELECT rid, project_id, contract_no, unit_key, entity_key, winner_type, agreed_price, sign_date,
                    ({rule_sum})::TINYINT AS rule_score, {', '.join('r_' + n for n, _ in RULES)} FROM sel)
                    TO '{ids_path}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    med = np.array(con.execute("SELECT " + ", ".join(f"median({f})" for f in FNAMES) + " FROM sel").fetchone(), dtype=np.float32)
    nan_share = con.execute("SELECT " + ", ".join(f"avg(CASE WHEN {f} IS NULL OR isnan({f}) THEN 1.0 ELSE 0 END)" for f in FNAMES) + " FROM sel").fetchone()
    X = np.lib.format.open_memmap(d / "X.tmp.npy", mode="w+", dtype=np.float32, shape=(n, len(FNAMES)))
    step = 500_000
    for lo in range(0, n, step):
        cols = con.execute(f"SELECT {', '.join(FNAMES)} FROM sel WHERE rid >= {lo} AND rid < {lo + step} ORDER BY rid").fetchnumpy()
        blk = np.column_stack([np.asarray(cols[f], dtype=np.float32) if not hasattr(cols[f], "filled")
                               else cols[f].filled(np.nan).astype(np.float32) for f in FNAMES])
        bad = ~np.isfinite(blk)
        blk[bad] = np.take(med, np.where(bad)[1])
        X[lo:lo + len(blk)] = blk
    X.flush(); del X
    (d / "X.tmp.npy").replace(d / "X.npy")
    json.dump({"n": n, "features": FNAMES, "impute_median": med.tolist(), "nan_share": list(nan_share)},
              open(d / "prep.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ fit
def stage_fit(d: Path):
    from sklearn.ensemble import IsolationForest
    X = np.load(d / "X.npy", mmap_mode="r")
    for s in SEEDS:
        f = d / f"iforest_seed{s}.joblib"
        if f.exists():
            continue
        rng = np.random.default_rng(s)
        sub = np.asarray(X[np.sort(rng.choice(len(X), size=min(FIT_SAMPLE, len(X)), replace=False))])
        m = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES, random_state=s, n_jobs=2).fit(sub)
        joblib.dump(m, f)
        log(f"fit seed {s}")


# ------------------------------------------------------------------ score
def stage_score(d: Path, budget_s: float):
    X = np.load(d / "X.npy", mmap_mode="r")
    t0 = time.time()
    n_chunks = (len(X) + CHUNK - 1) // CHUNK
    for s in SEEDS:
        if (d / f"score_seed{s}.npy").exists():
            continue
        m = joblib.load(d / f"iforest_seed{s}.joblib")
        for i in range(n_chunks):
            f = d / f"score_seed{s}_{i}.npy"
            if f.exists():
                continue
            if time.time() - t0 > budget_s:
                log("score: หมดเวลาสำหรับรอบนี้ — รันซ้ำเพื่อทำต่อ")
                return False
            np.save(f, (-m.score_samples(np.asarray(X[i * CHUNK:(i + 1) * CHUNK]))).astype(np.float32))
            log(f"score seed {s} chunk {i + 1}/{n_chunks}")
        np.save(d / f"score_seed{s}.npy", np.concatenate([np.load(d / f"score_seed{s}_{i}.npy") for i in range(n_chunks)]))
        for i in range(n_chunks):
            (d / f"score_seed{s}_{i}.npy").unlink()
    return True


# ------------------------------------------------------------------ shap
def stage_shap(d: Path):
    import shap
    X = np.load(d / "X.npy", mmap_mode="r")
    sc = np.load(d / "score_seed0.npy")
    if not (d / "shap_rows.npy").exists():
        top = np.argsort(-sc)[:SHAP_TOP]
        rng = np.random.default_rng(42)
        rest = np.setdiff1d(np.arange(len(sc)), top)
        rows = np.concatenate([top, rng.choice(rest, SHAP_RAND, replace=False)])
        np.save(d / "shap_rows.npy", rows)
    rows = np.load(d / "shap_rows.npy")
    Xs = np.asarray(X[np.sort(rows)])
    order = np.argsort(np.argsort(rows))  # map กลับลำดับเดิม
    Xs = Xs[order]
    for s in SEEDS:
        f = d / f"shap_seed{s}.npy"
        if f.exists():
            continue
        m = joblib.load(d / f"iforest_seed{s}.joblib")
        t0 = time.time()
        ex = shap.TreeExplainer(m)
        sv = ex.shap_values(Xs)
        # TreeSHAP ของ IsolationForest อธิบาย "path length" (สูง = ปกติ) → กลับเครื่องหมายให้ค่าบวก = ดันให้ผิดปกติ
        np.save(f, (-sv).astype(np.float32))
        log(f"shap seed {s}: {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ inject
def stage_inject(d: Path):
    f = d / "inject.json"
    if f.exists():
        return
    X = np.load(d / "X.npy", mmap_mode="r")
    sc0 = np.load(d / "score_seed0.npy")
    m = joblib.load(d / "iforest_seed0.joblib")
    ids = pd.read_parquet(d / "ids.parquet", columns=["rule_score"])
    thr = np.quantile(sc0, 0.99)
    rng = np.random.default_rng(7)
    j = {n: i for i, n in enumerate(FNAMES)}
    # ใช้แถว "ปกติ" (score ต่ำกว่า median) เป็นฐาน
    base_pool = np.where(sc0 < np.median(sc0))[0]
    res = {}
    scen = {
        "ซอยสัญญาใต้เพดาน": {"gap_to_500k": 0.004, "log_split_n_day": np.log1p(6), "log_split_sum_day": np.log1p(2_990_000),
                             "log_split_n_month": np.log1p(8), "is_round_1000": 0},
        "ราคาสูงกว่ากลุ่ม+ไม่มีส่วนลด": {"price_z_peer": 4.0, "price_eq_ref": 1, "ratio_price_ref": 1.0, "ratio_price_budget": 1.0,
                                          "is_round_1000": 1},
        "ผูกขาดผู้ขายในหน่วยงาน": {"pair_value_share": 0.9, "unit_hhi": 0.82, "log_pair_n": np.log1p(150),
                                  "log_vendor_units": np.log1p(1), "log_vendor_prov": np.log1p(1)},
    }
    for name, mods in scen.items():
        rows = rng.choice(base_pool, 500, replace=False)
        Xi = np.asarray(X[np.sort(rows)]).copy()
        for k, v in mods.items():
            Xi[:, j[k]] = v
        si = -m.score_samples(Xi)
        pct = (np.searchsorted(np.sort(sc0), si) / len(sc0))
        res[name] = {"recall_top1pct": float((si >= thr).mean()), "median_percentile": float(np.median(pct)),
                     "recall_top5pct": float((si >= np.quantile(sc0, 0.95)).mean())}
        log(f"inject {name}: {res[name]}")
    # baseline (rule) ทำได้แค่ตาม flag — สำหรับเทียบ: สัดส่วนสัญญาจริงที่ rule_score สูง
    res["_rule_score_dist"] = ids["rule_score"].value_counts().sort_index().to_dict()
    json.dump(res, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


# ------------------------------------------------------------------ report
def stage_report(year, d: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    ids = pd.read_parquet(d / "ids.parquet", columns=["rid", "project_id", "contract_no", "entity_key", "winner_type", "agreed_price", "rule_score"]
                          + [f"r_{n}" for n, _ in RULES]).sort_values("rid").set_index("rid")
    scores = {s: np.load(d / f"score_seed{s}.npy") for s in SEEDS}
    ids["if_score"] = np.mean([scores[s] for s in SEEDS], axis=0)
    ids["if_pct"] = ids["if_score"].rank(pct=True)
    n = len(ids)
    k1 = int(n * 0.01)

    # ความเสถียรข้าม seed
    tops = {s: set(np.argsort(-scores[s])[:k1]) for s in SEEDS}
    jacc = {f"{a}-{b}": len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]}
    rank_corr = {f"{a}-{b}": float(spearmanr(scores[a][::50], scores[b][::50]).correlation) for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]}

    # SHAP
    rows = np.load(d / "shap_rows.npy")
    sv = {s: np.load(d / f"shap_seed{s}.npy") for s in SEEDS}
    imp = {s: np.abs(sv[s]).mean(axis=0) for s in SEEDS}
    imp_top = {s: np.abs(sv[s][:SHAP_TOP]).mean(axis=0) for s in SEEDS}
    shap_rank_corr = {f"{a}-{b}": float(spearmanr(imp[a], imp[b]).correlation) for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]}
    # ความเสถียรรายสัญญา: top-3 เหตุผลตรงกันข้าม seed กี่ %
    top3 = {s: np.argsort(-sv[s][:SHAP_TOP], axis=1)[:, :3] for s in SEEDS}
    local_overlap = float(np.mean([len(set(top3[0][i]) & set(top3[1][i])) / 3 for i in range(SHAP_TOP)]))
    # additivity: sum(shap) ≈ score - E[score]
    X = np.load(d / "X.npy", mmap_mode="r")
    gimp = pd.Series(np.mean([imp_top[s] for s in SEEDS], axis=0), index=FNAMES).sort_values(ascending=False)

    # เทียบกับ baseline
    top_if = ids.nlargest(k1, "if_score")
    rule_hi = ids["rule_score"] >= 2
    overlap = {
        "share_top1pct_with_rule>=2": float((top_if["rule_score"] >= 2).mean()),
        "share_all_with_rule>=2": float(rule_hi.mean()),
        "spearman_if_vs_rule": float(spearmanr(ids["if_score"][::50], ids["rule_score"][::50]).correlation),
        "mean_if_pct_by_rule": ids.groupby("rule_score")["if_pct"].mean().round(3).to_dict(),
        "flag_rate_top1pct": {c: float(top_if[c].mean()) for c in ids.columns if c.startswith("r_")},
        "flag_rate_all": {c: float(ids[c].mean()) for c in ids.columns if c.startswith("r_")},
    }
    inj = json.load(open(d / "inject.json", encoding="utf-8"))

    # คำอธิบายรายสัญญา top 20 (pseudonymize บุคคล/ร้าน)
    pos = {r: i for i, r in enumerate(rows)}
    top20 = ids.nlargest(20, "if_score").copy()
    def reasons(r):
        i = pos.get(r)
        if i is None:
            return "-"
        v = sv[0][i]
        o = np.argsort(-v)[:3]
        return "; ".join(f"{FDESC[FNAMES[k]]} = {X[r, k]:.3g} (+{v[k]:.3f})" for k in o)
    top20["เหตุผล 3 อันดับ (SHAP)"] = [reasons(r) for r in top20.index]
    top20["ผู้ขาย"] = [e if wt in ("company", "partnership", "government") else f"(บุคคล/ร้าน #{abs(hash(e)) % 100000})"
                       for e, wt in zip(top20["entity_key"], top20["winner_type"])]

    # บันทึกคะแนน
    out = ids[["project_id", "contract_no", "if_score", "if_pct", "rule_score"] + [c for c in ids.columns if c.startswith("r_")]]
    out.reset_index().to_parquet(PROCESSED / f"scores_{year}_{SCOPE}.parquet", index=False)

    # รูป
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    gi = gimp.iloc[::-1]
    ax.barh(list(gi.index), gi.values, color="#3b6ea8")
    ax.set_xlabel("mean |SHAP| (top 600 anomalies, avg 3 seeds)")
    ax.set_title(f"Global feature importance — FY{year}, scope={SCOPE}")
    fig.tight_layout(); fig.savefig(FIGS / f"shap_global_{year}_{SCOPE}.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ids["if_score"], bins=200, color="#3b6ea8"); ax.axvline(np.quantile(ids["if_score"], 0.99), color="#c0392b", ls="--")
    ax.set_yscale("log"); ax.set_xlabel("anomaly score (mean of 3 seeds)"); ax.set_title(f"Score distribution FY{year} (red = top 1%)")
    fig.tight_layout(); fig.savefig(FIGS / f"score_hist_{year}_{SCOPE}.png", dpi=150); plt.close(fig)

    rep = {"n": n, "k1": k1, "jaccard_top1pct": jacc, "spearman_scores": rank_corr, "shap_global_rank_corr": shap_rank_corr,
           "shap_local_top3_overlap_seed0_1": local_overlap, "global_importance": gimp.round(4).to_dict(),
           "baseline_comparison": overlap, "injection": inj}
    json.dump(rep, open(REPORTS / f"model_{year}_{SCOPE}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

    L = [f"# Isolation Forest + SHAP — ปีงบ {year} · scope = {SCOPE} ({SCOPES[SCOPE]})", "",
         f"- จำนวนสัญญา: **{n:,}** · ฟีเจอร์ {len(FNAMES)} ตัว · IF {N_ESTIMATORS} ต้น, max_samples={MAX_SAMPLES}, fit บนตัวอย่าง {FIT_SAMPLE:,} แถว × {len(SEEDS)} seed; คะแนน = ค่าเฉลี่ย 3 seed",
         f"- top 1% = {k1:,} สัญญา", "", f"![](figures/score_hist_{year}_{SCOPE}.png)", "",
         "## ความสำคัญของฟีเจอร์ (mean |SHAP| ใน top 600)", f"![](figures/shap_global_{year}_{SCOPE}.png)", "", "| ฟีเจอร์ | ความหมาย | mean abs SHAP |", "|---|---|---|"]
    L += [f"| {k} | {FDESC[k]} | {v:.4f} |" for k, v in gimp.items()]
    L += ["", "## ความเสถียร (stability)",
          f"- Jaccard ของชุด top 1% ข้าม seed: {', '.join(f'{k}: {v:.3f}' for k, v in jacc.items())}",
          f"- Spearman ของคะแนนข้าม seed: {', '.join(f'{k}: {v:.3f}' for k, v in rank_corr.items())}",
          f"- Spearman ของลำดับความสำคัญ SHAP (global) ข้าม seed: {', '.join(f'{k}: {v:.3f}' for k, v in shap_rank_corr.items())}",
          f"- เหตุผล top-3 ต่อสัญญาตรงกันระหว่าง seed 0 กับ 1: **{local_overlap:.1%}** (เฉลี่ยใน top 600)", "",
          "## เทียบกับ baseline (rule score = จำนวน flag)",
          f"- rule ที่ใช้: {', '.join(n for n, _ in RULES)}",
          f"- สัดส่วนที่ rule_score ≥ 2: ทั้งหมด {overlap['share_all_with_rule>=2']:.3f} vs ใน IF top 1% {overlap['share_top1pct_with_rule>=2']:.3f}",
          f"- Spearman(IF score, rule score) = {overlap['spearman_if_vs_rule']:.3f}",
          f"- percentile เฉลี่ยของ IF แยกตาม rule_score: {overlap['mean_if_pct_by_rule']}",
          "", "| flag | อัตราในทั้งหมด | อัตราใน IF top 1% |", "|---|---|---|"]
    L += [f"| {c[2:]} | {overlap['flag_rate_all'][c]:.4f} | {overlap['flag_rate_top1pct'][c]:.4f} |" for c in overlap["flag_rate_all"]]
    L += ["", "## Injection test (ใส่ความผิดปกติสังเคราะห์ลงในสัญญาปกติ 500 แถวต่อแบบ)", "| สถานการณ์ | recall@top1% | recall@top5% | percentile มัธยฐาน |", "|---|---|---|---|"]
    L += [f"| {k} | {v['recall_top1pct']:.3f} | {v['recall_top5pct']:.3f} | {v['median_percentile']:.3f} |" for k, v in inj.items() if not k.startswith("_")]
    L += ["", "## 20 สัญญาคะแนนสูงสุด (ผู้ขายบุคคล/ร้านแทนด้วยรหัส)", "| project_id | ราคา | ผู้ขาย | rule | เหตุผล 3 อันดับ (SHAP) |", "|---|---|---|---|---|"]
    L += [f"| {r.project_id} | {r.agreed_price:,.0f} | {r['ผู้ขาย']} | {r.rule_score} | {r['เหตุผล 3 อันดับ (SHAP)']} |" for _, r in top20.iterrows()]
    L += ["", "> คะแนนสูง = สัญญาที่ **แตกต่าง** จากสัญญาส่วนใหญ่ในชุดนี้ ไม่ใช่ข้อสรุปว่ามีการทุจริต ต้องให้ผู้เชี่ยวชาญตรวจ (precision@k)"]
    (REPORTS / f"model_{year}_{SCOPE}.md").write_text("\n".join(L), encoding="utf-8")
    log(f"report → reports/model_{year}_{SCOPE}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2568")
    ap.add_argument("--stage", default="all", choices=["all", "prep", "fit", "score", "shap", "inject", "report"])
    ap.add_argument("--scope", default="le500k", choices=list(SCOPES))
    ap.add_argument("--budget", type=float, default=1e9, help="เวลาสูงสุด (วินาที) สำหรับขั้น score ต่อรอบ")
    a = ap.parse_args()
    global SCOPE
    SCOPE = a.scope
    d = tagdir(a.year)
    st = a.stage
    if st in ("all", "prep"): stage_prep(a.year, d)
    if st in ("all", "fit"): stage_fit(d)
    if st in ("all", "score"):
        if not stage_score(d, a.budget):
            return
    if st in ("all", "shap"): stage_shap(d)
    if st in ("all", "inject"): stage_inject(d)
    if st in ("all", "report"): stage_report(a.year, d)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
