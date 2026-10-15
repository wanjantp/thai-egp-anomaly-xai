"""
bunching.py — วัดการกระจุกตัวของมูลค่าสัญญาใต้เพดานวงเงิน (bunching estimator)

ขั้นที่ 1  aggregate : contracts_YYYY → data/processed/bunching_bins.parquet
           นับสัญญาตาม (ปีงบ, เดือนลงนาม, ช่วงก่อน/หลังปฏิรูป, กลุ่มวิธี, ช่วงราคา 1,000 บาท)
           ช่วงราคาใช้ "ขอบบน" (ceil) เพราะเพดานเป็น "ไม่เกิน" → 500,000 พอดีอยู่ในช่วงใต้เพดาน
           นับคู่ขนานกับ "งบประมาณ" เพื่อแยกการตั้งงบที่เพดาน vs การกดราคาให้ต่ำกว่าเพดาน
ขั้นที่ 2  estimate  : bunching estimator แบบ Chetty et al. (2011) / Kleven & Waseem (2013)
           - รวมเป็นช่วง 5,000 บาท (หรือ 1,000 สำหรับเพดาน 100k)
           - fit polynomial อันดับ p บนจำนวนสัญญานอกหน้าต่าง [T - zL, T + zU] พร้อม dummy เลขกลม (หาร 10k/50k/100k ลงตัว)
           - excess mass B = Σ(จริง − counterfactual) ใน [T − zL, T];  b = B / counterfactual เฉลี่ยต่อช่วงในหน้าต่างใต้เพดาน
           - missing mass M = Σ(counterfactual − จริง) ใน (T, T + zU]
           - CI จาก bootstrap residual
           → reports/bunching.md + reports/figures/bunching_*.png

เพดาน (ตรวจตัวบทแล้ว 14 ต.ค. 2569; ดู Project doc research-design.md)
  ก่อน 23 ส.ค. 2560: ตกลงราคา ≤ 100,000 (ระเบียบสำนักนายกฯ 2535); สอบราคา ≤ 2,000,000
  ตั้งแต่ 23 ส.ค. 2560: เฉพาะเจาะจง ≤ 500,000 (พ.ร.บ. 2560 ม.56(2)(ข)); วงเงินเล็กน้อย ≤ 100,000

ใช้งาน
  python src\\bunching.py aggregate            # ทุกปีที่มี data/interim/contracts_YYYY
  python src\\bunching.py estimate
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
BINS = PROCESSED / "bunching_bins.parquet"
REFORM = "2017-08-23"          # พ.ร.บ. จัดซื้อจัดจ้างฯ 2560 มีผล
ADMIN_SHIFT = "2015-10-01"     # จุดเปลี่ยนที่เห็นในข้อมูล: ตกลงราคาได้ถึง 500k ตามหนังสือ กวพ. (มาตรการปี 2558–59; วันที่มีผลแน่นอนยังไม่ยืนยัน)
# regime: pre = ตกลงราคา ≤ 100k (ระเบียบ 2535) | transition = ตกลงราคา ≤ 500k ตามหนังสือเวียน | act = เฉพาะเจาะจง ≤ 500k (พ.ร.บ. 2560)
REGIME_SQL = f"""CASE WHEN sign_date < DATE '{ADMIN_SHIFT}' THEN 'pre'
                      WHEN sign_date < DATE '{REFORM}' THEN 'transition' ELSE 'act' END"""
KNOWN_T = {100_000: (10_000, 20_000), 500_000: (50_000, 100_000), 2_000_000: (200_000, 400_000)}
MAX_PRICE = 3_000_000

# กลุ่มวิธี: discretion = วิธีที่ใช้ดุลพินิจ (ก่อน: ตกลงราคา, หลัง: เฉพาะเจาะจง), competitive = สอบราคา/ประกวด/e-bidding/e-market/คัดเลือก
METHOD_CLS = """CASE
  WHEN method IN ('ตกลงราคา', 'เฉพาะเจาะจง') THEN 'discretion'
  WHEN method IN ('สอบราคา', 'ประกวดราคา', 'คัดเลือก') OR method LIKE 'ประกวดราคา%' OR method LIKE 'ตลาดอิเล็กทรอนิกส์%' THEN 'competitive'
  WHEN method IN ('พิเศษ', 'กรณีพิเศษ') THEN 'special'
  ELSE 'other' END"""


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def connect():
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{(Path(tempfile.gettempdir()) / 'duckdb_egp').as_posix()}'")
    con.execute("SET threads=2; SET memory_limit='2GB'")
    return con


# ------------------------------------------------------------------ aggregate
def aggregate():
    con = connect()
    years = sorted(int(p.name.split("_")[1]) for p in INTERIM.glob("contracts_25??") if p.is_dir())
    parts = []
    for y in years:
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        t0 = time.time()
        df = con.execute(f"""
            WITH c AS (
              SELECT DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)
                     fiscal_year, sign_date, method, agreed_price, budget, ref_price, agency
              FROM read_parquet('{src}')
              WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE}
            )
            SELECT fiscal_year,
                   strftime(date_trunc('month', sign_date), '%Y-%m') AS ym,
                   {REGIME_SQL} AS regime,
                   {METHOD_CLS} AS mcls,
                   CASE WHEN agency LIKE 'องค์การบริหารส่วน%' OR agency LIKE 'เทศบาล%' OR agency = 'กรุงเทพมหานคร' OR agency = 'เมืองพัทยา'
                        THEN 'local' ELSE 'central' END AS gov,
                   (ceil(agreed_price / 1000) * 1000)::INT AS bin,
                   count(*)::INT AS n,
                   sum(CASE WHEN budget = agreed_price THEN 1 ELSE 0 END)::INT AS n_price_eq_budget,
                   sum(CASE WHEN ref_price = agreed_price THEN 1 ELSE 0 END)::INT AS n_price_eq_ref,
                   sum(CASE WHEN budget > 500000 THEN 1 ELSE 0 END)::INT AS n_budget_gt500k,
                   sum(CASE WHEN budget > 100000 THEN 1 ELSE 0 END)::INT AS n_budget_gt100k
            FROM c GROUP BY ALL
        """).df()
        # การกระจายของ "งบประมาณ" (ไม่ใช่ราคา) — ใช้ตรวจว่าการกระจุกเกิดตั้งแต่ตอนตั้งงบหรือไม่
        dfb = con.execute(f"""
            WITH c AS (
              SELECT DISTINCT ON (project_id) fiscal_year, sign_date, method, budget, agency
              FROM read_parquet('{src}')
              WHERE sign_date IS NOT NULL AND budget > 0 AND budget <= {MAX_PRICE}
            )
            SELECT fiscal_year, strftime(date_trunc('month', sign_date), '%Y-%m') AS ym,
                   {REGIME_SQL} AS regime,
                   {METHOD_CLS} AS mcls, 'all' AS gov,
                   (ceil(budget / 1000) * 1000)::INT AS bin, count(*)::INT AS n
            FROM c GROUP BY ALL
        """).df()
        df["var"] = "price"; dfb["var"] = "budget"
        parts += [df, dfb]
        log(f"aggregate {y}: {df['n'].sum():,} contracts, {time.time() - t0:.0f}s")
    out = pd.concat(parts, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(BINS, index=False)
    log(f"→ {BINS} ({len(out):,} rows, years {years})")


# ------------------------------------------------------------------ estimator
def bunch(counts: pd.Series, T: int, width: int, zL: int, zU: int, fit_lo: int, fit_hi: int,
          degree: int = 7, n_boot: int = 200, seed: int = 0, return_arrays: bool = False) -> dict:
    """counts: index = ขอบบนของช่วง 1,000 บาท, value = จำนวนสัญญา
    คืน excess mass ใต้เพดาน / missing mass เหนือเพดาน พร้อม bootstrap SE"""
    s = counts.copy()
    s.index = (np.ceil(s.index / width) * width).astype(int)
    s = s.groupby(level=0).sum()
    grid = np.arange(fit_lo + width, fit_hi + width, width)
    y = s.reindex(grid, fill_value=0).astype(float).values
    x = (grid - T) / (fit_hi - fit_lo)
    below = (grid > T - zL) & (grid <= T)
    above = (grid > T) & (grid <= T + zU)
    excl = below | above
    # ไม่ใช้ช่วงรอบเพดานจริงอื่น ๆ ในการ fit (สำคัญสำหรับ placebo)
    fit_mask = ~excl
    for T2, (zl2, zu2) in KNOWN_T.items():
        if T2 != T:
            fit_mask &= ~((grid > T2 - zl2) & (grid <= T2 + zu2))
    X = np.column_stack([x ** k for k in range(degree + 1)]
                        + [(grid % m == 0).astype(float) for m in (10_000, 50_000, 100_000) if m > width])
    # dummy เลขกลมในหน้าต่างใช้ค่าสัมประสิทธิ์จากนอกหน้าต่าง
    Xf, yf = X[fit_mask], y[fit_mask]
    beta, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
    cf = X @ beta
    resid = yf - Xf @ beta

    def stats(yv, cfv):
        B = (yv[below] - cfv[below]).sum()
        M = (cfv[above] - yv[above]).sum()
        h0 = cfv[below].mean()
        return B, M, B / h0 if h0 > 0 else np.nan

    B, M, b = stats(y, cf)
    rng = np.random.default_rng(seed)
    boots = []
    if return_arrays:
        sig = float(np.std(resid, ddof=X.shape[1])) if len(resid) > X.shape[1] else float(np.std(resid))
        return {"_grid": grid, "_y": y, "_cf": cf, "_sigma": sig}
    if n_boot == 0:
        return {"T": T, "width": width, "zL": zL, "zU": zU, "degree": degree, "fit_lo": fit_lo, "fit_hi": fit_hi,
                "excess_B": float(B), "missing_M": float(M), "b_ratio": float(b), "b_ratio_se": float("nan")}
    for _ in range(n_boot):
        yb = y.copy()
        yb[fit_mask] = (Xf @ beta) + rng.choice(resid, size=len(resid), replace=True)
        bb, *_ = np.linalg.lstsq(Xf, yb[fit_mask], rcond=None)
        boots.append(stats(y, X @ bb))
    boots = np.array(boots)
    se = np.nanstd(boots, axis=0)
    n_total = y.sum()
    return {"T": T, "width": width, "zL": zL, "zU": zU, "degree": degree,
            "excess_B": float(B), "excess_B_se": float(se[0]), "missing_M": float(M), "missing_M_se": float(se[1]),
            "b_ratio": float(b), "b_ratio_se": float(se[2]),
            "actual_below": float(y[below].sum()), "cf_below": float(cf[below].sum()),
            "n_in_range": float(n_total), "grid": grid.tolist(), "y": y.tolist(), "cf": cf.tolist()}


# ------------------------------------------------------------------ data-driven window
def choose_window(counts: pd.Series, T: int, width: int, fit_lo: int, fit_hi: int, degree: int = 7,
                  z_crit: float = 1.96, consec: int = 2, max_iter: int = 60) -> dict:
    """เลือกหน้าต่าง [T - zL, T + zU] จากข้อมูล (แนว Bosch, Dekker & Strohmaier 2020):
    เริ่มที่ zL = zU = 1 ช่วง → fit counterfactual นอกหน้าต่าง → ถ้าช่วงถัดลงไปใต้หน้าต่าง (consec ช่วงติดกัน)
    มีจำนวนจริง − counterfactual > z_crit·σ (σ = sd ของ residual นอกหน้าต่าง) ให้ขยาย zL อีก 1 ช่วง;
    ด้านบนทำกลับกัน (จำนวนจริงต่ำกว่า counterfactual เกิน z_crit·σ → ขยาย zU); วนจนทั้งสองด้านไม่เปลี่ยน"""
    zL, zU = width, width
    hist = []
    for it in range(max_iter):
        r = bunch(counts, T, width=width, zL=zL, zU=zU, fit_lo=fit_lo, fit_hi=fit_hi, degree=degree, n_boot=0, return_arrays=True)
        grid, y, cf, sigma = r["_grid"], r["_y"], r["_cf"], r["_sigma"]
        res = y - cf
        lo_edge = T - zL                      # ช่วงใต้หน้าต่าง: grid <= lo_edge (ขอบบน ≤ T - zL)
        below = [g for g in grid if g <= lo_edge][::-1][:consec]
        hi_edge = T + zU
        above = [g for g in grid if g > hi_edge][:consec]
        idx = {g: i for i, g in enumerate(grid)}
        grow_L = any(res[idx[g]] > z_crit * sigma for g in below) and T - zL - width > fit_lo + 3 * width
        grow_U = any(res[idx[g]] < -z_crit * sigma for g in above) and T + zU + width < fit_hi - 3 * width
        hist.append((zL, zU))
        if not grow_L and not grow_U:
            break
        zL += width if grow_L else 0
        zU += width if grow_U else 0
    return {"zL": zL, "zU": zU, "iterations": it + 1, "sigma": float(sigma), "converged": not (grow_L or grow_U)}


def sel(df, **kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= df[k].isin(v if isinstance(v, (list, tuple, set)) else [v]).values
    return df[m].groupby("bin")["n"].sum()


SPECS = {
    # เพดาน 500k: ช่วงละ 5,000; หน้าต่างใต้ 50k เหนือ 100k; fit 200k–1M
    500_000: dict(width=5_000, zL=50_000, zU=100_000, fit_lo=200_000, fit_hi=1_000_000),
    # เพดาน 100k: ช่วงละ 1,000; หน้าต่างใต้ 10k เหนือ 20k; fit 40k–250k
    100_000: dict(width=1_000, zL=10_000, zU=20_000, fit_lo=40_000, fit_hi=250_000),
    # เพดาน 2M (สอบราคา → ประกวดราคา ก่อนปฏิรูป): ช่วงละ 20,000
    2_000_000: dict(width=20_000, zL=200_000, zU=400_000, fit_lo=1_000_000, fit_hi=3_000_000),
}


def estimate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_parquet(BINS)
    pr = df[df["var"] == "price"]
    bu = df[df["var"] == "budget"]
    years = sorted(pr["fiscal_year"].unique())
    res = []

    def run(label, counts, T, **over):
        spec = {**SPECS[T], **over}
        if counts.sum() < 1000:
            return None
        r = bunch(counts, T, **spec)
        r.update(label=label)
        res.append(r)
        return r

    # 1) สามช่วงนโยบาย × เพดาน (ราคา ทุกวิธี)
    for regime in ("pre", "transition", "act"):
        for T in (100_000, 500_000, 2_000_000):
            run(f"{regime}|all methods|price|T={T:,}", sel(pr, regime=regime), T)
    # 2) รายปีงบ
    for y in years:
        for T in (100_000, 500_000):
            run(f"FY{y}|all methods|price|T={T:,}", sel(pr, fiscal_year=y), T)
    # 3) งบประมาณ vs ราคา (ช่วง act ที่ 500k; ช่วง pre ที่ 100k)
    run("act|all methods|budget|T=500,000", sel(bu, regime="act"), 500_000)
    run("pre|all methods|budget|T=100,000", sel(bu, regime="pre"), 100_000)
    # 4) ส่วนกลาง vs ท้องถิ่น (ช่วง act, 500k)
    for g in ("central", "local"):
        run(f"act|{g}|price|T=500,000", sel(pr, regime="act", gov=g), 500_000)
    # 5) placebo
    for T in (300_000, 700_000):
        SPECS[T] = dict(width=5_000, zL=25_000, zU=25_000, fit_lo=T - 150_000, fit_hi=T + 150_000, degree=5)
        run(f"act|all methods|price|PLACEBO T={T:,}", sel(pr, regime="act"), T)

    # ---- รายเดือน: สัดส่วนสัญญาใน (90k,100k] และ (450k,500k] ต่อสัญญาใน 50k–1M
    m = pr[(pr["bin"] > 50_000) & (pr["bin"] <= 1_000_000)].groupby(["ym", "bin"])["n"].sum().reset_index()
    tot = m.groupby("ym")["n"].sum()
    s100 = m[(m["bin"] > 90_000) & (m["bin"] <= 100_000)].groupby("ym")["n"].sum() / tot
    s500 = m[(m["bin"] > 450_000) & (m["bin"] <= 500_000)].groupby("ym")["n"].sum() / tot
    monthly = pd.DataFrame({"share_90_100k": s100, "share_450_500k": s500, "n_50k_1m": tot}).fillna(0)
    monthly.to_csv(REPORTS / "bunching_monthly.csv")

    # ---- รูป
    FIGS.mkdir(parents=True, exist_ok=True)
    def plot(r, fname, title):
        fig, ax = plt.subplots(figsize=(8, 4))
        g = np.array(r["grid"]) / 1000
        ax.bar(g, r["y"], width=r["width"] / 1000 * 0.9, color="#3b6ea8", label="actual")
        ax.plot(g, r["cf"], color="#c0392b", lw=1.5, label="counterfactual")
        ax.axvline(r["T"] / 1000, color="k", ls="--", lw=1)
        ax.axvspan((r["T"] - r["zL"]) / 1000, (r["T"] + r["zU"]) / 1000, color="grey", alpha=0.12)
        ax.set_xlabel("contract value (thousand THB, bin upper edge)"); ax.set_ylabel("contracts")
        ax.set_title(f"{title}\nb = {r['b_ratio']:.2f} (se {r['b_ratio_se']:.2f}); excess {r['excess_B']:,.0f}; missing {r['missing_M']:,.0f}", fontsize=9)
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGS / fname, dpi=140); plt.close(fig)
    lookup = {r["label"]: r for r in res}
    for regime in ("pre", "transition", "act"):
        for T in (100_000, 500_000):
            r = lookup.get(f"{regime}|all methods|price|T={T:,}")
            if r: plot(r, f"bunching_{regime}_{T // 1000}k.png", f"{regime}-reform, threshold {T:,}")
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = pd.to_datetime(monthly.index)
    ax.plot(idx, monthly["share_90_100k"], label="share in (90k,100k]", color="#3b6ea8")
    ax.plot(idx, monthly["share_450_500k"], label="share in (450k,500k]", color="#c0392b")
    for (d, t), yf in zip(((ADMIN_SHIFT, " FY2559: negotiated-price cap 500k (circular 299)"), (REFORM, " 2017-08-23 Procurement Act")), (0.95, 0.85)):
        ax.axvline(pd.Timestamp(d), color="k", ls="--", lw=1); ax.text(pd.Timestamp(d), ax.get_ylim()[1] * yf, t, fontsize=8)
    ax.set_ylabel("share of contracts 50k–1M"); ax.legend(fontsize=8); ax.set_title("Monthly share of contracts just below each threshold (all methods)")
    fig.tight_layout(); fig.savefig(FIGS / "bunching_monthly.png", dpi=140); plt.close(fig)

    # ---- รายงาน
    tab = pd.DataFrame([{k: r[k] for k in ("label", "b_ratio", "b_ratio_se", "excess_B", "excess_B_se", "missing_M", "missing_M_se", "actual_below", "cf_below")} for r in res])
    tab.to_csv(REPORTS / "bunching_results.csv", index=False)
    L = ["# Bunching ที่เพดานวงเงิน", "",
         f"- ข้อมูล: ปีงบ {', '.join(map(str, years))} (สัญญาราคา ≤ 3 ล้าน, ตัดแถวซ้ำ)",
         f"- ช่วงนโยบาย: pre (< {ADMIN_SHIFT}, ตกลงราคา ≤ 100k) · transition ({ADMIN_SHIFT} – {REFORM}, ตกลงราคา ≤ 500k ตามหนังสือเวียน) · act (≥ {REFORM}, เฉพาะเจาะจง ≤ 500k)",
         "- b = excess mass / counterfactual เฉลี่ยต่อช่วงในหน้าต่างใต้เพดาน (≈ จำนวน 'ช่วง' ของสัญญาที่ย้ายมากองใต้เพดาน); se จาก bootstrap residual 200 รอบ",
         "- **ยังเป็นผลรอบแรก** — ยังไม่ได้ทดสอบความไวต่อ degree/หน้าต่าง/ขนาดช่วง; ปีงบ 2563–2565 ยังไม่มีในข้อมูล", "",
         "| การประมาณ | b | se(b) | excess | missing | จริงใต้เพดาน | counterfactual |", "|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        L.append(f"| {r['label']} | {r['b_ratio']:.2f} | {r['b_ratio_se']:.2f} | {r['excess_B']:,.0f} | {r['missing_M']:,.0f} | {r['actual_below']:,.0f} | {r['cf_below']:,.0f} |")
    L += [""] + [f"![](figures/bunching_{r}_{t}k.png)" for t in (100, 500) for r in ("pre", "transition", "act")] + ["![](figures/bunching_monthly.png)"]
    (REPORTS / "bunching.md").write_text("\n".join(L), encoding="utf-8")
    log("→ reports/bunching.md, bunching_results.csv, bunching_monthly.csv, figures/bunching_*.png")


# ------------------------------------------------------------------ robustness
ROBUST_GRID = {
    # (regime, T): ตัวเลือกแต่ละมิติ — ค่าหลักอยู่ใน SPECS
    ("act", 500_000):        dict(width=[2_500, 5_000, 10_000], degree=[3, 5, 7, 9], zL=[25_000, 50_000, 75_000],
                                  zU=[50_000, 100_000, 150_000], fit=[(200_000, 1_000_000), (300_000, 800_000), (150_000, 1_200_000)]),
    ("transition", 500_000): dict(width=[2_500, 5_000, 10_000], degree=[3, 5, 7, 9], zL=[25_000, 50_000, 75_000],
                                  zU=[50_000, 100_000, 150_000], fit=[(200_000, 1_000_000), (300_000, 800_000), (150_000, 1_200_000)]),
    ("pre", 100_000):        dict(width=[500, 1_000, 2_000], degree=[3, 5, 7, 9], zL=[5_000, 10_000, 15_000],
                                  zU=[10_000, 20_000, 30_000], fit=[(40_000, 250_000), (50_000, 200_000), (30_000, 300_000)]),
    ("act", 100_000):        dict(width=[500, 1_000, 2_000], degree=[3, 5, 7, 9], zL=[5_000, 10_000, 15_000],
                                  zU=[10_000, 20_000, 30_000], fit=[(40_000, 250_000), (50_000, 200_000), (30_000, 300_000)]),
    ("act", 300_000):        dict(width=[2_500, 5_000, 10_000], degree=[3, 5, 7], zL=[25_000, 50_000],
                                  zU=[25_000, 50_000], fit=[(150_000, 450_000), (120_000, 480_000)]),   # placebo
}


def robust():
    import itertools
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_parquet(BINS)
    pr = df[df["var"] == "price"]
    rows = []
    for (regime, T), g in ROBUST_GRID.items():
        counts = sel(pr, regime=regime)
        for w, d, zl, zu, (lo, hi) in itertools.product(g["width"], g["degree"], g["zL"], g["zU"], g["fit"]):
            if lo >= T - zl or hi <= T + zu:
                continue
            r = bunch(counts, T, width=w, zL=zl, zU=zu, fit_lo=lo, fit_hi=hi, degree=d, n_boot=0)
            r.update(regime=regime, label=f"{regime}|T={T:,}")
            rows.append(r)
    R = pd.DataFrame(rows)
    # Δz = b × ขนาดช่วง (บาท) — ไม่ขึ้นกับขนาดช่วงราคา (b เองแปรผกผันกับขนาดช่วง)
    R["dz"] = R["b_ratio"] * R["width"]
    R["dz_over_T"] = R["dz"] / R["T"]
    R.to_csv(REPORTS / "bunching_robustness.csv", index=False)
    base = {}
    for (regime, T) in ROBUST_GRID:
        sp = {**SPECS.get(T, dict(width=5_000, zL=25_000, zU=25_000, fit_lo=T - 150_000, fit_hi=T + 150_000, degree=5))}
        r0 = bunch(sel(pr, regime=regime), T, **sp)
        r0["dz"], r0["dz_se"] = r0["b_ratio"] * sp["width"], r0["b_ratio_se"] * sp["width"]
        base[(regime, T)] = r0
    L = ["# Bunching — ทดสอบความไว (robustness)", "",
         "คำนวณใหม่ทุกชุดค่าผสมของ ขนาดช่วงราคา × degree ของ polynomial × หน้าต่างใต้/เหนือเพดาน × ช่วงที่ใช้ fit (ไม่ bootstrap รายชุด)",
         "",
         "**หน่วยที่ใช้:** Δz = b × ขนาดช่วง (บาท) = ความกว้างของช่วงราคาที่ \"ย้ายมากอง\" ใต้เพดาน (Kleven 2016) — ไม่ขึ้นกับขนาดช่วงราคา; "
         "b ดิบแปรผกผันกับขนาดช่วง (2.5k → 46.6, 5k → 23.9, 10k → 12.4 ในช่วง act ที่ 500k) จึงไม่ควรรายงาน b ดิบเทียบข้ามสเปก", "",
         "| ช่วง / เพดาน | ชุด | Δz ค่าหลัก (se) | Δz/T | min | p10 | median | p90 | max | % ชุดที่ Δz > 2·se(ค่าหลัก) | excess B median [p10, p90] |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (regime, T), b0 in base.items():
        g = R[(R["regime"] == regime) & (R["T"] == T)]
        x = g["dz"]; thr = 2 * b0["dz_se"]
        L.append(f"| {regime} / {T:,}{' (placebo)' if T == 300_000 else ''} | {len(g)} | {b0['dz']:,.0f} ({b0['dz_se']:,.0f}) | {b0['dz'] / T:.2f} | "
                 f"{x.min():,.0f} | {x.quantile(.1):,.0f} | {x.median():,.0f} | {x.quantile(.9):,.0f} | {x.max():,.0f} | {(x > thr).mean():.0%} | "
                 f"{g['excess_B'].median():,.0f} [{g['excess_B'].quantile(.1):,.0f}, {g['excess_B'].quantile(.9):,.0f}] |")
    L += ["", "## ความไวแยกตามมิติ (ช่วง act, เพดาน 500k): median Δz (บาท) ต่อค่าในแต่ละมิติ", ""]
    a = R[(R["regime"] == "act") & (R["T"] == 500_000)]
    for dim in ("width", "degree", "zL", "zU", "fit_lo"):
        L.append(f"- {dim}: " + ", ".join(f"{k:,} → {v:,.0f}" for k, v in a.groupby(dim)["dz"].median().items()))
    L += ["", "ข้อสังเกต: ผลไวที่สุดต่อหน้าต่างใต้เพดาน zL (การกระจุกค่อย ๆ เพิ่มตั้งแต่ ~450k จึงไม่ใช่ spike เดียว) → ควรเลือก zL ด้วยวิธีที่กำหนดจากข้อมูล และรายงานช่วงทั้งหมด",
          "", "![](figures/bunching_spec_curve.png)"]
    (REPORTS / "bunching_robustness.md").write_text("\n".join(L), encoding="utf-8")
    fig, axes = plt.subplots(1, len(ROBUST_GRID), figsize=(15, 3.6))
    for ax, ((regime, T), b0) in zip(axes, base.items()):
        x = np.sort(R[(R["regime"] == regime) & (R["T"] == T)]["dz"].values) / 1000
        ax.plot(np.arange(len(x)), x, ".", ms=3, color="#3b6ea8")
        ax.axhline(b0["dz"] / 1000, color="#c0392b", lw=1)
        ax.axhspan((b0["dz"] - 2 * b0["dz_se"]) / 1000, (b0["dz"] + 2 * b0["dz_se"]) / 1000, color="#c0392b", alpha=0.1)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{regime}, T={T // 1000}k{' (placebo)' if T == 300_000 else ''}", fontsize=9)
        ax.set_xlabel("specification (sorted)", fontsize=8)
    axes[0].set_ylabel("Δz (thousand THB)")
    fig.tight_layout(); fig.savefig(FIGS / "bunching_spec_curve.png", dpi=140); plt.close(fig)
    log(f"robustness: {len(R)} specs → reports/bunching_robustness.md")


def datadriven():
    """หน้าต่างจากข้อมูล: ค่าหลัก (bootstrap) + ความไวต่อ degree/ขนาดช่วง/ช่วง fit + รายปี"""
    import itertools
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_parquet(BINS)
    pr = df[df["var"] == "price"]
    bu = df[df["var"] == "budget"]
    cases = [("pre", 100_000, pr, None), ("transition", 500_000, pr, None), ("act", 500_000, pr, None),
             ("act", 100_000, pr, None), ("act", 500_000, bu, "budget"), ("act", 300_000, pr, "placebo")]
    main_rows, grid_rows = [], []
    for regime, T, src, tag in cases:
        sp = SPECS.get(T, dict(width=5_000, fit_lo=T - 150_000, fit_hi=T + 150_000))
        counts = sel(src, regime=regime)
        w = choose_window(counts, T, sp["width"], sp["fit_lo"], sp["fit_hi"], degree=sp.get("degree", 7))
        r = bunch(counts, T, width=sp["width"], zL=w["zL"], zU=w["zU"], fit_lo=sp["fit_lo"], fit_hi=sp["fit_hi"], degree=sp.get("degree", 7))
        main_rows.append({"case": f"{regime}|T={T:,}" + (f"|{tag}" if tag else ""), "zL": w["zL"], "zU": w["zU"], "iter": w["iterations"],
                          "converged": w["converged"], "dz": r["b_ratio"] * sp["width"], "dz_se": r["b_ratio_se"] * sp["width"],
                          "excess_B": r["excess_B"], "excess_B_se": r["excess_B_se"], "missing_M": r["missing_M"], "T": T})
        if tag:
            continue
        widths = [sp["width"] // 2, sp["width"], sp["width"] * 2]
        fits = [(sp["fit_lo"], sp["fit_hi"]), (int(sp["fit_lo"] * 0.75), int(sp["fit_hi"] * 1.2)), (int(sp["fit_lo"] * 1.25), int(sp["fit_hi"] * 0.85))]
        for wd, dg, (lo, hi) in itertools.product(widths, [3, 5, 7, 9], fits):
            ww = choose_window(counts, T, wd, lo, hi, degree=dg)
            rr = bunch(counts, T, width=wd, zL=ww["zL"], zU=ww["zU"], fit_lo=lo, fit_hi=hi, degree=dg, n_boot=0)
            grid_rows.append({"case": f"{regime}|T={T:,}", "width": wd, "degree": dg, "fit_lo": lo, "zL": ww["zL"], "zU": ww["zU"],
                              "dz": rr["b_ratio"] * wd, "excess_B": rr["excess_B"]})
    # รายปี (500k ตั้งแต่ 2559; 100k ทุกปี)
    years = sorted(pr["fiscal_year"].unique())
    yr_rows = []
    for y in years:
        for T in (100_000, 500_000):
            if T == 500_000 and y < 2559:
                continue
            sp = SPECS[T]; counts = sel(pr, fiscal_year=y)
            w = choose_window(counts, T, sp["width"], sp["fit_lo"], sp["fit_hi"])
            r = bunch(counts, T, width=sp["width"], zL=w["zL"], zU=w["zU"], fit_lo=sp["fit_lo"], fit_hi=sp["fit_hi"], n_boot=100)
            yr_rows.append({"FY": y, "T": T, "zL": w["zL"], "zU": w["zU"], "dz": r["b_ratio"] * sp["width"], "dz_se": r["b_ratio_se"] * sp["width"],
                            "excess_B": r["excess_B"]})
    M, G, Y = pd.DataFrame(main_rows), pd.DataFrame(grid_rows), pd.DataFrame(yr_rows)
    M.to_csv(REPORTS / "bunching_dd_main.csv", index=False); G.to_csv(REPORTS / "bunching_dd_grid.csv", index=False); Y.to_csv(REPORTS / "bunching_dd_yearly.csv", index=False)
    old = pd.read_csv(REPORTS / "bunching_robustness.csv") if (REPORTS / "bunching_robustness.csv").exists() else None
    L = ["# Bunching — หน้าต่างกำหนดจากข้อมูล (data-driven window)", "",
         "วิธี: เริ่ม zL = zU = 1 ช่วง → fit counterfactual (polynomial + dummy เลขกลม) นอกหน้าต่าง → ขยาย zL ทีละช่วงถ้าหนึ่งใน 2 ช่วงถัดลงไปมี (จริง − counterfactual) > 1.96σ;"
         " ขยาย zU ถ้าหนึ่งใน 2 ช่วงถัดขึ้นไปมี (จริง − counterfactual) < −1.96σ (σ = sd residual นอกหน้าต่าง) วนจนคงที่ — แนว Bosch, Dekker & Strohmaier (2020)", "",
         "## ค่าหลัก (bootstrap 200)", "| กรณี | zL | zU | รอบ | Δz (se) | Δz/T | excess B (se) | missing M |", "|---|---|---|---|---|---|---|---|"]
    for _, r in M.iterrows():
        L.append(f"| {r['case']} | {r['zL']:,} | {r['zU']:,} | {r['iter']}{'' if r['converged'] else ' ⚠'} | {r['dz']:,.0f} ({r['dz_se']:,.0f}) | {r['dz'] / r['T']:.2f} | {r['excess_B']:,.0f} ({r['excess_B_se']:,.0f}) | {r['missing_M']:,.0f} |")
    L += ["", "## ความไวของ Δz เมื่อหน้าต่างกำหนดจากข้อมูล (36 สเปก/กรณี: ขนาดช่วง ×½ ×1 ×2 · degree 3/5/7/9 · ช่วง fit 3 แบบ)",
          "| กรณี | p10 | median | p90 | (p90−p10)/median | เทียบหน้าต่างตายตัว (p90−p10)/median | zL ที่เลือก (min–max) |", "|---|---|---|---|---|---|---|"]
    for case, g in G.groupby("case"):
        q = g["dz"].quantile([.1, .5, .9]).values
        spread_old = ""
        if old is not None:
            rg, T = case.split("|")[0], int(case.split("=")[1].replace(",", ""))
            o = old[(old["regime"] == rg) & (old["T"] == T)]["dz"]
            if len(o):
                qo = o.quantile([.1, .5, .9]).values; spread_old = f"{(qo[2] - qo[0]) / qo[1]:.2f}"
        L.append(f"| {case} | {q[0]:,.0f} | {q[1]:,.0f} | {q[2]:,.0f} | {(q[2] - q[0]) / q[1]:.2f} | {spread_old} | {g['zL'].min():,}–{g['zL'].max():,} |")
    L += ["", "## รายปีงบ", "| ปีงบ | เพดาน | zL | zU | Δz (se) | Δz/T | excess B |", "|---|---|---|---|---|---|---|"]
    for _, r in Y.iterrows():
        L.append(f"| {int(r['FY'])} | {int(r['T']):,} | {int(r['zL']):,} | {int(r['zU']):,} | {r['dz']:,.0f} ({r['dz_se']:,.0f}) | {r['dz'] / r['T']:.2f} | {r['excess_B']:,.0f} |")
    L += ["", "![](figures/bunching_dd_yearly.png)"]
    (REPORTS / "bunching_datadriven.md").write_text("\n".join(L), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for T, c in ((100_000, "#3b6ea8"), (500_000, "#c0392b")):
        d = Y[Y["T"] == T]
        ax.errorbar(d["FY"], d["dz"] / T, yerr=1.96 * d["dz_se"] / T, fmt="o-", color=c, capsize=3, label=f"threshold {T // 1000}k")
    ax.axvline(2558.5, color="k", ls="--", lw=1); ax.axvline(2560.9, color="k", ls=":", lw=1)
    ax.set_xlabel("fiscal year (B.E.)"); ax.set_ylabel("Δz / threshold"); ax.legend(fontsize=8)
    ax.set_title("Bunching (data-driven window), 95% CI — dashed: FY2559 circular; dotted: 2017 Act", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "bunching_dd_yearly.png", dpi=140); plt.close(fig)
    log("→ reports/bunching_datadriven.md")


# ------------------------------------------------------------------ difference-in-bunching
def _bins5k(counts: pd.Series, width: int, lo: int, hi: int) -> pd.Series:
    c = counts.copy()
    c.index = (np.ceil(c.index / width) * width).astype(int)
    c = c.groupby(level=0).sum()
    grid = np.arange(lo + width, hi + width, width)
    return c.reindex(grid, fill_value=0).astype(float)


def dib_estimate(y_tgt: np.ndarray, y_ref: np.ndarray, grid: np.ndarray, T: int, width: int,
                 scale_lo: int, scale_hi: int, zL: int, zU_list, shape: str = "const") -> dict:
    """counterfactual = รูปการกระจายช่วงอ้างอิง × สเกล; สเกลประมาณจากช่วง (scale_lo, scale_hi] ใต้เพดาน
    shape='const' : สเกลคงที่ = Σtgt/Σref ในช่วงสเกล
    shape='loglin': log(tgt/ref) = a + b·z fit ในช่วงสเกล แล้วต่อเส้นไปทั้งช่วง (รองรับการเลื่อนของราคาตามเวลา)"""
    m = (grid > scale_lo) & (grid <= scale_hi)
    if shape == "const":
        k = np.full(len(grid), y_tgt[m].sum() / max(y_ref[m].sum(), 1e-9))
    else:
        ok = m & (y_tgt > 0) & (y_ref > 0)
        a1, a0 = np.polyfit(grid[ok] / 1e6, np.log(y_tgt[ok] / y_ref[ok]), 1, w=np.sqrt(y_tgt[ok]))
        k = np.exp(a0 + a1 * grid / 1e6)
    cf = y_ref * k
    below = (grid > T - zL) & (grid <= T)
    B = (y_tgt[below] - cf[below]).sum()
    h0 = cf[below].mean() / width               # ความหนาแน่น counterfactual ต่อบาท
    out = {"excess_B": B, "dz": B / h0 if h0 > 0 else np.nan, "cf_below": cf[below].sum(), "act_below": y_tgt[below].sum()}
    for zu in zU_list:
        above = (grid > T) & (grid <= T + zu)
        out[f"missing_{zu // 1000}k"] = (cf[above] - y_tgt[above]).sum()
        out[f"missing_share_{zu // 1000}k"] = (cf[above] - y_tgt[above]).sum() / max(cf[above].sum(), 1e-9)
    out["_cf"] = cf
    return out


def dib():
    import itertools
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_parquet(BINS)
    pr = df[df["var"] == "price"]
    W, LO, HI, T = 5_000, 100_000, 2_000_000, 500_000
    ZU = [50_000, 100_000, 250_000, 500_000, 1_000_000, 1_500_000]
    ref = _bins5k(sel(pr, regime="pre"), W, LO, HI)
    grid = ref.index.values; y_ref = ref.values
    rng = np.random.default_rng(1)

    def run(y_tgt, label, T=T, zL=50_000, scale=(200_000, 400_000), shape="const", nboot=300, zU=ZU):
        r = dib_estimate(y_tgt, y_ref, grid, T, W, scale[0], scale[1], zL, zU, shape)
        bs = [dib_estimate(rng.poisson(y_tgt).astype(float), rng.poisson(y_ref).astype(float),
                           grid, T, W, scale[0], scale[1], zL, zU, shape) for _ in range(nboot)] if nboot else []
        row = {"label": label, "T": T, "zL": zL, "scale": f"{scale[0] // 1000}-{scale[1] // 1000}k", "shape": shape}
        for k_, v in r.items():
            if k_.startswith("_"):
                continue
            row[k_] = v
            if bs:
                row[k_ + "_se"] = float(np.nanstd([b[k_] for b in bs]))
        return row, r["_cf"]

    rows = []
    # ค่าหลัก: ช่วง transition / act ทั้งหมด และรายปี
    targets = {"transition (FY2559–60)": sel(pr, regime="transition"), "act (FY2560–68)": sel(pr, regime="act")}
    for y in range(2559, 2569):
        targets[f"FY{y}"] = sel(pr, fiscal_year=y)
    cfs = {}
    for lab, cnt in targets.items():
        yt = _bins5k(cnt, W, LO, HI).values
        row, cf = run(yt, lab)
        rows.append(row); cfs[lab] = (yt, cf)
    yt_act = cfs["act (FY2560–68)"][0]
    # placebo band: เลื่อนหน้าต่างหลอกกว้าง 50k ไปทั่วย่านสะอาด 200k–400k (อัตราส่วน act/pre คงที่ในย่านนี้)
    # สเกลจากส่วนที่เหลือของ 200k–400k → การกระจายของ Δz หลอก = ความไม่แน่นอนจากรูปการกระจาย (shape noise)
    def placebo_band(yt):
        out = []
        for Tp in range(250_000, 400_001, 5_000):
            m_scale = (grid > 200_000) & (grid <= 400_000) & ~((grid > Tp - 50_000) & (grid <= Tp))
            k = yt[m_scale].sum() / y_ref[m_scale].sum()
            cf = y_ref * k
            w_ = (grid > Tp - 50_000) & (grid <= Tp)
            out.append((yt[w_] - cf[w_]).sum() / (cf[w_].mean() / W))
        return np.array(out)
    pb = {lab: placebo_band(v[0]) for lab, v in cfs.items()}
    for row in rows:
        b_ = pb.get(row["label"])
        if b_ is not None:
            row["placebo_dz_sd"] = float(b_.std(ddof=1)); row["placebo_dz_mean"] = float(b_.mean())
            row["placebo_dz_absmax"] = float(np.abs(b_).max())
            row["dz_se_total"] = float(np.sqrt(row.get("dz_se", 0) ** 2 + row["placebo_dz_sd"] ** 2))
    # robustness ของค่าหลัก (act): ช่วงสเกล × รูปสเกล × zL
    rob = []
    for sc, shp, zl in itertools.product([(150_000, 400_000), (200_000, 400_000), (250_000, 425_000), (200_000, 350_000)],
                                         ["const", "loglin"], [25_000, 50_000, 75_000, 100_000]):
        r = dib_estimate(yt_act, y_ref, grid, T, W, sc[0], sc[1], zl, ZU, shp)
        rob.append({"scale": f"{sc[0] // 1000}-{sc[1] // 1000}k", "shape": shp, "zL": zl, "dz": r["dz"], "excess_B": r["excess_B"],
                    "missing_1000k": r["missing_1000k"], "missing_share_250k": r["missing_share_250k"]})
    D, Rb = pd.DataFrame(rows), pd.DataFrame(rob)
    D.to_csv(REPORTS / "bunching_dib.csv", index=False); Rb.to_csv(REPORTS / "bunching_dib_robust.csv", index=False)

    L = ["# Difference-in-bunching ที่เพดาน 500,000 บาท", "",
         "**แนวคิด:** ใช้รูปการกระจายของช่วง pre (ลงนามก่อน ต.ค. 2558 = ปีงบ 2558 — ก่อนเพดานดุลพินิจ 500k มีผล) เป็น counterfactual",
         "ปรับสเกลให้เท่ากับช่วงเป้าหมายในย่าน 200k–400k (ใต้หน้าต่าง) แล้วเทียบสัญญาจริงกับ counterfactual — ไม่ต้องสมมติรูปเหนือเพดานด้วย polynomial",
         "ข้อดีอีกข้อ: การกระจุกที่เลขกลม (เช่น 500,000 พอดี) มีอยู่แล้วในช่วง pre จึงถูกหักออกโดยอัตโนมัติ", "",
         "se จาก parametric bootstrap (Poisson ทั้งสองการกระจาย) 300 รอบ; ช่วงละ 5,000 บาท; หน้าต่างใต้เพดาน zL = 50,000", "",
         "**ความไม่แน่นอน:** se รวม = √(se จาก Poisson bootstrap² + sd ของ Δz หลอก²) โดย Δz หลอกได้จากการเลื่อนหน้าต่างกว้าง 50k ไปทั่วย่าน 200k–400k (31 ตำแหน่ง) — Poisson อย่างเดียวให้ se เล็กเกินจริงเพราะไม่รวมความต่างของรูปการกระจายระหว่างช่วงเวลา", "",
         "| เป้าหมาย | excess B | Δz บาท (se รวม) | Δz/T | placebo Δz: mean ± sd, |max| | missing 100k | missing 500k | missing 1M | missing 1.5M | สัดส่วนที่หายไปใน (500k, 750k] |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in D.iterrows():
        L.append(f"| {r['label']} | {r['excess_B']:,.0f} | {r['dz']:,.0f} ({r['dz_se_total']:,.0f}) | {r['dz'] / T:.2f} | "
                 f"{r['placebo_dz_mean']:,.0f} ± {r['placebo_dz_sd']:,.0f}, {r['placebo_dz_absmax']:,.0f} | "
                 f"{r['missing_100k']:,.0f} | {r['missing_500k']:,.0f} | {r['missing_1000k']:,.0f} | {r['missing_1500k']:,.0f} | {r['missing_share_250k']:.0%} |")
    L += ["", "**อ่านผล:** Δz จริงที่ 500k เทียบกับ |max| ของ Δz หลอกในย่านสะอาดของปีเดียวกัน — ถ้า Δz จริงสูงกว่ามาก แปลว่าไม่ใช่ผลจากความต่างของรูปการกระจาย",
          "ส่วนที่หายไปเหนือเพดาน (เทียบรูปช่วง pre) มากกว่าส่วนเกินใต้เพดาน → นอกจากการกดราคาลงมาใต้เพดาน น่าจะมีการแบ่งโครงการใหญ่เป็นหลายสัญญา (1 โครงการที่หายไป → หลายสัญญาเล็ก) และ/หรือองค์ประกอบของข้อมูลเปลี่ยน — ต้องแยกให้ได้"]
    L += ["", "## ความไวของค่าหลัก (act): ช่วงสเกล × รูปสเกล (คงที่ / log-linear) × zL", "",
          f"- Δz: min {Rb['dz'].min():,.0f} · p10 {Rb['dz'].quantile(.1):,.0f} · median {Rb['dz'].median():,.0f} · p90 {Rb['dz'].quantile(.9):,.0f} · max {Rb['dz'].max():,.0f}  ((p90−p10)/median = {(Rb['dz'].quantile(.9) - Rb['dz'].quantile(.1)) / Rb['dz'].median():.2f})",
          f"- excess B: median {Rb['excess_B'].median():,.0f} (p10 {Rb['excess_B'].quantile(.1):,.0f}, p90 {Rb['excess_B'].quantile(.9):,.0f})", ""]
    for dim in ("scale", "shape", "zL"):
        L.append(f"- {dim}: " + ", ".join(f"{k} → {v:,.0f}" for k, v in Rb.groupby(dim)["dz"].median().items()))
    L += ["", "![](figures/bunching_dib_act.png)", "![](figures/bunching_dib_yearly.png)"]
    (REPORTS / "bunching_dib.md").write_text("\n".join(L), encoding="utf-8")

    # รูป 1: act จริง vs counterfactual
    yt, cf = cfs["act (FY2560–68)"]
    m = grid <= 1_200_000
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(grid[m] / 1000, yt[m], width=W / 1000 * 0.9, color="#3b6ea8", label="actual (FY2560–68, post-Act)")
    ax.plot(grid[m] / 1000, cf[m], color="#c0392b", lw=1.6, label="counterfactual = pre-reform shape × scale")
    ax.axvline(500, color="k", ls="--", lw=1); ax.axvspan(200, 400, color="green", alpha=0.07, label="scaling range")
    ax.set_xlabel("contract value (thousand THB)"); ax.set_ylabel("contracts per 5k bin"); ax.legend(fontsize=8)
    ax.set_title("Difference-in-bunching at 500k", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "bunching_dib_act.png", dpi=140); plt.close(fig)
    # รูป 2: รายปี Δz/T
    Y = D[D["label"].str.startswith("FY")].copy(); Y["FY"] = Y["label"].str[2:].astype(int)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.errorbar(Y["FY"], Y["dz"] / T, yerr=1.96 * Y["dz_se_total"] / T, fmt="o-", color="#c0392b", capsize=3)
    ax.axvline(2560.9, color="k", ls=":", lw=1); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("fiscal year (B.E.)"); ax.set_ylabel("Δz / 500k"); ax.set_title("Difference-in-bunching by fiscal year (ref: FY2558), 95% CI incl. shape noise", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "bunching_dib_yearly.png", dpi=140); plt.close(fig)
    log("→ reports/bunching_dib.md")


# ------------------------------------------------------------------ event study (monthly difference-in-bunching)
ANNOUNCE = "2015-08-28"   # หนังสือ กวพ ว 299


def _dib_month(yt, yr, grid, T, W, zL, scale_lo, scale_hi, pl_lo, pl_hi, n_boot=100, rng=None):
    """Δz/T ของเดือนหนึ่ง เทียบรูปอ้างอิง yr; se = √(Poisson² + placebo sd²) — placebo เลื่อนหน้าต่าง zL ไปทั่ว (pl_lo, pl_hi]"""
    def one(y_t, y_r, Tx, scale_mask):
        k = y_t[scale_mask].sum() / max(y_r[scale_mask].sum(), 1e-9)
        cf = y_r * k
        w = (grid > Tx - zL) & (grid <= Tx)
        h0 = cf[w].mean() / W
        return (y_t[w] - cf[w]).sum() / h0 if h0 > 0 else np.nan
    sm = (grid > scale_lo) & (grid <= scale_hi)
    dz = one(yt, yr, T, sm)
    boots = [one(rng.poisson(yt).astype(float), rng.poisson(yr).astype(float), T, sm) for _ in range(n_boot)]
    pl = []
    for Tp in range(pl_lo + zL, pl_hi + 1, W):
        m = (grid > pl_lo) & (grid <= pl_hi) & ~((grid > Tp - zL) & (grid <= Tp))
        pl.append(one(yt, yr, Tp, m))
    se = float(np.sqrt(np.nanvar(boots) + np.nanvar(pl, ddof=1)))
    return dz / T, se / T, float(yt[sm].sum())


def event():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    df = pd.read_parquet(BINS)
    pr = df[df["var"] == "price"]
    rng = np.random.default_rng(3)
    months = sorted(pr["ym"].unique())
    months = [m for m in months if "2014-10" <= m <= "2025-09"]
    cfg = {
        500_000: dict(W=5_000, lo=100_000, hi=1_000_000, zL=50_000, scale=(200_000, 400_000), pl=(200_000, 400_000), ref_regime="pre"),
        100_000: dict(W=1_000, lo=30_000, hi=200_000, zL=10_000, scale=(40_000, 85_000), pl=(40_000, 85_000), ref_regime="act"),
    }
    rows = []
    for T, c in cfg.items():
        ref_all = pr[pr["regime"] == c["ref_regime"]]
        ref_by_m = {m: _bins5k(g.groupby("bin")["n"].sum(), c["W"], c["lo"], c["hi"]) for m, g in ref_all.groupby("ym")}
        ref_tot = sum(ref_by_m.values())
        grid = ref_tot.index.values
        for m in months:
            g = pr[pr["ym"] == m]
            yt = _bins5k(g.groupby("bin")["n"].sum(), c["W"], c["lo"], c["hi"]).values
            yr = (ref_tot - ref_by_m[m]).values if m in ref_by_m else ref_tot.values   # leave-one-month-out
            if yt.sum() < 500:
                continue
            v, se, n_sc = _dib_month(yt, yr, grid, T, c["W"], c["zL"], *c["scale"], *c["pl"], rng=rng)
            rows.append({"T": T, "ym": m, "dz_over_T": v, "se": se, "n_scale_range": n_sc, "ref": c["ref_regime"]})
    E = pd.DataFrame(rows)
    E.to_csv(REPORTS / "bunching_event_monthly.csv", index=False)

    # สรุปช่วง: ค่าเฉลี่ยถ่วง 1/se² ก่อน/หลังแต่ละจุดเปลี่ยน
    def wavg(d):
        w = 1 / d["se"] ** 2
        return (d["dz_over_T"] * w).sum() / w.sum(), np.sqrt(1 / w.sum())
    per = [("pre (ต.ค. 57 – ก.ย. 58)", "2014-10", "2015-09"), ("transition (ต.ค. 58 – ก.ค. 60)", "2015-10", "2017-07"),
           ("act ปีแรก (ก.ย. 60 – ก.ย. 61)", "2017-09", "2018-09"), ("act (ต.ค. 61 – ก.ย. 68)", "2018-10", "2025-09")]
    L = ["# Event study รายเดือน: difference-in-bunching", "",
         "- **500k:** อ้างอิง = รูปช่วง pre (ก่อน ต.ค. 2558; เดือนในช่วงอ้างอิงใช้ leave-one-month-out); สเกลย่าน 200k–400k; หน้าต่าง 50k",
         "- **100k:** อ้างอิง = รูปช่วง act (หลัง ส.ค. 2560 ซึ่งยังมีเกณฑ์วงเงินเล็กน้อย 100k) → ค่า = การกระจุก *เกินระดับหลัง พ.ร.บ.*; สเกลย่าน 40k–85k; หน้าต่าง 10k",
         "- se = √(Poisson bootstrap² + sd ของ Δz หลอกในย่านสเกล²); เส้นประ = ประกาศ ว 299 (28 ส.ค. 2558), ต้นปีงบ 2559 (1 ต.ค. 2558), พ.ร.บ. (23 ส.ค. 2560)", "",
         "| ช่วง | 500k: Δz/T (se) | 100k: Δz/T เทียบระดับหลัง พ.ร.บ. (se) |", "|---|---|---|"]
    for name, a, b in per:
        out = []
        for T in (500_000, 100_000):
            d = E[(E["T"] == T) & (E["ym"] >= a) & (E["ym"] <= b)]
            mu, se = wavg(d) if len(d) else (np.nan, np.nan)
            out.append(f"{mu:.3f} ({se:.3f})")
        L.append(f"| {name} | {out[0]} | {out[1]} |")
    # เดือนรอบจุดเปลี่ยน
    L += ["", "## เดือนรอบจุดเปลี่ยน (500k)", "| เดือน | Δz/T | se | สัญญาในย่านสเกล |", "|---|---|---|---|"]
    for _, r in E[(E["T"] == 500_000) & (E["ym"] >= "2015-06") & (E["ym"] <= "2016-01")].iterrows():
        L.append(f"| {r['ym']} | {r['dz_over_T']:.3f} | {r['se']:.3f} | {r['n_scale_range']:,.0f} |")
    L += ["", "## เดือนกันยายน vs เดือนอื่น (500k, ช่วง act ต.ค. 2560 – ก.ย. 2568)"]
    d = E[(E["T"] == 500_000) & (E["ym"] >= "2017-10")]
    sep, oth = d[d["ym"].str.endswith("-09")], d[~d["ym"].str.endswith("-09")]
    L.append(f"- กันยายน: เฉลี่ย {sep['dz_over_T'].mean():.3f} (n = {len(sep)}); เดือนอื่น: {oth['dz_over_T'].mean():.3f} (n = {len(oth)})")
    L += ["", "![](figures/bunching_event_monthly.png)"]
    (REPORTS / "bunching_event.md").write_text("\n".join(L), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, T, col, ttl in ((axes[0], 500_000, "#c0392b", "500k (ref: pre-reform shape)"), (axes[1], 100_000, "#3b6ea8", "100k (ref: post-Act shape)")):
        d = E[E["T"] == T].copy(); d["t"] = pd.to_datetime(d["ym"])
        ax.fill_between(d["t"], d["dz_over_T"] - 1.96 * d["se"], d["dz_over_T"] + 1.96 * d["se"], color=col, alpha=0.18, lw=0)
        ax.plot(d["t"], d["dz_over_T"], "-", color=col, lw=1.2)
        sp = d[d["ym"].str.endswith("-09")]
        ax.plot(sp["t"], sp["dz_over_T"], "o", ms=3.5, color="k", label="September")
        ax.axhline(0, color="k", lw=0.6)
        for x, ls in ((ANNOUNCE, ":"), (ADMIN_SHIFT, "--"), (REFORM, "--")):
            ax.axvline(pd.Timestamp(x), color="k", ls=ls, lw=1)
        ax.set_ylabel("Δz / T"); ax.set_title(ttl, fontsize=9); ax.legend(fontsize=7, loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Monthly difference-in-bunching (95% CI incl. shape noise) — dotted: circular 299 (2015-08-28); dashed: FY2559 start, 2017 Act", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "bunching_event_monthly.png", dpi=140); plt.close(fig)
    log(f"event: {len(E)} month×threshold → reports/bunching_event.md")


# ------------------------------------------------------------------ splitting (การแบ่งสัญญา)
SPLIT_BINS = PROCESSED / "bunching_split_bins.parquet"
SPLIT_FLAGS = PROCESSED / "bunching_split_flags.parquet"


def split_aggregate():
    """กลุ่ม = (หน่วยงาน, หน่วยงานย่อย, ผู้ขาย[ชื่อ normalize + จังหวัด], วันลงนาม) และระดับเดือน
    1) การกระจายของ 'มูลค่ารวมต่อกลุ่ม' (รวมสัญญาที่ถูกแบ่งกลับเป็นก้อนเดียว) → ใช้ทำ difference-in-bunching ซ้ำ
    2) ต่อเดือน: สัญญาใกล้เพดาน (0.9T, T] และย่านควบคุม (0.4T, 0.8T] กี่ฉบับอยู่ในกลุ่มวันเดียวกัน ≥ 2 สัญญาที่รวมกัน > T"""
    con = connect()
    years = sorted(int(p.name.split("_")[1]) for p in INTERIM.glob("contracts_25??") if p.is_dir())
    bins, flags = [], []
    for y in years:
        t0 = time.time()
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
            SELECT DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)
                   sign_date, agreed_price, agency || '|' || coalesce(sub_agency, '') AS unit,
                   CASE WHEN vendor_name_key IS NULL THEN NULL ELSE vendor_name_key || '|' || coalesce(province, '') END AS vkey,
                   {REGIME_SQL} AS regime
            FROM read_parquet('{src}')
            WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE}""")
        for level, keyexpr in (("day", "sign_date"), ("month", "date_trunc('month', sign_date)")):
            b = con.execute(f"""
                WITH g AS (
                  SELECT regime, strftime(min(sign_date), '%Y-%m') ym, sum(agreed_price) v, count(*) k
                  FROM c WHERE vkey IS NOT NULL GROUP BY unit, vkey, {keyexpr}, regime
                  UNION ALL
                  SELECT regime, strftime(sign_date, '%Y-%m'), agreed_price, 1 FROM c WHERE vkey IS NULL
                )
                SELECT '{level}' AS level, {y} AS fiscal_year, ym, regime,
                       (ceil(least(v, {MAX_PRICE * 3}) / 1000) * 1000)::BIGINT AS bin, count(*)::INT AS n, sum(k)::INT AS n_contracts
                FROM g GROUP BY ALL""").df()
            bins.append(b)
        f = con.execute(f"""
            WITH g AS (SELECT unit, vkey, sign_date, count(*) k, sum(agreed_price) v, max(agreed_price) mx
                       FROM c WHERE vkey IS NOT NULL GROUP BY 1, 2, 3),
            j AS (SELECT c.*, g.k, g.v FROM c LEFT JOIN g USING (unit, vkey, sign_date)),
            t AS (SELECT *, CASE WHEN regime = 'pre' THEN 100000 ELSE 500000 END AS T FROM j)
            SELECT {y} AS fiscal_year, strftime(sign_date, '%Y-%m') ym, regime, T,
                   sum(CASE WHEN agreed_price > 0.9 * T AND agreed_price <= T THEN 1 ELSE 0 END)::INT n_near,
                   sum(CASE WHEN agreed_price > 0.9 * T AND agreed_price <= T AND k >= 2 AND v > T THEN 1 ELSE 0 END)::INT n_near_split,
                   sum(CASE WHEN agreed_price > 0.4 * T AND agreed_price <= 0.8 * T THEN 1 ELSE 0 END)::INT n_ctrl,
                   sum(CASE WHEN agreed_price > 0.4 * T AND agreed_price <= 0.8 * T AND k >= 2 AND v > T THEN 1 ELSE 0 END)::INT n_ctrl_split,
                   -- เทียบเพดาน 500k ทุกช่วง (คงที่) เพื่อดูการเปลี่ยนรอบ ต.ค. 2558
                   sum(CASE WHEN agreed_price > 450000 AND agreed_price <= 500000 THEN 1 ELSE 0 END)::INT n_near500,
                   sum(CASE WHEN agreed_price > 450000 AND agreed_price <= 500000 AND k >= 2 AND v > 500000 THEN 1 ELSE 0 END)::INT n_near500_split,
                   sum(CASE WHEN agreed_price > 200000 AND agreed_price <= 400000 THEN 1 ELSE 0 END)::INT n_ctrl500,
                   sum(CASE WHEN agreed_price > 200000 AND agreed_price <= 400000 AND k >= 2 AND v > 500000 THEN 1 ELSE 0 END)::INT n_ctrl500_split
            FROM t GROUP BY ALL""").df()
        flags.append(f)
        log(f"split_aggregate {y}: {time.time() - t0:.0f}s")
    pd.concat(bins, ignore_index=True).to_parquet(SPLIT_BINS, index=False)
    pd.concat(flags, ignore_index=True).to_parquet(SPLIT_FLAGS, index=False)
    log(f"→ {SPLIT_BINS.name}, {SPLIT_FLAGS.name}")


def split_analysis():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SB = pd.read_parquet(SPLIT_BINS)
    PB = pd.concat([pd.read_parquet(f) for f in sorted(SPLIT_PLACEBO_DIR.glob("*.parquet"))], ignore_index=True)
    W, LO, HI, T = 5_000, 100_000, 2_000_000, 500_000
    ZU = [100_000, 500_000, 1_000_000, 1_500_000]
    df = pd.read_parquet(BINS); pr = df[df["var"] == "price"]
    res = []
    curves = {}
    for level_name, ref_cnt, act_cnt in (
        ("contract (เดิม)", sel(pr, regime="pre"), sel(pr, regime="act")),
        ("รวมกลุ่ม: หน่วยงาน×ผู้ขาย×วัน", SB[(SB.level == "day") & (SB.regime == "pre")].groupby("bin")["n"].sum(),
         SB[(SB.level == "day") & (SB.regime == "act")].groupby("bin")["n"].sum()),
        ("รวมกลุ่ม: หน่วยงาน×ผู้ขาย×เดือน", SB[(SB.level == "month") & (SB.regime == "pre")].groupby("bin")["n"].sum(),
         SB[(SB.level == "month") & (SB.regime == "act")].groupby("bin")["n"].sum()),
        ("placebo: สลับผู้ขายสุ่มภายในหน่วยงาน×เดือน", PB[PB.regime == "pre"].groupby("bin")["n"].sum(),
         PB[PB.regime == "act"].groupby("bin")["n"].sum())):
        yr = _bins5k(ref_cnt, W, LO, HI); grid = yr.index.values
        ya = _bins5k(act_cnt, W, LO, HI).values
        r = dib_estimate(ya, yr.values, grid, T, W, 200_000, 400_000, 50_000, ZU, "const")
        res.append({"level": level_name, "excess_B": r["excess_B"], "dz_over_T": r["dz"] / T,
                    **{f"missing_{z // 1000}k": r[f"missing_{z // 1000}k"] for z in ZU},
                    "missing_share_250k": None})
        curves[level_name] = (grid, ya, r["_cf"])
    R = pd.DataFrame(res)
    F = pd.read_parquet(SPLIT_FLAGS)
    m = F.groupby("ym")[["n_near500", "n_near500_split", "n_ctrl500", "n_ctrl500_split", "n_near", "n_near_split", "n_ctrl", "n_ctrl_split"]].sum()
    m["share_near500"] = m["n_near500_split"] / m["n_near500"]
    m["share_ctrl500"] = m["n_ctrl500_split"] / m["n_ctrl500"]
    m["diff500"] = m["share_near500"] - m["share_ctrl500"]
    m = m.loc["2014-10":"2025-09"]
    m.to_csv(REPORTS / "bunching_split_monthly.csv")
    def per(a, b):
        d = m.loc[a:b]
        return (d["n_near500_split"].sum() / d["n_near500"].sum(), d["n_ctrl500_split"].sum() / d["n_ctrl500"].sum())
    periods = [("pre (ต.ค. 57 – ก.ย. 58)", "2014-10", "2015-09"), ("transition (ต.ค. 58 – ก.ค. 60)", "2015-10", "2017-07"),
               ("act (ก.ย. 60 – ก.ย. 68)", "2017-09", "2025-09")]
    L = ["# การแบ่งสัญญา (splitting) กับการกระจุกใต้เพดาน 500k", "",
         "## 1) ทดสอบการรวมกลับ (re-aggregation)",
         "รวมสัญญาที่ หน่วยงาน×หน่วยงานย่อย×ผู้ขาย (ชื่อ normalize + จังหวัด) เดียวกัน ในวันเดียวกัน / เดือนเดียวกัน เป็นก้อนเดียว แล้วทำ difference-in-bunching ซ้ำ (อ้างอิง = ช่วง pre ที่รวมแบบเดียวกัน, สเกล 200k–400k)",
         "ถ้าส่วนที่หายไปเหนือเพดานลดลงมากหลังรวม → การแบ่งสัญญาอธิบายได้ส่วนหนึ่ง", "",
         "| ระดับ | excess ใต้เพดาน | Δz/T | missing ถึง 100k | ถึง 500k | ถึง 1M | ถึง 1.5M |", "|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        L.append(f"| {r['level']} | {r['excess_B']:,.0f} | {r['dz_over_T']:.2f} | {r['missing_100k']:,.0f} | {r['missing_500k']:,.0f} | {r['missing_1000k']:,.0f} | {r['missing_1500k']:,.0f} |")
    L += ["", "| ระดับ | missing ถึง 1.5M ÷ excess |", "|---|---|"]
    for _, r in R.iterrows():
        L.append(f"| {r['level']} | {r['missing_1500k'] / r['excess_B']:.2f} |")
    L += ["", "หมายเหตุ: หน่วยนับหลังรวมกลุ่มคือ 'กลุ่ม' ไม่ใช่ 'สัญญา' จึงเทียบกันด้วยอัตราส่วน missing/excess", "",
          "## 2) สัดส่วนสัญญาที่อยู่ในกลุ่ม 'วันเดียวกัน ≥ 2 สัญญา รวม > 500k'", "",
          "| ช่วง | สัญญาใกล้เพดาน (450k, 500k] | ย่านควบคุม (200k, 400k] | ส่วนต่าง |", "|---|---|---|---|"]
    for name, a, b in periods:
        near, ctrl = per(a, b)
        L.append(f"| {name} | {near:.1%} | {ctrl:.1%} | {near - ctrl:+.1%} |")
    # 3) missing mass เมื่อยึดย่านบน (ไม่ได้รับผลจากเพดาน 500k) แทน 200k–400k
    g = pr.groupby(["fiscal_year", "bin"])["n"].sum().unstack(0).fillna(0)
    band = lambda a, b: g[(g.index > a) & (g.index <= b)].sum()
    L += ["", "## 3) ย่านอ้างอิงสำหรับ missing mass: 200k–400k ไม่สะอาด",
          "อัตราส่วนต่อย่าน 200k–400k ลดลงต่อเนื่องแม้ที่ 2M–3M (ซึ่งไม่เกี่ยวกับเพดาน 500k) เพราะจำนวนสัญญา 200k–400k เองโต ~2.5 เท่า (2558→2568) "
          "ขณะที่จำนวนสัญญา 2M–3M คงที่ ~15k/ปี → การสเกลด้วย 200k–400k ทำให้ missing เหนือเพดานสูงเกินจริง",
          "จึงคำนวณใหม่: expected(500k–2M) = [สัดส่วน 500k–2M ÷ ย่านยึด ของปี 2558] × ย่านยึดของแต่ละปี", ""]
    for anc in ((2_000_000, 3_000_000), (2_500_000, 3_000_000)):
        A = band(*anc); M = band(500_000, 2_000_000); E = band(450_000, 500_000); C = band(200_000, 400_000)
        r0 = M[2558] / A[2558]; ex = r0 * A
        L += [f"**ย่านยึด {anc[0] / 1e6:.1f}M–{anc[1] / 1e6:.0f}M**", "",
              "| ปีงบ | สัญญา 500k–2M จริง | คาด | missing | % | สัญญา (450k,500k] | สัญญา (200k,400k] |", "|---|---|---|---|---|---|---|"]
        for y in g.columns:
            L.append(f"| {y} | {M[y]:,.0f} | {ex[y]:,.0f} | {ex[y] - M[y]:,.0f} | {(ex[y] - M[y]) / ex[y]:.0%} | {E[y]:,.0f} | {C[y]:,.0f} |")
        yy = [y for y in g.columns if y >= 2561]
        L += ["", f"รวม 2561–2568: missing ≈ {sum(ex[y] - M[y] for y in yy):,.0f} สัญญา", ""]
    L += ["", "![](figures/bunching_split.png)"]
    (REPORTS / "bunching_split.md").write_text("\n".join(L), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax = axes[0]
    en = {0: "contracts", 1: "clusters: unit x vendor x day", 2: "clusters: unit x vendor x month", 3: "placebo: random vendor within unit x month"}
    for i, ((name, (g, ya, cf)), col) in enumerate(zip(curves.items(), ("#3b6ea8", "#c0392b", "#2e8b57", "#999999"))):
        name = en[i]
        mk = g <= 1_500_000
        ax.plot(g[mk] / 1000, (ya[mk] - cf[mk]) / cf[mk].clip(min=1), lw=1, color=col, label=name)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(500, color="k", ls="--", lw=1); ax.set_ylim(-1.05, 5)
    ax.set_xlabel("value (thousand THB)"); ax.set_ylabel("(actual − counterfactual) / counterfactual"); ax.legend(fontsize=7)
    ax.set_title("Post-Act vs pre-reform shape: contracts vs re-aggregated clusters", fontsize=9)
    ax = axes[1]; t = pd.to_datetime(m.index)
    ax.plot(t, m["share_near500"], color="#c0392b", lw=1.2, label="near cap (450k,500k]")
    ax.plot(t, m["share_ctrl500"], color="#3b6ea8", lw=1.2, label="control (200k,400k]")
    for x in (ADMIN_SHIFT, REFORM):
        ax.axvline(pd.Timestamp(x), color="k", ls="--", lw=1)
    ax.set_ylabel("share in same-day clusters summing > 500k"); ax.legend(fontsize=7)
    ax.set_title("Same-day split clusters around the 500k cap", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "bunching_split.png", dpi=140); plt.close(fig)
    log("→ reports/bunching_split.md")


SPLIT_PLACEBO_DIR = PROCESSED / "_split_placebo"


def split_placebo():
    """Placebo: สลับผู้ขายแบบสุ่มภายใน (หน่วยงาน, เดือน) — ขนาดกลุ่มเท่าเดิมทุกประการ แต่ตัดความเชื่อมโยงผู้ขายจริง
    ถ้าการรวมกลุ่มจริงลด missing มากกว่าการรวมแบบสุ่ม → ไม่ใช่ผลเชิงกลไกของการรวม"""
    SPLIT_PLACEBO_DIR.mkdir(exist_ok=True)
    con = connect()
    con.execute("SELECT setseed(0.42)")
    years = sorted(int(p.name.split("_")[1]) for p in INTERIM.glob("contracts_25??") if p.is_dir())
    t_all = time.time()
    for y in years:
        out = SPLIT_PLACEBO_DIR / f"{y}.parquet"
        if out.exists():
            continue
        if time.time() - t_all > 120:
            log("หยุดพัก (time budget) — รันซ้ำเพื่อทำต่อ"); return
        t0 = time.time()
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
              SELECT DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)
                     agreed_price, agency || '|' || coalesce(sub_agency, '') AS unit, date_trunc('month', sign_date) AS mo,
                     vendor_name_key || '|' || coalesce(province, '') AS vkey, {REGIME_SQL} AS regime
              FROM read_parquet('{src}')
              WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE} AND vendor_name_key IS NOT NULL""")
        parts = []
        NB = 8
        for k in range(NB):
            parts.append(con.execute(f"""
            WITH r AS (SELECT agreed_price, unit, mo, vkey, regime,
                         row_number() OVER (PARTITION BY unit, mo ORDER BY random()) rn_rand,
                         row_number() OVER (PARTITION BY unit, mo ORDER BY vkey, agreed_price) rn_key
                  FROM c WHERE hash(unit) % {NB} = {k}),
            sh AS (SELECT a.regime, a.agreed_price, a.unit, a.mo, b.vkey AS vfake
                   FROM r a JOIN r b ON a.unit = b.unit AND a.mo = b.mo AND a.rn_rand = b.rn_key),
            g AS (SELECT regime, sum(agreed_price) v FROM sh GROUP BY unit, vfake, mo, regime)
            SELECT regime, (ceil(least(v, {MAX_PRICE * 3}) / 1000) * 1000)::BIGINT AS bin, count(*)::INT AS n
            FROM g GROUP BY ALL""").df())
        b = pd.concat(parts).groupby(["regime", "bin"], as_index=False)["n"].sum()
        b.insert(0, "fiscal_year", y); b.insert(0, "level", "month_placebo")
        b.to_parquet(out, index=False)
        log(f"split_placebo {y}: {time.time() - t0:.0f}s")
    log("split_placebo ครบ")


UNIT_BANDS = PROCESSED / "bunching_unit_bands.parquet"
BAND_EDGES = [0, 30_000, 100_000, 200_000, 400_000, 450_000, 500_000, 1_000_000, 2_000_000, 3_000_000]


def unit_bands():
    """นับสัญญาและมูลค่า ต่อ หน่วยงาน×หน่วยงานย่อย × ปีงบ × ช่วงราคา (ไว้ทำ balanced panel ตรวจความครอบคลุม e-GP)"""
    con = connect()
    years = sorted(int(p.name.split("_")[1]) for p in INTERIM.glob("contracts_25??") if p.is_dir())
    case = "CASE " + " ".join(f"WHEN agreed_price <= {b} THEN {b}" for b in BAND_EDGES[1:]) + " END"
    out = []
    for y in years:
        t0 = time.time()
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        out.append(con.execute(f"""
            WITH c AS (SELECT DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)
                              agency, sub_agency, agreed_price, sign_date
                       FROM read_parquet('{src}')
                       WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE})
            SELECT {y} AS fiscal_year, agency, coalesce(sub_agency, '') AS sub_agency, ({case})::INT AS band_hi,
                   CASE WHEN agency LIKE 'องค์การบริหารส่วน%' OR agency LIKE 'เทศบาล%' OR agency = 'กรุงเทพมหานคร' OR agency = 'เมืองพัทยา'
                        THEN 'local' ELSE 'central' END AS gov,
                   count(*)::INT n, sum(agreed_price) v
            FROM c GROUP BY ALL""").df())
        log(f"unit_bands {y}: {time.time() - t0:.0f}s")
    pd.concat(out, ignore_index=True).to_parquet(UNIT_BANDS, index=False)
    log(f"→ {UNIT_BANDS.name}")


BAND_LABELS = {30_000: "0-30k", 100_000: "30-100k", 200_000: "100-200k", 400_000: "200-400k", 450_000: "400-450k",
               500_000: "450-500k", 1_000_000: "500k-1M", 2_000_000: "1-2M", 3_000_000: "2-3M"}


def coverage():
    """ช่วง 200k–400k โตเพราะความครอบคลุม e-GP หรือพฤติกรรม?
    (1) balanced panel ของหน่วยงานที่มีข้อมูลครบ 11 ปี (2) บัญชีการย้ายจำนวน/มูลค่าข้ามช่วงราคา ยึด 2M–3M (3) DiB แยกส่วนกลาง/ท้องถิ่น"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    U = pd.read_parquet(UNIT_BANDS)
    U["unit"] = U["agency"] + "|" + U["sub_agency"]
    U["band"] = U["band_hi"].map(BAND_LABELS)
    order = list(BAND_LABELS.values())
    nyears = U["fiscal_year"].nunique()
    pres = U.groupby("unit")["fiscal_year"].nunique()
    bal = set(pres[pres == nyears].index)
    U["bal"] = U["unit"].isin(bal)
    piv = lambda S, col: S.pivot_table(index="fiscal_year", columns="band", values=col, aggfunc="sum")[order]
    L = ["# ช่วง 200k–400k โตเพราะความครอบคลุม e-GP หรือพฤติกรรม?", "",
         f"หน่วย = หน่วยงาน×หน่วยงานย่อย; balanced panel = มีสัญญาทุกปี 2558–2568 ({len(bal):,} หน่วย, ครอบคลุม 81–92% ของสัญญาทุกช่วงราคา)", "",
         "## 1) จำนวนหน่วยงานต่อปี", "", "| ปีงบ | ส่วนกลาง | ท้องถิ่น |", "|---|---|---|"]
    nu = U.groupby(["fiscal_year", "gov"])["unit"].nunique().unstack()
    for y, r in nu.iterrows():
        L.append(f"| {y} | {r['central']:,} | {r['local']:,} |")
    L += ["", "## 2) ดัชนีจำนวนสัญญา (2558 = 100)"]
    idx = {}
    for name, S in (("ทั้งหมด", U), ("balanced ทั้งหมด", U[U.bal]), ("balanced ส่วนกลาง", U[U.bal & (U.gov == "central")]),
                    ("balanced ท้องถิ่น", U[U.bal & (U.gov == "local")])):
        n = piv(S, "n"); ix = n / n.loc[2558] * 100; idx[name] = ix
        L += ["", f"**{name}**", "", "| ปีงบ | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|"]
        for y, r in ix.iterrows():
            L.append(f"| {y} | " + " | ".join(f"{v:.0f}" for v in r) + " |")
    L += ["", "## 3) บัญชีการย้าย: จริง − คาด เฉลี่ยต่อปี 2561–2568",
          "คาด = สัดส่วนต่อช่วง 2M–3M ของปี 2558 × จำนวน (มูลค่า) ช่วง 2M–3M ของแต่ละปี", ""]
    yrs = [y for y in range(2561, 2569)]
    for name, S in (("ทั้งหมด", U), ("balanced ส่วนกลาง", U[U.bal & (U.gov == "central")]), ("balanced ท้องถิ่น", U[U.bal & (U.gov == "local")])):
        n, v = piv(S, "n"), piv(S, "v")
        en = n["2-3M"].values[:, None] * (n.loc[2558] / n.loc[2558, "2-3M"]).values[None, :]
        ev = v["2-3M"].values[:, None] * (v.loc[2558] / v.loc[2558, "2-3M"]).values[None, :]
        dn = (n - en).loc[yrs].mean(); dv = ((v - ev) / 1e9).loc[yrs].mean(); rr = (n / en).loc[yrs].mean()
        L += [f"**{name}**", "", "| | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|",
              "| สัญญา (พัน) | " + " | ".join(f"{x / 1e3:+,.1f}" for x in dn) + " |",
              "| มูลค่า (พันล้านบาท) | " + " | ".join(f"{x:+,.1f}" for x in dv) + " |",
              "| จริง/คาด | " + " | ".join(f"{x:.2f}" for x in rr) + " |", ""]
        if name == "balanced ส่วนกลาง":
            lost = dv[["0-30k", "30-100k", "500k-1M", "1-2M"]].sum(); gain = dv[["100-200k", "200-400k", "400-450k", "450-500k"]].sum()
            L += [f"มูลค่าที่หายจาก ≤100k และ 500k–2M = {lost:,.1f} พันล้าน/ปี; เพิ่มใน 100k–500k = {gain:+,.1f} พันล้าน/ปี", ""]
    df = pd.read_parquet(BINS); pr = df[(df["var"] == "price") & (df["gov"] != "all")]
    W, LO, HI, T = 5_000, 100_000, 2_000_000, 500_000
    L += ["## 4) Difference-in-bunching แยกส่วนกลาง/ท้องถิ่น (act vs pre, สเกล 200k–400k)", "",
          "| กลุ่ม | excess | Δz/T | missing ≤1.5M |", "|---|---|---|---|"]
    for g in ("central", "local", None):
        S = pr if g is None else pr[pr["gov"] == g]
        yr = _bins5k(S[S.regime == "pre"].groupby("bin")["n"].sum(), W, LO, HI)
        ya = _bins5k(S[S.regime == "act"].groupby("bin")["n"].sum(), W, LO, HI)
        r = dib_estimate(ya.values, yr.values, yr.index.values, T, W, 200_000, 400_000, 50_000, [1_500_000], "const")
        L.append(f"| {g or 'all'} | {r['excess_B']:,.0f} | {r['dz'] / T:.3f} | {r['missing_1500k']:,.0f} |")
    L += ["", "![](figures/bunching_coverage.png)"]
    (REPORTS / "bunching_coverage.md").write_text("\n".join(L), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(order)))
    for ax, name, ttl in ((axes[0], "balanced ส่วนกลาง", "Central govt (balanced panel)"), (axes[1], "balanced ท้องถิ่น", "Local govt (balanced panel)")):
        ix = idx[name]
        for b_, c in zip(order, cols):
            ax.plot(ix.index, ix[b_], color=c, lw=2.2 if b_ in ("200-400k", "450-500k", "2-3M") else 1, label=b_)
        ax.axhline(100, color="k", lw=0.6); ax.set_yscale("log"); ax.set_title(ttl, fontsize=10); ax.set_xlabel("fiscal year (BE)")
    axes[0].set_ylabel("contracts, index FY2558 = 100 (log)"); axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIGS / "bunching_coverage.png", dpi=140); plt.close(fig)
    log("→ reports/bunching_coverage.md")


CONSOL_DIR = PROCESSED / "_consolidation"
GOV_SQL = """CASE WHEN agency LIKE 'องค์การบริหารส่วน%' OR agency LIKE 'เทศบาล%' OR agency = 'กรุงเทพมหานคร' OR agency = 'เมืองพัทยา'
                  THEN 'local' ELSE 'central' END"""


def consolidation_aggregate():
    """กลุ่ม = หน่วยงาน×หน่วยงานย่อย×ผู้ขาย × (วัน | เดือน); ต่อเดือน×gov×ช่วงยอดรวมกลุ่ม:
    จำนวนกลุ่ม, กลุ่มที่มีสัญญาเดียว, กลุ่ม ≥2 สัญญาที่ทุกฉบับ ≤100k (ลายเซ็นการแบ่งที่ 100k), จำนวนสัญญารวม"""
    CONSOL_DIR.mkdir(exist_ok=True)
    con = connect()
    years = sorted(int(p.name.split("_")[1]) for p in INTERIM.glob("contracts_25??") if p.is_dir())
    t_all = time.time()
    for y in years:
        out = CONSOL_DIR / f"{y}.parquet"
        if out.exists():
            continue
        if time.time() - t_all > 120:
            log("หยุดพัก (time budget) — รันซ้ำเพื่อทำต่อ"); return
        t0 = time.time()
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
            SELECT DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)
                   sign_date, agreed_price, agency || '|' || coalesce(sub_agency, '') AS unit,
                   vendor_name_key || '|' || coalesce(province, '') AS vkey, {GOV_SQL} AS gov
            FROM read_parquet('{src}')
            WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE} AND vendor_name_key IS NOT NULL""")
        parts = []
        for level, keyexpr in (("day", "sign_date"), ("month", "date_trunc('month', sign_date)")):
            parts.append(con.execute(f"""
                WITH g AS (SELECT gov, min(sign_date) d0, count(*) k, sum(agreed_price) v, max(agreed_price) mx
                           FROM c GROUP BY unit, vkey, gov, {keyexpr})
                SELECT '{level}' AS level, strftime(d0, '%Y-%m') AS ym, gov,
                       CASE WHEN v <= 100000 THEN 'le100k' WHEN v <= 500000 THEN '100k-500k'
                            WHEN v <= 2000000 THEN '500k-2M' ELSE 'gt2M' END AS tband,
                       count(*)::INT AS n_cl, sum(CASE WHEN k = 1 THEN 1 ELSE 0 END)::INT AS n_single,
                       sum(CASE WHEN k >= 2 AND mx <= 100000 THEN 1 ELSE 0 END)::INT AS n_split100,
                       sum(CASE WHEN k >= 2 AND mx <= 500000 THEN 1 ELSE 0 END)::INT AS n_split500,
                       sum(k)::BIGINT AS n_contracts, sum(v) AS v
                FROM g GROUP BY ALL""").df())
        b = pd.concat(parts, ignore_index=True); b.insert(0, "fiscal_year", y)
        b.to_parquet(out, index=False)
        log(f"consolidation {y}: {time.time() - t0:.0f}s")
    log("consolidation_aggregate ครบ")


def consolidation():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = pd.concat([pd.read_parquet(f) for f in sorted(CONSOL_DIR.glob("*.parquet"))], ignore_index=True)
    C = C[(C.ym >= "2014-10") & (C.ym <= "2025-09")].copy()
    C["per"] = np.where(C.ym < "2015-10", "pre", np.where(C.ym < "2017-09", "transition", "act"))
    L = ["# การแบ่ง/รวมการซื้อ: หลักฐานตรงจากกลุ่ม หน่วยงาน×ผู้ขาย×(วัน|เดือน)", "",
         "กลุ่ม = หน่วยงาน×หน่วยงานย่อย×ผู้ขาย(ชื่อ normalize+จังหวัด) ในวันเดียวกัน (หรือเดือนเดียวกัน); จัดตามยอดรวมของกลุ่ม",
         "- **แบ่งที่ 100k** = กลุ่ม ≥2 สัญญา ที่ทุกฉบับ ≤100k; **แบ่งที่ 500k** = กลุ่ม ≥2 สัญญา ที่ทุกฉบับ ≤500k",
         "- กลุ่มควบคุม = กลุ่มยอดรวม ≤100k (ไม่ได้รับผลจากเพดานใด ๆ)", ""]
    for lev in ("day", "month"):
        S = C[C.level == lev]
        t = S.groupby(["gov", "tband", "per"])[["n_cl", "n_single", "n_split100", "n_split500"]].sum()
        L += [f"## ระดับ{'วัน' if lev == 'day' else 'เดือน'}: % ของกลุ่ม", "",
              "| gov | ยอดรวมกลุ่ม | ช่วง | กลุ่ม | สัญญาเดียว | แบ่งที่ 100k | แบ่งที่ 500k |", "|---|---|---|---|---|---|---|"]
        for (g, tb, pe), r in t.iterrows():
            if tb == "gt2M":
                continue
            L.append(f"| {g} | {tb} | {pe} | {r.n_cl:,} | {r.n_single / r.n_cl:.1%} | {r.n_split100 / r.n_cl:.1%} | {r.n_split500 / r.n_cl:.1%} |")
        L.append("")
    # เทียบกลุ่มที่ถูกแบ่งเกินอัตราปี 2558 กับ missing (ยึด 2M–3M)
    pr = pd.read_parquet(BINS); pr = pr[(pr["var"] == "price") & (pr["gov"] != "all")]
    g = pr.groupby(["fiscal_year", "bin"])["n"].sum().unstack(0).fillna(0)
    band = lambda a, b: g[(g.index > a) & (g.index <= b)].sum()
    A, M = band(2_000_000, 3_000_000), band(500_000, 2_000_000); miss = M[2558] / A[2558] * A - M
    L += ["## กลุ่มยอดรวม 500k–2M ที่ถูกแบ่งเป็นสัญญา ≤500k (ส่วนเกินจากอัตราปี 2558) เทียบสัญญา 500k–2M ที่หายไป", "",
          "| ปีงบ | แบ่ง (วัน) | แบ่ง (เดือน) | missing ยึด 2M–3M |", "|---|---|---|---|"]
    ex = {}
    for lev in ("day", "month"):
        S = C[(C.level == lev) & (C.tband == "500k-2M")].groupby("fiscal_year")[["n_cl", "n_split500"]].sum()
        ex[lev] = S.n_split500 - S.loc[2558, "n_split500"] / S.loc[2558, "n_cl"] * S.n_cl
    for y in miss.index:
        L.append(f"| {y} | {ex['day'][y]:,.0f} | {ex['month'][y]:,.0f} | {miss[y]:,.0f} |")
    yy = [y for y in miss.index if y >= 2561]
    L += ["", f"รวม 2561–68: แบ่ง (วัน) {ex['day'][yy].sum():,.0f} ({ex['day'][yy].sum() / miss[yy].sum():.0%} ของ missing) · "
          f"แบ่ง (เดือน) {ex['month'][yy].sum():,.0f} ({ex['month'][yy].sum() / miss[yy].sum():.0%}) · missing {miss[yy].sum():,.0f}", "",
          "![](figures/bunching_consolidation.png)"]
    (REPORTS / "bunching_consolidation.md").write_text("\n".join(L), encoding="utf-8")
    S = C[C.level == "day"]
    m = S.groupby(["ym", "gov", "tband"])[["n_cl", "n_split100", "n_split500"]].sum().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, gv in zip(axes, ("central", "local")):
        for tb, col_, num, lab in (("100k-500k", "#3b6ea8", "n_split100", "100k-500k clusters split into <=100k pieces"),
                                   ("500k-2M", "#c0392b", "n_split500", "500k-2M clusters split into <=500k pieces"),
                                   ("le100k", "#888888", "n_split100", "control: <=100k clusters with >=2 contracts")):
            d = m[(m.gov == gv) & (m.tband == tb)].set_index("ym").sort_index()
            ax.plot(pd.to_datetime(d.index), d[num] / d["n_cl"], color=col_, lw=1.2, label=lab)
        for x in (ADMIN_SHIFT, REFORM):
            ax.axvline(pd.Timestamp(x), color="k", ls="--", lw=1)
        ax.set_title(f"{gv} govt: same-day unit x vendor clusters", fontsize=9)
    axes[0].set_ylabel("share of clusters"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "bunching_consolidation.png", dpi=140); plt.close(fig)
    log("→ reports/bunching_consolidation.md")


# ประเภทงานจากชื่อโครงการ (หยาบ ๆ ใช้ควบคุมการเปรียบเทียบส่วนลด)
# ตัดคำนำหน้า "ประกวดราคา..." ของชื่อโครงการวิธีแข่งขันออกก่อน (เดิมทำให้ซื้อ/จ้างแบบแข่งขันตกไปอยู่ other)
_KIND_NAME = r"""regexp_replace(coalesce(project_name, ''), '^\s*(ประกวดราคา(อิเล็กทรอนิกส์)?|ประกาศ(ประกวดราคา)?|เชิญชวน|การ)\s*', '')"""
KIND_SQL = (r"""CASE
  WHEN regexp_matches({NM}, 'ซื้อยา|เวชภัณฑ์|การแพทย์|วัคซีน|ทันตกรรม|น้ำยา|รีเอเจนต์|(?i)\b(mg|tab|tablet|inj|cap|syrup)\b') THEN 'medical'
  WHEN regexp_matches({NM}, 'ก่อสร้าง|ถนน|คสล|ค\.ส\.ล|ท่อ|ปรับปรุง|ซ่อมแซม|ขุดลอก|อาคาร|สะพาน|ไฟฟ้า|ประปา|ฝาย|ระบายน้ำ|ลาดยาง|บ่อบาดาล') THEN 'works'
  WHEN regexp_matches({NM}, 'อาหาร|นม') THEN 'food'
  WHEN {NM} LIKE 'ซื้อ%' THEN 'goods'
  WHEN {NM} LIKE 'จ้าง%' OR {NM} LIKE 'เช่า%' THEN 'services'
  ELSE 'other' END""").replace("{NM}", _KIND_NAME)

HARM_DIR = PROCESSED / "_split_harm"


def split_harm_aggregate():
    """ระดับกลุ่ม หน่วยงาน×ผู้ขาย×วัน: ประเภท (single / split500 / split100 / multi_other), วิธี, ความเหมือนของชื่อโครงการ,
    ส่วนลดจากราคากลาง (เฉพาะกลุ่มที่ทุกสัญญามีราคากลาง > 0 และ 0.3 ≤ ราคา/ราคากลาง ≤ 1.5)
    ทำทีละเดือน (กลุ่มอยู่ภายในวันเดียว จึงแบ่งตามเดือนได้พอดี) เพื่อไม่ให้หน่วยความจำเต็ม"""
    HARM_DIR.mkdir(exist_ok=True)
    years = sorted(int(p.name.split("_")[1]) for p in INTERIM.glob("contracts_25??") if p.is_dir())
    t_all = time.time()
    con = connect()
    con.execute("SET preserve_insertion_order=false")
    for y in years:
        out = HARM_DIR / f"{y}.parquet"
        if out.exists():
            continue
        if time.time() - t_all > 100:
            log("หยุดพัก (time budget) — รันซ้ำเพื่อทำต่อ"); return
        t0 = time.time()
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        months = [r[0] for r in con.execute(f"""SELECT DISTINCT date_trunc('month', sign_date) FROM read_parquet('{src}')
                                                 WHERE sign_date IS NOT NULL ORDER BY 1""").fetchall()]
        parts, samples = [], []
        for m0 in months:
            con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
                SELECT DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)
                       sign_date, agreed_price, ref_price, project_id, left(coalesce(project_name, ''), 80) AS pname,
                       agency || '|' || coalesce(sub_agency, '') AS unit, vendor_name_key || '|' || coalesce(province, '') AS vkey,
                       {GOV_SQL} AS gov, {REGIME_SQL} AS regime, {METHOD_CLS} AS mcls, {KIND_SQL} AS kind
                FROM read_parquet('{src}')
                WHERE sign_date >= DATE '{m0}' AND sign_date < DATE '{m0}' + INTERVAL 1 MONTH
                  AND agreed_price > 0 AND agreed_price <= {MAX_PRICE} AND vendor_name_key IS NOT NULL""")
            con.execute("""CREATE OR REPLACE TEMP TABLE cl AS
                WITH p0 AS (SELECT unit, vkey, sign_date, min(pname) p0 FROM c GROUP BY ALL)
                SELECT unit, vkey, sign_date, any_value(gov) gov, any_value(regime) regime, count(*) k,
                       sum(agreed_price) v, sum(ref_price) r, max(agreed_price) mx,
                       bool_and(ref_price > 0) ref_ok, count(DISTINCT project_id) n_proj, count(DISTINCT pname) n_names,
                       CASE WHEN min(kind) = max(kind) THEN min(kind) ELSE 'mixed' END AS kind,
                       min(CASE WHEN pname = p0 THEN 1.0 ELSE jaro_winkler_similarity(pname, p0) END) jw_min,
                       CASE WHEN bool_and(mcls = 'discretion') THEN 'discretion' WHEN bool_and(mcls = 'competitive') THEN 'competitive'
                            ELSE 'mixed' END AS mtype
                FROM c JOIN p0 USING (unit, vkey, sign_date) GROUP BY unit, vkey, sign_date""")
            parts.append(con.execute(f"""
                WITH t AS (SELECT *,
                       CASE WHEN v <= 100000 THEN 'le100k' WHEN v <= 500000 THEN '100k-500k' WHEN v <= 2000000 THEN '500k-2M' ELSE 'gt2M' END tband,
                       CASE WHEN k = 1 THEN 'single' WHEN mx <= 100000 AND v > 100000 THEN 'split100'
                            WHEN mx <= 500000 AND v > 500000 THEN 'split500' ELSE 'multi_other' END ctype,
                       ref_ok AND v / r BETWEEN 0.3 AND 1.5 AS rv FROM cl)
                SELECT {y} AS fiscal_year, '{pd.Timestamp(m0).strftime("%Y-%m")}' AS ym, regime, gov, tband, ctype, mtype, kind,
                       count(*)::INT n_cl, sum(k)::BIGINT n_contracts, sum(v) v,
                       sum(CASE WHEN rv THEN 1 ELSE 0 END)::INT n_rv, sum(CASE WHEN rv THEN v END) v_rv, sum(CASE WHEN rv THEN r END) r_rv,
                       sum(CASE WHEN rv THEN 1 - v / r END) disc_sum, sum(CASE WHEN rv AND abs(v - r) < 1 THEN 1 ELSE 0 END)::INT n_eqref,
                       sum(CASE WHEN n_proj = 1 THEN 1 ELSE 0 END)::INT n_sameproj, sum(CASE WHEN n_names = 1 THEN 1 ELSE 0 END)::INT n_samename,
                       sum(CASE WHEN jw_min >= 0.9 THEN 1 ELSE 0 END)::INT n_simname
                FROM t GROUP BY ALL""").df())
            if y in (2558, 2563, 2568):
                con.execute("SELECT setseed(0.7)")
                samples.append(con.execute("""
                    WITH pick AS (SELECT * FROM cl WHERE k >= 2 AND ((mx <= 500000 AND v > 500000 AND v <= 2000000)
                                                                  OR (mx <= 100000 AND v > 100000 AND v <= 500000))
                                  ORDER BY random() LIMIT 25)
                    SELECT p.sign_date, p.gov, p.regime, p.k, round(p.v) v, round(p.r) r, p.n_proj, p.n_names, round(p.jw_min, 3) jw_min, p.mtype, p.kind,
                           string_agg(c.pname || ' [' || CAST(round(c.agreed_price) AS BIGINT) || ']', ' || ') AS names
                    FROM pick p JOIN c USING (unit, vkey, sign_date) GROUP BY ALL""").df())
        pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
        if samples:
            pd.concat(samples, ignore_index=True).to_csv(REPORTS / f"split_clusters_sample_{y}.csv", index=False, encoding="utf-8-sig")
        log(f"split_harm {y}: {time.time() - t0:.0f}s")
    log("split_harm_aggregate ครบ")


SQ_DIR = PROCESSED / "_split_quality"
# ทำชื่อโครงการให้เทียบกันได้: ตัดตัวเลข (ไทย/อารบิก) วงเล็บ เครื่องหมาย และคำบอกลำดับ/งวด
NAME_NORM = r"""regexp_replace(regexp_replace(lower(coalesce(project_name, '')),
                 '(งวดที่|งวด|รายการที่|ครั้งที่|ลำดับที่|ชุดที่|จำนวน|โครงการ|[0-9๐-๙]+)', '', 'g'),
                 '[^ก-๙a-z]', '', 'g')"""


def split_quality_aggregate():
    """ช่วง act (ปีงบ 2561–2568): จัดสัญญาเป็นกลุ่ม หน่วยงาน×ผู้ขาย×วันเดียวกัน แล้วแยก
    split500 = ≥2 สัญญา ยอดรวม (500k, 2M] ทุกฉบับ ≤500k; multi_under = ≥2 สัญญา ยอดรวม ≤500k; single = สัญญาเดียว
    เก็บ (1) ส่วนลดจากราคากลางตามช่วงราคาของสัญญา (2) ความเหมือนของชื่อโครงการในกลุ่ม"""
    SQ_DIR.mkdir(exist_ok=True)
    t_all = time.time()
    for y in range(2561, 2569):
        out_c, out_n = SQ_DIR / f"contracts_{y}.parquet", SQ_DIR / f"names_{y}.parquet"
        if out_c.exists() and out_n.exists():
            continue
        if time.time() - t_all > 110:
            log("หยุดพัก (time budget) — รันซ้ำเพื่อทำต่อ"); return
        t0 = time.time()
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        dbf = Path(tempfile.gettempdir()) / f"egp_sq_{y}.duckdb"
        if dbf.exists():
            dbf.unlink()
        con = duckdb.connect(str(dbf))
        con.execute(f"SET temp_directory='{(Path(tempfile.gettempdir()) / 'duckdb_egp').as_posix()}'")
        con.execute("SET threads=2; SET memory_limit='1500MB'; SET preserve_insertion_order=false")
        NP = 6  # แบ่งตาม hash(project_id) — แถวซ้ำมี project_id เดียวกันจึงแบ่งได้ตรง
        for part in range(NP):
            head = "CREATE OR REPLACE TABLE c AS" if part == 0 else "INSERT INTO c"
            con.execute(f"""{head}
            SELECT DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)
                   project_id, {NAME_NORM} AS nm, project_type, method, sign_date, agreed_price,
                   CASE WHEN ref_price > 0 THEN ref_price END AS ref,
                   agency || '|' || coalesce(sub_agency, '') AS unit,
                   vendor_name_key || '|' || coalesce(province, '') AS vkey, {GOV_SQL} AS gov
            FROM read_parquet('{src}')
            WHERE sign_date >= DATE '{REFORM}' AND agreed_price > 0 AND agreed_price <= {MAX_PRICE} AND vendor_name_key IS NOT NULL
              AND hash(project_id) % {NP} = {part}""")
        con.execute("""CREATE OR REPLACE TABLE g AS
            SELECT unit, vkey, sign_date, any_value(gov) AS gov, count(*) k, sum(agreed_price) v, max(agreed_price) mx
            FROM c GROUP BY 1, 2, 3""")
        # ความเหมือนของชื่อ: คำนวณเฉพาะกลุ่ม ≥2 สัญญา (เล็กกว่ามาก)
        con.execute("""CREATE OR REPLACE TABLE gn AS
            SELECT unit, vkey, sign_date, count(DISTINCT project_id) n_proj, count(DISTINCT nm) n_nm,
                   count(DISTINCT left(nm, 15)) n_pre, count(DISTINCT project_type) n_type
            FROM c SEMI JOIN (SELECT unit, vkey, sign_date FROM g WHERE k >= 2) USING (unit, vkey, sign_date)
            GROUP BY 1, 2, 3""")
        cat = """CASE WHEN g.k = 1 THEN 'single'
                      WHEN g.v > 500000 AND g.v <= 2000000 AND g.mx <= 500000 THEN 'split500'
                      WHEN g.v <= 500000 THEN 'multi_under' ELSE 'multi_other' END"""
        con.execute(f"""SELECT {y} AS fiscal_year, c.gov, {cat} AS cat,
                   CASE WHEN c.method LIKE '%e-bidding%' THEN 'ebidding' WHEN c.method LIKE '%เฉพาะเจาะจง%' THEN 'specific' ELSE 'other' END AS meth,
                   CASE WHEN c.agreed_price <= 500000 THEN (ceil(c.agreed_price / 50000) * 50000)::INT
                        WHEN c.agreed_price <= 1000000 THEN 1000000 WHEN c.agreed_price <= 2000000 THEN 2000000 ELSE 3000000 END AS pbin,
                   count(*)::INT n, count(c.ref)::INT n_ref,
                   sum(CASE WHEN c.ref IS NOT NULL THEN c.agreed_price END) sum_price_r, sum(c.ref) sum_ref,
                   sum(CASE WHEN c.ref IS NOT NULL AND c.agreed_price >= c.ref THEN 1 ELSE 0 END)::INT n_nodisc,
                   sum(c.agreed_price) sum_price,
                   -- ส่วนลดแบบทนค่าผิดปกติ: ใช้เฉพาะราคากลาง 0.5–2 เท่าของราคาตกลง
                   sum(CASE WHEN c.ref BETWEEN 0.5 * c.agreed_price AND 2 * c.agreed_price THEN 1 ELSE 0 END)::INT n_ok,
                   sum(CASE WHEN c.ref BETWEEN 0.5 * c.agreed_price AND 2 * c.agreed_price THEN (c.ref - c.agreed_price) / c.ref END) sum_disc_ok,
                   sum(CASE WHEN c.ref BETWEEN 0.5 * c.agreed_price AND 2 * c.agreed_price THEN c.agreed_price END) sum_price_ok,
                   sum(CASE WHEN c.ref BETWEEN 0.5 * c.agreed_price AND 2 * c.agreed_price THEN c.ref END) sum_ref_ok,
                   sum(CASE WHEN c.ref BETWEEN 0.5 * c.agreed_price AND 2 * c.agreed_price AND c.agreed_price >= c.ref THEN 1 ELSE 0 END)::INT n_nodisc_ok
            FROM c JOIN g USING (unit, vkey, sign_date, gov) GROUP BY ALL""").df().to_parquet(out_c, index=False)
        con.execute(f"""SELECT {y} AS fiscal_year, gov, {cat} AS cat, count(*)::INT n_cl, sum(k)::BIGINT n_k,
                   sum(CASE WHEN n_proj = 1 THEN 1 ELSE 0 END)::INT same_project,
                   sum(CASE WHEN n_nm = 1 THEN 1 ELSE 0 END)::INT same_name,
                   sum(CASE WHEN n_pre = 1 THEN 1 ELSE 0 END)::INT same_prefix15,
                   sum(CASE WHEN n_type = 1 THEN 1 ELSE 0 END)::INT same_type
            FROM g JOIN gn USING (unit, vkey, sign_date) WHERE k >= 2 GROUP BY ALL""").df().to_parquet(out_n, index=False)
        con.close(); dbf.unlink(missing_ok=True)
        log(f"split_quality {y}: {time.time() - t0:.0f}s")
    log("split_quality_aggregate ครบ")


def split_harm():
    """รายงาน: ชื่อโครงการในกลุ่มที่ถูกแบ่ง + ส่วนลดจากราคากลาง เทียบสัญญาแข่งขัน 500k–2M → reports/bunching_split_harm.md"""
    H = pd.concat([pd.read_parquet(f) for f in sorted(HARM_DIR.glob("*.parquet"))], ignore_index=True)
    H = H[(H.ym >= "2014-10") & (H.ym <= "2025-09")]

    def summ(d):
        return pd.Series({"clusters": d.n_cl.sum(), "k_mean": d.n_contracts.sum() / d.n_cl.sum(),
                          "disc_mean": d.disc_sum.sum() / max(d.n_rv.sum(), 1), "disc_vw": 1 - d.v_rv.sum() / d.r_rv.sum(),
                          "eqref": d.n_eqref.sum() / max(d.n_rv.sum(), 1), "sameproj": d.n_sameproj.sum() / d.n_cl.sum(),
                          "simname": d.n_simname.sum() / d.n_cl.sum()})
    L = ["# ความเสียหายจากการแบ่งสัญญา: ชื่อโครงการและส่วนลดจากราคากลาง", "",
         "กลุ่ม = หน่วยงาน×หน่วยงานย่อย×ผู้ขาย ในวันเดียวกัน; split500 = ≥2 สัญญา ยอดรวม (500k, 2M] ทุกฉบับ ≤500k;",
         "ส่วนลดคำนวณเฉพาะกลุ่มที่ทุกสัญญามีราคากลาง และ 0.3 ≤ ราคา/ราคากลาง ≤ 1.5; ชื่อคล้าย = Jaro–Winkler ≥ 0.9 กับชื่อแรกของกลุ่ม", ""]
    A = H[H.regime == "act"]
    t = A.groupby(["tband", "ctype", "mtype"]).apply(summ)
    t = t[t.clusters >= 5000]
    L += ["## 1) ช่วง act: ส่วนลดตามประเภทกลุ่ม", "", "| ยอดรวม | ประเภท | วิธี | กลุ่ม | สัญญา/กลุ่ม | ส่วนลดเฉลี่ย | ส่วนลดถ่วงมูลค่า | ราคา = ราคากลาง | ชื่อคล้าย |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (tb, ct, mt), r in t.iterrows():
        L.append(f"| {tb} | {ct} | {mt} | {r.clusters:,.0f} | {r.k_mean:.2f} | {r.disc_mean:.1%} | {r.disc_vw:.1%} | {r.eqref:.0%} | {r.simname:.0%} |")
    B = H[H.tband == "500k-2M"].assign(sp=lambda d: (d.ctype == "split500") * d.n_cl)
    sh = B.groupby(["kind", "regime"])[["n_cl", "sp"]].sum(); sh = (sh.sp / sh.n_cl).unstack()
    S = A[(A.tband == "500k-2M") & (A.ctype == "split500") & (A.mtype == "discretion")]
    C = A[(A.tband == "500k-2M") & (A.ctype == "single") & (A.mtype == "competitive")]
    gs = S.groupby("kind")[["n_cl", "v", "v_rv", "r_rv"]].sum(); gc = C.groupby("kind")[["n_cl", "v_rv", "r_rv"]].sum()
    gs["disc_split"] = 1 - gs.v_rv / gs.r_rv; gs["disc_comp"] = (1 - gc.v_rv / gc.r_rv).reindex(gs.index)
    gs["gap"] = gs.disc_comp - gs.disc_split
    gs["excess_frac"] = ((sh["act"] - sh["pre"]) / sh["act"]).reindex(gs.index).clip(lower=0)
    gs["foregone"] = gs.r_rv * gs.gap * (gs.v / gs.v_rv) / 1e9
    gs["foregone_excess"] = gs.foregone * gs.excess_frac
    L += ["", "## 2) ยอดรวม 500k–2M: กลุ่มที่แบ่ง (เฉพาะเจาะจง) vs สัญญาเดียวที่แข่งขัน (e-bidding/คัดเลือก) แยกประเภทงาน", "",
          "| ประเภท | % กลุ่มที่แบ่ง pre → act | กลุ่มแบ่ง | ส่วนลด แบ่ง | ส่วนลด แข่งขัน | ส่วนต่าง | มูลค่ากลุ่มแบ่ง (พันล้าน) | ส่วนลดที่เสียไป ส่วนเกิน (พันล้าน) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, r in gs.iterrows():
        if r.n_cl < 1000:
            continue
        L.append(f"| {k} | {sh.loc[k, 'pre']:.0%} → {sh.loc[k, 'act']:.0%} | {r.n_cl:,.0f} | {r.disc_split:.1%} | "
                 + (f"{r.disc_comp:.1%} | {r.gap:+.1%} | {r.v / 1e9:,.1f} | {r.foregone_excess:,.1f} |" if pd.notna(r.gap) else f"– | – | {r.v / 1e9:,.1f} | – |"))
    nm = gs.drop(index=[i for i in ("medical", "mixed", "other") if i in gs.index])
    L += ["", f"**ส่วนลดที่เสียไป (ไม่รวมยา/กลุ่มผสม, เฉพาะส่วนที่เกินอัตราการแบ่งยุค pre) ≈ {nm.foregone_excess.sum():,.1f} พันล้านบาท ในปีงบ 2560–2568 "
          f"(ทั้งหมดก่อนหักอัตรา pre {nm.foregone.sum():,.1f}); งานก่อสร้าง/ซ่อมแซม {gs.loc['works', 'foregone_excess']:,.1f}**", "",
          "ข้อควรระวัง: เทียบแบบตรง ๆ ไม่ได้ควบคุมขนาด/ลักษณะงาน; สมมติว่าถ้าไม่แบ่งจะได้ส่วนลดเท่าสัญญาแข่งขัน; ราคากลางตั้งโดยหน่วยงานเอง; "
          "การจัดประเภทจากชื่อโครงการหยาบ และรูปแบบชื่อยุค pre ต่างจากยุค act (ประเภท other ยุค pre สูง) จึงอัตรา pre รายประเภทไม่แม่นยำ", ""]
    M = H[H.ctype != "single"].groupby(["regime", "tband", "ctype"]).apply(summ)
    M = M[M.clusters >= 5000]
    L += ["## 3) ความเหมือนของชื่อโครงการในกลุ่ม ≥2 สัญญา", "", "| ยุค | ยอดรวม | ประเภท | กลุ่ม | โครงการเดียวกัน | ชื่อคล้าย |", "|---|---|---|---|---|---|"]
    for (rg, tb, ct), r in M.iterrows():
        L.append(f"| {rg} | {tb} | {ct} | {r.clusters:,.0f} | {r.sameproj:.1%} | {r.simname:.0%} |")
    L += ["", "- สัญญาที่ถูกแบ่งแทบทั้งหมดลงทะเบียนเป็น **คนละโครงการ** (project_id ต่างกัน 99.7%)",
          "- ชื่อคล้ายกัน (≥0.9) ~52% ใกล้เคียงกลุ่มหลายสัญญาที่ไม่ข้ามเพดาน (~53%) → ชื่อไม่ได้แยกการแบ่งออกจากการซื้อหลายรายการปกติ",
          "- ตัวอย่าง (`reports/split_clusters_sample_*.csv`): มีทั้งงานเดียวกันแยกช่วง (ขุดลอกคลองซอยหลายสาย, ถนนหลายสาย), ซื้อน้ำมันแยกกอง, และยาแยกรายการ"]
    (REPORTS / "bunching_split_harm.md").write_text("\n".join(L), encoding="utf-8")
    log("→ reports/bunching_split_harm.md")


HARM_CELLS_DIR = PROCESSED / "_harm_cells"


def harm_cells_aggregate():
    """ระดับกลุ่ม (หน่วยงาน×ผู้ขาย×วัน) ยอดรวม (500k, 2M] ช่วง act: เก็บเฉพาะ split500 (ทุกฉบับ ≤500k, เฉพาะเจาะจง)
    และสัญญาเดียวที่แข่งขัน พร้อม unit, จังหวัด, ประเภทงาน, ยอดรวม, ราคากลางรวม — ใช้ประมาณส่วนลดแบบ fixed effects"""
    HARM_CELLS_DIR.mkdir(exist_ok=True)
    t_all = time.time()
    con = connect()
    con.execute("SET preserve_insertion_order=false")
    for y in range(2561, 2569):
        out = HARM_CELLS_DIR / f"{y}.parquet"
        if out.exists():
            continue
        if time.time() - t_all > 100:
            log("หยุดพัก (time budget) — รันซ้ำเพื่อทำต่อ"); return
        t0 = time.time()
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        months = [r[0] for r in con.execute(f"""SELECT DISTINCT date_trunc('month', sign_date) FROM read_parquet('{src}')
                                                 WHERE sign_date >= DATE '{REFORM}' ORDER BY 1""").fetchall()]
        parts = []
        for m0 in months:
            con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
                SELECT DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)
                       sign_date, agreed_price, ref_price, agency, coalesce(sub_agency, '') AS sub_agency, province,
                       agency || '|' || coalesce(sub_agency, '') AS unit, vendor_name_key || '|' || coalesce(province, '') AS vkey,
                       {GOV_SQL} AS gov, {METHOD_CLS} AS mcls, {KIND_SQL} AS kind
                FROM read_parquet('{src}')
                WHERE sign_date >= DATE '{m0}' AND sign_date < DATE '{m0}' + INTERVAL 1 MONTH
                  AND agreed_price > 0 AND agreed_price <= {MAX_PRICE} AND vendor_name_key IS NOT NULL""")
            parts.append(con.execute(f"""
                WITH cl AS (SELECT unit, vkey, sign_date, any_value(agency) agency, any_value(province) province, any_value(gov) gov,
                                   count(*) k, sum(agreed_price) v, sum(ref_price) r, max(agreed_price) mx, bool_and(ref_price > 0) ref_ok,
                                   CASE WHEN min(kind) = max(kind) THEN min(kind) ELSE 'mixed' END AS kind,
                                   bool_and(mcls = 'discretion') all_disc, bool_and(mcls = 'competitive') all_comp
                            FROM c GROUP BY unit, vkey, sign_date)
                SELECT {y} AS fiscal_year, unit, agency, province, gov, vkey, sign_date, kind, k, v, r,
                       CASE WHEN k >= 2 AND mx <= 500000 AND all_disc THEN 'split' ELSE 'comp' END AS grp,
                       ref_ok AND v / r BETWEEN 0.3 AND 1.5 AS rv
                FROM cl WHERE v > 500000 AND v <= 2000000
                  AND ((k >= 2 AND mx <= 500000 AND all_disc) OR (k = 1 AND all_comp))""").df())
        pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
        log(f"harm_cells {y}: {time.time() - t0:.0f}s")
    log("harm_cells_aggregate ครบ")


def _fe_ols(d, cell_cols, X_cols, add_year=False):
    """OLS แบบ absorb fixed effects (demean ภายใน cell), SE cluster ระดับหน่วยงาน; ใช้เฉพาะ cell ที่มีทั้งสองกลุ่ม"""
    d = d.copy()
    d["cell"] = d[cell_cols].astype(str).agg("|".join, axis=1).astype("category").cat.codes
    g = d.groupby("cell")["split"].agg(["min", "max"])
    keep = g[(g["min"] == 0) & (g["max"] == 1)].index
    d = d[d["cell"].isin(keep)]
    X = d[X_cols].copy()
    if add_year:
        X = pd.concat([X, pd.get_dummies(d["fiscal_year"], prefix="y", drop_first=True).astype(float)], axis=1)
    M = pd.concat([d[["disc"]], X], axis=1)
    M = M - M.groupby(d["cell"].values).transform("mean")
    y, Xm = M["disc"].values, M[X.columns].values
    XtX_inv = np.linalg.inv(Xm.T @ Xm)
    b = XtX_inv @ (Xm.T @ y)
    e = y - Xm @ b
    S = pd.DataFrame(Xm * e[:, None]).groupby(d["unit"].values).sum().values
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    return b[0], float(np.sqrt(V[0, 0])), len(d), len(keep)


def harm_fe():
    """ช่องว่างส่วนลด (กลุ่มที่แบ่ง − สัญญาเดียวที่แข่งขัน) แบบ fixed effects + ส่วนลดที่เสียไป → reports/bunching_harm_fe.md"""
    D = pd.concat([pd.read_parquet(f) for f in sorted(HARM_CELLS_DIR.glob("*.parquet"))], ignore_index=True)
    D = D[D["rv"]].copy()
    D["disc"] = 1 - D["v"] / D["r"]; D["split"] = (D["grp"] == "split").astype(float); D["logv"] = np.log(D["v"])
    specs = [("ไม่มี (เฉพาะปี)", ["fiscal_year"], False), ("ประเภทงาน×ปี", ["kind", "fiscal_year"], False),
             ("หน่วยงาน×ประเภทงาน + ปี", ["unit", "kind"], True), ("หน่วยงาน×ปี×ประเภทงาน", ["unit", "fiscal_year", "kind"], False)]
    L = ["# ช่องว่างส่วนลดจากราคากลาง: กลุ่มที่แบ่ง vs สัญญาเดียวที่แข่งขัน (fixed effects)", "",
         "หน่วย = กลุ่ม หน่วยงาน×ผู้ขาย×วัน ยอดรวม (500k, 2M] ช่วง act (ปีงบ 2561–2568); กลุ่มที่แบ่ง = ≥2 สัญญาเฉพาะเจาะจง ทุกฉบับ ≤500k;",
         "เทียบกับ สัญญาเดียววิธีแข่งขัน; ส่วนลด = 1 − ราคารวม/ราคากลางรวม (เฉพาะ 0.3 ≤ ราคา/ราคากลาง ≤ 1.5); SE cluster ระดับหน่วยงาน;",
         "fixed effects ใช้เฉพาะ cell ที่มีทั้งสองกลุ่ม", "",
         "| fixed effects | ควบคุม log มูลค่า | ช่องว่าง (β) | SE | n กลุ่ม | cells |", "|---|---|---|---|---|---|"]
    for name, cc, ay in specs:
        for xc in (["split"], ["split", "logv"]):
            b, se, n, nc = _fe_ols(D, cc, xc, ay)
            L.append(f"| {name} | {'✓' if 'logv' in xc else ''} | {b:+.1%} | {se:.2%} | {n:,} | {nc:,} |")
    H = pd.concat([pd.read_parquet(f) for f in sorted(HARM_DIR.glob("*.parquet"))], ignore_index=True)
    B = H[H.tband == "500k-2M"].assign(sp=lambda d: (d.ctype == "split500") * d.n_cl)
    sh = B.groupby(["kind", "regime"])[["n_cl", "sp"]].sum(); sh = (sh.sp / sh.n_cl).unstack()
    L += ["", "## รายประเภทงาน (FE หน่วยงาน×ปี, ควบคุม log มูลค่า) และส่วนลดที่เสียไป", "",
          "ส่วนลดที่เสียไป = |β| × ราคากลางรวมของกลุ่มที่แบ่ง × สัดส่วนส่วนเกิน ((อัตราแบ่ง act − pre)/act)", "",
          "| ประเภท | β | SE | % กลุ่ม 500k–2M ที่แบ่ง pre → act | ราคากลางรวมกลุ่มแบ่ง (พันล้าน) | ส่วนลดที่เสียไป (พันล้าน) |", "|---|---|---|---|---|---|"]
    S = D[D["split"] == 1]
    tot = 0.0
    for k in ("works", "goods", "services", "food", "medical"):
        d = D[D.kind == k]
        b, se, n, nc = _fe_ols(d, ["unit", "fiscal_year"], ["split", "logv"])
        ref = S.loc[S.kind == k, "r"].sum() * (S.loc[S.kind == k, "v"].sum() / S.loc[S.kind == k, "v"].sum())
        ex = max((sh.loc[k, "act"] - sh.loc[k, "pre"]) / sh.loc[k, "act"], 0)
        fg = -b * ref * ex / 1e9
        if k != "medical":
            tot += fg
        L.append(f"| {k} | {b:+.1%} | {se:.2%} | {sh.loc[k, 'pre']:.0%} → {sh.loc[k, 'act']:.0%} | {ref / 1e9:,.1f} | {fg:,.1f} |")
    yrs = D["fiscal_year"].nunique()
    L += ["", f"**รวม (ไม่รวมยา) ≈ {tot:,.1f} พันล้านบาท ในปีงบ 2561–2568 (~{tot / yrs:,.1f} พันล้าน/ปี)**", "",
          "ข้อควรระวัง: ไม่รวมกลุ่มประเภทผสม (mixed) และกลุ่มที่ราคากลางผิดปกติ (ประมาณ 1%); สมมติว่าถ้าไม่แบ่งจะได้ส่วนลดเท่าสัญญาแข่งขันในหน่วยงาน-ปี-ประเภทเดียวกัน; "
          "ราคากลางตั้งโดยหน่วยงาน — ถ้าหน่วยงานตั้งราคากลางสูงเผื่อไว้สำหรับงานแข่งขัน ช่องว่างจะสูงเกินจริง; อัตราแบ่งยุค pre มาจากปีงบ 2558 ปีเดียว"]
    (REPORTS / "bunching_harm_fe.md").write_text("\n".join(L), encoding="utf-8")
    log("→ reports/bunching_harm_fe.md")


DID_DIR = PROCESSED / "_did_discount"


def did_aggregate():
    """DiD ส่วนลด (ต.ค. 2557 – ก.ย. 2561): หน่วยงาน×เดือน×ช่วงราคา(ของสัญญา) → จำนวน, ส่วนแบ่งวิธีดุลพินิจ,
    ส่วนลดจากราคากลาง (0.3 ≤ ราคา/ราคากลาง ≤ 1.5) และจากงบประมาณ (0.3 ≤ ราคา/งบ ≤ 1.5)"""
    DID_DIR.mkdir(exist_ok=True)
    con = connect()
    con.execute("SET preserve_insertion_order=false")
    t_all = time.time()
    band = """CASE WHEN agreed_price <= 30000 THEN '0-30k' WHEN agreed_price <= 100000 THEN '30k-100k'
                   WHEN agreed_price <= 500000 THEN '100k-500k' WHEN agreed_price <= 2000000 THEN '500k-2M' ELSE '2M-3M' END"""
    for y in range(2558, 2562):
        out = DID_DIR / f"{y}.parquet"
        if out.exists():
            continue
        if time.time() - t_all > 110:
            log("หยุดพัก (time budget) — รันซ้ำเพื่อทำต่อ"); return
        t0 = time.time()
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        con.execute(f"""COPY (
            WITH c AS (SELECT DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)
                              agency || '|' || coalesce(sub_agency, '') AS unit, {GOV_SQL} AS gov,
                              strftime(sign_date, '%Y-%m') AS ym, agreed_price, ref_price, budget, {METHOD_CLS} AS mcls,
                              {KIND_SQL} AS kind
                       FROM read_parquet('{src}')
                       WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE}),
                 d AS (SELECT *, {band} AS band,
                              ref_price > 0 AND agreed_price / ref_price BETWEEN 0.3 AND 1.5 AS ok_r,
                              budget > 0 AND agreed_price / budget BETWEEN 0.3 AND 1.5 AS ok_b FROM c)
            SELECT {y} AS fiscal_year, unit, gov, ym, band, kind, count(*)::INT n,
                   sum(CASE WHEN mcls = 'discretion' THEN 1 ELSE 0 END)::INT n_disc,
                   sum(CASE WHEN mcls = 'competitive' THEN 1 ELSE 0 END)::INT n_comp,
                   sum(CASE WHEN ok_r THEN 1 ELSE 0 END)::INT n_r, sum(CASE WHEN ok_r THEN 1 - agreed_price / ref_price END) sd_r,
                   sum(CASE WHEN ok_r THEN agreed_price END) sp_r, sum(CASE WHEN ok_r THEN ref_price END) sr_r,
                   sum(CASE WHEN ok_r AND agreed_price >= ref_price THEN 1 ELSE 0 END)::INT n_eq_r,
                   sum(CASE WHEN ok_b THEN 1 ELSE 0 END)::INT n_b, sum(CASE WHEN ok_b THEN 1 - agreed_price / budget END) sd_b,
                   sum(CASE WHEN ok_b THEN agreed_price END) sp_b, sum(CASE WHEN ok_b THEN budget END) sb_b,
                   sum(CASE WHEN ok_b AND agreed_price >= budget THEN 1 ELSE 0 END)::INT n_eq_b
            FROM d GROUP BY ALL) TO '{out.as_posix()}' (FORMAT parquet)""")
        log(f"did_aggregate {y}: {time.time() - t0:.0f}s")
    log("did_aggregate ครบ")


def _twfe(d, y, w, X, fe1, fe2, cl, iters=30, tol=1e-10, return_V=False):
    """WLS ที่ absorb 2 ชุด fixed effects ด้วย alternating projections (numpy bincount), SE cluster"""
    W = d[w].astype(float).values
    M = d[[y] + X].astype(float).values.copy()
    keys = [pd.factorize(d[f])[0] for f in (fe1, fe2)]
    wsum = [np.bincount(k, weights=W) for k in keys]
    for _ in range(iters):
        delta = 0.0
        for k, ws in zip(keys, wsum):
            for j in range(M.shape[1]):
                mu = np.bincount(k, weights=W * M[:, j]) / np.where(ws > 0, ws, 1)
                M[:, j] -= mu[k]
                delta = max(delta, float(np.abs(mu).max()))
        if delta < tol:
            break
    yv, Xv = M[:, 0], M[:, 1:]
    sw = np.sqrt(W)
    Xw, yw = Xv * sw[:, None], yv * sw
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    b = XtX_inv @ (Xw.T @ yw)
    e = yw - Xw @ b
    g = pd.factorize(d[cl])[0]
    S = np.vstack([np.bincount(g, weights=Xw[:, j] * e) for j in range(Xw.shape[1])]).T
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    if return_V:
        return b, np.sqrt(np.diag(V)), V
    return b, np.sqrt(np.diag(V))


def did_discount():
    """ผลของการเปลี่ยนวิธีจากแข่งขันเป็นดุลพินิจต่อส่วนลด: กลุ่ม 100k–500k (เปลี่ยน ต.ค. 2558) vs กลุ่มควบคุม
    30k–100k (ดุลพินิจตลอด) และ 2M–3M (แข่งขันตลอด); FE หน่วยงาน×ช่วงราคา + เดือน; ถ่วงด้วยจำนวนสัญญา; SE cluster หน่วยงาน"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = pd.concat([pd.read_parquet(f) for f in sorted(DID_DIR.glob("*.parquet"))], ignore_index=True)
    D = D[(D.ym >= "2014-10") & (D.ym <= "2018-09")].copy()
    D["per"] = np.select([D.ym < "2015-10", D.ym < "2017-09"], ["pre", "transition"], "act")
    cell = D.groupby(["unit", "ym", "band", "per"], as_index=False)[["n", "n_disc", "n_r", "sd_r", "n_eq_r", "n_b", "sd_b", "n_eq_b"]].sum()
    cell["disc_share"] = cell.n_disc / cell.n
    L = ["# ผลของวิธีดุลพินิจต่อส่วนลด: difference-in-differences รอบ ต.ค. 2558", "",
         "กลุ่มทดลอง = สัญญา 100k–500k (สอบราคา → ตกลงราคา เมื่อ ต.ค. 2558); ควบคุม A = 30k–100k (ตกลงราคาตลอด); ควบคุม B = 2M–3M (แข่งขันตลอด)",
         "ส่วนลด = 1 − ราคา/ราคากลาง (0.3–1.5) และ 1 − ราคา/งบประมาณ (0.3–1.5; งบตั้งก่อนเลือกวิธี จึงไม่ถูกบิดจากการตั้งราคากลางเผื่อ)",
         "หน่วยวิเคราะห์ = หน่วยงาน×เดือน×ช่วงราคา, FE หน่วยงาน×ช่วงราคา + เดือน×กลุ่มควบคุม, ถ่วงด้วยจำนวนสัญญา, SE cluster หน่วยงาน; ช่วง pre + transition (ต.ค. 2557 – ส.ค. 2560)", "",
         "## 1) ค่าเฉลี่ยดิบ", "", "| ช่วงราคา | ช่วงเวลา | สัญญา | % ดุลพินิจ | ส่วนลด/ราคากลาง | ส่วนลด/งบ | ราคา ≥ งบ |", "|---|---|---|---|---|---|---|"]
    g = D.groupby(["band", "per"])[["n", "n_disc", "n_r", "sd_r", "n_b", "sd_b", "n_eq_b"]].sum()
    for (b_, pe), r in g.iterrows():
        if b_ == "0-30k":
            continue
        L.append(f"| {b_} | {pe} | {r.n:,.0f} | {r.n_disc / r.n:.0%} | {r.sd_r / r.n_r:.1%} | {r.sd_b / r.n_b:.1%} | {r.n_eq_b / r.n_b:.0%} |")
    W = cell[cell.per != "act"].copy()
    W["post"] = (W.ym >= "2015-10").astype(float)
    res, es = [], {}
    L += ["", "## 2) DiD (ช่วง pre + transition)", "",
          "| ควบคุม | first stage: Δ%ดุลพินิจ | DiD ส่วนลด/ราคากลาง | DiD ส่วนลด/งบ | Wald: ผลต่อ 1 หน่วยดุลพินิจ (งบ) |", "|---|---|---|---|---|"]
    for ctrl in ("30k-100k", "2M-3M"):
        d = W[W.band.isin(["100k-500k", ctrl])].copy()
        d["treat"] = (d.band == "100k-500k").astype(float); d["tp"] = d.treat * d.post
        d["ub"] = d.unit + "|" + d.band
        out = {}
        for yv, wv, sdc in (("fs", "n", None), ("dr", "n_r", "sd_r"), ("db", "n_b", "sd_b")):
            dd = d[d[wv] > 0].copy()
            dd["yy"] = dd.disc_share if sdc is None else dd[sdc] / dd[wv]
            b, se = _twfe(dd, "yy", wv, ["tp"], "ub", "ym", "unit")
            out[yv] = (b[0], se[0])
        res.append((ctrl, out))
        L.append(f"| {ctrl} | {out['fs'][0]:+.3f} ({out['fs'][1]:.3f}) | {out['dr'][0]:+.2%} ({out['dr'][1]:.2%}) | {out['db'][0]:+.2%} ({out['db'][1]:.2%}) | {out['db'][0] / out['fs'][0]:+.2%} |")
        # event study (ส่วนลด/งบ) อ้างอิง ก.ย. 2558
        dd = d[d.n_b > 0].copy(); dd["yy"] = dd.sd_b / dd.n_b
        months = sorted(dd.ym.unique()); ref = "2015-09"
        X = []
        for m in months:
            if m == ref:
                continue
            c = f"e_{m}"; dd[c] = dd.treat * (dd.ym == m); X.append(c)
        b, se = _twfe(dd, "yy", "n_b", X, "ub", "ym", "unit", iters=15)
        e = pd.DataFrame({"ym": [x[2:] for x in X] + [ref], "b": list(b) + [0.0], "se": list(se) + [0.0]}).sort_values("ym")
        pre_mean = e.loc[e.ym < "2015-10", "b"].mean()
        e["b"] = e["b"] - pre_mean  # อ้างอิงค่าเฉลี่ยช่วง pre ทั้งหมด (ก.ย. 2558 เป็นเดือนผิดปกติ)
        es[ctrl] = e
        post = e.loc[(e.ym >= "2015-10") & (e.ym < "2017-09"), "b"]
        L.append(f"| {ctrl} | event study: ค่าเฉลี่ยหลัง ต.ค. 2558 เทียบค่าเฉลี่ย pre = {post.mean():+.2%}; เดือนแรก (ต.ค. 2558) {e.loc[e.ym == '2015-10', 'b'].iloc[0]:+.2%} | | | |")
    L += ["", "ตีความ: ถ้าการซื้อหนึ่งรายการเปลี่ยนจากแข่งขันเป็นดุลพินิจ ส่วนลดจากงบประมาณลดลงตามคอลัมน์ Wald",
          "เทียบกับช่องว่าง FE ของกลุ่มที่แบ่ง vs แข่งขัน (−12.5 pp ราคากลาง) ใน `bunching_harm_fe.md`", "",
          "![](figures/did_discount_event.png)"]
    (REPORTS / "did_discount.md").write_text("\n".join(L), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 4))
    for (ctrl, e), col in zip(es.items(), ("#3b6ea8", "#c0392b")):
        t = pd.to_datetime(e.ym)
        ax.plot(t, e.b, color=col, lw=1.3, label=f"100k-500k vs {ctrl}")
        ax.fill_between(t, e.b - 1.96 * e.se, e.b + 1.96 * e.se, color=col, alpha=0.15)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(pd.Timestamp(ADMIN_SHIFT), color="k", ls="--", lw=1)
    ax.set_ylabel("discount vs budget: treated - control\n(relative to pre-period mean)"); ax.legend(fontsize=8)
    ax.set_title("Event study: discount when 100k-500k purchases moved from competitive to discretionary (Oct 2015)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "did_discount_event.png", dpi=140); plt.close(fig)
    log("→ reports/did_discount.md")


PLACEBO_T_DIR = PROCESSED / "_split_placebo_T"


def split_placebo_T_aggregate():
    """Placebo เพดาน: กลุ่ม หน่วยงาน×ผู้ขาย×วัน ที่ทุกชิ้นอยู่ในช่วง (500k, T'] และยอดรวม > T' (T' = 1M ไม่เคยเป็นเพดาน; 2M = เพดานสอบราคาก่อน พ.ร.บ.)
    เทียบกับการแบ่งที่ 500k (ทุกชิ้น ≤500k ยอดรวม 500k–2M) — รายเดือน"""
    PLACEBO_T_DIR.mkdir(exist_ok=True)
    con = connect()
    con.execute("SET preserve_insertion_order=false")
    t_all = time.time()
    for y in range(2558, 2569):
        out = PLACEBO_T_DIR / f"{y}.parquet"
        if out.exists():
            continue
        if time.time() - t_all > 110:
            log("หยุดพัก (time budget) — รันซ้ำเพื่อทำต่อ"); return
        t0 = time.time()
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        con.execute(f"""COPY (
            WITH c AS (SELECT DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)
                              sign_date, agreed_price, agency || '|' || coalesce(sub_agency, '') AS unit,
                              vendor_name_key || '|' || coalesce(province, '') AS vkey, {GOV_SQL} AS gov
                       FROM read_parquet('{src}')
                       WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE} AND vendor_name_key IS NOT NULL),
                 g AS (SELECT gov, strftime(sign_date, '%Y-%m') ym, count(*) k, sum(agreed_price) v, max(agreed_price) mx, min(agreed_price) mn
                       FROM c GROUP BY unit, vkey, sign_date, gov)
            SELECT {y} AS fiscal_year, ym, gov,
                   sum(CASE WHEN v > 500000 AND v <= 2000000 THEN 1 ELSE 0 END)::INT n_500,
                   sum(CASE WHEN k >= 2 AND mx <= 500000 AND v > 500000 AND v <= 2000000 THEN 1 ELSE 0 END)::INT s_500,
                   sum(CASE WHEN v > 1000000 AND v <= 4000000 THEN 1 ELSE 0 END)::INT n_1m,
                   sum(CASE WHEN k >= 2 AND mn > 500000 AND mx <= 1000000 AND v > 1000000 THEN 1 ELSE 0 END)::INT s_1m,
                   sum(CASE WHEN v > 2000000 AND v <= 8000000 THEN 1 ELSE 0 END)::INT n_2m,
                   sum(CASE WHEN k >= 2 AND mn > 500000 AND mx <= 2000000 AND v > 2000000 THEN 1 ELSE 0 END)::INT s_2m,
                   sum(CASE WHEN v > 100000 AND v <= 500000 THEN 1 ELSE 0 END)::INT n_100,
                   sum(CASE WHEN k >= 2 AND mx <= 100000 AND v > 100000 AND v <= 500000 THEN 1 ELSE 0 END)::INT s_100
            FROM g GROUP BY ALL) TO '{out.as_posix()}' (FORMAT parquet)""")
        log(f"split_placebo_T {y}: {time.time() - t0:.0f}s")
    log("split_placebo_T_aggregate ครบ")


def split_placebo_T():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = pd.concat([pd.read_parquet(f) for f in sorted(PLACEBO_T_DIR.glob("*.parquet"))], ignore_index=True)
    P = P[(P.ym >= "2014-10") & (P.ym <= "2025-09")]
    m = P.groupby("ym")[["n_100", "s_100", "n_500", "s_500", "n_1m", "s_1m", "n_2m", "s_2m"]].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    for t, col, lab in (("100", "#3b6ea8", "100k: clusters 100k-500k, all pieces <=100k"),
                        ("500", "#c0392b", "500k: clusters 500k-2M, all pieces <=500k"),
                        ("2m", "#8e44ad", "2M: clusters 2M-8M, all pieces in (500k, 2M]"),
                        ("1m", "#7f8c8d", "placebo 1M: clusters 1M-4M, all pieces in (500k, 1M]")):
        r = (m["s_" + t] / m["n_" + t]).rolling(3, center=True, min_periods=1).mean()
        ax.plot(pd.to_datetime(r.index), r, color=col, lw=1.4, label=lab)
    for x in (ADMIN_SHIFT, REFORM):
        ax.axvline(pd.Timestamp(x), color="k", ls="--", lw=1)
    ax.set_ylabel("share of same-day unit x vendor clusters (3-month avg)"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Splitting tracks thresholds: rises where burden rose, falls where it disappeared, flat at a non-threshold", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "split_placebo_thresholds.png", dpi=140); plt.close(fig)
    log("→ figures/split_placebo_thresholds.png")


UNIT_BINS_DIR = PROCESSED / "_unit_bins5k"


def unit_bins_aggregate():
    """จำนวนสัญญา ต่อ หน่วยงาน×ปีงบ×ช่วงเวลา×ช่อง 5k (100k–2M) — สำหรับ cluster bootstrap ระดับหน่วยงาน"""
    UNIT_BINS_DIR.mkdir(exist_ok=True)
    con = connect()
    con.execute("SET preserve_insertion_order=false")
    t_all = time.time()
    for y in range(2558, 2569):
        out = UNIT_BINS_DIR / f"{y}.parquet"
        if out.exists():
            continue
        if time.time() - t_all > 120:
            log("หยุดพัก (time budget) — รันซ้ำเพื่อทำต่อ"); return
        t0 = time.time()
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        con.execute(f"""COPY (
            WITH c AS (SELECT DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)
                              agency || '|' || coalesce(sub_agency, '') AS unit, sign_date, agreed_price
                       FROM read_parquet('{src}')
                       WHERE sign_date IS NOT NULL AND agreed_price > 100000 AND agreed_price <= 2000000)
            SELECT {y} AS fiscal_year, unit, {REGIME_SQL} AS regime,
                   (ceil(agreed_price / 5000) * 5000)::INT AS bin, count(*)::INT AS n
            FROM c GROUP BY ALL) TO '{out.as_posix()}' (FORMAT parquet)""")
        log(f"unit_bins {y}: {time.time() - t0:.0f}s")
    log("unit_bins_aggregate ครบ")


def cluster_bootstrap(R=500, seed=2026):
    """Cluster (หน่วยงาน) Poisson bootstrap ของ difference-in-bunching ที่ 500k: สุ่มน้ำหนัก Poisson(1) ต่อหน่วยงาน
    ใช้ร่วมกันทุกช่วงเวลา (คงความสัมพันธ์ภายในหน่วยงานข้ามปี) → se แบบ cluster; รวมกับ shape noise จาก placebo เดิม"""
    from scipy import sparse
    U = pd.concat([pd.read_parquet(f) for f in sorted(UNIT_BINS_DIR.glob("*.parquet"))], ignore_index=True)
    W, LO, HI, T = 5_000, 100_000, 2_000_000, 500_000
    grid = np.arange(LO + W, HI + W, W)
    gi = {g: i for i, g in enumerate(grid)}
    U = U[U["bin"].isin(gi)]
    uid, units = pd.factorize(U["unit"])
    U["uid"] = uid; U["gid"] = U["bin"].map(gi)
    nU, nG = len(units), len(grid)
    groups = {"pre": U.regime == "pre", "transition (FY2559–60)": U.regime == "transition", "act (FY2560–68)": U.regime == "act"}
    for y in range(2559, 2569):
        groups[f"FY{y}"] = (U.fiscal_year == y) & (U.regime != "pre")
    M = {k: sparse.csr_matrix((U.loc[m, "n"].values.astype(float), (U.loc[m, "uid"].values, U.loc[m, "gid"].values)), shape=(nU, nG))
         for k, m in groups.items()}
    est = lambda yt, yr: dib_estimate(yt, yr, grid, T, W, 200_000, 400_000, 50_000, [1_500_000], "const")
    point = {k: est(np.asarray(M[k].sum(0)).ravel(), np.asarray(M["pre"].sum(0)).ravel()) for k in groups if k != "pre"}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point}
    for r0 in range(0, R, 100):
        Wt = rng.poisson(1.0, size=(min(100, R - r0), nU)).astype(float)
        C = {k: np.asarray((sparse.csr_matrix(Wt) @ M[k]).todense()) for k in groups}
        for i in range(Wt.shape[0]):
            for k in point:
                o = est(C[k][i], C["pre"][i])
                draws[k].append((o["dz"], o["excess_B"], o["missing_1500k"]))
    old = pd.read_csv(REPORTS / "bunching_dib.csv").set_index("label")
    rows = []
    for k, o in point.items():
        dr = np.array(draws[k])
        se_cl = dr[:, 0].std(ddof=1)
        psd = old.loc[k, "placebo_dz_sd"] if k in old.index else np.nan
        rows.append({"label": k, "dz": o["dz"], "dz_over_T": o["dz"] / T, "se_contract_poisson": old.loc[k, "dz_se"] if k in old.index else np.nan,
                     "se_cluster": se_cl, "placebo_sd": psd, "se_total_cluster": np.sqrt(se_cl ** 2 + psd ** 2),
                     "ci95_lo_T": np.percentile(dr[:, 0], 2.5) / T, "ci95_hi_T": np.percentile(dr[:, 0], 97.5) / T,
                     "excess_B": o["excess_B"], "excess_se_cluster": dr[:, 1].std(ddof=1)})
    Rr = pd.DataFrame(rows)
    Rr.to_csv(REPORTS / "bunching_cluster_boot.csv", index=False)
    L = [f"# Cluster bootstrap ระดับหน่วยงาน ({R} รอบ, Poisson weights, {nU:,} หน่วยงาน)", "",
         "difference-in-bunching ที่ 500k (อ้างอิง pre, สเกล 200k–400k, zL 50k); se รวม = √(se_cluster² + placebo_sd²)", "",
         "| ช่วง | Δz (บาท) | Δz/T | se สุ่มรายสัญญา (เดิม) | se cluster | se รวม | 95% CI (cluster) Δz/T | excess | se excess |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in Rr.iterrows():
        L.append(f"| {r['label']} | {r.dz:,.0f} | {r.dz_over_T:.3f} | {r.se_contract_poisson:,.0f} | {r.se_cluster:,.0f} | {r.se_total_cluster:,.0f} | "
                 f"[{r.ci95_lo_T:.3f}, {r.ci95_hi_T:.3f}] | {r.excess_B:,.0f} | {r.excess_se_cluster:,.0f} |")
    (REPORTS / "bunching_cluster_boot.md").write_text("\n".join(L), encoding="utf-8")
    log("→ reports/bunching_cluster_boot.md")


def dib_spec_curve():
    """Specification curve ของ difference-in-bunching (act vs pre ที่ 500k): ช่วงสเกล × zL × ความกว้างช่อง × ส่วนของปีอ้างอิง × รูปสเกล"""
    import itertools
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_parquet(BINS); pr = df[(df["var"] == "price") & (df["gov"] != "all")]
    T, LO, HI = 500_000, 100_000, 2_000_000
    pre = pr[pr.regime == "pre"]
    refs = {"full": pre, "H1": pre[pre.ym <= "2015-03"], "H2": pre[pre.ym >= "2015-04"], "no Aug-Sep": pre[pre.ym <= "2015-07"]}
    tgt = pr[pr.regime == "act"].groupby("bin")["n"].sum()
    rows = []
    for (slo, shi), zL, W, rk, shape in itertools.product([(150_000, 400_000), (200_000, 400_000), (200_000, 450_000), (250_000, 400_000)],
                                                       [25_000, 50_000, 75_000, 100_000], [2_000, 5_000, 10_000], list(refs), ["const", "loglin"]):
        yr = _bins5k(refs[rk].groupby("bin")["n"].sum(), W, LO, HI); ya = _bins5k(tgt, W, LO, HI)
        o = dib_estimate(ya.values, yr.values, yr.index.values, T, W, slo, shi, zL, [1_500_000], shape)
        rows.append({"scale": f"{slo // 1000}-{shi // 1000}k", "zL": zL, "W": W, "ref": rk, "shape": shape,
                     "dz_over_T": o["dz"] / T, "excess": o["excess_B"]})
    R = pd.DataFrame(rows).sort_values("dz_over_T").reset_index(drop=True)
    R.to_csv(REPORTS / "bunching_dib_spec_curve.csv", index=False)
    q = R.dz_over_T.quantile([0, 0.05, 0.5, 0.95, 1]).round(3).tolist()
    L = [f"# Specification curve: difference-in-bunching ที่ 500k (act vs pre), {len(R)} specs", "",
         f"Δz/T: min {q[0]} · p5 {q[1]} · **median {q[2]}** · p95 {q[3]} · max {q[4]}; ทุก spec > 0: {bool((R.dz_over_T > 0).all())}", "",
         "ค่ากลาง Δz/T ตามตัวเลือก:", ""]
    for col in ("scale", "zL", "W", "ref", "shape"):
        L.append(f"- {col}: " + ", ".join(f"{k} = {v:.3f}" for k, v in R.groupby(col)["dz_over_T"].median().items()))
    L += ["", "![](figures/bunching_dib_spec_curve.png)"]
    (REPORTS / "bunching_dib_spec_curve.md").write_text("\n".join(L), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.scatter(np.arange(len(R)), R.dz_over_T, s=6, c=np.where(R["shape"] == "const", "#3b6ea8", "#c0392b"))
    ax.axhline(R.dz_over_T.median(), color="k", lw=0.8, ls="--")
    ax.set_xlabel("specification (sorted)"); ax.set_ylabel("dz / T"); ax.set_ylim(bottom=0)
    ax.set_title(f"Difference-in-bunching at 500k across {len(R)} specifications (blue = constant scale, red = log-linear)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "bunching_dib_spec_curve.png", dpi=140); plt.close(fig)
    log("→ reports/bunching_dib_spec_curve.md")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("aggregate", "all"):
        aggregate()
    if step in ("estimate", "all"):
        estimate()
    if step == "robust":
        robust()
    if step == "datadriven":
        datadriven()
    if step == "dib":
        dib()
    if step == "event":
        event()
    if step == "split_aggregate":
        split_aggregate()
    if step == "split_placebo":
        split_placebo()
    if step == "unit_bands":
        unit_bands()
    if step == "coverage":
        coverage()
    if step == "consolidation_aggregate":
        consolidation_aggregate()
    if step == "consolidation":
        consolidation()
    if step == "split_harm_aggregate":
        split_harm_aggregate()
    if step == "split_quality_aggregate":
        split_quality_aggregate()
    if step == "split_harm":
        split_harm()
    if step == "harm_cells_aggregate":
        harm_cells_aggregate()
    if step == "harm_fe":
        harm_fe()
    if step == "did_aggregate":
        did_aggregate()
    if step == "did_discount":
        did_discount()
    if step == "split_placebo_T_aggregate":
        split_placebo_T_aggregate()
    if step == "split_placebo_T":
        split_placebo_T()
    if step == "unit_bins_aggregate":
        unit_bins_aggregate()
    if step == "cluster_bootstrap":
        cluster_bootstrap()
    if step == "dib_spec_curve":
        dib_spec_curve()
    if step == "split":
        split_analysis()
