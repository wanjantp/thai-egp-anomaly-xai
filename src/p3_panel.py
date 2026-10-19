"""Paper 3, step 1 of the locked pre-analysis plan (project doc claude/p3-pre-analysis-plan.md, locked 19 Oct 2026).

Builds the analysis panel. Writes the unit-level outcome pseudo-outcomes and the pre-period covariates to SEPARATE
files and never relates them to each other (the only joint output is the eligibility / selection table, which uses
cell counts, not outcome values). No CATE, no regression of outcomes on covariates.

  python src/p3_panel.py [--rebuild]      (EGP_DUCKDB_MEMORY=16GB EGP_DUCKDB_THREADS=8 recommended)

Outputs, data/processed/p3/:
  build/*_YYYY.parquet  per-year aggregates (cache; --rebuild recomputes)
  tau.parquet   O1 / O1-P / R4 / R5 / R6 unit DiD of recorded discount: spec, window, m, unit, tau, v, fs, cell counts
  o2.parquet    O2 splitting effect: window, m, unit, tau, v
  o3.parquet    O3 (exploratory) top-vendor share change: window, unit, tau
  X.parquet     covariates measured Oct 2014 - Jul 2015 + policy weights B_u, N_u (no outcome)
  x_meta.json   covariate list, mechanical drops, mapping notes
reports/p3_panel.md / .json: eligible N per spec, selection table, O2 power gate, build checks (aggregates only)

Definitions follow the plan: unit = agency|sub_agency; bands by agreed price; recorded discount
d = 1 - agreed/budget (budget > 0, ratio in [0.3, 1.5]); pre Oct 2014 - Sep 2015; post main Oct-Nov 2015,
alt Oct 2015 - 22 Aug 2017 (contracts signed before the 2017 Act), alt_late Dec 2015 - 22 Aug 2017.
Purchases (O1-P, O2, N_u) = unit x vendor key x sign date groups (paper 1); contracts without a vendor key are
their own purchase. Duplicate rows are dropped as in paper 1, with a fixed order so the kept row is deterministic.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from bunching import GOV_SQL, MAX_PRICE, METHOD_CLS, PROCESSED, REPORTS, connect, log
from revision_checks import src

OUT = PROCESSED / "p3"
BUILD = OUT / "build"
YEARS = range(2558, 2562)
START, ACT = "2014-10-01", "2017-08-23"
X_END = "2015-08-01"                     # covariates: contracts signed before 1 Aug 2015 (Oct 2014 - Jul 2015)
PRE, PRE_R4 = ("2014-10", "2015-09"), ("2014-10", "2015-07")
WINDOWS = {"main": ("2015-10", "2015-11"), "alt": ("2015-10", "2017-08"), "alt_late": ("2015-12", "2017-08")}
TREAT, CTRL, CTRL2 = "100k-500k", "30k-100k", "2M-3M"
MS = (1, 3, 5)
N_MIN = 2000
O2_REF = 0.423 - 0.119                   # paper 1: same-vendor same-day split share 11.9% (2015) -> 42.3% (2018-25)
O2_MDE_MAX = O2_REF / 2
# R5: 2015-16 village / tambon stimulus projects; matched on the project name with spaces removed and Thai digits
# converted. Locked with the plan; not changed after estimation starts.
STIM_RE = "ตำบลละ|หมู่บ้านละ|กองทุนหมู่บ้าน|ประชารัฐ|ส่งเสริมความเป็นอยู่"
DEDUP = ("DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)",
         "ORDER BY project_id, contract_no, winner_name, agreed_price, sign_date, budget NULLS LAST, ref_price NULLS LAST, "
         "agency, sub_agency NULLS LAST, vendor_name_key NULLS LAST, province NULLS LAST, project_type NULLS LAST, "
         "method NULLS LAST, project_name NULLS LAST")


def band(col):
    return f"""CASE WHEN {col} <= 30000 THEN '0-30k' WHEN {col} <= 100000 THEN '30k-100k'
                    WHEN {col} <= 500000 THEN '100k-500k' WHEN {col} <= 2000000 THEN '500k-2M'
                    WHEN {col} <= {MAX_PRICE} THEN '2M-3M' ELSE '>3M' END"""


REGIONS = {
    "north": "เชียงราย เชียงใหม่ น่าน พะเยา แพร่ แม่ฮ่องสอน ลำปาง ลำพูน อุตรดิตถ์",
    "northeast": "กาฬสินธุ์ ขอนแก่น ชัยภูมิ นครพนม นครราชสีมา บึงกาฬ บุรีรัมย์ มหาสารคาม มุกดาหาร ยโสธร ร้อยเอ็ด เลย "
                 "สกลนคร สุรินทร์ ศรีสะเกษ หนองคาย หนองบัวลำภู อุดรธานี อุบลราชธานี อำนาจเจริญ",
    "central": "กำแพงเพชร ชัยนาท นครนายก นครปฐม นครสวรรค์ นนทบุรี ปทุมธานี พระนครศรีอยุธยา พิจิตร พิษณุโลก "
               "เพชรบูรณ์ ลพบุรี สมุทรปราการ สมุทรสงคราม สมุทรสาคร สิงห์บุรี สุโขทัย สุพรรณบุรี สระบุรี อ่างทอง อุทัยธานี",
    "east": "จันทบุรี ฉะเชิงเทรา ชลบุรี ตราด ปราจีนบุรี ระยอง สระแก้ว",
    "west": "กาญจนบุรี ตาก ประจวบคีรีขันธ์ เพชรบุรี ราชบุรี",
    "south": "กระบี่ ชุมพร ตรัง นครศรีธรรมราช นราธิวาส ปัตตานี พังงา พัทลุง ภูเก็ต ระนอง สตูล สงขลา สุราษฎร์ธานี ยะลา",
}
PROV2REG = {p: r for r, ps in REGIONS.items() for p in ps.split()}
assert len(PROV2REG) == 76  # + Bangkok = 77 provinces


def region(p):
    if not isinstance(p, str):
        return None
    p = p.replace(" ", "").removeprefix("จังหวัด")
    return "bangkok" if p.startswith("กรุงเทพ") else PROV2REG.get(p)


def worktype(t):
    t = t if isinstance(t, str) else ""
    return "works" if "ก่อสร้าง" in t else ("goods" if "ซื้อ" in t else "services")


# ---------------------------------------------------------------- per-year aggregates (DuckDB)
def build_year(con, y):
    con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
        SELECT {DEDUP[0]} agency, agency || '|' || coalesce(sub_agency, '') AS unit, {GOV_SQL} AS gov,
               sign_date, strftime(sign_date, '%Y-%m') AS ym, agreed_price, budget, ref_price, province, project_type,
               {METHOD_CLS} AS mcls,
               CASE WHEN vendor_name_key IS NULL THEN NULL ELSE vendor_name_key || '|' || coalesce(province, '') END AS vkey,
               regexp_matches(translate(replace(coalesce(project_name, ''), ' ', ''), '๐๑๒๓๔๕๖๗๘๙', '0123456789'),
                              '{STIM_RE}') AS stim
        FROM read_parquet('{src(y)}')
        WHERE sign_date >= DATE '{START}' AND sign_date < DATE '{ACT}' AND agreed_price > 0 AND agreed_price <= {MAX_PRICE}
        {DEDUP[1]}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cd AS
        SELECT *, {band('agreed_price')} AS band,
               coalesce(budget > 0 AND agreed_price / budget BETWEEN 0.3 AND 1.5, false) AS ok,
               CASE WHEN budget > 0 THEN 1 - agreed_price / budget END AS disc
        FROM c""")
    q = {
        # contract-level cells (O1, R4, R5, R6, first stage, B_u, months active)
        "cells_c": """SELECT unit, ym, band, stim, count(*)::INT n_all, sum((mcls = 'discretion')::INT)::INT n_disc,
                             sum(ok::INT)::INT n, sum(CASE WHEN ok THEN disc END) s1, sum(CASE WHEN ok THEN disc * disc END) s2,
                             sum(CASE WHEN budget > 0 THEN budget END) bsum
                      FROM cd GROUP BY ALL""",
        # purchase-level cells (O1-P, N_u)
        "cells_p": f"""WITH p AS (SELECT unit, coalesce(vkey, '#' || rowid::VARCHAR) pk, sign_date, min(ym) ym, bool_or(stim) stim,
                                       sum(agreed_price) v, sum(budget) b, bool_and(coalesce(budget, 0) > 0) bok
                                FROM c GROUP BY unit, pk, sign_date),
                            q AS (SELECT *, {band('v')} AS band, coalesce(bok AND v / b BETWEEN 0.3 AND 1.5, false) AS ok,
                                         CASE WHEN bok THEN 1 - v / b END AS disc FROM p)
                       SELECT unit, ym, band, stim, count(*)::INT n_all, sum(ok::INT)::INT n,
                              sum(CASE WHEN ok THEN disc END) s1, sum(CASE WHEN ok THEN disc * disc END) s2
                       FROM q GROUP BY ALL""",
        # O2 splitting flags at the binding threshold (T = 100k before Oct 2015, 500k after)
        "split": """WITH g AS (SELECT unit, vkey, sign_date, count(*) k, sum(agreed_price) v FROM c WHERE vkey IS NOT NULL GROUP BY ALL),
                         j AS (SELECT c.unit, c.ym, c.agreed_price, coalesce(g.k, 1) k, coalesce(g.v, c.agreed_price) v,
                                      CASE WHEN c.sign_date < DATE '2015-10-01' THEN 100000 ELSE 500000 END T
                               FROM c LEFT JOIN g ON c.unit = g.unit AND c.vkey = g.vkey AND c.sign_date = g.sign_date),
                         f AS (SELECT *, agreed_price > 0.9 * T AND agreed_price <= T AS near,
                                      agreed_price > 0.4 * T AND agreed_price <= 0.8 * T AS ctl, k >= 2 AND v > T AS sp FROM j)
                    SELECT unit, ym, sum(near::INT)::INT n_near, sum((near AND sp)::INT)::INT n_near_split,
                           sum(ctl::INT)::INT n_ctrl, sum((ctl AND sp)::INT)::INT n_ctrl_split
                    FROM f GROUP BY ALL""",
        # O3 (exploratory): treated-band value by vendor
        "vt": f"""SELECT unit, ym, vkey, sum(agreed_price) v FROM cd WHERE band = '{TREAT}' AND vkey IS NOT NULL GROUP BY ALL""",
        # covariates, Oct 2014 - Jul 2015
        "xu": f"""SELECT unit, any_value(agency) agency, any_value(gov) gov, count(*)::INT n_contracts, sum(agreed_price) val,
                         sum(CASE WHEN ok AND band = '{TREAT}' THEN disc END) s_t, sum((ok AND band = '{TREAT}')::INT)::INT n_t,
                         sum(CASE WHEN ok AND band = '{CTRL}' THEN disc END) s_c, sum((ok AND band = '{CTRL}')::INT)::INT n_c,
                         sum((ok AND agreed_price = budget)::INT)::INT n_zero, sum(ok::INT)::INT n_ok,
                         sum((band = '{TREAT}' AND mcls = 'discretion')::INT)::INT n_t_disc, sum((band = '{TREAT}')::INT)::INT n_t_all,
                         sum((agreed_price > 90000 AND agreed_price <= 100000)::INT)::INT n_b90,
                         sum((agreed_price > 40000 AND agreed_price <= 80000)::INT)::INT n_b40,
                         sum(CASE WHEN band = '{TREAT}' THEN agreed_price ELSE 0 END) v_t, sum((ref_price > 0)::INT)::INT n_ref
                  FROM cd WHERE sign_date < DATE '{X_END}' GROUP BY unit""",
        "xprov": f"SELECT unit, province, count(*)::INT n FROM c WHERE sign_date < DATE '{X_END}' GROUP BY ALL",
        "xtype": f"SELECT unit, project_type, count(*)::INT n FROM c WHERE sign_date < DATE '{X_END}' GROUP BY ALL",
        "xvend": f"""SELECT unit, vkey, count(*)::INT n, sum(agreed_price) v FROM c
                     WHERE sign_date < DATE '{X_END}' AND vkey IS NOT NULL GROUP BY ALL""",
    }
    for name, sql in q.items():
        con.execute(f"COPY ({sql}) TO '{(BUILD / f'{name}_{y}.parquet').as_posix()}' (FORMAT parquet)")


def load(name):
    return pd.concat([pd.read_parquet(BUILD / f"{name}_{y}.parquet") for y in YEARS], ignore_index=True)


# ---------------------------------------------------------------- unit-level effects (outcome side only)
def period(ym, pre, post):
    return np.select([(ym >= pre[0]) & (ym <= pre[1]), (ym >= post[0]) & (ym <= post[1])], ["pre", "post"], "")


def did(cells, ctrl, pre, post, m, fs=False):
    d = cells[cells.band.isin([TREAT, ctrl])].copy()
    d["per"] = period(d.ym, pre, post)
    d = d[d.per != ""]
    cols = ["n", "s1", "s2"] + (["n_all", "n_disc"] if fs else [])
    g = d.groupby(["unit", "band", "per"], as_index=False)[cols].sum()
    g["mean"] = g.s1 / g.n.where(g.n > 0)
    g["vm"] = np.where(g.n > 1, (g.s2 - g.n * g["mean"] ** 2) / (g.n - 1), np.nan) / g.n
    g["key"] = np.where(g.band == TREAT, "T", "C") + "_" + g.per
    w = g.pivot(index="unit", columns="key", values=["n", "mean", "vm"] + (["n_all", "n_disc"] if fs else []))
    keys = ["T_post", "T_pre", "C_post", "C_pre"]
    if not all(("n", k) in w.columns for k in keys):
        return pd.DataFrame()
    ok = np.logical_and.reduce([w[("n", k)].fillna(0) >= m for k in keys])
    w = w[ok]
    r = pd.DataFrame({"unit": w.index,
                      "tau": ((w[("mean", "T_post")] - w[("mean", "T_pre")]) - (w[("mean", "C_post")] - w[("mean", "C_pre")])).values,
                      "v": sum(w[("vm", k)].fillna(0) for k in keys).values})
    for k in keys:
        r[f"n_{k}"] = w[("n", k)].values.astype(int)
    if fs:
        r["fs"] = (w[("n_disc", "T_post")] / w[("n_all", "T_post")] - w[("n_disc", "T_pre")] / w[("n_all", "T_pre")]).values
    return r


def o2_effects(S, post, m):
    d = S.copy()
    d["per"] = period(d.ym, PRE, post)
    g = d[d.per != ""].groupby(["unit", "per"])[["n_near", "n_near_split", "n_ctrl", "n_ctrl_split"]].sum().unstack("per")
    need = [("n_near", "pre"), ("n_near", "post"), ("n_ctrl", "pre"), ("n_ctrl", "post")]
    if not all(k in g.columns for k in need):
        return pd.DataFrame()
    g = g[np.logical_and.reduce([g[k].fillna(0) >= m for k in need])]
    p = {(a, per): g[(f"{a}_split", per)] / g[(a, per)] for a in ("n_near", "n_ctrl") for per in ("pre", "post")}
    tau = (p[("n_near", "post")] - p[("n_ctrl", "post")]) - (p[("n_near", "pre")] - p[("n_ctrl", "pre")])
    v = sum(p[(a, per)] * (1 - p[(a, per)]) / g[(a, per)] for a in ("n_near", "n_ctrl") for per in ("pre", "post"))
    return pd.DataFrame({"unit": g.index, "tau": tau.values, "v": v.values})


def o3_effects(V, post):
    d = V.copy()
    d["per"] = period(d.ym, PRE, post)
    g = d[d.per != ""].groupby(["unit", "per", "vkey"]).v.sum()
    tot = g.groupby(["unit", "per"]).sum()
    top = (g.groupby(["unit", "per"]).max() / tot).unstack("per").dropna()
    return pd.DataFrame({"unit": top.index, "tau": (top["post"] - top["pre"]).values}) if len(top) else pd.DataFrame()


def mde(v, n):
    return 2.8 * np.sqrt(2 * np.mean(v) / (n / 5)) if n >= 5 else np.nan


# ---------------------------------------------------------------- covariates (pre-period only)
COVARIATES = ["local", "agency_cat", "region", "log_n_contracts", "log_value", "months_active",
              "disc_treated_pre", "disc_control_pre", "zero_discount_share", "discretion_share_treated",
              "bunching_index_100k", "split_rate_100k", "log_n_vendors", "vendor_hhi", "top_vendor_share",
              "repeat_vendor_share", "share_goods", "share_services", "share_works", "value_share_treated",
              "ref_price_share"]
DROPPED_BY_DEFINITION = {"egp_coverage": "item 22 (share of months with >= 1 contract) equals months_active / 10 — "
                                         "identical to item 6, dropped as a mechanical duplicate"}


def covariates(Cc, Cp, S):
    xu = load("xu").groupby("unit").agg(
        agency=("agency", "first"), gov=("gov", "first"),
        **{c: (c, "sum") for c in ["n_contracts", "val", "s_t", "n_t", "s_c", "n_c", "n_zero", "n_ok", "n_t_disc",
                                   "n_t_all", "n_b90", "n_b40", "v_t", "n_ref"]})
    X = pd.DataFrame(index=xu.index)
    X["agency"] = xu.agency
    X["local"] = (xu.gov == "local").astype(int)
    cnt = X.groupby("agency").size()
    X["agency_cat"] = np.where(X.agency.map(cnt) >= 30, X.agency, "other")
    pv = load("xprov").groupby(["unit", "province"], dropna=False).n.sum().reset_index()
    pv = pv.sort_values(["unit", "n", "province"], ascending=[True, False, True], na_position="last").drop_duplicates("unit")
    X["province_modal"] = pv.set_index("unit").province.reindex(X.index)
    X["region"] = X.province_modal.map(region)
    X["log_n_contracts"] = np.log(xu.n_contracts)
    X["log_value"] = np.log(xu.val)
    xm = Cc[Cc.ym <= PRE_R4[1]].groupby("unit").ym.nunique()
    X["months_active"] = xm.reindex(X.index).fillna(0)
    X["disc_treated_pre"] = xu.s_t / xu.n_t.where(xu.n_t > 0)
    X["disc_control_pre"] = xu.s_c / xu.n_c.where(xu.n_c > 0)
    X["zero_discount_share"] = xu.n_zero / xu.n_ok.where(xu.n_ok > 0)
    X["discretion_share_treated"] = xu.n_t_disc / xu.n_t_all.where(xu.n_t_all > 0)
    X["bunching_index_100k"] = xu.n_b90 / xu.n_b40.where(xu.n_b40 > 0)
    s = S[S.ym <= PRE_R4[1]].groupby("unit")[["n_near", "n_near_split", "n_ctrl", "n_ctrl_split"]].sum()
    X["split_rate_100k"] = (s.n_near_split / s.n_near.where(s.n_near > 0) - s.n_ctrl_split / s.n_ctrl.where(s.n_ctrl > 0)).reindex(X.index)
    vd = load("xvend").groupby(["unit", "vkey"])[["n", "v"]].sum().reset_index()
    vd["sh"] = vd.v / vd.groupby("unit").v.transform("sum")
    vg = vd.groupby("unit").agg(nv=("vkey", "size"), hhi=("sh", lambda x: float((x ** 2).sum())), top=("sh", "max"),
                                rep=("n", lambda x: float((x >= 3).mean())))
    X["log_n_vendors"] = np.log(vg.nv).reindex(X.index)
    X["vendor_hhi"] = vg.hhi.reindex(X.index)
    X["top_vendor_share"] = vg.top.reindex(X.index)
    X["repeat_vendor_share"] = vg.rep.reindex(X.index)
    tp = load("xtype")
    tp["wt"] = tp.project_type.map(worktype)
    wt = tp.groupby(["unit", "wt"]).n.sum().unstack(fill_value=0)
    for k in ("goods", "services", "works"):
        X[f"share_{k}"] = (wt.get(k, 0) / wt.sum(axis=1)).reindex(X.index)
    X["value_share_treated"] = xu.v_t / xu.val
    X["ref_price_share"] = xu.n_ref / xu.n_contracts
    # policy weights (pre period Oct 2014 - Sep 2015, per year): treated-band budget and treated-band purchases
    pre_c = Cc[(Cc.ym >= PRE[0]) & (Cc.ym <= PRE[1]) & (Cc.band == TREAT)].groupby("unit").bsum.sum()
    pre_p = Cp[(Cp.ym >= PRE[0]) & (Cp.ym <= PRE[1]) & (Cp.band == TREAT)].groupby("unit").n_all.sum()
    W = pd.DataFrame({"B_u": pre_c, "N_u": pre_p})
    X = X.join(W, how="outer")
    X.index.name = "unit"
    return X.reset_index(), tp


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    BUILD.mkdir(parents=True, exist_ok=True)
    con = None
    for y in YEARS:
        if a.rebuild or not (BUILD / f"xvend_{y}.parquet").exists():
            con = con or connect()
            t1 = time.time()
            build_year(con, y)
            log(f"p3_panel build {y}: {time.time() - t1:.0f}s")
    Cc, Cp, S, V = load("cells_c"), load("cells_p"), load("split"), load("vt")

    # outcome side ------------------------------------------------------------------------------------------
    specs = {"O1": (Cc, PRE, CTRL, True), "O1P": (Cp, PRE, CTRL, False), "R4": (Cc, PRE_R4, CTRL, True),
             "R5": (Cc[~Cc.stim], PRE, CTRL, True), "R6": (Cc, PRE, CTRL2, True)}
    taus = []
    for spec, (cells, pre, ctrl, fs) in specs.items():
        for w, post in WINDOWS.items():
            for m in (MS if spec != "R6" else (3,)):
                r = did(cells, ctrl, pre, post, m, fs)
                if len(r):
                    taus.append(r.assign(spec=spec, window=w, m=m))
    T = pd.concat(taus, ignore_index=True)
    T.to_parquet(OUT / "tau.parquet", index=False)
    O2 = pd.concat([o2_effects(S, post, m).assign(window=w, m=m) for w, post in WINDOWS.items() for m in MS], ignore_index=True)
    O2.to_parquet(OUT / "o2.parquet", index=False)
    O3 = pd.concat([o3_effects(V, post).assign(window=w) for w, post in WINDOWS.items()], ignore_index=True)
    O3.to_parquet(OUT / "o3.parquet", index=False)

    # covariate side ----------------------------------------------------------------------------------------
    X, tp = covariates(Cc, Cp, S)
    X.to_parquet(OUT / "X.parquet", index=False)

    # eligibility counts and mechanical drops (counts only) --------------------------------------------------
    loc = X.set_index("unit").local
    cnt = (T.assign(local=T.unit.map(loc)).groupby(["spec", "window", "m"])
           .agg(N=("unit", "size"), N_local=("local", "sum"), N_noX=("local", lambda s: int(s.isna().sum()))).reset_index())
    cnt["N_central"] = cnt.N - cnt.N_local - cnt.N_noX
    elig = {w: set(T[(T.spec == "O1") & (T.window == w) & (T.m == 3)].unit) for w in ("main", "alt")}
    Xi = X.set_index("unit")
    Xm = Xi.reindex(sorted(elig["main"]))
    drops = {}
    for c in COVARIATES:
        miss = float(Xm[c].isna().mean())
        const = Xm[c].dropna().nunique() <= 1
        if miss > 0.5 or const:
            drops[c] = f"missing {miss:.0%}" if miss > 0.5 else "constant"
    meta = {"covariates": COVARIATES, "dropped_by_definition": DROPPED_BY_DEFINITION, "mechanical_drops": drops,
            "drop_rule": "computed on O1 main-window m=3 eligible units: > 50% missing or constant",
            "policy_weights": {"B_u": "treated-band (contract level, 100k-500k) budget, Oct 2014 - Sep 2015",
                               "N_u": "treated-band purchases (unit x vendor x day, banded by purchase total), Oct 2014 - Sep 2015"},
            "x_window": "contracts signed 1 Oct 2014 - 31 Jul 2015", "stimulus_regex": STIM_RE,
            "worktype_map": "'ก่อสร้าง' -> works; else 'ซื้อ' -> goods; else services",
            "region_source": "modal project province (จังหวัด field) of the unit's contracts in the X window; 6 regions + Bangkok"}
    (OUT / "x_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # selection table: covariate means, all units vs eligible (no outcome values)
    num = [c for c in COVARIATES if c not in ("agency_cat", "region")]
    sel = pd.DataFrame({"all units (X window)": Xi[num].mean(),
                        "eligible main m=3": Xi.reindex(sorted(elig["main"]))[num].mean(),
                        "eligible alt m=3": Xi.reindex(sorted(elig["alt"]))[num].mean()})
    sel.loc["N"] = [Xi.local.notna().sum(), len(elig["main"]), len(elig["alt"])]
    reg = pd.DataFrame({k: Xi.reindex(sorted(s)).region.fillna("unmapped").value_counts(normalize=True)
                        for k, s in (("all units (X window)", set(Xi.index)), ("eligible main m=3", elig["main"]),
                                     ("eligible alt m=3", elig["alt"]))}).fillna(0)

    # O2 power gate (noise only)
    g = O2[(O2.window == "main") & (O2.m == 3)]
    o2_mde = mde(g.v.values, len(g)) if len(g) else np.nan
    o2_pass = bool(len(g) >= N_MIN and o2_mde <= O2_MDE_MAX)

    # build checks
    types = tp.groupby("project_type", dropna=False).n.sum().sort_values(ascending=False).head(15).reset_index()
    types["class"] = types.project_type.map(worktype)
    stim = Cc[Cc.stim].groupby("ym").n_all.sum()
    stim_tab = pd.DataFrame({"stimulus-matched contracts": stim,
                             "share of all": stim / Cc.groupby("ym").n_all.sum().reindex(stim.index)}).reset_index()
    prov_rate = float(Xi.region.notna().mean())
    unm = Xi[Xi.region.isna()].province_modal.fillna("(missing)").value_counts().head(10)

    res = {"o2_gate": {"N": int(len(g)), "MDE": o2_mde, "MDE_max": O2_MDE_MAX, "pass": o2_pass},
           "counts": cnt.to_dict("records"), "mechanical_drops": drops, "province_mapped_share": prov_rate,
           "seconds": round(time.time() - t0)}
    (REPORTS / "p3_panel.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    L = ["# Paper 3 panel build (plan: project doc `claude/p3-pre-analysis-plan.md`, locked 19 Oct 2026)", "",
         "Outcome pseudo-outcomes and covariates are written to separate files; nothing here relates them.", "",
         f"## O2 power gate (noise only) — **{'pass' if o2_pass else 'fail → O2 at group level only, H5 dropped'}**",
         f"Main window, m = 3: N = {len(g):,}, MDE = {o2_mde * 100:.2f} pp; rule N ≥ {N_MIN:,} and MDE ≤ "
         f"{O2_MDE_MAX * 100:.1f} pp (half of paper 1's split-share change, {O2_REF * 100:.1f} pp).", "",
         "## Eligible units per specification", "", cnt.to_markdown(index=False), "",
         "## Covariates", f"Mechanical drops (rule: > 50% missing or constant among main-window eligible units): "
         f"{drops if drops else 'none'}. By definition: {DROPPED_BY_DEFINITION['egp_coverage']}.", "",
         "### Selection table (covariate means)", "", sel.to_markdown(floatfmt=".3f"), "",
         "### Region shares", "", reg.to_markdown(floatfmt=".3f"), "",
         "## Build checks", f"Province → region mapped for {prov_rate:.1%} of units. Most frequent unmapped values:", "",
         unm.to_frame("units").to_markdown(), "", "Work-type mapping (top 15 project types, all X-window contracts):", "",
         types.to_markdown(index=False), "", f"Stimulus keyword matches by month (regex `{STIM_RE}`):", "",
         stim_tab.to_markdown(index=False, floatfmt=".4f"), "", f"Run time {res['seconds']} s."]
    (REPORTS / "p3_panel.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:8]), flush=True)
    print(cnt[(cnt.m == 3)].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
