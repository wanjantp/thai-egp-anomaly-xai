#!/usr/bin/env python3
"""
Robustness checks requested by the Reviewer-2 audit of the GIQ manuscript (16 Oct 2026).

  did     M3 + M7 : DiD on discounts — extensive margin (share at or above budget), intensive margin
                    (discount among contracts below budget), balanced panel of buying units
  anchor  M1      : contracts missing above THB 500k anchored on 2–3M, 3–5M and 5–10M bands,
                    with and without specific-method contracts above 500k (legal exceptions)
  boot    M5      : agency-clustered Poisson bootstrap (500 draws) for split shares, missing contracts,
                    the share of missing contracts explained by excess splitting, and the alert share
  vendor  M6      : FY2568 — split share using full 13-digit tax IDs vs the normalized-name vendor key

Usage (Anaconda Prompt, env egp-xai):
    python src\\revision_checks.py            # all steps (each step caches its aggregate; rerun resumes)
    python src\\revision_checks.py boot       # one step
Output: reports/revision_checks.md (+ reports/revision_*.csv). Reads data/interim and data/processed only.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bunching import (DID_DIR, GOV_SQL, INTERIM, MAX_PRICE, METHOD_CLS, PROCESSED, REGIME_SQL, REPORTS,
                      UNIT_BANDS, _twfe, connect, log)

OUT_DIR = PROCESSED / "_revision"
REPORT = REPORTS / "revision_checks.md"
SECTIONS = {}
DEDUP = "DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)"


def years():
    return sorted(int(p.name.split("_")[1]) for p in INTERIM.glob("contracts_25??") if p.is_dir())


def src(y):
    return (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()


# ---------------------------------------------------------------- M3 + M7: DiD variants
def did():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()
    band = """CASE WHEN agreed_price <= 30000 THEN '0-30k' WHEN agreed_price <= 100000 THEN '30k-100k'
                   WHEN agreed_price <= 500000 THEN '100k-500k' WHEN agreed_price <= 2000000 THEN '500k-2M' ELSE '2M-3M' END"""
    parts = []
    for y in range(2558, 2562):
        f = OUT_DIR / f"did_int_{y}.parquet"
        if not f.exists():
            t0 = time.time()
            con.execute(f"""COPY (
                WITH c AS (SELECT {DEDUP} agency || '|' || coalesce(sub_agency, '') AS unit,
                                  strftime(sign_date, '%Y-%m') AS ym, agreed_price, budget, {METHOD_CLS} AS mcls
                           FROM read_parquet('{src(y)}')
                           WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE}),
                     d AS (SELECT *, {band} AS band, budget > 0 AND agreed_price / budget BETWEEN 0.3 AND 1.5 AS ok FROM c)
                SELECT unit, ym, band, count(*)::INT n,
                       sum(CASE WHEN mcls = 'discretion' THEN 1 ELSE 0 END)::INT n_disc,
                       sum(CASE WHEN ok THEN 1 ELSE 0 END)::INT n_b,
                       sum(CASE WHEN ok THEN 1 - agreed_price / budget END) sd_b,
                       sum(CASE WHEN ok AND agreed_price >= budget THEN 1 ELSE 0 END)::INT n_ge,
                       sum(CASE WHEN ok AND agreed_price < budget THEN 1 ELSE 0 END)::INT n_lt,
                       sum(CASE WHEN ok AND agreed_price < budget THEN 1 - agreed_price / budget END) sd_lt
                FROM d GROUP BY ALL) TO '{f.as_posix()}' (FORMAT parquet)""")
            log(f"did_int {y}: {time.time() - t0:.0f}s")
        parts.append(pd.read_parquet(f))
    D = pd.concat(parts, ignore_index=True)
    D = D[(D.ym >= "2014-10") & (D.ym <= "2017-08")].copy()
    D["post"] = (D.ym >= "2015-10").astype(float)
    bal = None
    if UNIT_BANDS.exists():
        U = pd.read_parquet(UNIT_BANDS, columns=["fiscal_year", "agency", "sub_agency"])
        U["unit"] = U["agency"] + "|" + U["sub_agency"].fillna("")
        pres = U.groupby("unit")["fiscal_year"].nunique()
        bal = set(pres[pres == pres.max()].index)
    rows = []
    for sample in ("all", "balanced"):
        S = D if sample == "all" else (D[D.unit.isin(bal)] if bal else None)
        if S is None:
            continue
        for ctrl in ("30k-100k", "2M-3M"):
            d = S[S.band.isin(["100k-500k", ctrl])].copy()
            d["treat"] = (d.band == "100k-500k").astype(float)
            d["tp"] = d.treat * d.post
            d["ub"] = d.unit + "|" + d.band
            out = {"sample": sample, "control": ctrl, "units": d.unit.nunique()}
            specs = (("first_stage", "n", lambda x: x.n_disc / x.n),
                     ("discount_vs_budget", "n_b", lambda x: x.sd_b / x.n_b),
                     ("share_at_or_above_budget", "n_b", lambda x: x.n_ge / x.n_b),
                     ("discount_if_below_budget", "n_lt", lambda x: x.sd_lt / x.n_lt))
            for name, w, fn in specs:
                dd = d[d[w] > 0].copy()
                dd["yy"] = fn(dd)
                b, se = _twfe(dd, "yy", w, ["tp"], "ub", "ym", "unit")
                out[name], out[name + "_se"] = b[0], se[0]
            out["wald_discount"] = out["discount_vs_budget"] / out["first_stage"]
            rows.append(out)
    R = pd.DataFrame(rows)
    R.to_csv(REPORTS / "revision_did.csv", index=False)
    L = ["## M3 + M7 — DiD on discounts: margins and balanced panel", "",
         "Treated = THB 0.1–0.5M; Oct 2014 – Aug 2017; unit×band and month FE; weights = contracts; SE clustered by unit.",
         "Extensive margin = share of contracts at or above budget; intensive margin = discount among contracts below budget.", "",
         "| sample | control | units | first stage | discount vs budget | Wald | share ≥ budget | discount if below budget |",
         "|---|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        L.append(f"| {r.sample} | {r.control} | {r.units:,} | {r.first_stage:+.3f} ({r.first_stage_se:.3f}) | "
                 f"{r.discount_vs_budget:+.2%} ({r.discount_vs_budget_se:.2%}) | {r.wald_discount:+.2%} | "
                 f"{r.share_at_or_above_budget:+.3f} ({r.share_at_or_above_budget_se:.3f}) | "
                 f"{r.discount_if_below_budget:+.2%} ({r.discount_if_below_budget_se:.2%}) |")
    if bal:
        L.append(f"\nBalanced panel = units with contracts in every fiscal year ({len(bal):,} units).")
    SECTIONS["did"] = L
    log("did done")


# ---------------------------------------------------------------- M1: alternative anchors for missing contracts
def anchor():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()
    f = OUT_DIR / "anchor_bands.parquet"
    if not f.exists():
        parts = []
        for y in years():
            t0 = time.time()
            parts.append(con.execute(f"""
                WITH c AS (SELECT {DEDUP} agreed_price, method FROM read_parquet('{src(y)}')
                           WHERE sign_date IS NOT NULL AND agreed_price > 500000 AND agreed_price <= 10000000)
                SELECT {y} AS fy,
                       CASE WHEN agreed_price <= 2000000 THEN '0.5-2M' WHEN agreed_price <= 3000000 THEN '2-3M'
                            WHEN agreed_price <= 5000000 THEN '3-5M' ELSE '5-10M' END AS band,
                       method LIKE '%เฉพาะเจาะจง%' AS specific, count(*)::INT AS n
                FROM c GROUP BY ALL""").df())
            log(f"anchor {y}: {time.time() - t0:.0f}s")
        pd.concat(parts, ignore_index=True).to_parquet(f, index=False)
    A = pd.read_parquet(f)
    ex = _excess_split()
    rows = []
    for variant, S in (("all methods", A), ("excluding specific method", A[~A.specific])):
        N = S.groupby(["fy", "band"])["n"].sum().unstack().fillna(0)
        for anc in ("2-3M", "3-5M", "5-10M"):
            r15 = N.loc[2558, "0.5-2M"] / N.loc[2558, anc]
            miss = r15 * N[anc] - N["0.5-2M"]
            yy = [y for y in N.index if 2561 <= y <= 2568]
            m = miss[yy].sum()
            rows.append({"variant": variant, "anchor": anc, "missing_2018_25": m,
                         "missing_share_2025": miss[2568] / (r15 * N.loc[2568, anc]),
                         "explained_day": ex["day"] / m if ex else np.nan, "explained_month": ex["month"] / m if ex else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(REPORTS / "revision_anchor.csv", index=False)
    L = ["## M1 — Contracts missing above THB 500,000 under alternative anchor bands", "",
         "missing_t = (N_0.5–2M / N_anchor in FY2558) × N_anchor,t − N_0.5–2M,t; summed over FY2561–2568 (fiscal 2018–2025).",
         f"Excess split clusters 2018–2025 (from _consolidation): same day {ex.get('day', float('nan')):,.0f}, same month {ex.get('month', float('nan')):,.0f}.", "",
         "| contracts | anchor | missing 2018–25 | missing share 2025 | explained (day) | explained (month) |", "|---|---|---|---|---|---|"]
    for r in R.itertuples():
        L.append(f"| {r.variant} | {r.anchor} | {r.missing_2018_25:,.0f} | {r.missing_share_2025:.0%} | {r.explained_day:.0%} | {r.explained_month:.0%} |")
    SECTIONS["anchor"] = L
    log("anchor done")


def _excess_split():
    d = PROCESSED / "_consolidation"
    if not d.exists():
        return {}
    C = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))], ignore_index=True)
    out = {}
    for lev in ("day", "month"):
        S = C[(C.level == lev) & (C.tband == "500k-2M")].groupby("fiscal_year")[["n_cl", "n_split500"]].sum()
        e = S.n_split500 - S.loc[2558, "n_split500"] / S.loc[2558, "n_cl"] * S.n_cl
        out[lev] = e[[y for y in e.index if 2561 <= y <= 2568]].sum()
    return out


# ---------------------------------------------------------------- M5: agency-clustered bootstrap
def boot(R=500, seed=2026):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()
    for y in years():
        f = OUT_DIR / f"boot_{y}.parquet"
        if f.exists():
            continue
        t0 = time.time()
        con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
            SELECT {DEDUP} sign_date, agreed_price, agency || '|' || coalesce(sub_agency, '') AS unit,
                   vendor_name_key || '|' || coalesce(province, '') AS vkey, {REGIME_SQL} AS regime
            FROM read_parquet('{src(y)}')
            WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE} AND vendor_name_key IS NOT NULL""")
        parts = []
        for level, key in (("day", "sign_date"), ("month", "date_trunc('month', sign_date)")):
            parts.append(con.execute(f"""
                WITH g AS (SELECT unit, regime, count(*) k, sum(agreed_price) v, max(agreed_price) mx
                           FROM c GROUP BY unit, vkey, regime, {key})
                SELECT unit, regime, '{level}' AS level,
                       sum(CASE WHEN v > 500000 AND v <= 2000000 THEN 1 ELSE 0 END)::INT AS cl500,
                       sum(CASE WHEN v > 500000 AND v <= 2000000 AND k >= 2 AND mx <= 500000 THEN 1 ELSE 0 END)::INT AS sp500,
                       sum(CASE WHEN v > 100000 AND v <= 500000 THEN 1 ELSE 0 END)::INT AS cl100,
                       sum(CASE WHEN v > 100000 AND v <= 500000 AND k >= 2 AND mx <= 100000 THEN 1 ELSE 0 END)::INT AS sp100
                FROM g GROUP BY ALL""").df())
        b = pd.concat(parts, ignore_index=True)
        n = con.execute("""SELECT unit, regime,
                sum(CASE WHEN agreed_price > 500000 AND agreed_price <= 2000000 THEN 1 ELSE 0 END)::INT AS n0520,
                sum(CASE WHEN agreed_price > 2000000 AND agreed_price <= 3000000 THEN 1 ELSE 0 END)::INT AS n23
                FROM c GROUP BY ALL""").df()
        n["level"] = "contracts"
        out = pd.concat([b, n], ignore_index=True)
        out.insert(0, "fy", y)
        out.to_parquet(f, index=False)
        log(f"boot aggregate {y}: {time.time() - t0:.0f}s")
    B = pd.concat([pd.read_parquet(f) for f in sorted(OUT_DIR.glob("boot_25??.parquet"))], ignore_index=True).fillna(0)
    uid, units = pd.factorize(B.unit)
    B["uid"] = uid
    nU = len(units)
    act = (B.regime == "act") & B.fy.between(2561, 2568)

    def mat(mask, col):
        s = B.loc[mask].groupby("uid")[col].sum()
        v = np.zeros(nU); v[s.index.values] = s.values
        return v

    V = {}
    for lev in ("day", "month"):
        m = B.level == lev
        V[f"pre_cl500_{lev}"], V[f"pre_sp500_{lev}"] = mat(m & (B.regime == "pre"), "cl500"), mat(m & (B.regime == "pre"), "sp500")
        V[f"act_cl500_{lev}"], V[f"act_sp500_{lev}"] = mat(m & act, "cl500"), mat(m & act, "sp500")
        V[f"pre_cl100_{lev}"], V[f"pre_sp100_{lev}"] = mat(m & (B.regime == "pre"), "cl100"), mat(m & (B.regime == "pre"), "sp100")
        V[f"act_cl100_{lev}"], V[f"act_sp100_{lev}"] = mat(m & act, "cl100"), mat(m & act, "sp100")
    mc = B.level == "contracts"
    V["pre_n0520"], V["pre_n23"] = mat(mc & (B.regime == "pre"), "n0520"), mat(mc & (B.regime == "pre"), "n23")
    V["act_n0520"], V["act_n23"] = mat(mc & act, "n0520"), mat(mc & act, "n23")

    def stats(w):
        s = {k: (w * v).sum() for k, v in V.items()}
        o = {}
        for lev in ("day", "month"):
            r_pre = s[f"pre_sp500_{lev}"] / s[f"pre_cl500_{lev}"]
            o[f"split500_pre_{lev}"] = r_pre
            o[f"split500_act_{lev}"] = s[f"act_sp500_{lev}"] / s[f"act_cl500_{lev}"]
            o[f"split100_pre_{lev}"] = s[f"pre_sp100_{lev}"] / s[f"pre_cl100_{lev}"]
            o[f"split100_act_{lev}"] = s[f"act_sp100_{lev}"] / s[f"act_cl100_{lev}"]
            o[f"excess_{lev}"] = s[f"act_sp500_{lev}"] - r_pre * s[f"act_cl500_{lev}"]
            o[f"alert_excess_share_{lev}"] = o[f"excess_{lev}"] / s[f"act_sp500_{lev}"]
        o["missing"] = s["pre_n0520"] / s["pre_n23"] * s["act_n23"] - s["act_n0520"]
        for lev in ("day", "month"):
            o[f"explained_{lev}"] = o[f"excess_{lev}"] / o["missing"]
        return o

    point = stats(np.ones(nU))
    rng = np.random.default_rng(seed)
    draws = [stats(rng.poisson(1.0, nU).astype(float)) for _ in range(R)]
    Dr = pd.DataFrame(draws)
    rows = [{"stat": k, "estimate": v, "se": Dr[k].std(ddof=1), "ci_lo": Dr[k].quantile(.025), "ci_hi": Dr[k].quantile(.975)}
            for k, v in point.items()]
    T = pd.DataFrame(rows)
    T.to_csv(REPORTS / "revision_boot.csv", index=False)
    L = [f"## M5 — Agency-clustered Poisson bootstrap ({R} draws, {nU:,} buying units)", "",
         "pre = signed before Oct 2015; act = signed from 23 Aug 2017 in FY2561–2568 (fiscal 2018–2025). Splits use clusters with totals in (0.5M, 2M] (500k) and (0.1M, 0.5M] (100k).",
         "Missing contracts here are anchored on 2–3M with the pre-reform ratio, pooled over 2018–2025 (consistent with the bunching tables).", "",
         "| statistic | estimate | SE | 95% CI |", "|---|---|---|---|"]
    for r in T.itertuples():
        big = abs(r.estimate) > 10
        fmt = (lambda x: f"{x:,.0f}") if big else (lambda x: f"{x:.3f}")
        L.append(f"| {r.stat} | {fmt(r.estimate)} | {fmt(r.se)} | [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}] |")
    SECTIONS["boot"] = L
    log("boot done")


# ---------------------------------------------------------------- M6: tax ID vs name key (FY2568)
def vendor(y=2568):
    con = connect()
    cols = set(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{src(y)}')").df()["column_name"])
    if not {"tax_id", "tax_id_kind", "vendor_name_key"} <= cols:
        SECTIONS["vendor"] = ["## M6 — skipped: tax_id / tax_id_kind / vendor_name_key not found in contracts parquet"]
        return
    con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
        SELECT {DEDUP} sign_date, agreed_price, agency || '|' || coalesce(sub_agency, '') AS unit,
               tax_id, vendor_name_key || '|' || coalesce(province, '') AS nkey
        FROM read_parquet('{src(y)}')
        WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE}
          AND tax_id_kind = 'full' AND vendor_name_key IS NOT NULL""")
    rows = []
    for keyname, key in (("tax ID", "tax_id"), ("name key", "nkey")):
        for level, dk in (("day", "sign_date"), ("month", "date_trunc('month', sign_date)")):
            r = con.execute(f"""WITH g AS (SELECT count(*) k, sum(agreed_price) v, max(agreed_price) mx
                                          FROM c GROUP BY unit, {key}, {dk})
                SELECT sum(CASE WHEN v > 500000 AND v <= 2000000 THEN 1 ELSE 0 END) AS cl,
                       sum(CASE WHEN v > 500000 AND v <= 2000000 AND k >= 2 AND mx <= 500000 THEN 1 ELSE 0 END) AS sp
                FROM g""").fetchone()
            rows.append({"key": keyname, "level": level, "clusters": r[0], "split": r[1], "share": (r[1] / r[0]) if r[0] else float("nan")})
    R = pd.DataFrame(rows)
    R.to_csv(REPORTS / "revision_vendor.csv", index=False)
    L = [f"## M6 — Vendor key check, FY{y} contracts with a full 13-digit tax ID", "",
         "| vendor key | window | clusters 0.5–2M | split at 500k | share |", "|---|---|---|---|---|"]
    L += [f"| {r.key} | {r.level} | {r.clusters:,.0f} | {r.split:,.0f} | {r.share:.1%} |" for r in R.itertuples()]
    SECTIONS["vendor"] = L
    log("vendor done")


def write():
    L = ["# Revision checks (Reviewer-2 audit, 16 Oct 2026)", ""]
    for k in ("did", "anchor", "boot", "vendor"):
        if k in SECTIONS:
            L += SECTIONS[k] + [""]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    log(f"→ {REPORT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    steps = sys.argv[1:] or ["did", "anchor", "boot", "vendor"]
    for s in steps:
        t0 = time.time()
        {"did": did, "anchor": anchor, "boot": boot, "vendor": vendor}[s]()
        log(f"{s}: {time.time() - t0:.0f}s")
        write()   # write after every step so a later failure keeps earlier results
