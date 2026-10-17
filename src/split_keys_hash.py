#!/usr/bin/env python3
"""
How much splitting does the same-vendor definition miss? (GIQ manuscript, v3 follow-up to Caires et al., 2025)

The main measure groups contracts by buying unit × vendor × day. Caires et al. (2025) find that Portuguese buyers
spread split contracts across sellers, so a same-vendor measure is a lower bound. This script recomputes the split
share and the excess split purchases under wider keys that ignore the vendor:

  vendor_day    unit × vendor × signing day                (main measure)
  vendor_month  unit × vendor × signing month              (robustness measure in the paper)
  kind_day      unit × type of work × signing day          (type of work from the project name: works, goods,
                                                             services, medical, food, other)
  unit_day      unit × signing day                         (widest: any purchases by one unit on one day)

For each key: split share at THB 500,000 (clusters with totals in (0.5M, 2M], ≥2 contracts, each ≤ 500k), at the
abolished THB 100,000 line (totals in (0.1M, 0.5M], each ≤ 100k) and at the THB 1 million placebo (totals in (1M, 4M],
≥2 contracts, each in (0.5M, 1M]); excess split purchases at 500k in fiscal 2018–2025 over the pre-reform rate; their
ratio to the contracts missing above the threshold (THB 2–3 million anchor); and, for the vendor-free keys, the share
of split clusters with two or more vendors.

Usage (Anaconda Prompt, env egp-xai):  python src\\split_keys.py
Output: reports/split_keys.md (+ .csv). Cache: data/processed/_splitkeys/. Reads data/interim only.
Unit and vendor keys are 64-bit hashes of the text keys to keep memory low.
"""
import sys
import time

import pandas as pd

from bunching import INTERIM, KIND_SQL, MAX_PRICE, PROCESSED, REGIME_SQL, REPORTS, connect, log

OUT_DIR = PROCESSED / "_splitkeys"
DEDUP = "DISTINCT ON (project_id, contract_no, winner_name, agreed_price, sign_date)"
KEYS = {"vendor_day": "unit, vkey, sign_date", "vendor_month": "unit, vkey, date_trunc('month', sign_date)",
        "kind_day": "unit, kind, sign_date", "unit_day": "unit, sign_date"}


def years():
    return sorted(int(p.name.split("_")[1]) for p in INTERIM.glob("contracts_25??") if p.is_dir())


def aggregate(con, y):
    f = OUT_DIR / f"split_keys_{y}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    t0 = time.time()
    src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
    con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
        SELECT {DEDUP} sign_date, agreed_price, hash(agency || '|' || coalesce(sub_agency, '')) AS unit,
               hash(vendor_name_key || '|' || coalesce(province, '')) AS vkey, {KIND_SQL} AS kind, {REGIME_SQL} AS regime
        FROM read_parquet('{src}')
        WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= {MAX_PRICE} AND vendor_name_key IS NOT NULL""")
    parts = []
    for key, cols in KEYS.items():
        parts.append(con.execute(f"""
            WITH g AS (SELECT regime, count(*) k, sum(agreed_price) v, max(agreed_price) mx, min(agreed_price) mn,
                              count(DISTINCT vkey) nv
                       FROM c GROUP BY regime, {cols})
            SELECT '{key}' AS key, regime,
                   sum(CASE WHEN v > 500000 AND v <= 2000000 THEN 1 ELSE 0 END)::BIGINT AS cl500,
                   sum(CASE WHEN v > 500000 AND v <= 2000000 AND k >= 2 AND mx <= 500000 THEN 1 ELSE 0 END)::BIGINT AS sp500,
                   sum(CASE WHEN v > 500000 AND v <= 2000000 AND k >= 2 AND mx <= 500000 AND nv >= 2 THEN 1 ELSE 0 END)::BIGINT AS sp500_multi,
                   sum(CASE WHEN v > 100000 AND v <= 500000 THEN 1 ELSE 0 END)::BIGINT AS cl100,
                   sum(CASE WHEN v > 100000 AND v <= 500000 AND k >= 2 AND mx <= 100000 THEN 1 ELSE 0 END)::BIGINT AS sp100,
                   sum(CASE WHEN v > 1000000 AND v <= 4000000 THEN 1 ELSE 0 END)::BIGINT AS cl1m,
                   sum(CASE WHEN v > 1000000 AND v <= 4000000 AND k >= 2 AND mx <= 1000000 AND mn > 500000 THEN 1 ELSE 0 END)::BIGINT AS sp1m
            FROM g GROUP BY ALL""").df())
    parts.append(con.execute("""SELECT 'contracts' AS key, regime,
            sum(CASE WHEN agreed_price > 500000 AND agreed_price <= 2000000 THEN 1 ELSE 0 END)::BIGINT AS n0520,
            sum(CASE WHEN agreed_price > 2000000 AND agreed_price <= 3000000 THEN 1 ELSE 0 END)::BIGINT AS n23
            FROM c GROUP BY ALL""").df())
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "fy", y)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f, index=False)
    log(f"FY{y}: {time.time() - t0:.0f}s")
    return out


def main():
    con = connect()
    con.execute("SET preserve_insertion_order = false")
    A = pd.concat([aggregate(con, y) for y in years()], ignore_index=True).fillna(0)
    pre = A.regime == "pre"
    act = (A.regime == "act") & A.fy.between(2561, 2568)
    trans = A.regime == "transition"
    C = A[A.key == "contracts"]
    cp, ca = C[pre.loc[C.index]], C[act.loc[C.index]]
    missing = (cp.n0520.sum() / cp.n23.sum()) * ca.n23.sum() - ca.n0520.sum()
    rows = []
    for key in KEYS:
        K = A[A.key == key]
        r = {"key": key}
        for lab, m in (("pre", pre), ("transition", trans), ("act", act)):
            s = K[m.loc[K.index]].sum(numeric_only=True)
            r[f"split500_{lab}"] = s.sp500 / s.cl500
            r[f"split100_{lab}"] = s.sp100 / s.cl100
            r[f"split1m_{lab}"] = s.sp1m / s.cl1m
        sp, sa = K[pre.loc[K.index]].sum(numeric_only=True), K[act.loc[K.index]].sum(numeric_only=True)
        r["excess500_2018_25"] = sa.sp500 - (sp.sp500 / sp.cl500) * sa.cl500
        r["explained"] = r["excess500_2018_25"] / missing
        r["multi_vendor_share_act"] = sa.sp500_multi / sa.sp500 if sa.sp500 else float("nan")
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(REPORTS / "split_keys.csv", index=False)
    L = ["# Split purchases under keys with and without the vendor", "",
         "pre = signed before the reform regime change; act = signed from 23 Aug 2017, fiscal 2018–2025 (B.E. 2561–2568).",
         f"Contracts missing above THB 500,000, fiscal 2018–2025 (THB 2–3 million anchor, pre-reform ratio): {missing:,.0f}.", "",
         "| key | 500k pre | 500k transition | 500k act | 100k pre | 100k act | 1M placebo pre | 1M placebo act | excess split 500k, 2018–25 | share of missing explained | split clusters with ≥2 vendors (act) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        L.append(f"| {r.key} | {r.split500_pre:.1%} | {r.split500_transition:.1%} | {r.split500_act:.1%} | {r.split100_pre:.1%} | "
                 f"{r.split100_act:.1%} | {r.split1m_pre:.1%} | {r.split1m_act:.1%} | {r.excess500_2018_25:,.0f} | {r.explained:.0%} | "
                 f"{r.multi_vendor_share_act:.1%} |")
    L += ["", "Wider keys also catch unrelated purchases made on the same day; the excess over the pre-reform rate nets out the",
          "part that is constant over time. Placebo movement at THB 1 million under a wide key would signal that the key picks up",
          "general changes in purchasing rather than threshold-driven splitting."]
    out = REPORTS / "split_keys.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
