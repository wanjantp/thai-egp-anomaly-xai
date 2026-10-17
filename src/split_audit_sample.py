#!/usr/bin/env python3
"""
Validation sample of same-day split clusters (statistical review item 7, GIQ v7).

Draws a random sample of split clusters at THB 500,000 (one buying unit, one vendor, one signing day, two or more
contracts of at most THB 500,000 summing to THB 0.5–2 million) from fiscal 2015 (pre-reform) and from post-Act years,
and writes the contracts of each sampled cluster for manual coding from the project names:
  A  one purchase divided: the same item or work in lots, sections, sites or periods
  B  separate needs: different items or works that happen to share a vendor and a day
  C  cannot tell from the names
Vendor names are not written out (personal data under the PDPA); clusters are identified by a sequence number.

Usage (Anaconda Prompt, env egp-xai):  python src\\split_audit_sample.py
Output: reports/split_audit_sample.csv (one row per contract; add a `code` column per cluster when coding).
"""
import sys

import pandas as pd

from bunching import REGIME_SQL, REPORTS, connect, log
from revision_checks import src

SAMPLE = {2558: 100, 2561: 25, 2563: 25, 2566: 25, 2568: 25}   # pre-reform year, post-Act years
SEED = 0.2026


def main():
    con = connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SELECT setseed({SEED})")
    out = []
    for y, k in SAMPLE.items():
        regime = "pre" if y == 2558 else "act"
        d = con.execute(f"""
            WITH c AS (SELECT project_id, project_name, project_type, method, agreed_price, budget, sign_date,
                              agency || '|' || coalesce(sub_agency, '') AS unit, agency,
                              vendor_name_key || '|' || coalesce(province, '') AS vkey
                       FROM read_parquet('{src(y)}')
                       WHERE sign_date IS NOT NULL AND agreed_price > 0 AND agreed_price <= 3000000
                         AND vendor_name_key IS NOT NULL AND {REGIME_SQL} = '{regime}'),
                 g AS (SELECT unit, vkey, sign_date FROM c GROUP BY ALL
                       HAVING count(*) >= 2 AND max(agreed_price) <= 500000
                          AND sum(agreed_price) > 500000 AND sum(agreed_price) <= 2000000),
                 s AS (SELECT *, row_number() OVER () AS cl FROM (SELECT * FROM g ORDER BY random() LIMIT {k}))
            SELECT {y} AS fy, s.cl, c.agency, c.sign_date, c.project_id, c.project_name, c.project_type, c.method,
                   c.agreed_price, c.budget
            FROM s JOIN c USING (unit, vkey, sign_date) ORDER BY s.cl, c.agreed_price DESC""").df()
        out.append(d)
        log(f"{y}: {d.cl.nunique()} clusters, {len(d)} contracts")
    D = pd.concat(out, ignore_index=True)
    D.insert(0, "cluster", D.fy.astype(str) + "-" + D.cl.astype(str).str.zfill(3))
    D.drop(columns="cl").to_csv(REPORTS / "split_audit_sample.csv", index=False, encoding="utf-8-sig")
    log(f"→ {REPORTS / 'split_audit_sample.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
