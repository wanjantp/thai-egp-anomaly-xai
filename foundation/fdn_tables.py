"""Foundation tables shared by papers 3-9, built once from data/interim and versioned.

Output folder (default $EGP/data/shared/foundation_v1):
  project_names.parquet  name_id, name, n, y0, y1          distinct normalised project names (papers 4, 5, 6)
  vendor_names.parquet   vendor_id, name, n, n_keys, ...   distinct normalised vendor names (paper 8)
  edges/fy=YYYY.parquet  one row per contract: buyer unit, vendor entity, date, method, prices, name_id
                         (papers 3, 6, 9; same unit_key/entity_key definitions as src/features.py)
  panel_unit_year.parquet, series_agency_month.parquet     (papers 3, 9)
  manifest_<step>.json   rows, years, git commit, time

Steps (each skips its output if it already exists; delete the file to rebuild):
  python foundation/fdn_tables.py names|edges|panels [--years 2568] [--out DIR]
Contains vendor names: keep the output under $EGP/data (permission 700), never in git.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import duckdb

EGP = Path(os.environ.get("EGP", str(Path.home() / "egp"))).expanduser()
INTERIM = EGP / "data" / "interim"
REPO = Path(__file__).resolve().parents[1]

NORM_P = "regexp_replace(trim(project_name), '\\s+', ' ', 'g')"
NORM_V = "regexp_replace(trim(winner_name), '\\s+', ' ', 'g')"
# same definitions as src/features.py
METHOD_CAT = """CASE
  WHEN method = 'เฉพาะเจาะจง' THEN 'specific'
  WHEN method LIKE 'ประกวดราคาอิเล็กทรอนิกส์%' THEN 'ebidding'
  WHEN method = 'คัดเลือก' THEN 'selection'
  WHEN method LIKE 'ตลาดอิเล็กทรอนิกส์%' THEN 'emarket'
  WHEN method LIKE 'จ้างที่ปรึกษา%' THEN 'consult'
  WHEN method LIKE 'จ้างออกแบบ%' THEN 'design'
  ELSE 'other' END"""
ENTITY = """CASE WHEN vendor_name_key IS NULL THEN NULL
  WHEN winner_type IN ('company', 'partnership', 'government') THEN vendor_name_key
  ELSE vendor_name_key || '|' || coalesce(province, '') END"""
UNIT = "agency || ' || ' || coalesce(sub_agency, '')"


def connect():
    con = duckdb.connect()
    tmp = os.environ.get("TMPDIR", str(EGP / "scratch"))
    con.execute(f"SET threads={int(os.environ.get('EGPQ_THREADS', '4'))}; "
                f"SET memory_limit='{os.environ.get('FDN_MEMORY', '12GB')}'; "
                f"SET temp_directory='{tmp}/fdn_duckdb'; SET preserve_insertion_order=false")
    return con


def years_available(only):
    ys = sorted(p.name.split("_")[1] for p in INTERIM.glob("contracts_*") if p.is_dir() and any(p.glob("*.parquet")))
    return [y for y in ys if not only or y in only]


def files(years):
    return "[" + ", ".join(f"'{(INTERIM / f'contracts_{y}' / '*.parquet').as_posix()}'" for y in years) + "]"


def manifest(out, step, info):
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    info.update(step=step, commit=commit, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), interim=str(INTERIM))
    (out / f"manifest_{step}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(json.dumps(info, ensure_ascii=False), flush=True)


def names(con, out, years):
    t0 = time.time()
    f = out / "project_names.parquet"
    if not f.exists():
        con.execute(f"""COPY (
            SELECT (row_number() OVER (ORDER BY n DESC, name) - 1)::BIGINT AS name_id, name, n, y0, y1 FROM (
              SELECT {NORM_P} AS name, count(*) AS n, min(fiscal_year) AS y0, max(fiscal_year) AS y1
              FROM read_parquet({files(years)}, union_by_name=true)
              WHERE project_name IS NOT NULL AND {NORM_P} <> '' GROUP BY 1)
          ) TO '{f.as_posix()}.part' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000)""")
        os.replace(f"{f}.part", f)
        print(f"project names done in {time.time() - t0:.0f} s", flush=True)
    g = out / "vendor_names.parquet"
    if not g.exists():
        con.execute(f"""COPY (
            SELECT (row_number() OVER (ORDER BY n DESC, name) - 1)::BIGINT AS vendor_id, * FROM (
              SELECT {NORM_V} AS name, count(*) AS n, count(DISTINCT vendor_name_key) AS n_keys,
                     mode(winner_type) AS winner_type, mode(tax_id_kind) AS tax_id_kind,
                     count(DISTINCT province) AS n_provinces, count(DISTINCT {UNIT}) AS n_units,
                     min(fiscal_year) AS y0, max(fiscal_year) AS y1
              FROM read_parquet({files(years)}, union_by_name=true)
              WHERE winner_name IS NOT NULL AND {NORM_V} <> '' GROUP BY 1)
          ) TO '{g.as_posix()}.part' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000)""")
        os.replace(f"{g}.part", g)
    info = {"years": years,
            "project_names": con.execute(f"SELECT count(*), sum(n) FROM '{f.as_posix()}'").fetchone(),
            "vendor_names": con.execute(f"SELECT count(*), sum(n) FROM '{g.as_posix()}'").fetchone(),
            "seconds": round(time.time() - t0)}
    manifest(out, "names", info)


def edges(con, out, years):
    pn = out / "project_names.parquet"
    if not pn.exists():
        raise SystemExit("run the names step first")
    d = out / "edges"
    d.mkdir(exist_ok=True)
    rows = {}
    for y in years:
        t0 = time.time()
        f = d / f"fy={y}.parquet"
        if not f.exists():
            con.execute(f"""COPY (
                SELECT c.fiscal_year, try_cast(c.sign_date AS DATE) AS sign_date, c.agency, c.sub_agency, c.province,
                       {UNIT} AS unit_key, {ENTITY} AS entity_key, c.vendor_name_key, c.winner_type, c.tax_id_kind,
                       c.method, {METHOD_CAT} AS method_cat, c.project_type, c.budget, c.ref_price, c.agreed_price,
                       c.contract_value, c.project_id, c.contract_no, p.name_id
                FROM read_parquet({files([y])}, union_by_name=true) c
                LEFT JOIN read_parquet('{pn.as_posix()}') p ON p.name = regexp_replace(trim(c.project_name), '\\s+', ' ', 'g')
              ) TO '{f.as_posix()}.part' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000)""")
            os.replace(f"{f}.part", f)
        rows[y] = con.execute(f"SELECT count(*), count(name_id) FROM '{f.as_posix()}'").fetchone()
        print(f"edges {y}: rows, with name_id = {rows[y]}  ({time.time() - t0:.0f} s)", flush=True)
    manifest(out, "edges", {"years": years, "rows_and_named": rows})


def panels(con, out, years):
    src = "[" + ", ".join(f"'{(out / 'edges' / f'fy={y}.parquet').as_posix()}'" for y in years) + "]"
    f = out / "panel_unit_year.parquet"
    if not f.exists():
        con.execute(f"""COPY (
            SELECT unit_key, any_value(agency) AS agency, fiscal_year, count(*) AS n, sum(agreed_price) AS value,
                   count(DISTINCT entity_key) AS n_vendors,
                   avg((method_cat = 'specific')::INT) AS sh_specific, avg((method_cat = 'ebidding')::INT) AS sh_ebidding,
                   avg((method_cat = 'selection')::INT) AS sh_selection,
                   sum((agreed_price > 450000 AND agreed_price <= 500000)::INT) AS n_450_500k,
                   sum((agreed_price > 500000 AND agreed_price <= 550000)::INT) AS n_500_550k,
                   sum((agreed_price > 90000 AND agreed_price <= 100000)::INT) AS n_90_100k,
                   median(agreed_price) AS median_price,
                   avg(CASE WHEN budget > 0 THEN 1 - agreed_price / budget END) AS mean_discount_vs_budget
            FROM read_parquet({src}) GROUP BY unit_key, fiscal_year
          ) TO '{f.as_posix()}.part' (FORMAT parquet, COMPRESSION zstd)""")
        os.replace(f"{f}.part", f)
    g = out / "series_agency_month.parquet"
    if not g.exists():
        con.execute(f"""COPY (
            SELECT agency, fiscal_year, date_trunc('month', sign_date) AS month, count(*) AS n, sum(agreed_price) AS value,
                   sum(agreed_price) FILTER (WHERE method_cat = 'specific') AS value_specific,
                   sum(agreed_price) FILTER (WHERE method_cat = 'ebidding') AS value_ebidding,
                   sum(budget) AS budget
            FROM read_parquet({src}) WHERE sign_date IS NOT NULL GROUP BY 1, 2, 3
          ) TO '{g.as_posix()}.part' (FORMAT parquet, COMPRESSION zstd)""")
        os.replace(f"{g}.part", g)
    manifest(out, "panels", {"years": years,
                             "panel_unit_year": con.execute(f"SELECT count(*) FROM '{f.as_posix()}'").fetchone()[0],
                             "series_agency_month": con.execute(f"SELECT count(*) FROM '{g.as_posix()}'").fetchone()[0]})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["names", "edges", "panels", "all"])
    ap.add_argument("--years", nargs="*", help="limit to these fiscal years (smoke test)")
    ap.add_argument("--out", default=str(EGP / "data" / "shared" / "foundation_v1"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    years = years_available(a.years)
    if not years:
        raise SystemExit(f"no contracts_* folders with parquet under {INTERIM}")
    print(f"interim {INTERIM}; years {years}; out {out}", flush=True)
    con = connect()
    for step in (["names", "edges", "panels"] if a.step == "all" else [a.step]):
        {"names": names, "edges": edges, "panels": panels}[step](con, out, years)


if __name__ == "__main__":
    main()
