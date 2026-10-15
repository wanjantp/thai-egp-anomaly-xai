"""
clean.py — ทำความสะอาดข้อมูลสัญญา e-GP จาก stage parquet (ผลของ profile_raw.py)
          → data/interim/contracts_YYYY/*.parquet (typed, ชื่อคอลัมน์อังกฤษ, 1 ไฟล์ต่อไฟล์ต้นทาง) + reports/clean_YYYY.md
          อ่านทั้งปี: duckdb  read_parquet('data/interim/contracts_2567/*.parquet')  /  polars.scan_parquet(...)

กฎที่ใช้ (อ้างอิง PROJECT_CONTEXT.md ข้อ 2.2)
  - ทิ้งแถวที่ค่าเลื่อนคอลัมน์: ปีงบประมาณ != ปี หรือ สถานะสัญญา ไม่ใช่ {ระหว่างดำเนินการ, สิ้นสุดสัญญา}
  - วันที่ พ.ศ. 2 หลัก + เดือนย่อไทย ทั้งรูป `14-พ.ค.-67` และ `14 พ.ค. 68` → DATE ค.ศ.
    end_date < sign_date หรือ ห่างเกิน 15 ปี → null + flag
  - ตัวเลข: ตัด comma แล้ว try_cast; ราคากลาง <= 0 → null (ratio คำนวณในขั้น features)
  - เลขนิติบุคคล: full (13 หลัก) / masked (9 หลัก + xxxx → เก็บ prefix 9 หลัก) / sci (E+ กู้ไม่ได้) / placeholder / missing / other
  - vendor_name_key: ชื่อผู้ชนะ normalize (ตัดคำนำหน้า/ต่อท้ายนิติบุคคล, ช่องว่าง, เครื่องหมาย) — คีย์หลักข้ามปี
    เพราะปี 2566–67 ไม่มีเลข 13 หลักเลย
  - winner_type: company / partnership / shop / individual / government / other (จากคำนำหน้าชื่อ)

ใช้งาน
  python src\clean.py                    # ทุกปี
  python src\clean.py --years 2568 --limit 200000   # ทดลอง
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import tempfile

import duckdb

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data" / "interim" / "stage"
OUT = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"

# คอลัมน์ไทย (stage) → อังกฤษ
RENAME = {
    "ลำดับ": "seq", "รหัสโครงการ": "project_id", "ชื่อโครงการ": "project_name", "ชื่อประเภทโครงการ": "project_type",
    "ชื่อหน่วยงาน": "agency", "ชื่อหน่วยงานย่อย": "sub_agency", "วิธีจัดซื้อฯ": "method", "กลุ่มวิธีจัดซื้อฯ": "method_group",
    "วันที่ประกาศ": "announce_date_raw", "งบประมาณ(บาท)": "budget_raw", "ราคากลาง(บาท)": "ref_price_raw",
    "ราคาตกลงซื้อ/จ้าง": "agreed_price_raw", "ปีงบประมาณ": "fiscal_year_raw", "วันที่เกิดรายการ": "txn_date_raw",
    "จังหวัด": "province", "จังหวัด (Eng)": "province_en", "เขต/อำเภอ": "district", "เขต/อำเภอ (Eng)": "district_en",
    "แขวง/ตำบล": "subdistrict", "แขวง/ตำบล (Eng)": "subdistrict_en", "สถานะโครงการ": "project_status",
    "พิกัดของโครงการ": "coord_raw", "ละติจูดโครงการ": "lat_raw", "ลองจิจูดโครงการ": "lon_raw",
    "เลขนิติบุคคล": "tax_id_raw", "ชื่อผู้ชนะ": "winner_name", "เลขที่สัญญา": "contract_no",
    "วันที่ลงนามสัญญา": "sign_date_raw", "วันที่สิ้นสุดสัญญา": "end_date_raw", "งบสัญญา(บาท)": "contract_value_raw",
    "สถานะสัญญา": "contract_status", "_src_file": "src_file",
}

THAI_MONTHS = {
    "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4, "พ.ค.": 5, "มิ.ย.": 6,
    "ก.ค.": 7, "ส.ค.": 8, "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
    # เผื่อรูปไม่มีจุด / เต็ม
    "มค": 1, "กพ": 2, "มีค": 3, "เมย": 4, "พค": 5, "มิย": 6, "กค": 7, "สค": 8, "กย": 9, "ตค": 10, "พย": 11, "ธค": 12,
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4, "พฤษภาคม": 5, "มิถุนายน": 6, "กรกฎาคม": 7,
    "สิงหาคม": 8, "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
}
VALID_STATUS = ("ระหว่างดำเนินการ", "สิ้นสุดสัญญา")

# คำนำหน้า/ต่อท้ายที่ตัดออกจากชื่อผู้ชนะเพื่อทำคีย์ (เรียงยาว→สั้น)
STRIP_TOKENS = [
    "ห้างหุ้นส่วนจำกัด", "ห้างหุ้นส่วนสามัญนิติบุคคล", "ห้างหุ้นส่วนสามัญ", "บริษัทมหาชนจำกัด", "บริษัท", "จำกัด (มหาชน)",
    "จำกัด(มหาชน)", "(มหาชน)", "มหาชน", "จำกัด", "หจก.", "หจก", "บจก.", "บจก", "บมจ.", "บมจ", "หสน.", "หสม.",
    "ร้านค้า", "ร้าน", "กิจการร่วมค้า", "co., ltd.", "co.,ltd.", "co.,ltd", "co. ltd", "company limited", "limited",
    "ltd.", "ltd", "inc.", "inc", "corporation", "corp.", "public company",
]


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def month_case(expr: str) -> str:
    return "CASE " + " ".join(f"WHEN {expr} = '{k}' THEN {v}" for k, v in THAI_MONTHS.items()) + " END"


def date_sql(col: str) -> str:
    """แปลง '14-พ.ค.-67' / '14 พ.ค. 68' / '14 พ.ค. 2568' → DATE (ค.ศ.) ; แปลงไม่ได้ → NULL"""
    parts = f"regexp_extract(trim({col}), '^(\\d{{1,2}})[-\\s]+([^\\s-]+)[-\\s]+(\\d{{2,4}})$', ['d', 'm', 'y'])"
    return f"""(
        WITH p AS (SELECT {parts} AS s)
        SELECT NULL
    )""" if False else f"""
        try_cast(
          printf('%04d-%02d-%02d',
            CASE WHEN length(({parts}).y) = 4 THEN try_cast(({parts}).y AS INT) - 543
                 WHEN length(({parts}).y) = 2 THEN 1957 + try_cast(({parts}).y AS INT)      -- 25yy พ.ศ. → ค.ศ.
            END,
            {month_case(f'({parts}).m')},
            try_cast(({parts}).d AS INT))
        AS DATE)"""


def num_sql(col: str) -> str:
    return f"try_cast(replace(trim({col}), ',', '') AS DOUBLE)"


def name_key_sql(col: str) -> str:
    """normalize ชื่อผู้ชนะ: lower → ตัด token นิติบุคคล → ตัดเครื่องหมาย/ช่องว่างทั้งหมด"""
    e = f"lower(trim({col}))"
    for t in STRIP_TOKENS:
        e = f"replace({e}, '{t.lower()}', ' ')"
    e = f"regexp_replace({e}, '[\\s\\.\\,\\-_/\\\\()\\[\\]\"''\\*&+:;!?]+', '', 'g')"
    return f"nullif({e}, '')"


def build_sql(year: str, src_glob: str, limit: int | None) -> str:
    cols = ", ".join(f"{q(th)} AS {en}" for th, en in RENAME.items())
    dedup_cols = ", ".join(en for en in RENAME.values() if en not in ("seq", "src_file"))
    lim = f"LIMIT {limit}" if limit else ""
    return f"""
    WITH raw0 AS (
        SELECT {cols} FROM read_parquet('{src_glob}') {lim}
    ),
    -- ตัดแถวซ้ำทั้งแถว (ไม่นับ ลำดับ/ไฟล์) — ปี 2561 มี ~83k แถว
    raw AS (
        SELECT DISTINCT ON ({dedup_cols}) * FROM raw0
    ),
    kept AS (
        SELECT *,
               (fiscal_year_raw = '{year}' AND contract_status IN {VALID_STATUS}) AS _ok
        FROM raw
    ),
    typed AS (
        SELECT
            try_cast(seq AS BIGINT)                         AS seq,
            trim(project_id)                                AS project_id,
            trim(project_name)                              AS project_name,
            trim(project_type)                              AS project_type,
            trim(agency)                                    AS agency,
            trim(sub_agency)                                AS sub_agency,
            trim(method)                                    AS method,
            trim(method_group)                              AS method_group,
            {year}::INT                                     AS fiscal_year,
            {date_sql('announce_date_raw')}                 AS announce_date,
            {date_sql('txn_date_raw')}                      AS txn_date,
            {date_sql('sign_date_raw')}                     AS sign_date,
            {date_sql('end_date_raw')}                      AS end_date_parsed,
            {num_sql('budget_raw')}                         AS budget,
            nullif({num_sql('ref_price_raw')}, 0)           AS ref_price,
            ({num_sql('ref_price_raw')} = 0)                AS ref_price_is_zero,
            {num_sql('agreed_price_raw')}                   AS agreed_price,
            {num_sql('contract_value_raw')}                 AS contract_value,
            nullif(trim(province), '-')                     AS province,
            nullif(trim(district), '-')                     AS district,
            nullif(trim(subdistrict), '-')                  AS subdistrict,
            trim(project_status)                            AS project_status,
            trim(contract_status)                           AS contract_status,
            CASE WHEN {num_sql('lat_raw')} BETWEEN 5 AND 21 AND {num_sql('lon_raw')} BETWEEN 97 AND 106
                 THEN {num_sql('lat_raw')} END              AS lat,
            CASE WHEN {num_sql('lat_raw')} BETWEEN 5 AND 21 AND {num_sql('lon_raw')} BETWEEN 97 AND 106
                 THEN {num_sql('lon_raw')} END              AS lon,
            trim(tax_id_raw)                                AS tax_id_raw,
            nullif(trim(winner_name), '')                   AS winner_name,
            nullif(trim(contract_no), '')                   AS contract_no,
            announce_date_raw, txn_date_raw, sign_date_raw, end_date_raw,
            src_file
        FROM kept WHERE _ok
    ),
    enriched AS (
        SELECT *,
            -- วันสิ้นสุดต้องไม่ก่อนวันลงนาม และไม่เกิน 15 ปี
            CASE WHEN end_date_parsed IS NOT NULL AND sign_date IS NOT NULL
                      AND (end_date_parsed < sign_date OR end_date_parsed > sign_date + INTERVAL 15 YEAR)
                 THEN NULL ELSE end_date_parsed END         AS end_date,
            (end_date_parsed IS NOT NULL AND sign_date IS NOT NULL
                 AND (end_date_parsed < sign_date OR end_date_parsed > sign_date + INTERVAL 15 YEAR)) AS end_date_invalid,
            -- เลขนิติบุคคล
            CASE
              WHEN tax_id_raw IS NULL OR tax_id_raw = '' OR tax_id_raw = '-'      THEN 'missing'
              WHEN regexp_matches(tax_id_raw, '^0{{13}}$|^0{{12}}1$')              THEN 'placeholder'
              WHEN regexp_matches(tax_id_raw, '^[0-9]{{13}}$')                     THEN 'full'
              WHEN regexp_matches(tax_id_raw, '^[0-9]{{9}}x{{4}}$')                THEN 'masked'
              WHEN regexp_matches(tax_id_raw, '^[0-9]\\.?[0-9]*E\\+[0-9]+$')       THEN 'sci'
              ELSE 'other' END                                                   AS tax_id_kind,
            CASE WHEN regexp_matches(tax_id_raw, '^[0-9]{{13}}$') AND NOT regexp_matches(tax_id_raw, '^0{{13}}$|^0{{12}}1$')
                 THEN tax_id_raw END                                             AS tax_id,
            CASE WHEN regexp_matches(tax_id_raw, '^[0-9]{{9}}')
                      AND NOT regexp_matches(tax_id_raw, '^0{{13}}$|^0{{12}}1$')
                 THEN left(tax_id_raw, 9) END                                    AS tax_id_prefix9,
            -- ประเภทผู้ชนะจากคำนำหน้า
            CASE
              WHEN winner_name IS NULL                                                        THEN 'missing'
              WHEN regexp_matches(winner_name, '^(บริษัท|บจก|บมจ)|จำกัด')                        THEN 'company'
              WHEN regexp_matches(winner_name, '^(ห้างหุ้นส่วน|หจก|หสน|หสม)')                     THEN 'partnership'
              WHEN regexp_matches(winner_name, '^(ร้าน|กิจการร่วมค้า)')                          THEN 'shop'
              WHEN regexp_matches(winner_name, '^(นาย|นาง|นางสาว|น\\.ส\\.|ว่าที่|ด\\.ช\\.|ด\\.ญ\\.|พ\\.ต\\.|ร\\.ต\\.|จ\\.ส\\.|ส\\.ต\\.|พล\\.)') THEN 'individual'
              WHEN regexp_matches(winner_name, '^(องค์การ|สำนักงาน|กรม|มหาวิทยาลัย|โรงพยาบาล|การไฟฟ้า|การประปา|เทศบาล|องค์การบริหาร|สหกรณ์|มูลนิธิ|วิทยาลัย|โรงเรียน|สถาบัน|กอง)') THEN 'government'
              ELSE 'other' END                                                   AS winner_type,
            {name_key_sql('winner_name')}                                        AS vendor_name_key
        FROM typed
    )
    SELECT * EXCLUDE (end_date_parsed) FROM enriched
    """


def clean_year(con, year: str, limit: int | None) -> dict:
    """ประมวลผลทีละไฟล์ stage → data/interim/contracts_YYYY/<file>.parquet (ข้ามไฟล์ที่ทำแล้ว)
    แล้วสรุปรายงานจากทั้งโฟลเดอร์"""
    src_dir = STAGE / f"contract_{year}"
    if not src_dir.exists():
        raise SystemExit(f"ไม่พบ {src_dir} — รัน src/profile_raw.py --years {year} ก่อน")
    out_dir = OUT / (f"contracts_{year}" if not limit else f"contracts_{year}_sample{limit}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_raw = 0
    files = sorted(src_dir.glob("*.parquet"), key=lambda p: int(p.stem.rsplit("-", 1)[-1]))
    for f in files:
        dst = out_dir / f.name
        src = str(f).replace("\\", "/")
        n_raw += con.execute(f"SELECT count(*) FROM read_parquet('{src}') {'LIMIT ' + str(limit) if limit else ''}").fetchone()[0] if not limit else limit
        if dst.exists():
            continue
        t1 = time.time()
        tmp = dst.with_name(dst.name + ".tmp")
        con.execute(f"COPY ({build_sql(year, src, limit)}) TO '{str(tmp).replace(chr(92), '/')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        tmp.replace(dst)
        print(f"   {f.name}: {time.time()-t1:.0f}s", flush=True)
    out_glob = str(out_dir / "*.parquet").replace("\\", "/")
    for kind in ("VIEW", "TABLE"):
        try:
            con.execute(f"DROP {kind} IF EXISTS c")
        except duckdb.CatalogException:
            pass
    con.execute(f"CREATE VIEW c AS SELECT * FROM read_parquet('{out_glob}')")
    out = out_dir
    n = con.execute("SELECT count(*) FROM c").fetchone()[0]

    def cnt(where):
        return con.execute(f"SELECT count(*) FROM c WHERE {where}").fetchone()[0]

    rep = {
        "year": year, "limit": limit, "rows_in": n_raw, "rows_out": n, "dropped_shifted": n_raw - n,
        "date_parse_fail": {
            c: cnt(f"{c} IS NULL AND {c}_raw IS NOT NULL AND trim({c}_raw) NOT IN ('', '-')")
            for c in ["announce_date", "txn_date", "sign_date"]
        } | {"end_date": cnt("end_date IS NULL AND end_date_raw IS NOT NULL AND trim(end_date_raw) NOT IN ('', '-') AND NOT end_date_invalid")},
        "date_fail_examples": {
            c: [r[0] for r in con.execute(
                f"SELECT {c}_raw FROM c WHERE {c} IS NULL AND {c}_raw IS NOT NULL AND trim({c}_raw) NOT IN ('', '-') {'AND NOT end_date_invalid' if c == 'end_date' else ''} ORDER BY random() LIMIT 8").fetchall()]
            for c in ["txn_date", "sign_date", "end_date"]
        },
        "end_date_invalid": cnt("end_date_invalid"),
        "end_date_invalid_examples": con.execute(
            "SELECT sign_date_raw, end_date_raw FROM c WHERE end_date_invalid ORDER BY random() LIMIT 8").fetchall(),
        "sign_date_null": cnt("sign_date IS NULL"), "end_date_null": cnt("end_date IS NULL"),
        "ref_price_is_zero": cnt("ref_price_is_zero"), "ref_price_null": cnt("ref_price IS NULL"),
        "agreed_price_null": cnt("agreed_price IS NULL"), "budget_null": cnt("budget IS NULL"),
        "agreed_gt_ref": cnt("agreed_price > ref_price * 1.0001"),
        "lat_lon_present": cnt("lat IS NOT NULL"),
        "tax_id_kind": dict(con.execute("SELECT tax_id_kind, count(*) FROM c GROUP BY 1 ORDER BY 2 DESC").fetchall()),
        "winner_type": dict(con.execute("SELECT winner_type, count(*) FROM c GROUP BY 1 ORDER BY 2 DESC").fetchall()),
        "winner_type_other_examples": [r[0] for r in con.execute(
            "SELECT winner_name FROM c WHERE winner_type = 'other' ORDER BY random() LIMIT 15").fetchall()],
        "distinct_vendor_name_key": con.execute("SELECT count(DISTINCT vendor_name_key) FROM c").fetchone()[0],
        "distinct_winner_name": con.execute("SELECT count(DISTINCT winner_name) FROM c").fetchone()[0],
        "distinct_tax_id": con.execute("SELECT count(DISTINCT tax_id) FROM c").fetchone()[0],
        "distinct_project": con.execute("SELECT count(DISTINCT project_id) FROM c").fetchone()[0],
        "dup_project_contract": n - con.execute("SELECT count(*) FROM (SELECT DISTINCT project_id, contract_no FROM c)").fetchone()[0],
        # ชื่อต่างกันแต่คีย์เดียวกัน (ดูว่า normalize รวมมากไปไหม)
        "name_key_merge_examples": con.execute("""
            SELECT vendor_name_key, list(DISTINCT winner_name)[:4], count(DISTINCT winner_name) k
            FROM c WHERE vendor_name_key IS NOT NULL GROUP BY 1 HAVING k >= 2 ORDER BY k DESC LIMIT 12""").fetchall(),
        "key_with_multiple_tax_id": con.execute("""
            SELECT count(*) FROM (SELECT vendor_name_key FROM c WHERE tax_id IS NOT NULL
            GROUP BY 1 HAVING count(DISTINCT tax_id) > 1)""").fetchone()[0],
        "month_dist": dict(con.execute("SELECT month(sign_date), count(*) FROM c GROUP BY 1 ORDER BY 1").fetchall()),
    }
    rep["out"] = str(out)
    rep["sec"] = round(time.time() - t0, 1)
    rep["schema"] = [(r[0], r[1]) for r in con.execute("DESCRIBE c").fetchall()]
    return rep


def write_report(rep: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    y = rep["year"]
    tag = f"_sample{rep['limit']}" if rep["limit"] else ""
    (REPORTS / f"clean_{y}{tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    L = [f"# รายงานทำความสะอาดสัญญา ปีงบ {y}" + (f" (ตัวอย่าง {rep['limit']:,} แถว)" if rep["limit"] else ""), "",
         f"- แถวเข้า {rep['rows_in']:,} → ออก **{rep['rows_out']:,}** (ทิ้งแถวค่าเลื่อนคอลัมน์ {rep['dropped_shifted']:,}) ใช้เวลา {rep['sec']} s",
         f"- ไฟล์ผลลัพธ์: `{rep['out']}`", "",
         "## วันที่",
         f"- parse ไม่ได้ (ไม่รวมค่าว่าง/'-'): {rep['date_parse_fail']}",
         f"- ตัวอย่างที่ parse ไม่ได้: {rep['date_fail_examples']}",
         f"- end_date ผิด (ก่อนลงนาม/เกิน 15 ปี → null): **{rep['end_date_invalid']:,}** เช่น {rep['end_date_invalid_examples'][:5]}",
         f"- sign_date null: {rep['sign_date_null']:,} · end_date null: {rep['end_date_null']:,}",
         f"- เดือนที่ลงนาม (ค.ศ.): {rep['month_dist']}", "",
         "## ตัวเลข",
         f"- ราคากลาง = 0 → null: {rep['ref_price_is_zero']:,}; ราคากลาง null รวม {rep['ref_price_null']:,}; ราคาตกลง null {rep['agreed_price_null']:,}; งบ null {rep['budget_null']:,}",
         f"- ราคาตกลง > ราคากลาง: {rep['agreed_gt_ref']:,}",
         f"- มีพิกัดที่อยู่ในช่วงประเทศไทย: {rep['lat_lon_present']:,}", "",
         "## เลขนิติบุคคล", f"- ชนิด: {rep['tax_id_kind']}", f"- เลข 13 หลักไม่ซ้ำ: {rep['distinct_tax_id']:,}", "",
         "## ผู้ชนะ", f"- ประเภท: {rep['winner_type']}", f"- ตัวอย่าง 'other': {rep['winner_type_other_examples']}",
         f"- ชื่อไม่ซ้ำ {rep['distinct_winner_name']:,} → vendor_name_key ไม่ซ้ำ {rep['distinct_vendor_name_key']:,}",
         f"- key เดียวมีเลข 13 หลักมากกว่า 1 เลข: {rep['key_with_multiple_tax_id']:,}",
         "- ตัวอย่างชื่อที่ถูกรวมเป็นคีย์เดียว:"]
    for k, names, c in rep["name_key_merge_examples"]:
        L.append(f"  - `{k}` ({c} ชื่อ): {names}")
    L += ["", "## คีย์", f"- โครงการไม่ซ้ำ {rep['distinct_project']:,}; (project_id, contract_no) ซ้ำ {rep['dup_project_contract']:,} แถว", "",
          "## schema", "| คอลัมน์ | ชนิด |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rep["schema"]]
    (REPORTS / f"clean_{y}{tag}.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", default=["2566", "2567", "2568"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory", default="2GB")
    ap.add_argument("--db", default=None, help="ไฟล์ duckdb ชั่วคราว (ค่าเริ่มต้น in-memory)")
    a = ap.parse_args()
    con = duckdb.connect(a.db or ":memory:")
    con.execute(f"SET threads={a.threads}"); con.execute(f"SET memory_limit='{a.memory}'")
    con.execute(f"SET temp_directory='{(Path(tempfile.gettempdir()) / 'duckdb_egp').as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    for y in a.years:
        print(f"== clean {y}", flush=True)
        rep = clean_year(con, y, a.limit)
        write_report(rep)
        print(f"   {rep['rows_in']:,} → {rep['rows_out']:,} rows, {rep['sec']}s → {rep['out']}", flush=True)
    print("done")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
