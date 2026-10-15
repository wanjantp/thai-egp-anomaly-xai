"""
features.py — สร้างฟีเจอร์ red flag ที่ตีความได้ จาก data/interim/contracts_YYYY/*.parquet
             → data/processed/features_YYYY.parquet (1 แถวต่อสัญญา) + reports/features_YYYY.md

ฟีเจอร์ (ทุกตัวคำนวณภายในปีงบเดียว; aggregate ใช้ข้อมูลทั้งปีของปีนั้น)
ระดับสัญญา
  log_price            log10(ราคาตกลง)
  ratio_price_ref      ราคาตกลง / ราคากลาง (null ถ้าราคากลาง null หรืออยู่นอก [0.1, 10] → ref_price_suspect = 1)
  ratio_price_budget   ราคาตกลง / งบประมาณ
  ratio_ref_budget     ราคากลาง / งบประมาณ
  price_eq_ref         ราคาตกลง == ราคากลาง (ไม่มีส่วนลดเลย)
  price_gt_ref         ราคาตกลง > ราคากลาง
  price_z_peer         robust z ของ log_price ภายในกลุ่ม (project_type, method_cat) = (x - median)/(1.4826*MAD)
  is_round_1000        ราคาตกลงหาร 1,000 ลงตัว
  near_500k            วิธีเฉพาะเจาะจง และ 450,000 ≤ ราคา ≤ 500,000   (เพดานวิธีเฉพาะเจาะจงทั่วไป — ยังไม่ยืนยันข้อยกเว้นรายหน่วยงาน)
  near_100k            95,000 ≤ ราคา ≤ 100,000
  gap_to_500k          (500,000 − ราคา)/500,000 เฉพาะวิธีเฉพาะเจาะจงที่ราคา ≤ 500,000 (ใกล้ 0 = ชิดเพดาน)
  duration_days        วันที่สิ้นสุด − วันที่ลงนาม
  announce_to_sign_days (มีเฉพาะ e-bidding/คัดเลือก)
  fiscal_month         1 = ต.ค. … 12 = ก.ย.
  is_fy_q4             ก.ค.–ก.ย. ;  is_sep  ก.ย.
  days_to_fy_end       จำนวนวันจากวันลงนามถึง 30 ก.ย. ของปีงบ
ระดับหน่วยงานย่อย (unit = agency + sub_agency)
  unit_n_contracts, unit_n_vendors
  unit_hhi             HHI (0–1) ของมูลค่าตาม entity_key — **ไม่รวมสัญญาหมวดการแพทย์ (is_medical)**
  unit_top1_share      ส่วนแบ่งมูลค่าของผู้ขายอันดับ 1 (ไม่รวมหมวดการแพทย์)
  unit_share_specific  สัดส่วนสัญญาวิธีเฉพาะเจาะจงของหน่วย
  unit_share_near500k  สัดส่วนสัญญาใกล้เพดานของหน่วย
ระดับคู่ผู้ขาย–หน่วยงาน
  pair_n_contracts     จำนวนสัญญาของผู้ขายนี้กับหน่วยนี้ทั้งปี
  pair_value_share     ส่วนแบ่งมูลค่าของผู้ขายนี้ในหน่วยนี้
  split_n_month / split_sum_month      จำนวน/มูลค่ารวม สัญญา (หน่วย, ผู้ขาย, เดือนลงนาม) เดียวกัน  ┐ ไม่นับหมวด
  split_n_sameday / split_sum_sameday  จำนวน/มูลค่ารวม สัญญา (หน่วย, ผู้ขาย, วันลงนาม) เดียวกัน     ┘ การแพทย์
  split_flag           ≥2 สัญญาเฉพาะเจาะจง (≤500k) ผู้ขายเดียว หน่วยเดียว วันเดียว รวมกัน > 500k (สัญญาณซอยสัญญา)
  split_flag_month     แบบเดียวกันแต่ภายในเดือน (กว้างกว่า, มี false positive จากการสั่งของประจำ)
ระดับผู้ขาย
  vendor_n_contracts, vendor_n_units, vendor_n_provinces, vendor_log_value
หมวดหมู่ (เก็บไว้สำหรับ one-hot/กรอง ไม่ใช่ฟีเจอร์ตัวเลข)
  method_cat {specific, ebidding, selection, emarket, consult, design, other}
  project_type, winner_type, tax_id_kind
  is_medical           หมวดยา/เวชภัณฑ์/วัสดุการแพทย์: ชื่อโครงการ (ไทย + ชื่อยาอังกฤษ+หน่วยยา) หรือหน่วยงานย่อยฝ่ายเภสัช/คลังยา
                       หรือผู้ขายที่ ≥50% ของสัญญาเป็นหมวดนี้ (is_med_name = เฉพาะจากชื่อ/หน่วย)

entity_key = vendor_name_key สำหรับ company/partnership/government
           = vendor_name_key || '|' || จังหวัด สำหรับ shop/individual/other (กันชื่อร้านซ้ำข้ามจังหวัด — PROJECT_CONTEXT 2.2-9b)

ใช้งาน
  python src\\features.py                     # ทุกปี
  python src\\features.py --years 2568 --db D:\\tmp\\feat.duckdb
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
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

CEIL_SPECIFIC = 500_000
# หมวดยา/เวชภัณฑ์/วัสดุการแพทย์ — ตัด ยาง/ยาน/ยาว/ยาม และ "พยาบาล" ออกโดยระบุบริบทหน้า "ยา"
MED_NAME_RX = (r"(ซื้อยา([^งนวม]|$)|รายการยา|ยาในบัญชี|ยานอกบัญชี|ยาและ|ยาฉีด|ยาเม็ด|ยาน้ำ|ยาแคปซูล|"
               r"เวชภัณฑ์|วัคซีน|เภสัช|วัสดุการแพทย์|วัสดุทางการแพทย์|วัสดุทันตกรรม|อาหารทางการแพ|"
               r"วัสดุวิทยาศาสตร์การแพทย์|น้ำยาตรวจ|สารน้ำ)")
# ชื่อยาสามัญภาษาอังกฤษ: "ซื้อ" ตามด้วยอักษรละติน และมีหน่วยยา
MED_LATIN_RX = r"^ซื้อ\s*[A-Za-z0-9].*(?i:\b(mg|mcg|ml|tab|tablet|cap|caps|inj|amp|vial|syr|syrup|cream|gm|iu|bag|bot)\b)"
MED_UNIT_RX = r"(เภสัช|คลังยา|เวชภัณฑ์|ยาและเวช)"
MED_VENDOR_SHARE = 0.5   # ผู้ขายที่ ≥ 50% ของสัญญาเป็นหมวดการแพทย์ → ทุกสัญญาของผู้ขายนั้นนับเป็นหมวดการแพทย์

NUM_FEATURES = [
    "log_price", "ratio_price_ref", "ratio_price_budget", "ratio_ref_budget", "price_eq_ref", "price_gt_ref",
    "price_z_peer", "is_round_1000", "near_500k", "near_100k", "gap_to_500k", "duration_days",
    "announce_to_sign_days", "fiscal_month", "is_fy_q4", "is_sep", "days_to_fy_end",
    "unit_n_contracts", "unit_n_vendors", "unit_hhi", "unit_top1_share", "unit_share_specific", "unit_share_near500k",
    "pair_n_contracts", "pair_value_share", "split_n_month", "split_sum_month", "split_n_sameday", "split_sum_sameday",
    "split_flag", "split_flag_month", "ref_price_suspect",
    "vendor_n_contracts", "vendor_n_units", "vendor_n_provinces", "vendor_log_value",
]


def _has(con, name: str) -> bool:
    return con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchone()[0] > 0


def build(con, year: str) -> None:
    """แต่ละขั้นเก็บเป็นตารางในไฟล์ duckdb (--db) → ถ้าโดนตัดกลางคัน รันซ้ำจะทำต่อจากขั้นที่ค้าง
    ตารางผูกกับปีด้วย meta table; เปลี่ยนปีจะล้างตารางเก่า"""
    src = str(INTERIM / f"contracts_{year}" / "*.parquet").replace("\\", "/")
    t = lambda: time.time()
    con.execute("CREATE TABLE IF NOT EXISTS _meta (k VARCHAR, v VARCHAR)")
    cur = con.execute("SELECT v FROM _meta WHERE k = 'year'").fetchone()
    if not cur or cur[0] != year:
        for tb in ("base", "peer", "unit", "pair", "split_m", "split_d", "vendor", "joined"):
            con.execute(f"DROP TABLE IF EXISTS {tb}")
        con.execute("DELETE FROM _meta"); con.execute("INSERT INTO _meta VALUES ('year', ?)", [year])

    t0 = t()
    if not _has(con, "base"):
      con.execute(f"""
    CREATE TABLE base AS
    SELECT
        project_id, contract_no, fiscal_year, agency, sub_agency, province, district,
        project_name, project_type, method, winner_name, winner_type, tax_id_kind, tax_id,
        vendor_name_key, sign_date, end_date, txn_date, announce_date,
        budget, ref_price, agreed_price, contract_value,
        CASE
          WHEN method = 'เฉพาะเจาะจง'                              THEN 'specific'
          WHEN method LIKE 'ประกวดราคาอิเล็กทรอนิกส์%'             THEN 'ebidding'
          WHEN method = 'คัดเลือก'                                  THEN 'selection'
          WHEN method LIKE 'ตลาดอิเล็กทรอนิกส์%'                    THEN 'emarket'
          WHEN method LIKE 'จ้างที่ปรึกษา%'                         THEN 'consult'
          WHEN method LIKE 'จ้างออกแบบ%'                            THEN 'design'
          ELSE 'other' END                                          AS method_cat,
        (regexp_matches(project_name, '{MED_NAME_RX}') OR regexp_matches(project_name, '{MED_LATIN_RX}')
         OR regexp_matches(coalesce(sub_agency, ''), '{MED_UNIT_RX}'))     AS is_med_name,
        agency || ' || ' || coalesce(sub_agency, '')               AS unit_key,
        CASE WHEN vendor_name_key IS NULL THEN NULL
             WHEN winner_type IN ('company', 'partnership', 'government') THEN vendor_name_key
             ELSE vendor_name_key || '|' || coalesce(province, '') END AS entity_key
    FROM read_parquet('{src}')
    """)
      # ปรับด้วยสัดส่วนของผู้ขาย (ผู้ขายยาเฉพาะทางขายของที่ชื่อโครงการไม่บอกว่าเป็นยา)
      con.execute(f"""
    CREATE OR REPLACE TABLE base AS
    SELECT * EXCLUDE (is_med_name), is_med_name,
           (is_med_name OR coalesce(avg(is_med_name::INT) OVER (PARTITION BY entity_key), 0) >= {MED_VENDOR_SHARE}) AS is_medical
    FROM base
    """)
    print(f"   base: {t()-t0:.0f}s", flush=True)

    t0 = t()
    if not _has(con, "vendor"):
      for tb in ("peer", "unit", "pair", "split_m", "split_d"):
        con.execute(f"DROP TABLE IF EXISTS {tb}")
      # robust z ของ log price ในกลุ่ม peer
      con.execute("""
    CREATE OR REPLACE TABLE peer AS
    WITH x AS (SELECT project_type, method_cat, log10(agreed_price) lp FROM base WHERE agreed_price > 0),
    m AS (SELECT project_type, method_cat, median(lp) med FROM x GROUP BY 1, 2)
    SELECT m.project_type, m.method_cat, m.med, median(abs(x.lp - m.med)) mad
    FROM x JOIN m USING (project_type, method_cat) GROUP BY 1, 2, 3
    """)
    # หน่วยงานย่อย
      con.execute(f"""
    CREATE OR REPLACE TABLE unit AS
    WITH nd AS (SELECT unit_key, entity_key, sum(agreed_price) v FROM base
                WHERE NOT is_medical AND entity_key IS NOT NULL GROUP BY 1, 2),
    tot AS (SELECT unit_key, sum(v) tv FROM nd GROUP BY 1),
    sh AS (SELECT nd.unit_key, v / nullif(tv, 0) s FROM nd JOIN tot USING (unit_key)),
    conc AS (SELECT unit_key, sum(s * s) hhi, max(s) top1 FROM sh GROUP BY 1),
    cnt AS (SELECT unit_key, count(*) n, count(DISTINCT entity_key) nv,
                   avg(CASE WHEN method_cat = 'specific' THEN 1.0 ELSE 0 END) sh_spec,
                   avg(CASE WHEN method_cat = 'specific' AND agreed_price BETWEEN 450000 AND {CEIL_SPECIFIC} THEN 1.0 ELSE 0 END) sh_near
            FROM base GROUP BY 1)
    SELECT cnt.unit_key, n unit_n_contracts, nv unit_n_vendors, hhi unit_hhi, top1 unit_top1_share,
           sh_spec unit_share_specific, sh_near unit_share_near500k
    FROM cnt LEFT JOIN conc USING (unit_key)
    """)
    # คู่ผู้ขาย–หน่วย
      con.execute("""
    CREATE OR REPLACE TABLE pair AS
    WITH p AS (SELECT unit_key, entity_key, count(*) n, sum(agreed_price) v FROM base
               WHERE entity_key IS NOT NULL GROUP BY 1, 2),
    u AS (SELECT unit_key, sum(agreed_price) uv FROM base GROUP BY 1)
    SELECT p.unit_key, entity_key, n pair_n_contracts, v / nullif(uv, 0) pair_value_share
    FROM p JOIN u USING (unit_key)
    """)
      con.execute("""
    CREATE OR REPLACE TABLE split_m AS
    SELECT unit_key, entity_key, date_trunc('month', sign_date) mth, count(*) split_n_month, sum(agreed_price) split_sum_month
    FROM base WHERE entity_key IS NOT NULL AND sign_date IS NOT NULL AND NOT is_medical GROUP BY 1, 2, 3
    """)
      con.execute("""
    CREATE OR REPLACE TABLE split_d AS
    SELECT unit_key, entity_key, sign_date, count(*) split_n_sameday, sum(agreed_price) split_sum_sameday,
           sum(CASE WHEN method_cat = 'specific' AND agreed_price <= 500000 THEN 1 ELSE 0 END) split_n_sameday_specific
    FROM base WHERE entity_key IS NOT NULL AND sign_date IS NOT NULL AND NOT is_medical GROUP BY 1, 2, 3
    """)
      con.execute("""
    CREATE OR REPLACE TABLE vendor AS
    SELECT entity_key, count(*) vendor_n_contracts, count(DISTINCT unit_key) vendor_n_units,
           count(DISTINCT province) vendor_n_provinces, log10(1 + sum(agreed_price)) vendor_log_value
    FROM base WHERE entity_key IS NOT NULL GROUP BY 1
    """)
    print(f"   aggregates: {t()-t0:.0f}s", flush=True)

    t0 = t()
    fy_end = f"DATE '{int(year) - 543}-09-30'"
    out = PROCESSED / f"features_{year}.parquet"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tmp = str(out).replace("\\", "/") + ".tmp"
    if not _has(con, "joined"):
      con.execute(f"""
    CREATE TABLE joined AS
    SELECT
        b.project_id, b.contract_no, b.fiscal_year, b.agency, b.sub_agency, b.unit_key, b.province, b.district,
        b.project_type, b.method, b.method_cat, b.winner_name, b.winner_type, b.tax_id_kind, b.tax_id,
        b.entity_key, b.is_medical, b.is_med_name, b.sign_date, b.end_date, b.budget, b.ref_price, b.agreed_price,
        -- ระดับสัญญา
        log10(nullif(b.agreed_price, 0))                                    AS log_price,
        CASE WHEN b.agreed_price / b.ref_price BETWEEN 0.1 AND 10 THEN b.agreed_price / b.ref_price END AS ratio_price_ref,
        CASE WHEN b.ref_price IS NOT NULL AND NOT (b.agreed_price / b.ref_price BETWEEN 0.1 AND 10) THEN 1 ELSE 0 END AS ref_price_suspect,
        b.agreed_price / nullif(b.budget, 0)                                AS ratio_price_budget,
        CASE WHEN b.ref_price / nullif(b.budget, 0) BETWEEN 0.1 AND 10 THEN b.ref_price / b.budget END AS ratio_ref_budget,
        CASE WHEN b.ref_price IS NULL THEN NULL WHEN b.agreed_price = b.ref_price THEN 1 ELSE 0 END AS price_eq_ref,
        CASE WHEN b.ref_price IS NULL THEN NULL WHEN b.agreed_price > b.ref_price * 1.0001 THEN 1 ELSE 0 END AS price_gt_ref,
        (log10(nullif(b.agreed_price, 0)) - pe.med) / nullif(1.4826 * pe.mad, 0) AS price_z_peer,
        CASE WHEN b.agreed_price % 1000 = 0 THEN 1 ELSE 0 END               AS is_round_1000,
        CASE WHEN b.method_cat = 'specific' AND b.agreed_price BETWEEN 450000 AND {CEIL_SPECIFIC} THEN 1 ELSE 0 END AS near_500k,
        CASE WHEN b.agreed_price BETWEEN 95000 AND 100000 THEN 1 ELSE 0 END AS near_100k,
        CASE WHEN b.method_cat = 'specific' AND b.agreed_price <= {CEIL_SPECIFIC}
             THEN ({CEIL_SPECIFIC} - b.agreed_price) / {CEIL_SPECIFIC} END  AS gap_to_500k,
        date_diff('day', b.sign_date, b.end_date)                           AS duration_days,
        date_diff('day', b.announce_date, b.sign_date)                      AS announce_to_sign_days,
        ((month(b.sign_date) + 2) % 12) + 1                                 AS fiscal_month,
        CASE WHEN month(b.sign_date) IN (7, 8, 9) THEN 1 ELSE 0 END         AS is_fy_q4,
        CASE WHEN month(b.sign_date) = 9 THEN 1 ELSE 0 END                  AS is_sep,
        date_diff('day', b.sign_date, {fy_end})                             AS days_to_fy_end,
        -- หน่วยงาน
        u.unit_n_contracts, u.unit_n_vendors, u.unit_hhi, u.unit_top1_share, u.unit_share_specific, u.unit_share_near500k,
        -- คู่
        p.pair_n_contracts, p.pair_value_share,
        coalesce(sm.split_n_month, 0) AS split_n_month, coalesce(sm.split_sum_month, 0) AS split_sum_month,
        coalesce(sd.split_n_sameday, 0) AS split_n_sameday, coalesce(sd.split_sum_sameday, 0) AS split_sum_sameday,
        -- ซอยสัญญา: ≥2 สัญญาเฉพาะเจาะจง (≤500k) กับผู้ขายเดียว หน่วยเดียว วันเดียว และรวมกันเกิน 500k (ไม่นับหมวดการแพทย์)
        CASE WHEN NOT b.is_medical AND b.method_cat = 'specific' AND b.agreed_price <= {CEIL_SPECIFIC}
                  AND sd.split_n_sameday_specific >= 2 AND sd.split_sum_sameday > {CEIL_SPECIFIC} THEN 1 ELSE 0 END AS split_flag,
        CASE WHEN NOT b.is_medical AND b.method_cat = 'specific' AND b.agreed_price <= {CEIL_SPECIFIC}
                  AND sm.split_n_month >= 2 AND sm.split_sum_month > {CEIL_SPECIFIC} THEN 1 ELSE 0 END AS split_flag_month,
        -- ผู้ขาย
        v.vendor_n_contracts, v.vendor_n_units, v.vendor_n_provinces, v.vendor_log_value
    FROM base b
    LEFT JOIN peer pe     ON pe.project_type IS NOT DISTINCT FROM b.project_type AND pe.method_cat = b.method_cat
    LEFT JOIN unit u      ON u.unit_key = b.unit_key
    LEFT JOIN pair p      ON p.unit_key = b.unit_key AND p.entity_key = b.entity_key
    LEFT JOIN split_m sm  ON sm.unit_key = b.unit_key AND sm.entity_key = b.entity_key AND sm.mth = date_trunc('month', b.sign_date)
    LEFT JOIN split_d sd  ON sd.unit_key = b.unit_key AND sd.entity_key = b.entity_key AND sd.sign_date = b.sign_date
    LEFT JOIN vendor v    ON v.entity_key = b.entity_key
    """)
      print(f"   join: {t()-t0:.0f}s", flush=True)
    t0 = t()
    con.execute(f"COPY joined TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    Path(tmp).replace(out)
    print(f"   write: {t()-t0:.0f}s → {out}", flush=True)


def report(con, year: str) -> dict:
    f = str(PROCESSED / f"features_{year}.parquet").replace("\\", "/")
    con.execute(f"CREATE OR REPLACE VIEW F AS SELECT * FROM read_parquet('{f}')")
    n = con.execute("SELECT count(*) FROM F").fetchone()[0]
    rows = []
    for c in NUM_FEATURES:
        r = con.execute(f"""SELECT avg(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0 END), avg({c}),
                   quantile_cont({c}, 0.01), quantile_cont({c}, 0.5), quantile_cont({c}, 0.99), min({c}), max({c}) FROM F""").fetchone()
        rows.append((c,) + tuple(None if x is None else round(float(x), 4) for x in r))
    rep = {"year": year, "n": n, "features": rows}
    rep["method_cat"] = con.execute("SELECT method_cat, count(*) FROM F GROUP BY 1 ORDER BY 2 DESC").fetchall()
    rep["is_medical"] = con.execute("SELECT is_medical, is_med_name, count(*), sum(agreed_price) FROM F GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
    rep["bunching_500k"] = con.execute(f"""SELECT floor(agreed_price/10000)*10000 b, count(*) FROM F
        WHERE method_cat='specific' AND agreed_price BETWEEN 400000 AND 609999 GROUP BY 1 ORDER BY 1""").fetchall()
    rep["split_flag_rate_by_winner_type"] = con.execute(
        "SELECT winner_type, avg(split_flag), avg(near_500k), count(*) FROM F GROUP BY 1 ORDER BY 4 DESC").fetchall()
    rep["top_units_hhi"] = con.execute("""SELECT unit_key, any_value(unit_n_contracts) n, any_value(unit_n_vendors) nv,
        round(any_value(unit_hhi), 3) hhi, round(any_value(unit_top1_share), 3) top1
        FROM F WHERE unit_n_contracts >= 200 GROUP BY 1 ORDER BY hhi DESC LIMIT 15""").fetchall()
    rep["top_split"] = con.execute("""SELECT unit_key, CASE WHEN winner_type IN ('individual', 'shop', 'other') THEN '(บุคคล/ร้าน #' || (hash(entity_key) % 100000)::VARCHAR || ')' ELSE entity_key END,
        strftime(sign_date, '%Y-%m-%d') d, any_value(split_n_sameday) n,
        round(any_value(split_sum_sameday)) s FROM F WHERE split_flag = 1
        GROUP BY 1, 2, 3 ORDER BY n DESC LIMIT 15""").fetchall()
    rep["split_flag_rate"] = con.execute("SELECT avg(split_flag), avg(split_flag_month) FROM F").fetchone()
    rep["corr_top"] = []
    return rep


def write_report(rep: dict) -> None:
    y = rep["year"]
    (REPORTS / f"features_{y}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    L = [f"# ฟีเจอร์ red flag ปีงบ {y}", "", f"- จำนวนสัญญา: **{rep['n']:,}**",
         f"- method_cat: {rep['method_cat']}",
         f"- หมวดการแพทย์ (is_medical, is_med_name, n, มูลค่า): {rep['is_medical']}", "",
         "## สรุปฟีเจอร์ตัวเลข", "| ฟีเจอร์ | null | mean | p01 | p50 | p99 | min | max |", "|---|---|---|---|---|---|---|---|"]
    L += ["| " + " | ".join("" if x is None else f"{x:,}" if isinstance(x, (int, float)) else str(x) for x in r) + " |" for r in rep["features"]]
    L += ["", "## การกระจุกตัวของราคาใต้เพดาน 500,000 บาท (วิธีเฉพาะเจาะจง, ช่วงละ 10,000)", "| ช่วงเริ่ม | n |", "|---|---|"]
    L += [f"| {int(b):,} | {c:,} |" for b, c in rep["bunching_500k"]]
    L += ["", "## อัตรา split_flag / near_500k ตามประเภทผู้ชนะ", "| winner_type | split_flag | near_500k | n |", "|---|---|---|---|"]
    L += [f"| {a} | {b:.4f} | {c:.4f} | {d:,} |" for a, b, c, d in rep["split_flag_rate_by_winner_type"]]
    L += ["", "## หน่วยงานย่อย (≥200 สัญญา) ที่ HHI สูงสุด (ไม่รวมหมวดการแพทย์)", "| หน่วย | n | ผู้ขาย | HHI | top1 |", "|---|---|---|---|---|"]
    L += [f"| {a} | {b:,} | {c:,} | {d} | {e} |" for a, b, c, d, e in rep["top_units_hhi"]]
    L += ["", f"## กลุ่ม (หน่วย, ผู้ขาย, วัน) ที่ split_flag มีจำนวนสัญญามากสุด — อัตรา split_flag {rep['split_flag_rate'][0]:.4f}, split_flag_month {rep['split_flag_rate'][1]:.4f}", "| หน่วย | ผู้ขาย (บุคคล/ร้าน ถูกแทนด้วยรหัส) | วันที่ | n | มูลค่ารวม |", "|---|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} | {e:,.0f} |" for a, b, c, d, e in rep["top_split"]]
    L += ["", "> หมายเหตุ: รายการข้างบนเป็น **สัญญาณที่ควรตรวจสอบ** ไม่ใช่ข้อสรุปว่ามีการทุจริต; ชื่อผู้ขายบุคคลธรรมดาต้อง pseudonymize ก่อนเผยแพร่"]
    (REPORTS / f"features_{y}.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", default=["2566", "2567", "2568"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory", default="3GB")
    ap.add_argument("--db", default=None, help="ไฟล์ duckdb ชั่วคราวบนดิสก์ (แนะนำ เพื่อให้ spill ได้)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--keep-db", action="store_true", help="ไม่ลบไฟล์ duckdb ชั่วคราวหลังเสร็จ")
    a = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)
    for y in a.years:
        print(f"== features {y}", flush=True)
        # ไฟล์ duckdb แยกต่อปี (DROP TABLE ไม่คืนพื้นที่ดิสก์) → ลบทิ้งเมื่อปีนั้นเสร็จ
        dbp = Path(a.db).with_name(f"{Path(a.db).stem}_{y}{Path(a.db).suffix or '.duckdb'}") if a.db else None
        con = duckdb.connect(str(dbp) if dbp else ":memory:")
        con.execute(f"SET threads={a.threads}"); con.execute(f"SET memory_limit='{a.memory}'")
        # temp ของ duckdb → โฟลเดอร์ temp ของระบบ (ค่าเริ่มต้น .tmp ในโฟลเดอร์ปัจจุบันช้ามากบน synced/network drive)
        con.execute(f"SET temp_directory='{(Path(tempfile.gettempdir()) / 'duckdb_egp').as_posix()}'")
        con.execute("SET preserve_insertion_order=false")
        if not a.report_only:
            build(con, y)
        rep = report(con, y)
        write_report(rep)
        con.close()
        if dbp and not a.keep_db:
            for f in (dbp, dbp.with_name(dbp.name + ".wal")):
                f.unlink(missing_ok=True)
        print(f"   report → reports/features_{y}.md", flush=True)
    print("done")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
