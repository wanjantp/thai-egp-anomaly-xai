"""
profile_raw.py — ตรวจสอบคุณภาพข้อมูลดิบ e-GP (data/raw) ด้วย duckdb โดยไม่โหลดทั้งไฟล์เข้า pandas

ขั้นตอน
  1. stage : อ่าน CSV ทีละไฟล์ (รับมือ header ไม่เท่ากัน / คอลัมน์ว่างท้ายแถว / BOM / quote)
             แล้วเขียน parquet แบบ all-varchar ลง data/interim/stage/<year>/<file>.parquet
             (ข้ามไฟล์ที่ทำแล้ว → รันซ้ำได้)
  2. profile: สรุปจำนวนแถว, ค่าว่าง, ค่าที่พบในคอลัมน์หมวดหมู่, รูปแบบวันที่, รูปแบบเลขนิติบุคคล,
             การแปลงตัวเลข → เขียน reports/profile_raw_<year>.md และ .json

ใช้งาน (Windows / conda env egp-xai)
  python src\profile_raw.py                 # ทุกปี
  python src\profile_raw.py --years 2567    # เฉพาะปี
  python src\profile_raw.py --sample 200000 # ทดลองกับ N แถวแรกของแต่ละไฟล์
  python src\profile_raw.py --winner        # profile egpwinner ด้วย
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import tempfile

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
STAGE = ROOT / "data" / "interim" / "stage"
REPORTS = ROOT / "reports"

# คอลัมน์มาตรฐาน 31 คอลัมน์ (ตามไฟล์ 2566/2567)
CANON = [
    "ลำดับ", "รหัสโครงการ", "ชื่อโครงการ", "ชื่อประเภทโครงการ", "ชื่อหน่วยงาน", "ชื่อหน่วยงานย่อย",
    "วิธีจัดซื้อฯ", "กลุ่มวิธีจัดซื้อฯ", "วันที่ประกาศ", "งบประมาณ(บาท)", "ราคากลาง(บาท)", "ราคาตกลงซื้อ/จ้าง",
    "ปีงบประมาณ", "วันที่เกิดรายการ", "จังหวัด", "จังหวัด (Eng)", "เขต/อำเภอ", "เขต/อำเภอ (Eng)",
    "แขวง/ตำบล", "แขวง/ตำบล (Eng)", "สถานะโครงการ", "พิกัดของโครงการ", "ละติจูดโครงการ", "ลองจิจูดโครงการ",
    "เลขนิติบุคคล", "ชื่อผู้ชนะ", "เลขที่สัญญา", "วันที่ลงนามสัญญา", "วันที่สิ้นสุดสัญญา", "งบสัญญา(บาท)", "สถานะสัญญา",
]
# ปี 2568: header ในไฟล์เชื่อถือไม่ได้ (ไฟล์ 1 มี 26 ชื่อ, ไฟล์ 2–10 มี 31 ชื่อ) แต่ข้อมูลจริงทุกแถวมี 28 ฟิลด์
# ตรวจจากค่าจริงแล้ว: ไม่มีคอลัมน์ (Eng) และตำแหน่ง 6/7 เป็น กลุ่มวิธี/วิธี (สลับกับปี 2566–2567)
LAYOUT_2568 = [
    "ลำดับ", "รหัสโครงการ", "ชื่อโครงการ", "ชื่อประเภทโครงการ", "ชื่อหน่วยงาน", "ชื่อหน่วยงานย่อย",
    "กลุ่มวิธีจัดซื้อฯ", "วิธีจัดซื้อฯ", "วันที่ประกาศ", "งบประมาณ(บาท)", "ราคากลาง(บาท)", "ราคาตกลงซื้อ/จ้าง",
    "ปีงบประมาณ", "วันที่เกิดรายการ", "จังหวัด", "เขต/อำเภอ", "แขวง/ตำบล", "สถานะโครงการ", "พิกัดของโครงการ",
    "ละติจูดโครงการ", "ลองจิจูดโครงการ", "เลขนิติบุคคล", "ชื่อผู้ชนะ", "เลขที่สัญญา", "วันที่ลงนามสัญญา",
    "วันที่สิ้นสุดสัญญา", "งบสัญญา(บาท)", "สถานะสัญญา",
]
POSITIONAL_LAYOUT = {"2568": LAYOUT_2568}
CAT_COLS = ["ชื่อประเภทโครงการ", "วิธีจัดซื้อฯ", "กลุ่มวิธีจัดซื้อฯ", "ปีงบประมาณ", "สถานะโครงการ", "สถานะสัญญา"]
DATE_COLS = ["วันที่ประกาศ", "วันที่เกิดรายการ", "วันที่ลงนามสัญญา", "วันที่สิ้นสุดสัญญา"]
NUM_COLS = ["งบประมาณ(บาท)", "ราคากลาง(บาท)", "ราคาตกลงซื้อ/จ้าง", "งบสัญญา(บาท)", "ละติจูดโครงการ", "ลองจิจูดโครงการ"]
KEY_COLS = ["รหัสโครงการ", "เลขที่สัญญา", "เลขนิติบุคคล"]


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def detect_encoding(path: Path) -> str:
    """ตรวจ BOM: ไฟล์ส่วนใหญ่ UTF-8 (มี/ไม่มี BOM) แต่ egp-winner-3.csv เป็น UTF-16 BE"""
    b = path.open("rb").read(4)
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        return "utf-16"
    return "utf-8-sig"


def read_header(path: Path) -> list[str]:
    with open(path, encoding=detect_encoding(path), newline="") as fh:
        return next(csv.reader(fh))


def utf8_copy(path: Path) -> Path:
    """duckdb อ่าน UTF-16 BE ไม่ได้ → แปลงเป็นสำเนา UTF-8 ใน data/interim/stage/_utf8 (ไม่แตะ data/raw)"""
    dst = STAGE.parent / "stage" / "_utf8" / path.name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(path, encoding="utf-16", newline="") as fi, open(dst, "w", encoding="utf-8", newline="") as fo:
            for chunk in iter(lambda: fi.read(1 << 20), ""):
                fo.write(chunk)
    return dst


def csv_source(path: Path, **extra) -> str:
    """สร้างนิพจน์ read_csv(...) ที่กำหนด dialect ชัดเจน ไม่พึ่ง sniffer"""
    if detect_encoding(path) == "utf-16":
        path = utf8_copy(path)
    enc = "utf-8"
    opts = dict(header="true", all_varchar="true", delim="','", quote="'\"'", escape="'\"'",
                encoding=f"'{enc}'", null_padding="true", strict_mode="false", parallel="false", ignore_errors="true", store_rejects="true", max_line_size="4000000")
    opts.update(extra)
    return f"read_csv({str(path)!r}, " + ", ".join(f"{k}={v}" for k, v in opts.items()) + ")"


def connect(threads: int, mem: str, db: Path | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db) if db else ":memory:")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{mem}'")
    con.execute(f"SET temp_directory='{(Path(tempfile.gettempdir()) / 'duckdb_egp').as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    return con


# ---------------------------------------------------------------- stage
def stage_file(con, src: Path, dst: Path, sample: int | None, log: dict, layout: list[str] | None = None) -> None:
    if dst.exists():
        log["skipped"] = True
        return
    header = read_header(src)
    if layout:  # ไม่เชื่อ header → ตั้งชื่อตามตำแหน่ง
        names, blanks = list(layout), 0
        log["header_ignored"] = True
    else:
        # ตั้งชื่อคอลัมน์ว่างเป็น _blank_i แล้วทิ้งทีหลัง
        names, blanks = [], 0
        for i, h in enumerate(header):
            h = h.strip()
            if h == "":
                blanks += 1
                h = f"_blank_{i}"
            names.append(h)
    present = [c for c in CANON if c in names]
    missing = [c for c in CANON if c not in names]
    unknown = [c for c in names if c not in CANON and not c.startswith("_blank_")]
    log.update(n_header=len(header), blank_cols=blanks, missing=missing, unknown=unknown)

    sel = ", ".join(q(c) if c in names else f"NULL::VARCHAR AS {q(c)}" for c in CANON)
    limit = f" LIMIT {sample}" if sample else ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    con.execute(
        f"""
        COPY (
          SELECT {sel}, '{src.name}' AS _src_file
          FROM {csv_source(src, header="false", skip="1", columns=repr({n: "VARCHAR" for n in names}))}{limit}
        ) TO {str(tmp)!r} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    tmp.replace(dst)
    # แถวที่ CSV เสีย (เช่น quote ไม่ปิดในชื่อโครงการ ปี 2568) ถูกข้าม → บันทึกจำนวนและตัวอย่างไว้
    try:
        rej = con.execute("SELECT line, error_type, left(csv_line, 120) FROM reject_errors LIMIT 5").fetchall()
        log["rejected_rows"] = con.execute("SELECT count(*) FROM reject_errors").fetchone()[0]
        if rej:
            log["rejected_examples"] = rej
        con.execute("DELETE FROM reject_errors"); con.execute("DELETE FROM reject_scans")
    except duckdb.Error:
        log["rejected_rows"] = 0
    log["rows"] = con.execute(f"SELECT count(*) FROM read_parquet({str(dst)!r})").fetchone()[0]


def stage_year(con, year: str, sample, out: dict) -> Path:
    src_dir = RAW / f"contract_{year}"
    files = sorted(src_dir.glob("*.csv"), key=lambda p: int(p.stem.rsplit("-", 1)[-1]))
    out["files"] = {}
    for f in files:
        t0 = time.time()
        dst = STAGE / f"contract_{year}" / (f.stem + ".parquet")
        log: dict = {}
        stage_file(con, f, dst, sample, log, POSITIONAL_LAYOUT.get(year))
        log["sec"] = round(time.time() - t0, 1)
        out["files"][f.name] = log
        print(f"  staged {f.name}: {log}", flush=True)
    return STAGE / f"contract_{year}"


# ---------------------------------------------------------------- profile
def _shape(col: str) -> str:
    return f"regexp_replace(regexp_replace(regexp_replace({q(col)}, '[0-9]', '9', 'g'), '[ก-๙]', 'ก', 'g'), '[A-Za-z]', 'a', 'g')"


def _num(col: str) -> str:
    return f"try_cast(replace({q(col)}, ',', '') AS DOUBLE)"


def load_year_table(con, year: str, pq_dir: Path) -> str:
    """โหลด parquet ของปีเข้าตาราง duckdb บนดิสก์ครั้งเดียว (พร้อมคอลัมน์ shape ที่คำนวณล่วงหน้า)"""
    tbl = f"t_{year}"
    exists = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [tbl]).fetchone()[0]
    if not exists:
        con.execute(f"CREATE TABLE {tbl} AS SELECT * FROM read_parquet({str(pq_dir / '*.parquet')!r})")
    return tbl


def profile_year(con, year: str, pq_dir: Path, out: dict, checkpoint: Path) -> None:
    """แต่ละส่วนบันทึกลง checkpoint json → ถ้าโดนตัดกลางคันรันใหม่จะทำต่อจากส่วนที่ค้าง"""
    if checkpoint.exists():
        out.update(json.loads(checkpoint.read_text(encoding="utf-8")))
    tbl = load_year_table(con, year, pq_dir)
    extra = [f"{_shape(c)} AS {q('shape_' + c)}" for c in DATE_COLS + ["เลขนิติบุคคล"]]
    extra += [f"{_num(c)} AS {q('num_' + c)}" for c in NUM_COLS]
    con.execute(f"CREATE OR REPLACE VIEW t AS SELECT *, {', '.join(extra)} FROM {tbl}")

    def step(key, fn):
        if key in out:
            return
        t0 = time.time()
        out[key] = fn()
        checkpoint.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"    {key}: {time.time()-t0:.0f}s", flush=True)

    step("n_rows", lambda: con.execute("SELECT count(*) FROM t").fetchone()[0])
    step("rows_per_file", lambda: dict(con.execute("SELECT _src_file, count(*) FROM t GROUP BY 1 ORDER BY 1").fetchall()))
    step("distinct", lambda: {
        "รหัสโครงการ": con.execute('SELECT count(DISTINCT "รหัสโครงการ") FROM t').fetchone()[0],
        "เลขที่สัญญา": con.execute('SELECT count(DISTINCT "เลขที่สัญญา") FROM t').fetchone()[0],
        "(รหัสโครงการ,เลขที่สัญญา)": con.execute('SELECT count(*) FROM (SELECT DISTINCT "รหัสโครงการ","เลขที่สัญญา" FROM t)').fetchone()[0],
        "(รหัสโครงการ,เลขที่สัญญา,เลขนิติบุคคล)": con.execute('SELECT count(*) FROM (SELECT DISTINCT "รหัสโครงการ","เลขที่สัญญา","เลขนิติบุคคล" FROM t)').fetchone()[0],
        "แถวซ้ำทั้งแถว(ไม่รวมลำดับ,_src_file)": out["n_rows"] - con.execute(
            "SELECT count(*) FROM (SELECT DISTINCT " + ", ".join(q(c) for c in CANON if c != "ลำดับ") + " FROM t)").fetchone()[0],
    })

    def null_share():
        nulls = con.execute("SELECT " + ", ".join(
            f"sum(CASE WHEN {q(c)} IS NULL OR trim({q(c)})='' OR {q(c)}='-' THEN 1 ELSE 0 END)" for c in CANON) + " FROM t").fetchone()
        return {c: round(v / out["n_rows"], 4) for c, v in zip(CANON, nulls)}
    step("null_share", null_share)

    def categories():
        r = {}
        for c in CAT_COLS:
            rows = con.execute(f"SELECT {q(c)}, count(*) n FROM t GROUP BY 1 ORDER BY n DESC LIMIT 40").fetchall()
            r[c] = {"n_distinct": con.execute(f"SELECT count(DISTINCT {q(c)}) FROM t").fetchone()[0], "top": rows}
        return r
    step("categories", categories)
    step("method_x_group", lambda: con.execute(
        'SELECT "วิธีจัดซื้อฯ", "กลุ่มวิธีจัดซื้อฯ", count(*) n FROM t GROUP BY 1,2 ORDER BY n DESC LIMIT 40').fetchall())
    step("method_x_group_by_file", lambda: con.execute(
        'SELECT _src_file, "วิธีจัดซื้อฯ", count(*) n FROM t GROUP BY 1,2 QUALIFY row_number() OVER (PARTITION BY _src_file ORDER BY n DESC) <= 3 ORDER BY 1, n DESC').fetchall())
    step("date_shapes", lambda: {c: con.execute(
        f"SELECT {q('shape_' + c)}, count(*) n, min({q(c)}) ex FROM t GROUP BY 1 ORDER BY n DESC LIMIT 15").fetchall() for c in DATE_COLS})
    step("date_examples", lambda: {c: [r[0] for r in con.execute(
        f"SELECT {q(c)} FROM t WHERE {q(c)} IS NOT NULL USING SAMPLE 12").fetchall()] for c in DATE_COLS})
    step("taxid_shapes", lambda: con.execute(
        f"SELECT {q('shape_เลขนิติบุคคล')}, count(*) n, min(\"เลขนิติบุคคล\") ex FROM t GROUP BY 1 ORDER BY n DESC LIMIT 20").fetchall())
    step("taxid_bad_examples", lambda: con.execute(
        """SELECT "เลขนิติบุคคล", "ชื่อผู้ชนะ" FROM t WHERE "เลขนิติบุคคล" IS NOT NULL
           AND "shape_เลขนิติบุคคล" <> '9999999999999' USING SAMPLE 15""").fetchall())

    def numeric():
        r = {}
        for c in NUM_COLS:
            n = q("num_" + c)
            row = con.execute(
                f"""SELECT count({q(c)}), sum(CASE WHEN {n} IS NULL AND {q(c)} IS NOT NULL AND trim({q(c)})<>'' THEN 1 ELSE 0 END),
                           min({n}), max({n}), median({n}), sum(CASE WHEN {n} <= 0 THEN 1 ELSE 0 END) FROM t""").fetchone()
            r[c] = dict(zip(["non_null", "unparsable", "min", "max", "median", "nonpositive"], row))
            r[c]["unparsable_examples"] = [x[0] for x in con.execute(
                f"SELECT {q(c)} FROM t WHERE {n} IS NULL AND {q(c)} IS NOT NULL AND trim({q(c)})<>'' USING SAMPLE 8").fetchall()]
        return r
    step("numeric", numeric)
    step("near_500k", lambda: con.execute(
        'SELECT count(*) FROM t WHERE "num_ราคาตกลงซื้อ/จ้าง" BETWEEN 450000 AND 500000').fetchone()[0])
    step("top_agencies", lambda: con.execute('SELECT "ชื่อหน่วยงาน", count(*) n FROM t GROUP BY 1 ORDER BY n DESC LIMIT 15').fetchall())
    step("top_winners", lambda: con.execute(
        """SELECT "เลขนิติบุคคล", any_value("ชื่อผู้ชนะ"), count(*) n FROM t
           WHERE "shape_เลขนิติบุคคล" = '9999999999999' AND "เลขนิติบุคคล" <> '0000000000000' GROUP BY 1 ORDER BY n DESC LIMIT 15""").fetchall())


def profile_winner(con, out: dict) -> None:
    files = sorted((RAW / "winner").glob("*.csv"))
    con.execute(
        "CREATE OR REPLACE VIEW w AS " + " UNION ALL ".join(
            f"SELECT {q('ลำดับ')}, {q('เลขประจำตัวนิติบุคคล')}, {q('ผู้ชนะการเสนอราคา')}, '{f.name}' AS filename FROM {csv_source(f)}"
            for f in files)
    )
    out["encodings"] = {f.name: detect_encoding(f) for f in files}
    out["columns"] = [read_header(f) for f in files][0]
    out["n_rows"] = con.execute("SELECT count(*) FROM w").fetchone()[0]
    out["rows_per_file"] = dict(con.execute("SELECT filename, count(*) FROM w GROUP BY 1 ORDER BY 1").fetchall())
    out["taxid_shapes"] = con.execute(
        """SELECT regexp_replace(regexp_replace("เลขประจำตัวนิติบุคคล", '[0-9]', '9', 'g'), '[A-Za-z]', 'a', 'g') shape,
                  count(*) n, min("เลขประจำตัวนิติบุคคล") ex FROM w GROUP BY 1 ORDER BY n DESC LIMIT 20"""
    ).fetchall()
    out["distinct_valid_taxid"] = con.execute(
        """SELECT count(DISTINCT "เลขประจำตัวนิติบุคคล") FROM w WHERE regexp_matches("เลขประจำตัวนิติบุคคล", '^[0-9]{13}$')"""
    ).fetchone()[0]
    out["distinct_names"] = con.execute('SELECT count(DISTINCT "ผู้ชนะการเสนอราคา") FROM w').fetchone()[0]
    out["dup_name_multi_taxid"] = con.execute(
        """SELECT count(*) FROM (SELECT "ผู้ชนะการเสนอราคา" FROM w
           WHERE regexp_matches("เลขประจำตัวนิติบุคคล", '^[0-9]{13}$') GROUP BY 1 HAVING count(DISTINCT "เลขประจำตัวนิติบุคคล") > 1)"""
    ).fetchone()[0]
    out["name_prefix"] = con.execute(
        """SELECT CASE WHEN "ผู้ชนะการเสนอราคา" LIKE 'บริษัท%' THEN 'บริษัท'
                       WHEN "ผู้ชนะการเสนอราคา" LIKE 'ห้างหุ้นส่วน%' OR "ผู้ชนะการเสนอราคา" LIKE 'หจก%' THEN 'ห้างหุ้นส่วน'
                       WHEN "ผู้ชนะการเสนอราคา" LIKE 'ร้าน%' THEN 'ร้าน'
                       WHEN "ผู้ชนะการเสนอราคา" LIKE 'นาย%' OR "ผู้ชนะการเสนอราคา" LIKE 'นาง%' OR "ผู้ชนะการเสนอราคา" LIKE 'น.ส.%' THEN 'บุคคลธรรมดา'
                       ELSE 'อื่น ๆ' END k, count(*) FROM w GROUP BY 1 ORDER BY 2 DESC"""
    ).fetchall()


# ---------------------------------------------------------------- report
def fmt_rows(rows, headers):
    s = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") if x is not None else "∅" for x in r) + " |\n"
    return s


def write_report(year: str, p: dict, stage_log: dict, sample) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"profile_raw_{year}.json").write_text(json.dumps({"stage": stage_log, "profile": p}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md = [f"# Profile ข้อมูลดิบสัญญา e-GP ปีงบ {year}", ""]
    md.append(f"สร้างเมื่อ {time.strftime('%Y-%m-%d %H:%M')}" + (f" — **ตัวอย่าง {sample:,} แถวแรกต่อไฟล์**" if sample else " — ข้อมูลเต็ม"))
    md += ["", "## ไฟล์และ header", fmt_rows(
        [(k, v.get("rows"), v.get("n_header"), v.get("blank_cols"), ", ".join(v.get("missing", [])) or "-", ", ".join(v.get("unknown", [])) or "-", v.get("sec"))
         for k, v in stage_log["files"].items()],
        ["ไฟล์", "แถว", "คอลัมน์ใน header", "คอลัมน์ว่าง", "คอลัมน์ที่ขาด", "คอลัมน์แปลกปลอม", "วินาที"])]
    d = p["distinct"]
    md += ["", "## จำนวนแถวและคีย์", f"- แถวทั้งหมด: **{p['n_rows']:,}**"] + [f"- {k}: {v:,}" for k, v in d.items()]
    md += ["", "## สัดส่วนค่าว่าง (NULL / '' / '-')", fmt_rows(sorted(p["null_share"].items(), key=lambda x: -x[1]), ["คอลัมน์", "สัดส่วนว่าง"])]
    md += ["", "## คอลัมน์หมวดหมู่"]
    for c, v in p["categories"].items():
        md += [f"### {c} (distinct = {v['n_distinct']:,})", fmt_rows(v["top"], ["ค่า", "n"])]
    md += ["## วิธีจัดซื้อฯ × กลุ่มวิธีจัดซื้อฯ (ตรวจการสลับคอลัมน์)", fmt_rows(p["method_x_group"], ["วิธีจัดซื้อฯ", "กลุ่มวิธีจัดซื้อฯ", "n"])]
    md += ["### ค่า วิธีจัดซื้อฯ ที่พบมากสุด 3 อันดับ ต่อไฟล์", fmt_rows(p["method_x_group_by_file"], ["ไฟล์", "วิธีจัดซื้อฯ", "n"])]
    md += ["", "## รูปแบบวันที่ (9=ตัวเลข, ก=อักษรไทย, a=อักษรละติน)"]
    for c, rows in p["date_shapes"].items():
        md += [f"### {c}", fmt_rows(rows, ["รูปแบบ", "n", "ตัวอย่าง"]), f"ตัวอย่างสุ่ม: {p['date_examples'][c]}", ""]
    md += ["## รูปแบบเลขนิติบุคคล", fmt_rows(p["taxid_shapes"], ["รูปแบบ", "n", "ตัวอย่าง"]),
           "### ตัวอย่างเลขที่ไม่ใช่ 13 หลัก", fmt_rows(p["taxid_bad_examples"], ["เลขนิติบุคคล", "ชื่อผู้ชนะ"])]
    md += ["", "## คอลัมน์ตัวเลข", fmt_rows(
        [(c, v["non_null"], v["unparsable"], v["min"], v["median"], v["max"], v["nonpositive"], v["unparsable_examples"]) for c, v in p["numeric"].items()],
        ["คอลัมน์", "ไม่ว่าง", "แปลงไม่ได้", "min", "median", "max", "≤0", "ตัวอย่างที่แปลงไม่ได้"])]
    md += [f"- สัญญาที่ราคาตกลงอยู่ในช่วง 450,000–500,000 บาท: **{p['near_500k']:,}** แถว", ""]
    md += ["## หน่วยงานที่มีแถวมากสุด", fmt_rows(p["top_agencies"], ["หน่วยงาน", "n"]),
           "## ผู้ชนะ (เลข 13 หลัก) ที่มีแถวมากสุด", fmt_rows(p["top_winners"], ["เลขนิติบุคคล", "ชื่อ", "n"])]
    (REPORTS / f"profile_raw_{year}.md").write_text("\n".join(md), encoding="utf-8")


def write_winner_report(p: dict) -> None:
    (REPORTS / "profile_raw_winner.json").write_text(json.dumps(p, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md = ["# Profile ข้อมูลดิบ egpwinner", "", f"- คอลัมน์: {p['columns']}", f"- แถวทั้งหมด: **{p['n_rows']:,}**",
          f"- เลขนิติบุคคล 13 หลักที่ไม่ซ้ำ: {p['distinct_valid_taxid']:,}", f"- ชื่อผู้ชนะไม่ซ้ำ: {p['distinct_names']:,}",
          f"- ชื่อเดียวกันแต่มีเลข 13 หลักมากกว่า 1 เลข: {p['dup_name_multi_taxid']:,}", "",
          "## แถวต่อไฟล์", fmt_rows(p["rows_per_file"].items(), ["ไฟล์", "n"]),
          "## รูปแบบเลขประจำตัวนิติบุคคล", fmt_rows(p["taxid_shapes"], ["รูปแบบ", "n", "ตัวอย่าง"]),
          "## ประเภทผู้ชนะจากคำนำหน้าชื่อ", fmt_rows(p["name_prefix"], ["ประเภท", "n"])]
    (REPORTS / "profile_raw_winner.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", default=["2566", "2567", "2568"])
    ap.add_argument("--sample", type=int, default=None, help="จำกัดแถวต่อไฟล์ (ทดลอง)")
    ap.add_argument("--winner", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory", default="2GB")
    ap.add_argument("--db", default=None, help="ที่เก็บไฟล์ duckdb ชั่วคราว (ค่าเริ่มต้น data/interim/stage/profile.duckdb; ใส่ดิสก์ local ถ้า data อยู่บน network drive)")
    args = ap.parse_args()
    if args.sample:
        global STAGE
        STAGE = STAGE.parent / f"stage_sample{args.sample}"
    STAGE.mkdir(parents=True, exist_ok=True)
    con = connect(args.threads, args.memory, Path(args.db) if args.db else STAGE / "profile.duckdb")
    for y in args.years:
        print(f"== ปี {y}", flush=True)
        stage_log: dict = {}
        pq_dir = stage_year(con, y, args.sample, stage_log)
        prof: dict = {}
        t0 = time.time()
        profile_year(con, y, pq_dir, prof, STAGE / f"profile_{y}.checkpoint.json")
        print(f"  profiled in {time.time()-t0:.0f}s: {prof['n_rows']:,} rows", flush=True)
        write_report(y, prof, stage_log, args.sample)
    if args.winner:
        print("== egpwinner", flush=True)
        prof = {}
        profile_winner(con, prof)
        write_winner_report(prof)
    print("done", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
