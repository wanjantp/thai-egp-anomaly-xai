#!/usr/bin/env python3
"""
ตรวจสัญญาที่ใช้วิธีเฉพาะเจาะจง (และคัดเลือก) แม้วงเงินเกิน 500,000 บาท หลัง พ.ร.บ. 2560
— ข้อยกเว้นตามมาตรา 56 วรรคหนึ่ง (1)(ซ) / (2)(ซ) และกฎกระทรวงพัสดุที่รัฐส่งเสริม (หนังสือ กค (กวจ) 0405.2/ว 123, 9 มี.ค. 2561)

คำถาม: กลุ่มนี้ใหญ่แค่ไหน และกระทบตัวเลข "สัญญาที่หายไปเหนือเพดาน" (ช่วง 0.5–2 ล้าน เทียบกับ 2–3 ล้าน) หรือไม่

ใช้งาน (Anaconda Prompt, env egp-xai; ~1–3 นาทีต่อปี):
    python src\\check_specific_above500k.py
ผลลัพธ์: reports/check_specific_above500k.md  (+ .csv)
อ่านเฉพาะ data/interim/contracts_YYYY/*.parquet ไม่แก้ไฟล์ข้อมูลใด ๆ
"""
import sys

import pandas as pd

from bunching import GOV_SQL, INTERIM, KIND_SQL, REFORM, REPORTS, connect, log

BANDS = """CASE WHEN agreed_price <= 500000 THEN 'le_0.5M'
                WHEN agreed_price <= 2000000 THEN '0.5-2M'
                WHEN agreed_price <= 3000000 THEN '2-3M'
                ELSE 'gt_3M' END"""
METH = """CASE WHEN method LIKE '%เฉพาะเจาะจง%' THEN 'specific'
               WHEN method LIKE '%คัดเลือก%' THEN 'selection'
               WHEN method LIKE '%e-bidding%' OR method LIKE 'ประกวดราคา%' THEN 'e-bidding'
               WHEN method LIKE 'ตลาดอิเล็กทรอนิกส์%' OR method LIKE '%e-market%' THEN 'e-market'
               ELSE 'other' END"""


def main():
    con = connect()
    years = sorted(int(p.name.split("_")[1]) for p in INTERIM.glob("contracts_25??") if p.is_dir())
    years = [y for y in years if y >= 2560]
    parts, kinds = [], []
    for y in years:
        src = (INTERIM / f"contracts_{y}" / "*.parquet").as_posix()
        log(f"FY{y} ...")
        base = f"""SELECT *, {BANDS} AS band, {METH} AS meth, {GOV_SQL} AS gov
                   FROM read_parquet('{src}')
                   WHERE sign_date >= DATE '{REFORM}' AND agreed_price > 0"""
        parts.append(con.execute(f"""SELECT {y} AS fy, band, meth, count(*) AS n, sum(agreed_price) AS value
                                     FROM ({base}) GROUP BY ALL""").df())
        kinds.append(con.execute(f"""SELECT {y} AS fy, gov, {KIND_SQL} AS kind, count(*) AS n, sum(agreed_price) AS value
                                     FROM ({base}) WHERE meth = 'specific' AND agreed_price > 500000 GROUP BY ALL""").df())
    d = pd.concat(parts, ignore_index=True)
    k = pd.concat(kinds, ignore_index=True)
    d["fy_ce"] = d["fy"] - 543
    k["fy_ce"] = k["fy"] - 543
    d.to_csv(REPORTS / "check_specific_above500k.csv", index=False, encoding="utf-8-sig")

    L = ["# วิธีเฉพาะเจาะจง/คัดเลือก เหนือ 500,000 บาท หลัง พ.ร.บ. (ข้อยกเว้นตาม ว 123)", "",
         f"สัญญาลงนามตั้งแต่ {REFORM}; ปีงบ {years[0]}–{years[-1]}", ""]
    # 1) สัดส่วนต่อปีในแต่ละช่วงราคา
    L += ["## 1) สัดส่วนวิธีในแต่ละช่วงราคา (จำนวนสัญญา)", "",
          "| ปีงบ (ค.ศ.) | ช่วง | สัญญาทั้งหมด | เฉพาะเจาะจง | % | คัดเลือก | % |", "|---|---|---|---|---|---|---|"]
    for (fy, band), g in d[d.band != "le_0.5M"].groupby(["fy_ce", "band"]):
        tot = g.n.sum()
        sp = g.loc[g.meth == "specific", "n"].sum()
        se = g.loc[g.meth == "selection", "n"].sum()
        L.append(f"| {fy} | {band} | {tot:,} | {sp:,} | {100*sp/tot:.1f} | {se:,} | {100*se/tot:.1f} |")
    # 2) ผลต่อการนับสัญญาที่หายไป: อัตราส่วน 0.5–2M ต่อ 2–3M แบบรวม/ไม่รวมเฉพาะเจาะจง
    L += ["", "## 2) อัตราส่วนสัญญา 0.5–2 ล้าน ต่อ 2–3 ล้าน (ฐานที่ใช้คำนวณ missing mass)", "",
          "| ปีงบ | รวมทุกวิธี | ไม่รวมเฉพาะเจาะจง | ต่างกัน (%) |", "|---|---|---|---|"]
    for fy, g in d.groupby("fy_ce"):
        a = g.loc[g.band == "0.5-2M", "n"].sum() / max(g.loc[g.band == "2-3M", "n"].sum(), 1)
        gs = g[g.meth != "specific"]
        b = gs.loc[gs.band == "0.5-2M", "n"].sum() / max(gs.loc[gs.band == "2-3M", "n"].sum(), 1)
        L.append(f"| {fy} | {a:.2f} | {b:.2f} | {100*(b-a)/a:+.1f} |")
    # 3) เฉพาะเจาะจง > 500k คืออะไร
    kk = k.groupby(["gov", "kind"], as_index=False)[["n", "value"]].sum().sort_values("n", ascending=False)
    L += ["", "## 3) เฉพาะเจาะจง > 500k แยกส่วนกลาง/ท้องถิ่น × ประเภทงาน (รวมทุกปี)", "",
          "| gov | ประเภท | สัญญา | มูลค่า (ล้านบาท) |", "|---|---|---|---|"]
    L += [f"| {r.gov} | {r.kind} | {r.n:,} | {r.value/1e6:,.0f} |" for r in kk.itertuples()]
    out = REPORTS / "check_specific_above500k.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"→ {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
