"""สรุปผล screen_group ทุกปี → reports/screen_group_summary.md  (python src\\summarize_screen_group.py)"""
import json

import numpy as np

from model_iforest import REPORTS

years = [y for y in range(2561, 2569) if (REPORTS / f"screen_group_{y}.json").exists()]
D = {y: json.load(open(REPORTS / f"screen_group_{y}.json", encoding="utf-8")) for y in years}
f2 = lambda v: f"{min(v):.2f}–{max(v):.2f} ({float(np.median(v)):.2f})"
fp = lambda v: f"{min(v):.1%}–{max(v):.1%} ({float(np.median(v)):.1%})"
L = [f"# สรุปตัวคัดกรองระดับการซื้อ (ceiling-aware) ปีงบ {years[0]}–{years[-1]} ({len(years)} ปี)", "",
     "ค่า = ต่ำสุด–สูงสุด (มัธยฐาน) ข้ามปี; budget80 = สัดส่วนสัญญาที่ต้องตรวจเพื่อพบ 80% ของการซื้อที่ฉีด", "",
     "- สัญญาในกลุ่มวันที่ยอดรวม > 500k: " + fp([D[y]["base_share_day_split"] for y in years])
     + "; กลุ่มเดือน: " + fp([D[y]["base_share_month_split"] for y in years]),
     "- HBOS ที่ fit ใหม่ตรงกับคะแนนที่บันทึกไว้: " + ", ".join(f"{y}: {D[y]['hbos_refit_ok']}" for y in years), "",
     "| สถานการณ์ | ตัวคัดกรอง | recall@top1% | recall@top5% | budget80 |", "|---|---|---|---|---|"]
for sc in D[years[0]]["injection"]:
    for m in D[years[0]]["injection"][sc]:
        g = lambda k: [D[y]["injection"][sc][m][k] for y in years]
        L.append(f"| {sc} | {m} | {f2(g('recall_top1pct'))} | {f2(g('recall_top5pct'))} | {fp(g('budget_for_recall80'))} |")
rules = [r for r in D[years[0]]["rule_base_rate"] if r != "r_price_z_peer_gt3"]
L += ["", "## Lift ของ red flag จริงใน top1%", "", "| ตัวคัดกรอง | " + " | ".join(r[2:] for r in rules) + " |", "|---" * (len(rules) + 1) + "|"]
for m in D[years[0]]["rule_lift_top1pct"]:
    L.append(f"| {m} | " + " | ".join(f"{min(D[y]['rule_lift_top1pct'][m][r] for y in years):.1f}–{max(D[y]['rule_lift_top1pct'][m][r] for y in years):.1f}" for r in rules) + " |")
L += ["", "หมายเหตุ: lift ของ split_flag ใน G-GATE ถูกจำกัดที่ 1/อัตราพื้นฐาน (top1% เป็นกลุ่มที่แบ่งทั้งหมด) — วนกลับโดยนิยาม",
      "", "Jaccard top1%: " + "; ".join(f"{k} " + f2([D[y]["jaccard_top1pct"][k] for y in years]) for k in D[years[-1]]["jaccard_top1pct"])]
(REPORTS / "screen_group_summary.md").write_text("\n".join(L), encoding="utf-8")
print("→ reports/screen_group_summary.md", years)
