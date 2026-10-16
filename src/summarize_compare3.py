"""สรุปผล model_compare3 ทุกปี → reports/model_compare3_summary.md  (python src\\summarize_compare3.py)"""
import json

import numpy as np

from model_iforest import REPORTS

years = [y for y in range(2561, 2569) if (REPORTS / f"model_compare3_{y}.json").exists()]
D = {y: json.load(open(REPORTS / f"model_compare3_{y}.json", encoding="utf-8")) for y in years}
rng = lambda v: f"{min(v):.2f}–{max(v):.2f} ({float(np.median(v)):.2f})"
L = [f"# สรุปเทียบโมเดลรอบ 3 (injection แบบสอดคล้อง) ปีงบ {years[0]}–{years[-1]} ({len(years)} ปี)", "",
     "ค่าในตาราง = ต่ำสุด–สูงสุด (มัธยฐาน) ข้ามปี", "",
     "- AE iterations: " + ", ".join(f"{y}: {D[y]['ae']['n_iter']}/{D[y]['ae']['max_iter']}" for y in years),
     "- LOF ใช้คะแนนรอบ 2: " + ", ".join(f"{y}: {D[y]['lof_reused']}" for y in years), ""]
for key, title in [("inj_consistent", "Injection แบบสอดคล้อง"), ("inj_naive", "Injection แบบเดิม (เขียนทับฟีเจอร์)")]:
    L += [f"## {title}", "", "| สถานการณ์ | โมเดล | recall@top1% | recall@top5% |", "|---|---|---|---|"]
    for sc in D[years[0]][key]:
        for m in D[years[0]][key][sc]:
            r1 = [D[y][key][sc][m]["recall_top1pct"] for y in years]; r5 = [D[y][key][sc][m]["recall_top5pct"] for y in years]
            L.append(f"| {sc} | {m} | {rng(r1)} | {rng(r5)} |")
    L.append("")
rules = [r for r in D[years[0]]["rule_base_rate"] if r != "r_price_z_peer_gt3"]
L += ["## Lift ของ red flag จริงใน top1% (AE iter 300)", "", "| โมเดล | " + " | ".join(r[2:] for r in rules) + " |", "|---" * (len(rules) + 1) + "|"]
for m in D[years[0]]["rule_lift_top1pct"]:
    L.append(f"| {m} | " + " | ".join(f"{min(D[y]['rule_lift_top1pct'][m][r] for y in years):.1f}–{max(D[y]['rule_lift_top1pct'][m][r] for y in years):.1f}" for r in rules) + " |")
L += ["", "## Jaccard top1%", ""] + [f"- {k}: " + rng([D[y]["jaccard_top1pct"].get(k, np.nan) for y in years]) for k in D[years[-1]]["jaccard_top1pct"]]
L += ["", "## รายละเอียดสถานการณ์ (ปีล่าสุด)", "", "```", json.dumps(D[years[-1]]["scenario_meta"], ensure_ascii=False, indent=1), "```"]
(REPORTS / "model_compare3_summary.md").write_text("\n".join(L), encoding="utf-8")
print("→ reports/model_compare3_summary.md", years)
