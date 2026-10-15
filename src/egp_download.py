#!/usr/bin/env python3
"""
ดาวน์โหลดข้อมูล e-GP (สัญญาจัดซื้อจัดจ้างภาครัฐ) และทะเบียนผู้ชนะ (egpwinner) จาก data.go.th

ใช้งาน:
    pip install requests
    python egp_download.py --list                      # ดูว่ามีปีงบไหน ไฟล์อะไรบ้าง (ไม่ดาวน์โหลด)
    python egp_download.py --years 2566 2567 2568      # ดาวน์โหลดสัญญาปีที่เลือก + egpwinner
    python egp_download.py --years all --dest D:/egp   # ทุกปีที่หาเจอ

หมายเหตุ:
- 1 ปีงบประมาณ ~5-6 GB (CSV 10-12 ไฟล์) ตรวจพื้นที่ดิสก์ก่อน
- รันซ้ำได้: ไฟล์ที่โหลดครบแล้วจะข้าม, ไฟล์ที่โหลดค้างจะต่อจากเดิม (HTTP Range)
- สร้าง manifest.csv บันทึก package / resource id / url / ขนาด / เวลาที่โหลด
"""
import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime

import requests

API = "https://data.go.th/api/3/action"
TITLE_PREFIX = "ข้อมูลโครงการจัดซื้อจัดจ้างจากระบบการจัดซื้อจัดจ้างภาครัฐ"
# slug ที่ตรวจพบแล้ว (ตั้งชื่อไม่สม่ำเสมอ: cgd- / cdg- / egp-contact-)
KNOWN_SLUGS = [
    "cgd-contract-2558", "cgd-contract-2559", "cgd-contract-2560", "cgd-contract-2561",
    "cgd-contract-2562", "cgd-contract-2563", "cdg-contract-2564", "cdg-contract-2565",
    "cdg-contract-2566", "cdg-contract-2567", "egp-contact-2568",
]
WINNER_SLUG = "egpwinner"
HEADERS = {"User-Agent": "egp-research-downloader/1.0"}


def api(action, **params):
    """เรียก CKAN API พร้อม retry (data.go.th มี rate limit 429 และ 5xx เป็นช่วง ๆ)"""
    for attempt in range(6):
        try:
            r = requests.get(f"{API}/{action}", params=params, headers=HEADERS, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            data = r.json()
            return data["result"] if data.get("success") else None
        except (requests.RequestException, ValueError) as e:
            wait = 10 * (attempt + 1)
            print(f"  ! {action} ล้มเหลว ({e}) รอ {wait}s แล้วลองใหม่", file=sys.stderr)
            time.sleep(wait)
    return None


def year_of(pkg):
    m = re.search(r"25\d\d", pkg.get("title", "")) or re.search(r"25\d\d", pkg["name"])
    return m.group(0) if m else None


def find_contract_packages():
    """รวม slug ที่รู้จัก + ค้นจากชื่อชุดข้อมูล เพื่อเก็บปีที่ยังไม่รู้ slug"""
    pkgs = {}
    for slug in KNOWN_SLUGS:
        p = api("package_show", id=slug)
        if p:
            pkgs[p["name"]] = p
        time.sleep(1)
    res = api("package_search", q=f'title:"{TITLE_PREFIX}"', rows=100)
    for p in (res or {}).get("results", []):
        t = p.get("title", "")
        if t.startswith(TITLE_PREFIX) and "ผลสรุป" not in t and year_of(p):
            pkgs.setdefault(p["name"], p)
    by_year = {}
    for p in pkgs.values():
        by_year.setdefault(year_of(p), []).append(p)
    return dict(sorted(by_year.items()))


def download(url, path):
    """สตรีมลงดิสก์ ต่อไฟล์ที่ค้างได้"""
    tmp = path + ".part"
    for attempt in range(8):
        done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        hdr = dict(HEADERS)
        if done:
            hdr["Range"] = f"bytes={done}-"
        try:
            with requests.get(url, headers=hdr, stream=True, timeout=120) as r:
                if r.status_code == 416:  # โหลดครบแล้ว
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                if done and r.status_code == 200:  # server ไม่รองรับ Range → เริ่มใหม่
                    done = 0
                total = int(r.headers.get("Content-Length", 0)) + done
                mode = "ab" if done else "wb"
                with open(tmp, mode) as f:
                    got = done
                    last = time.time()
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        if time.time() - last > 5:
                            pct = f"{got / total:5.1%}" if total else ""
                            print(f"    {got / 1e6:,.0f} MB {pct}", end="\r")
                            last = time.time()
            break
        except requests.RequestException as e:
            wait = 15 * (attempt + 1)
            print(f"\n  ! ดาวน์โหลดสะดุด ({e}) รอ {wait}s แล้วต่อจากเดิม", file=sys.stderr)
            time.sleep(wait)
    else:
        print(f"  ✗ ยอมแพ้: {url}", file=sys.stderr)
        return None
    os.replace(tmp, path)
    return os.path.getsize(path)


def safe_name(s):
    return re.sub(r'[\\/:*?"<>|\s]+', "_", s).strip("_")


def fetch_package(pkg, outdir, manifest, dry):
    os.makedirs(outdir, exist_ok=True)
    for res in pkg.get("resources", []):
        fmt = (res.get("format") or "").upper()
        url = res.get("url", "")
        if fmt not in ("CSV", "XLSX", "ZIP") and not url.lower().endswith((".csv", ".xlsx", ".zip")):
            continue
        ext = os.path.splitext(url.split("?")[0])[1] or f".{fmt.lower()}"
        fname = safe_name(res.get("name") or res["id"])
        if not fname.lower().endswith(ext.lower()):
            fname += ext
        path = os.path.join(outdir, fname)
        size_meta = res.get("size")
        print(f"  - {fname}  ({(size_meta or 0) / 1e6:,.0f} MB)")
        if dry:
            continue
        if os.path.exists(path) and (not size_meta or os.path.getsize(path) == size_meta):
            print("    ✓ มีแล้ว ข้าม")
            size = os.path.getsize(path)
        else:
            size = download(url, path)
            print(f"    ✓ เสร็จ {size / 1e6:,.0f} MB" if size else "    ✗ ไม่สำเร็จ")
        manifest.writerow([pkg["name"], pkg.get("title", ""), res["id"], res.get("name", ""),
                           url, size_meta or "", size or "", res.get("last_modified", ""),
                           datetime.now().isoformat(timespec="seconds")])


def main():
    ap = argparse.ArgumentParser(description="ดาวน์โหลด e-GP + egpwinner จาก data.go.th")
    ap.add_argument("--years", nargs="+", default=["2566", "2567", "2568"],
                    help="ปีงบ พ.ศ. เช่น 2566 2567 หรือ all (ค่าเริ่มต้น 2566-2568)")
    ap.add_argument("--dest", default="egp_data", help="โฟลเดอร์ปลายทาง")
    ap.add_argument("--list", action="store_true", help="แสดงรายการไฟล์อย่างเดียว ไม่ดาวน์โหลด")
    ap.add_argument("--no-winner", action="store_true", help="ไม่ต้องโหลด egpwinner")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="slug แพ็กเกจอื่นที่ต้องการโหลดเพิ่ม เช่น summary_cgdcontract (เก็บที่ <dest>/<slug>)")
    args = ap.parse_args()

    print("ค้นหาแพ็กเกจสัญญา e-GP ...")
    by_year = find_contract_packages()
    print("พบปีงบ:", ", ".join(by_year) or "(ไม่พบ)")
    years = list(by_year) if args.years == ["all"] else args.years

    os.makedirs(args.dest, exist_ok=True)
    mpath = os.path.join(args.dest, "manifest.csv")
    new = not os.path.exists(mpath)
    with open(mpath, "a", newline="", encoding="utf-8-sig") as mf:
        manifest = csv.writer(mf)
        if new:
            manifest.writerow(["package", "title", "resource_id", "resource_name", "url",
                               "size_meta", "size_downloaded", "last_modified", "downloaded_at"])
        for y in years:
            for pkg in by_year.get(y, []):
                print(f"\n[{y}] {pkg['name']} — {len(pkg.get('resources', []))} resources")
                fetch_package(pkg, os.path.join(args.dest, f"contract_{y}"), manifest, args.list)
            if y not in by_year:
                print(f"\n[{y}] ไม่พบแพ็กเกจ — ลองค้นชื่อ '{TITLE_PREFIX} ปีงบประมาณ {y}' บนเว็บ")
        for slug in args.extra:
            pk = api("package_show", id=slug)
            if pk:
                print(f"\n[extra] {slug} — {len(pk.get('resources', []))} resources")
                fetch_package(pk, os.path.join(args.dest, slug), manifest, args.list)
            else:
                print(f"\n[extra] ไม่พบแพ็กเกจ {slug}")
        if not args.no_winner:
            w = api("package_show", id=WINNER_SLUG)
            if w:
                print(f"\n[winner] {WINNER_SLUG}")
                fetch_package(w, os.path.join(args.dest, "winner"), manifest, args.list)

    print(f"\nเสร็จ — ดูรายการไฟล์ที่ {mpath}")


if __name__ == "__main__":
    main()
