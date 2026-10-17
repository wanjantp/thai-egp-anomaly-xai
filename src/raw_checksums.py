#!/usr/bin/env python3
"""
SHA-256 checksums of the raw files in data/raw (read-only), so that a replicator who downloads the same resources from
data.go.th (DATA_MANIFEST.csv) can confirm they hold the same bytes. Resumable: files already listed are skipped.

Usage (Anaconda Prompt, env egp-xai):  python src\\raw_checksums.py   (re-run until it reports all files done)
Output: DATA_SHA256.csv (path relative to the repository root, size_bytes, sha256).
"""
import csv
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "DATA_SHA256.csv"


def main(budget_s=150):
    done = {}
    if OUT.exists():
        with OUT.open(encoding="utf-8", newline="") as f:
            done = {r["path"]: r for r in csv.DictReader(f)}
    files = sorted(p for p in RAW.rglob("*") if p.is_file())
    t0 = time.time()
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if rel in done:
            continue
        if time.time() - t0 > budget_s:
            break
        h = hashlib.sha256()
        with p.open("rb") as f:                      # read-only
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
        done[rel] = {"path": rel, "size_bytes": p.stat().st_size, "sha256": h.hexdigest()}
        with OUT.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
            w.writeheader()
            w.writerows(done[k] for k in sorted(done))
    left = sum(p.relative_to(ROOT).as_posix() not in done for p in files)
    print(f"{len(done)} of {len(files)} files hashed; {left} left" + ("" if left else " — all files done"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
