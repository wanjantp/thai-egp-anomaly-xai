"""Compare two reports/ folders (e.g. PC results vs a rerun on Chopin), number by number.

  python foundation/compare_reports.py A_DIR B_DIR --out B_DIR/_vs_pc.md [--only model_ screen_group]

For each .md/.json/.csv file present in both folders: identical, or how many numbers differ and the
largest relative difference. Files in only one folder are listed. Standard library only.
"""
import argparse
import re
from pathlib import Path

NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def numbers(text):
    out = []
    for m in NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="file name prefixes to include")
    x = ap.parse_args()
    A, B = Path(x.a), Path(x.b)

    def listing(root):
        return {p.relative_to(root).as_posix() for p in root.rglob("*")
                if p.is_file() and p.suffix in (".md", ".json", ".csv")
                and (not x.only or any(p.name.startswith(o) for o in x.only))}

    la, lb = listing(A), listing(B)
    rows = []
    for f in sorted(la & lb):
        ta = (A / f).read_text(encoding="utf-8", errors="replace")
        tb = (B / f).read_text(encoding="utf-8", errors="replace")
        if ta == tb:
            rows.append((0.0, f, "identical", ""))
            continue
        na, nb = numbers(ta), numbers(tb)
        if len(na) != len(nb):
            rows.append((float("inf"), f, "structure differs", f"{len(na)} vs {len(nb)} numbers"))
            continue
        diffs = [(abs(p - q) / max(abs(p), abs(q), 1e-12), p, q) for p, q in zip(na, nb) if p != q]
        if not diffs:
            rows.append((0.0, f, "same numbers, text differs", ""))
            continue
        worst = max(diffs)
        rows.append((worst[0], f, f"{len(diffs)}/{len(na)} numbers differ",
                     f"max rel diff {worst[0]:.2e} ({worst[1]:g} vs {worst[2]:g})"))
    rows.sort(key=lambda r: -r[0])
    n_id = sum(1 for r in rows if r[2] == "identical")
    lines = [f"# Reports: {A} vs {B}", "",
             f"{len(rows)} files in both: {n_id} identical, {len(rows) - n_id} differ. "
             f"Only in A: {len(la - lb)}. Only in B: {len(lb - la)}.", "",
             "| file | result | detail |", "|---|---|---|"]
    lines += [f"| {f} | {r} | {d} |" for _, f, r, d in rows]
    if la - lb:
        lines += ["", "## Only in A", ""] + [f"- {f}" for f in sorted(la - lb)]
    if lb - la:
        lines += ["", "## Only in B", ""] + [f"- {f}" for f in sorted(lb - la)]
    Path(x.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
