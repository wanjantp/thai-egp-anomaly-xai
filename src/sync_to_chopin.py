"""Copy data folders from this PC to Chopin, resuming after dropped connections.

Uses only the ssh/sftp that ship with Windows (OpenSSH) and the Python standard library.
Each round lists what Chopin already has, then sends only missing files (put) and
finishes partial ones (reput, resume from the byte where the drop happened).
Rounds repeat until every file matches its size on the PC, or --rounds is reached.

Run from Anaconda Prompt in the repo folder:
    python src\\sync_to_chopin.py data/interim data/processed
    python src\\sync_to_chopin.py data/raw            # step 3, 44 GB
    python src\\sync_to_chopin.py data/interim --dry-run
    python src\\sync_to_chopin.py data/processed --exclude _screen_base.parquet   # also send *.joblib

Local  C:\\thai-egp-anomaly-xai\\data\\...  ->  Chopin  ~/egp/data/...
"""
import argparse
import fnmatch
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXCLUDE = ["_screen_base.parquet", "*.joblib", "*.duckdb", "*.duckdb.wal", "sizes_pc.txt"]
SSH_OPTS = ["-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
            "-o", "ConnectTimeout=20", "-o", "BatchMode=yes"]


def local_files(folders, exclude):
    out = {}
    for f in folders:
        base = REPO / f
        if not base.is_dir():
            sys.exit(f"Not a folder on this PC: {base}")
        for p in base.rglob("*"):
            if p.is_file() and not any(fnmatch.fnmatch(p.name, pat) for pat in exclude):
                out[p.relative_to(REPO).as_posix()] = p.stat().st_size
    return out


def remote_sizes(host, remote_root, folders):
    cmd = f"cd {remote_root} 2>/dev/null && find {' '.join(folders)} -type f -printf '%s %p\\n' 2>/dev/null; true"
    r = subprocess.run(["ssh", *SSH_OPTS, host, cmd], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise ConnectionError(r.stderr.strip() or f"ssh exit {r.returncode}")
    sizes = {}
    for line in r.stdout.splitlines():
        size, _, path = line.partition(" ")
        if path:
            sizes[path] = int(size)
    return sizes


def gb(n):
    return f"{n / 1024**3:.2f} GB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="folders relative to the repo, e.g. data/interim")
    ap.add_argument("--host", default="panupong@202.28.93.238")
    ap.add_argument("--remote-root", default="egp", help="folder under the Chopin home that maps to the repo root for data/")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=EXCLUDE, help="file name patterns to skip (default: %(default)s)")
    a = ap.parse_args()
    folders = [f.strip("/\\").replace("\\", "/") for f in a.folders]

    local = local_files(folders, a.exclude)
    total = sum(local.values())
    print(f"PC: {len(local)} files, {gb(total)} in {', '.join(folders)}")

    for rnd in range(1, a.rounds + 1):
        try:
            remote = remote_sizes(a.host, a.remote_root, folders)
        except ConnectionError as e:
            print(f"[round {rnd}] cannot list Chopin ({e}); retry in 30 s")
            time.sleep(30)
            continue
        todo = [(p, s, remote.get(p, 0)) for p, s in sorted(local.items()) if remote.get(p) != s]
        done = total - sum(s for _, s, _ in todo)
        print(f"[round {rnd}] on Chopin: {len(local) - len(todo)}/{len(local)} files OK ({gb(done)} of {gb(total)}); "
              f"{len(todo)} to send")
        if not todo:
            print("ALL FILES MATCH")
            return 0
        if a.dry_run:
            for p, s, r in todo:
                print(f"  {'reput' if 0 < r < s else 'put  '} {p}  PC {s:,}  Chopin {r:,}")
            return 0

        dirs = sorted({str(Path(p).parent.as_posix()) for p, _, _ in todo})
        mk = subprocess.run(["ssh", *SSH_OPTS, a.host,
                             "cd " + a.remote_root + " && mkdir -p " + " ".join(f"'{d}'" for d in dirs)])
        if mk.returncode != 0:
            print(f"[round {rnd}] mkdir failed; retry in 30 s")
            time.sleep(30)
            continue

        batch = ["progress"]
        for p, s, r in todo:
            verb = "reput" if 0 < r < s else "put"   # resume partial file; overwrite missing or wrong-size file
            batch.append(f'{verb} "{(REPO / p).as_posix()}" "{a.remote_root}/{p}"')
        t0 = time.time()
        rc = subprocess.run(["sftp", *SSH_OPTS, "-b", "-", a.host], input="\n".join(batch) + "\n",
                            text=True, encoding="utf-8").returncode
        print(f"[round {rnd}] sftp exit {rc} after {time.time() - t0:.0f} s")
        if rc != 0:
            time.sleep(10)
    print(f"Stopped after {a.rounds} rounds with files still missing; run the same command again to continue.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
