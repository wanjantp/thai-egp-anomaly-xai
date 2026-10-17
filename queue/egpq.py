#!/usr/bin/env python3
"""egpq: unattended job queue for Chopin (standard library only).

A queue file lists jobs in lanes. Each lane runs its jobs one at a time, in file order;
lanes run in parallel. A job starts when the jobs it depends on are done. When a job fails
or times out, its log is kept, the jobs that depend on it are skipped, and the lane moves on.
State is saved after every change, so a restarted runner resumes where it stopped.

Commands (EGP defaults to ~/egp; settings are read from $EGP/egpq.env):
  egpq.py run QUEUE [--bg] [--retry-failed] [--only-if-started]   start or resume (one runner per queue)
  egpq.py start QUEUE                          clear a stop, then run in the background
  egpq.py status [QUEUE]                       what is running, done, failed, waiting
  egpq.py log JOB [-n 60] [--queue QUEUE]      tail of a job log
  egpq.py pause | resume                       stop / allow starting new jobs (running jobs continue)
  egpq.py stop QUEUE                           kill running jobs and keep the runner from restarting
  egpq.py notify-test                          send a test message to the phone

Queue file: one job per line, 6 TAB-separated columns
  id  lane  after(comma list or -)  timeout_hours  cwd  bash command
Lane lines:  #!lane NAME threads=4 cpus=0-3 [gpu=0]
Each job gets: its own TMPDIR ($EGP/scratch/q/<job>), OMP/MKL/OPENBLAS threads = lane threads,
CPU affinity = lane cpus, CUDA_VISIBLE_DEVICES = lane gpu (empty for CPU lanes), EGPQ_THREADS.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

EGP = Path(os.environ.get("EGP", str(Path.home() / "egp"))).expanduser()
ENVFILE = EGP / "egpq.env"
RUNS = EGP / "runs" / "queue"
PAUSE = EGP / "PAUSE"
FINAL = {"done", "failed", "timeout", "skipped"}
BAD = {"failed", "timeout", "skipped"}
MIN_FREE_GB = float(os.environ.get("EGPQ_MIN_FREE_GB", "40"))


def load_envfile():
    if ENVFILE.exists():
        for line in ENVFILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), os.path.expandvars(v.strip().strip('"')))


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def notify(title, msg, prio=3, tags=""):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    head = {"Title": title.encode("ascii", "replace").decode(), "Priority": str(prio)}
    if tags:
        head["Tags"] = tags
    for attempt in range(3):
        try:
            urllib.request.urlopen(urllib.request.Request(f"{server}/{topic}", data=msg.encode("utf-8"), headers=head),
                                   timeout=20).read()
            return
        except Exception as e:  # network blips must never stop the queue
            err = e
            time.sleep(5 * (attempt + 1))
    print(f"[{now()}] ntfy failed: {err}", flush=True)


def parse_cpus(s):
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def parse_queue(path):
    lanes, jobs, ids = {}, [], set()
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("#!lane"):
            parts = line.split()[1:]
            lanes[parts[0]] = dict(p.split("=", 1) for p in parts[1:])
            continue
        if line.lstrip().startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 6:
            sys.exit(f"{path}:{n}: need 6 TAB-separated columns, found {len(cols)}")
        jid, lane, after, tmo, cwd, cmd = (c.strip() for c in cols)
        deps = [d for d in after.split(",") if d and d != "-"]
        if jid in ids:
            sys.exit(f"{path}:{n}: duplicate job id {jid}")
        if lane not in lanes:
            sys.exit(f"{path}:{n}: lane {lane} not declared with #!lane")
        for d in deps:
            if d not in ids:
                sys.exit(f"{path}:{n}: {jid} depends on {d}, which is not defined above it")
        ids.add(jid)
        jobs.append({"id": jid, "lane": lane, "after": deps, "timeout_h": float(tmo), "cwd": cwd, "cmd": cmd})
    return lanes, jobs


class Runner:
    def __init__(self, qfile, retry_failed=False):
        self.qfile = Path(qfile).resolve()
        self.name = self.qfile.stem
        self.lanes, self.jobs = parse_queue(self.qfile)
        self.byid = {j["id"]: j for j in self.jobs}
        self.dir = RUNS / self.name
        (self.dir / "jobs").mkdir(parents=True, exist_ok=True)
        self.sfile = self.dir / "state.json"
        self.state = json.loads(self.sfile.read_text()) if self.sfile.exists() else {}
        self.lock = threading.Lock()
        self.procs = {}
        self.stopping = False
        self.resumed = []
        for j in self.jobs:
            s = self.state.setdefault(j["id"], {"status": "pending"})
            if s["status"] in ("running", "interrupted"):
                self.resumed.append(j["id"])
                s["status"] = "pending"
            if retry_failed and s["status"] in BAD:
                s.update(status="pending", reason=None, rc=None)
        self.save()

    def save(self):
        with self.lock:
            tmp = self.sfile.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=1, ensure_ascii=False))
            tmp.replace(self.sfile)
            (self.dir / "status.txt").write_text(render_status(self), encoding="utf-8")

    def set(self, jid, **kw):
        with self.lock:
            self.state[jid].update(kw)
        self.save()

    def counts(self):
        c = {}
        for j in self.jobs:
            c[self.state[j["id"]]["status"]] = c.get(self.state[j["id"]]["status"], 0) + 1
        return c

    def summary(self):
        c = self.counts()
        failed = [j["id"] for j in self.jobs if self.state[j["id"]]["status"] in ("failed", "timeout")]
        running = [f"{j['id']} ({hms(time.time() - self.state[j['id']]['t0'])})" for j in self.jobs
                   if self.state[j["id"]]["status"] == "running"]
        lines = [", ".join(f"{k} {v}" for k, v in sorted(c.items())) + f" / {len(self.jobs)}"]
        if running:
            lines.append("running: " + "; ".join(running))
        if failed:
            lines.append("failed: " + ", ".join(failed[:12]) + (" ..." if len(failed) > 12 else ""))
        lines.append(f"disk free {shutil.disk_usage(EGP).free / 1e9:.0f} GB")
        return "\n".join(lines)

    # ---- one job ----
    def run_job(self, j):
        lane = self.lanes[j["lane"]]
        jid = j["id"]
        log = self.dir / "jobs" / f"{jid}.log"
        cwd = Path(os.path.expandvars(j["cwd"])).expanduser()
        env = os.environ.copy()
        th = lane.get("threads", "2")
        tmp = EGP / "scratch" / "q" / jid
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        env.update(OMP_NUM_THREADS=th, MKL_NUM_THREADS=th, OPENBLAS_NUM_THREADS=th, NUMEXPR_MAX_THREADS=th,
                   EGPQ_THREADS=th, EGPQ_JOB=jid, TMPDIR=str(tmp), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                   PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=lane.get("gpu", ""))
        cpus = lane.get("cpus")
        if cpus:
            parse_cpus(cpus)  # validate
        if not cwd.is_dir():
            self.set(jid, status="failed", rc=None, reason=f"folder not found: {cwd}", t0=time.time(), t1=time.time(),
                     log=str(log))
            log.write_text(f"folder not found: {cwd}\n")
            return "failed"
        commit = subprocess.run(["git", "-C", str(cwd), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip() or "-"
        t0 = time.time()
        self.set(jid, status="running", t0=t0, start=now(), log=str(log), commit=commit, reason=None, rc=None)
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(f"# job {jid}  lane {j['lane']}  host {socket.gethostname()}  start {now()}\n"
                     f"# cwd {cwd}  commit {commit}  threads {th}  cpus {lane.get('cpus', 'all')}  "
                     f"gpu {lane.get('gpu', '-')}  timeout {j['timeout_h']} h\n# cmd {j['cmd']}\n\n")
            fh.flush()
            argv = ["bash", "-c", j["cmd"]]
            if cpus and shutil.which("taskset"):
                argv = ["taskset", "-c", cpus] + argv
            p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            self.procs[jid] = p
            deadline = t0 + j["timeout_h"] * 3600
            timed_out = False
            while p.poll() is None:
                if time.time() > deadline and not timed_out:
                    timed_out = True
                    kill_group(p)
                time.sleep(3)
            self.procs.pop(jid, None)
            rc = p.returncode
            fh.write(f"\n# end {now()}  exit {rc}  elapsed {hms(time.time() - t0)}\n")
        shutil.rmtree(tmp, ignore_errors=True)
        if self.stopping:
            self.set(jid, status="interrupted", rc=rc, t1=time.time())
            return "interrupted"
        st = "timeout" if timed_out else ("done" if rc == 0 else "failed")
        self.set(jid, status=st, rc=rc, t1=time.time(), end=now())
        el = hms(time.time() - t0)
        if st == "done":
            notify(f"done {jid}", f"{el}\n{self.summary()}", prio=2, tags="white_check_mark")
        else:
            tail = log_tail(log, 6)
            notify(f"{st.upper()} {jid}", f"exit {rc} after {el}; dependents skipped, queue continues.\n"
                                          f"log: {log}\n---\n{tail}", prio=4, tags="x")
        return st

    # ---- one lane ----
    def lane_worker(self, lane):
        for j in [j for j in self.jobs if j["lane"] == lane]:
            if self.stopping:
                return
            jid = j["id"]
            if self.state[jid]["status"] in FINAL:
                continue
            self.set(jid, status="waiting")
            while not self.stopping:
                ds = {d: self.state[d]["status"] for d in j["after"]}
                bad = [d for d, s in ds.items() if s in BAD]
                if bad:
                    self.set(jid, status="skipped", reason="dependency not done: " + ", ".join(bad), end=now())
                    break
                if all(s == "done" for s in ds.values()) and not PAUSE.exists() and disk_ok():
                    self.run_job(j)
                    break
                time.sleep(15)

    def heartbeat(self):
        every = float(os.environ.get("EGPQ_HEARTBEAT_H", "8")) * 3600
        while not self.stopping:
            time.sleep(60)
            last = self.state.get("_heartbeat", 0) if isinstance(self.state.get("_heartbeat"), (int, float)) else 0
            if time.time() - last >= every:
                with self.lock:
                    self.state["_heartbeat"] = time.time()
                notify(f"queue {self.name}: status", self.summary() + gpu_line(), prio=3, tags="clipboard")
                self.save()

    def run(self):
        todo = [j for j in self.jobs if self.state[j["id"]]["status"] not in FINAL]
        if not todo:
            print("nothing to do: every job is finished")
            return 0
        (self.dir / "runner.pid").write_text(str(os.getpid()))
        c = self.counts()
        msg = f"{len(todo)} jobs to run, {c.get('done', 0)} already done. Lanes: {', '.join(self.lanes)}"
        if self.resumed:
            msg += f"\nResumed after interruption; restarting: {', '.join(self.resumed)}"
        notify(f"queue {self.name} started on {socket.gethostname()}", msg, prio=3, tags="rocket")
        print(f"[{now()}] {msg}", flush=True)
        signal.signal(signal.SIGTERM, self.on_signal)
        signal.signal(signal.SIGINT, self.on_signal)
        threading.Thread(target=self.heartbeat, daemon=True).start()
        ts = [threading.Thread(target=self.lane_worker, args=(ln,), daemon=True) for ln in self.lanes]
        for t in ts:
            t.start()
        while any(t.is_alive() for t in ts):
            time.sleep(2)
        if self.stopping:
            return 1
        c = self.counts()
        notify(f"queue {self.name} FINISHED", self.summary(), prio=4 if (c.get("failed") or c.get("timeout")) else 3,
               tags="checkered_flag")
        print(f"[{now()}] finished: {c}", flush=True)
        return 0

    def on_signal(self, *_):
        self.stopping = True
        for p in list(self.procs.values()):
            kill_group(p)


def kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
        for _ in range(30):
            if p.poll() is not None:
                return
            time.sleep(2)
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def disk_ok():
    free = shutil.disk_usage(EGP).free / 1e9
    if free >= MIN_FREE_GB:
        return True
    flag = EGP / "scratch" / ".lowdisk_notified"
    if not flag.exists() or time.time() - flag.stat().st_mtime > 6 * 3600:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
        notify("LOW DISK: queue waiting", f"{free:.0f} GB free (< {MIN_FREE_GB:.0f} GB). New jobs wait until space is freed.",
               prio=5, tags="warning")
    return False


def gpu_line():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return f"\nGPU util, mem: {r.stdout.strip()}" if r.returncode == 0 else ""
    except Exception:
        return ""


def log_tail(path, n):
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(l[-200:] for l in lines[-n:])
    except OSError:
        return ""


def render_status(r):
    out = [f"queue {r.name}   {now()}   host {socket.gethostname()}", r.summary() + gpu_line(),
           "PAUSED (new jobs wait; run `egpq resume`)" if PAUSE.exists() else "", ""]
    for lane, opts in r.lanes.items():
        out.append(f"== lane {lane}  ({' '.join(f'{k}={v}' for k, v in opts.items())})")
        for j in [j for j in r.jobs if j["lane"] == lane]:
            s = r.state[j["id"]]
            st = s["status"]
            if st == "running":
                extra = f"{hms(time.time() - s['t0'])} so far | {log_tail(s['log'], 1)[:110]}"
            elif st in ("done", "failed", "timeout") and s.get("t1"):
                extra = hms(s["t1"] - s["t0"]) + (f"  exit {s.get('rc')}" if st != "done" else "")
            elif st == "skipped":
                extra = s.get("reason", "")
            else:
                extra = ""
            mark = {"done": "ok ", "running": ">> ", "failed": "XX ", "timeout": "TO ", "skipped": "-- "}.get(st, "   ")
            out.append(f"  {mark}{j['id']:<34} {st:<9} {extra}")
    return "\n".join(out) + "\n"


def find_queue(name):
    if name:
        p = Path(name)
        return p.stem
    qs = sorted(RUNS.glob("*/state.json"), key=lambda p: p.stat().st_mtime) if RUNS.exists() else []
    if not qs:
        sys.exit("no queue has run yet")
    return qs[-1].parent.name


def main():
    load_envfile()
    os.environ.setdefault("EGP", str(EGP))
    os.environ.setdefault("REPO", str(EGP / "repo"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("queue"); r.add_argument("--bg", action="store_true")
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--only-if-started", action="store_true", help="for cron: do nothing until the queue was started once")
    s = sub.add_parser("start"); s.add_argument("queue"); s.add_argument("--retry-failed", action="store_true")
    st = sub.add_parser("status"); st.add_argument("queue", nargs="?")
    lg = sub.add_parser("log"); lg.add_argument("job"); lg.add_argument("-n", type=int, default=60)
    lg.add_argument("--queue")
    sp = sub.add_parser("stop"); sp.add_argument("queue")
    sub.add_parser("pause"); sub.add_parser("resume"); sub.add_parser("notify-test")
    a = ap.parse_args()

    if a.cmd in ("run", "start"):
        qname = Path(a.queue).stem
        qdir = RUNS / qname
        qdir.mkdir(parents=True, exist_ok=True)
        if a.cmd == "start":
            (qdir / "STOPPED").unlink(missing_ok=True)
        if getattr(a, "only_if_started", False) and not (qdir / "state.json").exists():
            return 0
        if (qdir / "STOPPED").exists():
            print(f"queue {qname} was stopped with `egpq stop`; use `egpq start {a.queue}` to run it again")
            return 0
        if a.cmd == "start" or a.bg:
            args = [sys.executable, os.path.abspath(__file__), "run", str(Path(a.queue).resolve())]
            if a.retry_failed:
                args.append("--retry-failed")
            with open(qdir / "runner.log", "a") as fh:
                p = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True, cwd=str(EGP))
            print(f"runner started in background (pid {p.pid}); log {qdir / 'runner.log'}")
            return 0
        lockf = open(qdir / "runner.lock", "w")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"queue {qname} is already running")
            return 0
        return Runner(a.queue, a.retry_failed).run()

    if a.cmd == "status":
        q = find_queue(a.queue)
        f = RUNS / q / "status.txt"
        lockf = RUNS / q / "runner.lock"
        alive = False
        if lockf.exists():
            with open(lockf, "w") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(fh, fcntl.LOCK_UN)
                except BlockingIOError:
                    alive = True
        print(f"runner: {'RUNNING' if alive else 'not running'}" + ("  (STOPPED by egpq stop)" if (RUNS / q / 'STOPPED').exists() else ""))
        print(f.read_text(encoding="utf-8") if f.exists() else "no status yet")
        return 0
    if a.cmd == "log":
        q = find_queue(a.queue)
        print(log_tail(RUNS / q / "jobs" / f"{a.job}.log", a.n))
        return 0
    if a.cmd == "pause":
        PAUSE.touch(); print("paused: running jobs continue, no new jobs start"); return 0
    if a.cmd == "resume":
        PAUSE.unlink(missing_ok=True); print("resumed"); return 0
    if a.cmd == "stop":
        qname = Path(a.queue).stem
        (RUNS / qname / "STOPPED").write_text(now())
        pid = (RUNS / qname / "runner.pid")
        if pid.exists():
            try:
                os.kill(int(pid.read_text()), signal.SIGTERM)
                print("runner told to stop; running jobs are killed and will rerun on `egpq start`")
            except ProcessLookupError:
                print("runner was not running")
        return 0
    if a.cmd == "notify-test":
        if not os.environ.get("NTFY_TOPIC"):
            sys.exit(f"NTFY_TOPIC is not set (put it in {ENVFILE})")
        notify("egpq test", f"Notifications from {socket.gethostname()} work.", prio=3, tags="wave")
        print("sent"); return 0


if __name__ == "__main__":
    sys.exit(main())
