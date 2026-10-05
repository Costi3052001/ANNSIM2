#!/usr/bin/env python
"""Run the whole study A-Z as a queue of stages.

Each stage runs as a subprocess with its own log.  A failing stage is recorded
and the queue **continues** with the next one (later stages work with whatever
earlier stages produced).  Status is persisted to ``results/pipeline_status.json``
so that ``--resume`` after an interruption (e.g. a machine restart) skips the
stages that already succeeded.  Training and evaluation are themselves
resumable/incremental, so re-running them only does the missing work.

Usage:
    python scripts/pipeline.py                 # full queue
    python scripts/pipeline.py --resume        # continue after an interruption
    python scripts/pipeline.py --only analyze paper
    python scripts/pipeline.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results" / "pipeline_status.json"
LOGS = ROOT / "results" / "logs"


def stages(cfg: str, jobs: int):
    py = sys.executable
    return [
        ("tests", [py, "-m", "pytest", "-q"], ROOT),
        ("calibrate", [py, "scripts/calibrate.py"], ROOT),
        ("train", [py, "scripts/run_experiments.py", "--config", cfg, "--jobs", str(jobs)], ROOT),
        ("evaluate", [py, "scripts/evaluate_all.py", "--config", cfg, "--jobs", str(jobs)], ROOT),
        ("analyze", [py, "scripts/analyze.py", "--config", cfg], ROOT),
        ("sensitivity", [py, "scripts/sensitivity.py", "--config", cfg, "--jobs", str(jobs)], ROOT),
        ("shield_variants", [py, "scripts/shield_variants.py", "--config", cfg, "--jobs", str(jobs)], ROOT),
        ("paper", ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], ROOT / "paper"),
    ]


def load_status():
    if STATUS.exists():
        return json.loads(STATUS.read_text())
    return {"stages": {}}


def save_status(st):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2))
    tmp.replace(STATUS)


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--resume", action="store_true", help="skip stages that already succeeded")
    ap.add_argument("--only", nargs="*", help="run only these stages")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    queue = stages(args.config, args.jobs)
    if args.list:
        for name, cmd, cwd in queue:
            print(f"{name:16s} {' '.join(cmd)}  (cwd={cwd.relative_to(ROOT) if cwd != ROOT else '.'})")
        return
    if args.only:
        queue = [q for q in queue if q[0] in args.only]

    st = load_status() if args.resume else {"stages": {}}
    st["started"] = st.get("started") if args.resume else now()
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] {len(queue)} stages queued: {', '.join(q[0] for q in queue)}", flush=True)

    for name, cmd, cwd in queue:
        prev = st["stages"].get(name, {})
        if args.resume and prev.get("status") == "ok":
            print(f"[pipeline] {name}: already ok, skipping", flush=True)
            continue
        st["stages"][name] = {"status": "running", "start": now(), "cmd": " ".join(cmd)}
        save_status(st)
        print(f"[pipeline] {name}: running ... (log: results/logs/{name}.log)", flush=True)
        t0 = time.time()
        with open(LOGS / f"{name}.log", "a") as log:
            log.write(f"\n===== {now()} : {' '.join(cmd)} =====\n")
            log.flush()
            try:
                rc = subprocess.call(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            except Exception as e:  # noqa: BLE001 - e.g. executable missing
                log.write(f"[pipeline] could not start: {e!r}\n")
                rc = -1
        status = "ok" if rc == 0 else "failed"
        st["stages"][name].update(status=status, returncode=rc, end=now(),
                                  minutes=round((time.time() - t0) / 60, 1))
        save_status(st)
        print(f"[pipeline] {name}: {status.upper()} (rc={rc}, {st['stages'][name]['minutes']} min)"
              + ("" if rc == 0 else " -- continuing with next stage"), flush=True)

    st["finished"] = now()
    save_status(st)
    failed = [k for k, v in st["stages"].items() if v.get("status") != "ok"]
    print(f"[pipeline] finished. failed stages: {failed or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
