#!/usr/bin/env python
"""Train every (method, seed) run of the protocol in parallel; resumable."""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safeacd.ppo import TrainConfig, train  # noqa: E402


def build_runs(cfg, only=None):
    tr = cfg["train"]
    common = dict(total_steps=tr["total_steps"], attackers=tuple(tr["attackers"]),
                  budgets=tuple(tr["budgets"]), out_dir=cfg["out_dir"], **tr.get("overrides", {}))
    runs = []
    # Seed-major order: every method finishes seed k before any starts seed k+1,
    # so a partially completed grid is still balanced across methods.
    for s in tr["seeds"]:
        for m in tr["methods"]:
            runs.append(TrainConfig(method=m, seed=s, shaping_beta=tr["shaping_beta"], **common))
    sw = cfg.get("shaping_sweep")
    if sw:
        for s in sw["seeds"]:
            for b in sw["betas"]:
                runs.append(TrainConfig(method="shaped", seed=s, shaping_beta=b, **common))
    if only:
        runs = [r for r in runs if r.method in only]
    return runs


def _job(tc: TrainConfig):
    done = Path(tc.out_dir) / tc.run_name / "model.pt"
    if done.exists():
        return f"skip {tc.run_name}"
    train(tc, verbose=True)
    return f"done {tc.run_name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--only", nargs="*", help="restrict to these methods")
    ap.add_argument("--steps", type=int, help="override total_steps (smoke tests)")
    ap.add_argument("--seeds", type=int, nargs="*", help="override seeds")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.steps:
        cfg["train"]["total_steps"] = args.steps
    if args.seeds is not None:
        cfg["train"]["seeds"] = args.seeds
        if cfg.get("shaping_sweep"):
            cfg["shaping_sweep"]["seeds"] = [s for s in cfg["shaping_sweep"]["seeds"] if s in args.seeds]
    runs = build_runs(cfg, args.only)
    print(f"{len(runs)} runs, {args.jobs} parallel jobs", flush=True)
    with Pool(args.jobs) as pool:
        for msg in pool.imap_unordered(_job, runs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
