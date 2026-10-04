#!/usr/bin/env python
"""Evaluate every trained run and the reference baselines with common random numbers."""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safeacd.evaluate import run_episodes  # noqa: E402
from safeacd.playbook import PlaybookAgent, RandomAgent, SleepAgent  # noqa: E402
from safeacd.ppo import TrainedPolicy  # noqa: E402
from safeacd.shield import Shield  # noqa: E402


def label(cfg):
    m = cfg["method"]
    return f"shaped-b{cfg['shaping_beta']:g}" if m == "shaped" else m


def _cached(cache: Path, fn):
    """Per-run cache so evaluation is incremental (re-running skips finished runs)."""
    if cache.exists():
        return pd.read_csv(cache).to_dict("records")
    rows = fn()
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cache, index=False)
    return rows


def _eval_run(args):
    run_dir, attackers, n_ep, deploy_shield, cache = args
    tag = f"{Path(run_dir).name}_n{n_ep}_{'-'.join(attackers)}{'_S' if deploy_shield else ''}.csv"
    return _cached(Path(cache) / tag, lambda: _eval_run_uncached(run_dir, attackers, n_ep, deploy_shield))


def _eval_run_uncached(run_dir, attackers, n_ep, deploy_shield):
    torch.set_num_threads(1)
    cfg = json.load(open(Path(run_dir) / "config.json"))
    rows = []
    variants = [(label(cfg), None)]
    if deploy_shield:
        variants.append((label(cfg) + "+S", True))
    for name, use_shield in variants:
        for att in attackers:
            pol = TrainedPolicy(run_dir, use_shield=use_shield, seed=cfg["seed"])
            for r in run_episodes(pol, att, n_ep, shield=Shield()):
                r.update(method=name, seed=cfg["seed"])
                rows.append(r)
    return rows


def _eval_baseline(args):
    name, attackers, n_ep, cache = args
    tag = f"baseline-{name}_n{n_ep}_{'-'.join(attackers)}.csv"
    return _cached(Path(cache) / tag, lambda: _eval_baseline_uncached(name, attackers, n_ep))


def _eval_baseline_uncached(name, attackers, n_ep):
    agents = {"playbook": PlaybookAgent, "sleep": SleepAgent, "random": lambda: RandomAgent(0)}
    rows = []
    for att in attackers:
        for r in run_episodes(agents[name](), att, n_ep, shield=Shield()):
            r.update(method=name, seed=0)
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--episodes", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    n_ep = args.episodes or ev["episodes"]
    run_dirs = sorted(p.parent for p in Path(cfg["out_dir"]).glob("*/model.pt"))
    cache = str(Path(cfg["eval_dir"]) / "per_run")
    jobs = []
    for rd in run_dirs:
        c = json.load(open(rd / "config.json"))
        jobs.append((str(rd), ev["attackers"], n_ep, c["method"] in ev["deploy_shield_for"], cache))
    rows = []
    with Pool(args.jobs) as pool:
        for i, part in enumerate(pool.imap_unordered(_eval_run, jobs)):
            rows += part
            print(f"evaluated {i + 1}/{len(jobs)}", flush=True)
        for part in pool.imap_unordered(_eval_baseline, [(b, ev["attackers"], n_ep, cache) for b in ev["baselines"]]):
            rows += part
    out = Path(cfg["eval_dir"])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "episodes.csv", index=False)
    print(f"wrote {len(rows)} episode records to {out / 'episodes.csv'}")


if __name__ == "__main__":
    main()
