#!/usr/bin/env python
"""Parameter-sensitivity analysis (methodology §3.3), evaluation-only.

Trained policies are re-evaluated under perturbed simulator parameters to test
whether the paper's conclusions survive model misspecification.  No
retraining: this measures robustness of *deployed* policies to an
environment that differs from the one they were trained in.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safeacd.env import EnvConfig  # noqa: E402
from safeacd.evaluate import run_episodes  # noqa: E402
from safeacd.playbook import PlaybookAgent  # noqa: E402
from safeacd.ppo import TrainedPolicy  # noqa: E402

BASE = EnvConfig()
VARIANTS = {
    "base": {},
    "detect -25%": dict(p_detect_exploit=BASE.p_detect_exploit * 0.75,
                        p_detect_escalate=BASE.p_detect_escalate * 0.75),
    "detect +25%": dict(p_detect_exploit=min(1.0, BASE.p_detect_exploit * 1.25),
                        p_detect_escalate=min(1.0, BASE.p_detect_escalate * 1.25)),
    "restore 1 step": dict(restore_steps=1),
    "restore 3 steps": dict(restore_steps=3),
    "false alerts x5": dict(p_false_alert=BASE.p_false_alert * 5),
    "exploit p=0.9": dict(p_exploit=0.9),
}
METHODS = ("typed", "lag", "shaped-b1")


def _job(args):
    run_dir, n_ep = args
    torch.set_num_threads(1)
    rows = []
    if run_dir == "playbook":
        label, seed, make = "playbook", 0, PlaybookAgent
    else:
        cfg = json.load(open(Path(run_dir) / "config.json"))
        label = cfg["method"] if cfg["method"] != "shaped" else f"shaped-b{cfg['shaping_beta']:g}"
        seed = cfg["seed"]
        make = lambda: TrainedPolicy(run_dir, seed=seed)  # noqa: E731
    for vname, kw in VARIANTS.items():
        env_cfg = replace(BASE, **kw)
        for att in ("bline", "meander"):
            for r in run_episodes(make(), att, n_ep, env_cfg=env_cfg):
                r.update(method=label, seed=seed, variant=vname)
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    budgets = cfg["train"]["budgets"]
    jobs = ["playbook"]
    for p in sorted(Path(cfg["out_dir"]).glob("*/model.pt")):
        c = json.load(open(p.parent / "config.json"))
        lab = c["method"] if c["method"] != "shaped" else f"shaped-b{c['shaping_beta']:g}"
        if lab in METHODS:
            jobs.append(str(p.parent))
    rows = []
    with Pool(args.jobs) as pool:
        for part in pool.imap_unordered(_job, [(j, args.episodes) for j in jobs]):
            rows += part
    df = pd.DataFrame(rows)
    out = Path(cfg["eval_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sensitivity_episodes.csv", index=False)

    df["hard"] = df.cost_evidence + df.cost_critical
    df["sat_all"] = ((df.cost_prod <= budgets[0]) & (df.cost_traffic <= budgets[1])
                     & (df.cost_evidence <= budgets[2]) & (df.cost_critical <= budgets[3]))
    seed_means = df.groupby(["variant", "method", "seed"])[["return", "cost_prod", "cost_traffic",
                                                            "hard", "sat_all"]].mean()
    summ = seed_means.groupby(["variant", "method"]).mean().round(2)
    summ.to_csv(out / "sensitivity_summary.csv")
    order = list(VARIANTS)
    piv = summ.reset_index().pivot(index="variant", columns="method", values=["return", "hard", "sat_all"])
    piv = piv.loc[[v for v in order if v in piv.index]]
    print(piv.to_string())

    # LaTeX table: return / hard violations / P(all satisfied) per variant.
    meths = ["playbook"] + [m for m in METHODS if m in summ.index.get_level_values(1)]
    names = {"playbook": "Playbook", "typed": "Typed", "lag": "Lag", "shaped-b1": "Shaped"}
    lines = ["\\begin{table}[t]\\centering",
             "\\caption{Sensitivity of deployed policies to simulator parameters (training-distribution "
             "attackers, 50 CRN episodes each, mean over seeds). Cells: security return / hard "
             "violations per episode / $P$(all constraints satisfied).}",
             "\\label{tab:sensitivity}\\small\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{l" + "r" * len(meths) + "}\\toprule",
             "Variant & " + " & ".join(names[m] for m in meths) + "\\\\\\midrule"]
    for v in order:
        cells = []
        for m in meths:
            if (v, m) in summ.index:
                r = summ.loc[(v, m)]
                cells.append(f"{r['return']:.0f} / {r['hard']:.2f} / {r['sat_all']:.2f}")
            else:
                cells.append("--")
        lines.append(f"{v} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule\\end{tabular}}\\end{table}"]
    tab = Path(cfg["table_dir"])
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "sensitivity.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
