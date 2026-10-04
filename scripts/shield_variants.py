#!/usr/bin/env python
"""Deployment-time shield variants for the RQ4 discussion (evaluation only).

H1 is monitor-relative: an image that post-dates the open ticket can still
miss a *new, undetected* compromise that starts between imaging and
reimaging.  Here the Typed policies are re-evaluated with H1 tightened by an
image-freshness window (restore must follow the image within k steps), on
the training attackers and the held-out Stealthy attacker.
"""
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
from safeacd.ppo import TrainedPolicy  # noqa: E402
from safeacd.shield import Shield  # noqa: E402

WINDOWS = {"H1 (no window)": None, "H1 + image age <= 3": 3, "H1 + image age <= 1": 1}


def _job(args):
    run_dir, n_ep, attackers = args
    torch.set_num_threads(1)
    seed = json.load(open(Path(run_dir) / "config.json"))["seed"]
    rows = []
    for name, k in WINDOWS.items():
        for att in attackers:
            pol = TrainedPolicy(run_dir, use_shield=True, seed=seed, shield=Shield(max_image_age=k))
            for r in run_episodes(pol, att, n_ep):
                r.update(method="typed", seed=seed, variant=name)
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    atts = cfg["eval"]["attackers"]
    runs = sorted(str(p.parent) for p in Path(cfg["out_dir"]).glob("typed_s*/model.pt"))
    rows = []
    with Pool(args.jobs) as pool:
        for part in pool.imap_unordered(_job, [(r, args.episodes, atts) for r in runs]):
            rows += part
    df = pd.DataFrame(rows)
    df["hard"] = df.cost_evidence + df.cost_critical
    df["hard_free"] = df.hard == 0
    df["dist"] = df.attacker.map(lambda a: "stealthy (OOD)" if a == "stealthy" else "training")
    out = Path(cfg["eval_dir"])
    df.to_csv(out / "shield_variants_episodes.csv", index=False)
    seedm = df.groupby(["dist", "variant", "seed"])[["return", "cost_prod", "cost_traffic",
                                                     "cost_evidence", "hard_free"]].mean()
    summ = seedm.groupby(["dist", "variant"]).mean()
    summ.round(3).to_csv(out / "shield_variants_summary.csv")
    print(summ.round(3).to_string())

    lines = ["\\begin{table}[t]\\centering",
             "\\caption{Typed policies deployed with H1 tightened by an image-freshness window "
             "(evaluation only; mean over seeds, 100 CRN episodes per attacker).}",
             "\\label{tab:freshness}\\small",
             "\\begin{tabular}{llrrrr}\\toprule",
             "Attackers & H1 variant & Return & Traffic & Evidence destroyed & $P$(no hard viol.)\\\\\\midrule"]
    le, eol = "$\\le$", "\\\\"
    for (dist, var), r in summ.iterrows():
        var_tex = var.replace("<=", le)
        lines.append(f"{dist} & {var_tex} & {r['return']:.1f} & {r.cost_traffic:.1f} & "
                     f"{r.cost_evidence:.3f} & {r.hard_free:.3f}{eol}")
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    Path(cfg["table_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["table_dir"]) / "freshness.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
