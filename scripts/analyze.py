#!/usr/bin/env python
"""Aggregate evaluation records into the paper's tables, statistics and figures.

Statistical protocol (see docs/methodology.md):
* unit of analysis for learned agents = training seed (per-seed mean over
  evaluation episodes); 95% CIs by percentile bootstrap over seeds (10k);
* method-vs-method tests: two-sided Mann-Whitney U on seed means, Holm-corrected;
* agent-vs-playbook tests exploit common random numbers: per-episode values
  (averaged over seeds) are paired with the playbook's on the same episode seed
  and compared with a Wilcoxon signed-rank test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

COSTS = ["prod", "traffic", "evidence", "critical"]

# Fixed entity -> style mapping (validated categorical palette; colour is never
# the only cue: every series also has its own marker and line style).
STYLE = {
    "typed":       dict(c="#2a78d6", m="o", ls="-",  label="Typed (Lag+Shield)"),
    "lag":         dict(c="#eb6834", m="s", ls="--", label="PPO-Lagrangian"),
    "shaped-b1":   dict(c="#1baf7a", m="^", ls="-.", label="Shaped ($\\beta$=1)"),
    "shield":      dict(c="#eda100", m="D", ls=":",  label="PPO+Shield"),
    "ppo":         dict(c="#e87ba4", m="v", ls=(0, (5, 1)), label="PPO (reward only)"),
    "playbook":    dict(c="#008300", m="P", ls="-",  label="SOC playbook"),
}
MAIN_ORDER = ["playbook", "ppo", "shaped-b1", "lag", "shield", "typed"]
ABLATION = ["ppo+S", "shaped-b1+S", "lag+S"]
NICE = {k: v["label"] for k, v in STYLE.items()}
NICE.update({"ppo+S": "PPO, shield at deploy", "lag+S": "PPO-Lag, shield at deploy",
             "shaped-b1+S": "Shaped, shield at deploy", "sleep": "No defence",
             "random": "Random"})


def boot_ci(x, n=10_000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (x.mean(), x.mean())
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return tuple(np.percentile(bs, [2.5, 97.5]))


def iqm(x):
    return stats.trim_mean(np.asarray(x, dtype=float), 0.25)


def per_seed(df, budgets):
    d = df.copy()
    d["sat_prod"] = d.cost_prod <= budgets[0]
    d["sat_traffic"] = d.cost_traffic <= budgets[1]
    d["sat_evidence"] = d.cost_evidence <= budgets[2]
    d["sat_critical"] = d.cost_critical <= budgets[3]
    d["sat_all"] = d[["sat_prod", "sat_traffic", "sat_evidence", "sat_critical"]].all(1)
    d["hard"] = d.cost_evidence + d.cost_critical
    cols = ["return", "impact_steps", "cost_prod", "cost_traffic", "cost_evidence",
            "cost_critical", "hard", "sat_prod", "sat_traffic", "sat_evidence",
            "sat_critical", "sat_all", "shield_flags"]
    return d.groupby(["method", "seed", "attacker"])[cols].mean().reset_index()


def summarize(ps, attackers):
    sub = ps[ps.attacker.isin(attackers)]
    seedavg = sub.groupby(["method", "seed"]).mean(numeric_only=True).reset_index()
    out = []
    for m, g in seedavg.groupby("method"):
        row = {"method": m, "n_seeds": len(g)}
        for c in g.columns:
            if c in ("method", "seed"):
                continue
            row[c] = g[c].mean()
            lo, hi = boot_ci(g[c].values)
            row[c + "_lo"], row[c + "_hi"] = lo, hi
        row["return_iqm"] = iqm(g["return"]) if len(g) >= 4 else g["return"].mean()
        out.append(row)
    return pd.DataFrame(out).set_index("method"), seedavg


def fmt(v, lo=None, hi=None, d=1, ci=True):
    if ci and lo is not None and not np.isclose(lo, hi):
        return f"{v:.{d}f} \\scriptsize[{lo:.{d}f}, {hi:.{d}f}]"
    return f"{v:.{d}f}"


def latex_table(summ, rows, caption, label, budgets):
    hdr = ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
           "\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\linewidth}{!}{%\n"
           "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
           "Method & Security return $\\uparrow$ & Downtime $\\downarrow$ & Blocked traffic $\\downarrow$ "
           "& Evidence destroyed $\\downarrow$ & Critical viol. $\\downarrow$ & $P$(all satisfied) $\\uparrow$\\\\\n"
           f" & & (budget {budgets[0]:g}) & (budget {budgets[1]:g}) & (budget 0) & (budget 0) & \\\\\n\\midrule\n")
    body = ""
    for m in rows:
        if m not in summ.index:
            continue
        r = summ.loc[m]
        body += (f"{NICE.get(m, m)} & {fmt(r['return'], r['return_lo'], r['return_hi'])} & "
                 f"{fmt(r.cost_prod, r.cost_prod_lo, r.cost_prod_hi)} & "
                 f"{fmt(r.cost_traffic, r.cost_traffic_lo, r.cost_traffic_hi)} & "
                 f"{fmt(r.cost_evidence, r.cost_evidence_lo, r.cost_evidence_hi, d=2)} & "
                 f"{fmt(r.cost_critical, r.cost_critical_lo, r.cost_critical_hi, d=2)} & "
                 f"{fmt(r.sat_all, r.sat_all_lo, r.sat_all_hi, d=2)}\\\\\n")
    return hdr + body + "\\bottomrule\n\\end{tabular}}\n\\end{table}\n"


def holm(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def tests(seedavg, ep, budgets, attackers):
    """Pre-registered comparisons (docs/methodology.md, Section 6)."""
    comps = [("typed", "lag"), ("typed", "shaped-b1"), ("typed", "shield"), ("typed", "ppo"),
             ("lag", "lag+S")]
    metrics = ["return", "cost_prod", "cost_traffic", "hard", "sat_all"]
    rows = []
    for a, b in comps:
        for met in metrics:
            xa = seedavg.loc[seedavg.method == a, met].values
            xb = seedavg.loc[seedavg.method == b, met].values
            if len(xa) < 2 or len(xb) < 2:
                continue
            if np.allclose(np.r_[xa, xb], np.r_[xa, xb][0]):
                p = 1.0
            else:
                p = stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue
            rows.append(dict(a=a, b=b, metric=met, mean_a=xa.mean(), mean_b=xb.mean(), p=p))
    # Paired CRN comparison with the playbook (episode level).
    e = ep[ep.attacker.isin(attackers)].copy()
    e["hard"] = e.cost_evidence + e.cost_critical
    pb = e[e.method == "playbook"].set_index(["attacker", "episode"])
    for m in ["typed", "lag", "shaped-b1"]:
        em = e[e.method == m].groupby(["attacker", "episode"]).mean(numeric_only=True)
        j = em.join(pb, rsuffix="_pb", how="inner")
        for met in ["return", "cost_prod", "cost_traffic"]:
            diff = j[met] - j[met + "_pb"]
            p = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(diff).pvalue
            rows.append(dict(a=m, b="playbook (paired CRN)", metric=met,
                             mean_a=j[met].mean(), mean_b=j[met + "_pb"].mean(), p=p))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df.p.values)
    return df


# --------------------------------------------------------------------------- #
def fig_pareto(summ, budgets, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in MAIN_ORDER:
        if m not in summ.index:
            continue
        r = summ.loc[m]
        harm = max(r.cost_prod / budgets[0], r.cost_traffic / budgets[1])
        hard = r.cost_evidence + r.cost_critical
        st = STYLE[m]
        ax.scatter(harm, r["return"], s=64, marker=st["m"],
                   facecolors=st["c"] if hard < 0.05 else "white",
                   edgecolors=st["c"], linewidths=2, zorder=3)
        ax.annotate(st["label"], (harm, r["return"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color="#0b0b0b")
    ax.axvline(1.0, color="#52514e", lw=1, ls="--")
    ax.text(1.03, ax.get_ylim()[0], "budget", fontsize=7, color="#52514e", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Collateral harm  max(downtime/budget, traffic/budget)  [log]")
    ax.set_ylabel("Security return (higher is better)")
    ax.grid(True, color="#e6e5e1", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Hollow marker = hard-constraint violations > 0.05 / episode", fontsize=7,
                 color="#52514e", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_learning(run_root, methods, budgets, path):
    panels = [("ep_return", "Security return", None),
              ("ep_prod", "Downtime / episode", budgets[0]),
              ("ep_traffic", "Blocked traffic / episode", budgets[1]),
              ("cum_hard", "Cumulative hard violations\n(training, log)", None)]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.6))
    for m in methods:
        logs = []
        for p in sorted(Path(run_root).glob("*/train_log.csv")):
            cfg = json.load(open(p.parent / "config.json"))
            lab = f"shaped-b{cfg['shaping_beta']:g}" if cfg["method"] == "shaped" else cfg["method"]
            if lab != m:
                continue
            d = pd.read_csv(p)
            d["cum_hard"] = d.cum_evidence + d.cum_critical
            logs.append(d.set_index("steps"))
        if not logs:
            continue
        st = STYLE[m]
        for ax, (col, title, bud) in zip(axes, panels):
            M = pd.concat([lg[col].rolling(5, min_periods=1).mean() for lg in logs], axis=1)
            mu = M.mean(1)
            lo = M.quantile(0.1, axis=1)
            hi = M.quantile(0.9, axis=1)
            ax.plot(M.index / 1e6, mu, color=st["c"], ls=st["ls"], lw=2, label=st["label"])
            ax.fill_between(M.index / 1e6, lo, hi, color=st["c"], alpha=0.15, lw=0)
    for ax, (col, title, bud) in zip(axes, panels):
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Env. steps (M)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, color="#e6e5e1", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if bud is not None:
            ax.axhline(bud, color="#52514e", lw=1, ls="--")
            ax.set_yscale("symlog", linthresh=max(bud, 1))
    axes[0].set_yscale("symlog", linthresh=10)
    axes[3].set_yscale("symlog", linthresh=1)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_shaping(summ, budgets, path):
    rows = [(float(m.split("-b")[1]), summ.loc[m]) for m in summ.index
            if m.startswith("shaped-b") and "+S" not in m]
    if len(rows) < 2:
        return
    rows.sort(key=lambda x: x[0])
    b = np.array([x[0] for x in rows])
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.5))
    series = [("return", "Security return", None),
              ("cost_traffic", "Blocked traffic", budgets[1]),
              ("hard", "Hard violations / ep.", None)]
    for ax, (col, title, bud) in zip(axes, series):
        y = np.array([r[col] for _, r in rows])
        lo = np.array([r[col + "_lo"] for _, r in rows])
        hi = np.array([r[col + "_hi"] for _, r in rows])
        st = STYLE["shaped-b1"]
        ax.errorbar(b, y, yerr=[y - lo, hi - y], color=st["c"], marker=st["m"], lw=2,
                    capsize=3, label="Shaped ($\\beta$ sweep)")
        for ref in ("typed", "lag"):
            if ref in summ.index:
                s2 = STYLE[ref]
                ax.axhline(summ.loc[ref, col], color=s2["c"], ls=s2["ls"], lw=1.5, label=s2["label"])
        if bud is not None:
            ax.axhline(bud, color="#52514e", lw=1, ls="--", label="budget")
        ax.set_xscale("log")
        ax.set_xlabel("penalty weight $\\beta$", fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, color="#e6e5e1", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    budgets = cfg["train"]["budgets"]
    train_att = cfg["train"]["attackers"]
    ep = pd.read_csv(Path(cfg["eval_dir"]) / "episodes.csv")
    ep["hard"] = ep.cost_evidence + ep.cost_critical
    ps = per_seed(ep, budgets)
    figd, tabd = Path(cfg["fig_dir"]), Path(cfg["table_dir"])
    figd.mkdir(parents=True, exist_ok=True)
    tabd.mkdir(parents=True, exist_ok=True)
    evald = Path(cfg["eval_dir"])

    summaries = {}
    for name, atts in [("train_dist", train_att), ("bline", ["bline"]), ("meander", ["meander"]),
                       ("stealthy", ["stealthy"])]:
        summ, seedavg = summarize(ps, atts)
        summ.to_csv(evald / f"summary_{name}.csv")
        summaries[name] = (summ, seedavg)

    s_tr, sa_tr = summaries["train_dist"]
    s_st, _ = summaries["stealthy"]
    (tabd / "main.tex").write_text(latex_table(
        s_tr, MAIN_ORDER,
        "Deployment performance under the training attacker distribution (B-line/Meander mixture). "
        "Mean over seeds with 95\\% bootstrap CI over seeds; 100 common-random-number episodes per attacker.",
        "tab:main", budgets))
    (tabd / "ood.tex").write_text(latex_table(
        s_st, MAIN_ORDER,
        "Out-of-distribution evaluation against the held-out low-and-slow (Stealthy) attacker.",
        "tab:ood", budgets))
    (tabd / "ablation.tex").write_text(latex_table(
        s_tr, ["ppo", "ppo+S", "shaped-b1", "shaped-b1+S", "lag", "lag+S", "typed"],
        "Ablation: shield applied only at deployment (+S) versus during training (Typed).",
        "tab:ablation", budgets))

    t = tests(sa_tr, ep, budgets, train_att)
    t.to_csv(evald / "tests.csv", index=False)

    fig_pareto(s_tr, budgets, figd / "pareto.pdf")
    fig_learning(cfg["out_dir"], ["ppo", "shaped-b1", "lag", "shield", "typed"], budgets,
                 figd / "learning.pdf")
    fig_shaping(s_tr, budgets, figd / "shaping.pdf")

    pd.set_option("display.width", 200)
    for name in ("train_dist", "stealthy"):
        s = summaries[name][0]
        cols = ["n_seeds", "return", "return_iqm", "cost_prod", "cost_traffic", "cost_evidence",
                "cost_critical", "sat_all", "impact_steps", "shield_flags"]
        print(f"\n=== {name} ===")
        print(s[[c for c in cols if c in s.columns]].round(3).to_string())
    print("\n=== tests (Holm-corrected) ===")
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
