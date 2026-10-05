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
    d["sat_all"] = d[["sat_prod", "sat_traffic", "sat_evidence", "sat_critical"]].all(axis=1)
    d["hard"] = d.cost_evidence + d.cost_critical
    d["hard_free"] = d.hard == 0
    cols = ["return", "impact_steps", "cost_prod", "cost_traffic", "cost_evidence",
            "cost_critical", "hard", "hard_free", "sat_prod", "sat_traffic", "sat_evidence",
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


def final_lambdas(run_root):
    """Mean final Lagrange multipliers per Lagrangian method (macros lamFinal<Method><Channel>)."""
    acc = {}
    for p in Path(run_root).glob("*/train_log.csv"):
        cfg = json.load(open(p.parent / "config.json"))
        if cfg["method"] not in ("lag", "typed") or not (p.parent / "model.pt").exists():
            continue
        last = pd.read_csv(p).iloc[-1]
        for ch in COSTS:
            acc.setdefault((cfg["method"], ch), []).append(last[f"lambda_{ch}"])
    out = {}
    for (m, ch), v in acc.items():
        out[f"lamFinal{m.capitalize()}{ch.capitalize()}"] = f"{np.mean(v):.1f}"
    return out


def hard_violation_ub(ep, attackers, alpha=0.05):
    """Exact (Clopper-Pearson) one-sided 95% upper bound on P(hard violation per episode),
    pooling all evaluation episodes of a method (seeds x episodes)."""
    e = ep[ep.attacker.isin(attackers)]
    out = {}
    for m, g in e.groupby("method"):
        n = len(g)
        k = int(((g.cost_evidence + g.cost_critical) > 0).sum())
        ub = 1.0 if k == n else stats.beta.ppf(1 - alpha, k + 1, n - k)
        out[m] = (k, n, ub)
    return out


def fmt(v, lo=None, hi=None, d=1, ci=True):
    if ci and lo is not None and not np.isclose(lo, hi):
        return f"{v:.{d}f} \\scriptsize[{lo:.{d}f}, {hi:.{d}f}]"
    return f"{v:.{d}f}"


def latex_table(summ, rows, caption, label, budgets):
    hdr = ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
           "\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\linewidth}{!}{%\n"
           "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
           "Method & Security return $\\uparrow$ & Downtime $\\downarrow$ & Blocked traffic $\\downarrow$ "
           "& Evidence destroyed $\\downarrow$ & Critical viol. $\\downarrow$ & $P$(no hard viol.) $\\uparrow$ "
           "& $P$(all satisfied) $\\uparrow$\\\\\n"
           f" & & (budget {budgets[0]:g}) & (budget {budgets[1]:g}) & (budget 0) & (budget 0) & & \\\\\n\\midrule\n")
    body = ""
    for m in rows:
        if m == "---":
            body += "\\midrule\n"
            continue
        if m not in summ.index:
            continue
        r = summ.loc[m]
        body += (f"{NICE.get(m, m)} & {fmt(r['return'], r['return_lo'], r['return_hi'])} & "
                 f"{fmt(r.cost_prod, r.cost_prod_lo, r.cost_prod_hi)} & "
                 f"{fmt(r.cost_traffic, r.cost_traffic_lo, r.cost_traffic_hi)} & "
                 f"{fmt(r.cost_evidence, r.cost_evidence_lo, r.cost_evidence_hi, d=2)} & "
                 f"{fmt(r.cost_critical, r.cost_critical_lo, r.cost_critical_hi, d=2)} & "
                 f"{fmt(r.hard_free, r.hard_free_lo, r.hard_free_hi, d=3)} & "
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
             ("lag", "lag+S"), ("shaped-b1", "shaped-b1+S")]
    metrics = ["return", "cost_prod", "cost_traffic", "hard", "hard_free", "sat_all"]
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
def _place_labels(ax, pts, fontsize=7):
    """Greedy non-overlapping direct labels.

    For each point (highest first) try candidate offsets around the marker and
    take the first whose text box overlaps neither earlier labels nor any
    marker; fall back to a leader line further down.
    """
    fig = ax.figure
    fig.canvas.draw()
    scale = fig.dpi / 72.0
    disp = [ax.transData.transform((x, y)) for (x, y, _) in pts]
    markers = [(px - 6, py - 6, px + 6, py + 6) for (px, py) in disp]
    placed = []

    def free(b):
        return all(b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3] for o in placed + markers)

    for i in sorted(range(len(pts)), key=lambda j: -pts[j][1]):
        x, y, text = pts[i]
        px, py = disp[i]
        w, h = 5.0 * scale * len(text), 9 * scale
        cands = [(8, 2), (8, -h - 2), (-8 - w, 2), (-8 - w, -h - 2), (8, h + 4), (8, -2 * h - 6),
                 (-8 - w, h + 4), (-8 - w, -2 * h - 6)]
        cands += [(8, -h - 2 - k * (h + 3)) for k in range(2, 10)]
        for dx, dy in cands:
            box = (px + dx, py + dy, px + dx + w, py + dy + h)
            if free(box):
                break
        placed.append(box)
        far = abs(dy) > 2 * h or dx < 0 and abs(dy) > h
        ax.annotate(text, (x, y), textcoords="offset points",
                    xytext=(dx / scale, dy / scale), fontsize=fontsize, color="#0b0b0b",
                    ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="#9a9893", lw=0.6) if far else None)


def fig_pareto(summ, budgets, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    pts = []
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
        pts.append((harm, r["return"], st["label"]))
    ax.axvline(1.0, color="#52514e", lw=1, ls="--")
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ax.set_xlim(min(lo, 0.1) / 3, hi * 12)
    ax.text(1.05, 0.96, "budget", transform=ax.get_xaxis_transform(), fontsize=7, color="#52514e",
            va="top")
    ax.set_xlabel("Collateral harm: max(downtime/budget, traffic/budget)  [log]")
    ax.set_ylabel("Security return (higher is better)")
    ax.grid(True, color="#e6e5e1", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Filled: no hard violations;  hollow: > 0.05 hard violations / episode",
                 fontsize=7, color="#52514e", loc="left")
    fig.tight_layout()
    _place_labels(ax, pts)
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
            mu = M.mean(axis=1)
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
            ax.set_ylim(bottom=0)
    axes[3].set_yscale("symlog", linthresh=1)
    axes[3].set_ylim(bottom=0)
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


def fmt_p(p):
    """p-value for use inside math mode: 3 s.f., scientific notation below 1e-3."""
    if p >= 1e-3:
        return f"{p:.3g}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


def write_macros(summaries, tests_df, path, ubs=None, extra=None):
    """Emit \\newcommand macros so every number in the paper text is generated."""
    words = {"0.1": "Tenth", "0.3": "ThreeTenths", "1": "", "3": "Three", "10": "Ten"}
    digit = dict(zip("0123456789.", ["Zero", "One", "Two", "Three", "Four", "Five", "Six",
                                     "Seven", "Eight", "Nine", "Pt"]))

    def key(m):
        """LaTeX-safe (letters only), collision-free macro key for a method label."""
        deploy = m.endswith("+S")
        base = m[:-2] if deploy else m
        if base.startswith("shaped-b"):
            b = base[len("shaped-b"):]
            k = "Shaped" + (words[b] if b in words else "B" + "".join(digit[c] for c in b))
        else:
            k = "".join(ch for ch in base if ch.isalpha()).capitalize()
        return k + ("Deploy" if deploy else "")
    lines = ["% Auto-generated by scripts/analyze.py -- do not edit by hand."]
    fields = {"return": ("R", 1), "return_iqm": ("RIqm", 1), "cost_prod": ("Prod", 1), "cost_traffic": ("Traf", 1),
              "cost_evidence": ("Evid", 2), "cost_critical": ("Crit", 2),
              "sat_all": ("Sat", 2), "hard_free": ("HardFree", 3), "impact_steps": ("Impact", 1)}
    for dist, (summ, _) in summaries.items():
        dkey = {"train_dist": "Train", "bline": "Bline", "meander": "Meander",
                "stealthy": "Ood"}[dist]
        for m in summ.index:
            for col, (abbr, d) in fields.items():
                if col in summ.columns:
                    lines.append(f"\\newcommand{{\\res{dkey}{key(m)}{abbr}}}{{{summ.loc[m, col]:.{d}f}}}")
    for dist, (summ, _) in summaries.items():
        dkey = {"train_dist": "Train", "bline": "Bline", "meander": "Meander",
                "stealthy": "Ood"}[dist]
        for m in summ.index:
            for col, abbr in (("hard_free", "HardFreePct"), ("sat_all", "SatPct")):
                if col in summ.columns:
                    lines.append(f"\\newcommand{{\\res{dkey}{key(m)}{abbr}}}{{{100 * summ.loc[m, col]:.1f}}}")
                    lines.append(f"\\newcommand{{\\res{dkey}{key(m)}{abbr}Viol}}"
                                 f"{{{100 * (1 - summ.loc[m, col]):.1f}}}")
    for name, val in (extra or {}).items():
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    for dist, ub in (ubs or {}).items():
        for m, (k, n, u) in ub.items():
            lines.append(f"\\newcommand{{\\ub{dist}{key(m)}}}{{{100 * u:.2f}}}")
            lines.append(f"\\newcommand{{\\nhard{dist}{key(m)}}}{{{k}/{n}}}")
    for _, r in tests_df.iterrows():
        k = key(r.a) + "Vs" + key(r.b.split(" ")[0]) + "".join(ch for ch in r.metric.title() if ch.isalpha())
        lines.append(f"\\newcommand{{\\p{k}}}{{{fmt_p(r.p_holm)}}}")
    Path(path).write_text("\n".join(lines) + "\n")


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
        s_tr, MAIN_ORDER + ["---"] + ABLATION,
        "Deployment performance under the training attacker distribution (B-line/Meander mixture). "
        "Mean over seeds with 95\\% bootstrap CI over seeds; 100 common-random-number episodes per attacker. "
        "Bottom block (RQ3): unshielded policies deployed with the shield attached.",
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
    ubs = {"Train": hard_violation_ub(ep, train_att), "Ood": hard_violation_ub(ep, ["stealthy"])}
    pd.DataFrame([dict(dist=d, method=m, k=k, n=n, ub95=u) for d, ub in ubs.items()
                  for m, (k, n, u) in ub.items()]).to_csv(evald / "hard_violation_bounds.csv", index=False)
    extra = final_lambdas(cfg["out_dir"])
    extra["nSeedsMain"] = int(s_tr.loc["typed", "n_seeds"]) if "typed" in s_tr.index else 0
    extra["nEpisodesEval"] = cfg["eval"]["episodes"]
    write_macros(summaries, t, Path(cfg["fig_dir"]).parent / "results_macros.tex", ubs, extra)

    fig_pareto(s_tr, budgets, figd / "pareto.pdf")
    fig_learning(cfg["out_dir"], ["ppo", "shaped-b1", "lag", "shield", "typed"], budgets,
                 figd / "learning.pdf")
    fig_shaping(s_tr, budgets, figd / "shaping.pdf")

    pd.set_option("display.width", 200)
    for name in ("train_dist", "stealthy"):
        s = summaries[name][0]
        cols = ["n_seeds", "return", "return_iqm", "cost_prod", "cost_traffic", "cost_evidence",
                "cost_critical", "hard_free", "sat_all", "impact_steps", "shield_flags"]
        print(f"\n=== {name} ===")
        print(s[[c for c in cols if c in s.columns]].round(3).to_string())
    print("\n=== tests (Holm-corrected) ===")
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
