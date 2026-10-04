# Defend Without Destroying — typed safety constraints for RL cyber defence

Research artefact for an **ANNSIM'27** submission (SCS Annual Modeling &
Simulation Conference, Wrocław, 17–21 May 2027; *Modeling and Simulation in
Cyber Security* track; paper deadline **18 Dec 2026**).

> A cyber-defence agent cannot simply maximise cumulative reward if its
> actions can disconnect production machines, block legitimate traffic, or
> destroy evidence.

**Idea.** Collateral harms have *types*:

- **Budget constraints**, where some harm is acceptable: production downtime
  and blocked legitimate traffic. These are handled by **Lagrangian
  relaxation** (PPO-Lagrangian).
- **Hard / irreversible constraints**, where any violation is unacceptable:
  reimaging a compromised host before it is imaged (destroys evidence), and
  isolating the safety-critical OT controller. These are handled by a
  **runtime shield** built only from SOC-observable signals.

The proposed agent (`typed`) combines the two. It is compared against
reward-only PPO, reward shaping (with a β sweep), Lagrangian-on-everything,
shield-only, and a SOC playbook, in a purpose-built simulator.

## Repository map

| Path | What it is |
|---|---|
| `safeacd/env.py` | **SafeACD-Sim**: 12-host CAGE-2-like network, red/green processes, 77 actions, separate reward and 4 cost channels |
| `safeacd/red.py` | B-line, Meander and held-out Stealthy attackers |
| `safeacd/shield.py` | Preemptive shield: H1 image-before-reimage, H2 never isolate the OpServer |
| `safeacd/ppo.py` | One PPO implementation for all conditions (`ppo`, `shaped`, `lag`, `shield`, `typed`), multi-head critic, Adam dual update |
| `safeacd/playbook.py` | SOC playbook, random and no-defence reference policies |
| `safeacd/evaluate.py` | Common-random-number evaluation harness |
| `configs/experiment.yaml` | The pre-registered experimental grid |
| `scripts/` | `calibrate.py`, `run_experiments.py`, `evaluate_all.py`, `analyze.py` |
| `docs/literature_review.md` | Related work, gap analysis, venue facts |
| `docs/methodology.md` | **Protocol**: RQs and hypotheses, V&V, statistics, threats, timeline |
| `paper/` | LaTeX draft (`main.tex`, `refs.bib`), generated `tables/` and `figures/` |
| `results/` | Run logs, evaluation records and summaries (generated) |

## Reproduce

```bash
pip install -e .[dev]          # CPU-only torch is enough
pytest -q                      # verification tests
python scripts/calibrate.py    # operational validation (paper Table 2)

python scripts/run_experiments.py --config configs/experiment.yaml --jobs 4   # resumable
python scripts/evaluate_all.py   --config configs/experiment.yaml --jobs 4
python scripts/analyze.py        --config configs/experiment.yaml             # tables, figures, tests
cd paper && latexmk -pdf main.tex
```

Smoke test, a few minutes end to end:

```bash
python scripts/run_experiments.py --steps 20000 --seeds 0 --jobs 4
python scripts/evaluate_all.py --episodes 5 && python scripts/analyze.py
```

See `docs/methodology.md` §7 for compute estimates and §9 for the timeline to
the deadline.
