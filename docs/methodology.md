# Methodology and execution protocol

This document is the paper's **pre-registered protocol**. Every number in the
paper must come from running the commands in §7 with `configs/experiment.yaml`.
Changes after the main runs start go in the change log (§10).

---

## 1. Problem statement

An autonomous cyber-defence agent that maximises a security reward learns
strategies that are secure on paper but operationally unacceptable. Three
examples:

- It severs the OT controller.
- It quarantines whole subnets.
- It reimages compromised hosts before they are imaged, which destroys the
  forensic evidence.

Folding these harms into the reward as penalties ("reward shaping") turns
safety into a tuning problem. Treating all of them as Lagrangian budgets only
guarantees safety *in expectation*. That is acceptable for downtime, and
unacceptable for irreversible harm.

**Thesis.** Constraints have *types*, and the safety mechanism should follow
the type:

| Type | Example | Semantics | Mechanism |
|---|---|---|---|
| Budget (soft, cumulative) | production downtime, blocked legitimate traffic | some harm is acceptable up to an operational budget per episode | CMDP, Lagrangian relaxation (PPO-Lag) |
| Hard (irreversible or safety-critical) | reimage without forensic image; isolate the OT safety controller | any violation is unacceptable and cannot be undone | runtime shield (preemptive action masking) built only from SOC-observable signals |

---

## 2. Research questions and hypotheses (fixed before the main runs)

**RQ1 – Harm of reward maximisation.** How much collateral harm does a
reward-only defender cause, and does reward shaping remove it without
per-scenario tuning?
- **H1a.** PPO (reward only) gets a higher security return than the SOC
  playbook while exceeding every constraint.
- **H1b.** Shaped PPO's compliance depends on the penalty weight β. Across
  β ∈ {0.1, 0.3, 1, 3, 10}, no single β satisfies both budgets and the hard
  constraints *and* keeps the security of the constrained methods.

**RQ2 – Typed vs. untyped constraint handling.** Does Typed (Lagrangian for
budgets + shield for hard rules) meet all constraints more often than
Lagrangian-on-everything (Lag) and shaping, at comparable security?
- **H2a.** P(all constraints satisfied per episode) is higher for Typed than
  for Lag and Shaped (β=1).
- **H2b.** Typed has (near-)zero hard violations during *training* as well as
  deployment. The only residuals come from the monitor-relative gap of rule
  H1 (§3.4).
- **H2c.** Typed's security return is not significantly worse than Lag's.

**RQ3 – Must the shield be in the training loop?**
- **H3.** Adding the shield only at deployment (PPO+S, Lag+S, Shaped+S)
  removes hard violations. It does so at a cost in security and/or budget
  compliance compared with training with the shield (Shield, Typed), because
  the masked policy has shifted away from the policy that was trained.

**RQ4 – Robustness to a held-out attacker.**
- **H4.** Against the unseen Stealthy attacker:
  - Typed's hard-constraint satisfaction is preserved, because the shield
    rules do not depend on the attacker.
  - Budget compliance degrades for *all* learned methods, because budgets
    are satisfied only in expectation under the training distribution.

---

## 3. Simulation model (SafeACD-Sim)

### 3.1 Conceptual model
A discrete-time, stochastic, partially observable simulation (`safeacd/env.py`).
It is formalised as a CMDP ⟨S, A, P, r, c₁..c₄, d₁..d₄, γ⟩ observed through
SOC signals (a POMDP for the defender).

- **Topology.** 12 hosts in 3 subnets, mirroring CAGE-2:
  - User0–4 (user subnet);
  - Ent0–2 (enterprise subnet);
  - OpServer + OpHost0–2 (operational subnet).

  There are two blockable zone links: user→enterprise and enterprise→operational.
- **Red agents (scripted).**
  - Attack-graph progression: exploit → escalate → impact. Exploits within a
    zone need a user-level foothold; crossing a zone needs a privileged one.
  - `bline`: goal-directed towards the OpServer.
  - `meander`: breadth-first.
  - `stealthy`: held out. It is B-line with 30% dwell and detection
    probabilities scaled by 0.4.
  - The initial foothold is a random user host and is undetected.
  - An attacker with no usable foothold re-phishes with probability 0.1 per
    step.
- **Green traffic.** 15 weighted business flows (web, file, auth, historian,
  control loop), each active with a fixed probability per step.
- **Blue actions (77).** Sleep, plus per-host Analyse, Forensics (image),
  Remove, Restore (reimage, offline for 2 steps), Isolate and Reconnect, plus
  Block/Unblock for each zone link.
- **Observation (87-d).** Per host:
  - SOC ticket severity (one-hot);
  - alert raised this step;
  - isolated flag;
  - reimaging flag;
  - whether a valid forensic image exists.

  Globally: link states and the time fraction. The true compromise state is
  never observed.
- **Signals.**
  - Reward = security only. It is minus the host-value-weighted compromise,
    minus 10 per OpServer impact step (CAGE-2 values).
  - Cost vector (defender-induced harm only):
    - `prod`: criticality-weighted host-steps offline;
    - `traffic`: weighted active legitimate flows not served;
    - `evidence`: reimages of tainted hosts whose current incident was never
      imaged;
    - `critical`: steps with the OpServer isolated.

### 3.2 Key modelling decisions (state them in the paper)
1. **Incident-scoped evidence.** The first compromise of a clean host opens
   an incident. An image taken after that preserves the incident's evidence
   until the host is reimaged.
2. **Costs are not part of the reward.** This makes harm measurable for
   every method, including those that ignore it.
3. **Common random numbers.** The red, green, detection and episode streams
   are independent `SeedSequence` children, so policies can be compared on
   the same evaluation episodes.

### 3.3 Verification and validation (V&V)
- **Verification** (`tests/`, 24 tests): determinism under a seed; cost
  semantics (image-then-reimage costs no evidence, reimage-without-image
  costs 1, the critical rule, blocked-link traffic); shield invariants; and
  smoke training for every method.
- **Face validity.** Topology, reward values and red behaviours mirror
  CybORG CAGE-2, the most-used ACD benchmark.
- **Operational validation with reference policies** (calibration run,
  100 episodes per attacker; per-episode mean security return, downtime and
  blocked traffic; no reference policy destroyed evidence or isolated the
  OpServer):

  | Policy | B-line R / prod / traffic | Meander R / prod / traffic |
  |---|---|---|
  | No defence | −1172.7 / 0 / 0 | −1131.0 / 0 / 0 |
  | Random | −528.8 / 328.8 / 1019.4 | −387.1 / 328.8 / 1019.4 |
  | SOC playbook | −78.8 / 12.5 / 52.4 | −13.3 / 3.2 / 31.0 |
  | "Isolate OpServer + block" (reward hacker) | −47.6 / 200 / 1319.2 (+100 critical) | −47.6 / 200 / 1319.2 (+100 critical) |

  This ordering is what a domain expert expects. The last row shows the core
  problem: the highest *security* return comes from the most harmful policy.
- **Two modelling artefacts found during pilots** (now fixed; report them in
  the paper as V&V lessons):
  1. A fixed initial foothold (the CAGE-2 convention) let every learner
     quarantine User0 forever and freeze the attacker.
  2. Alert-scoped evidence created a dead-lock: an attacker acting
     continuously on a host invalidated every new image.
- **Sensitivity analysis** (`scripts/sensitivity.py`; evaluation only, no
  retraining):
  - Trained Typed, Lag and Shaped policies, and the playbook, are re-run
    under perturbed parameters: detection ±25%, restore time 1 or 3 steps,
    false alerts ×5, and exploit success 0.9.
  - 50 CRN episodes per training attacker for each variant.
  - We report whether the *ordering* of methods on hard-violation rate and
    P(all satisfied) is preserved.
  - Retraining with perturbed budgets (×0.5, ×2) is optional, for the
    appendix or the artefact.

### 3.4 The shield (`safeacd/shield.py`)
- **H1 (image before reimage).** Restore(h) is allowed only if a forensic
  image of h exists that post-dates the currently open ticket on h and its
  last reimage.
- **H2 (never isolate the OT controller).** Isolate(OpServer) is never allowed.

The shield reads only the SOC's own records (tickets, image timestamps). So
the H1 guarantee is **monitor-relative**: a compromise that starts after an
image and is never detected can still lose artefacts on reimage. We measure
this residual and report it; we do not claim "zero violations by
construction".

---

## 4. Learning conditions

The PPO implementation is shared. Conditions differ *only* in how safety is
handled (`safeacd/ppo.py`).

| ID | Reward | Lagrangian channels | Shield in training |
|---|---|---|---|
| `ppo` | security | – | no |
| `shaped` | security − β·(1·prod + 1·traffic + 10·evidence + 10·critical) | – | no |
| `lag` | security | all 4 (d = 10, 50, 0, 0) | no |
| `shield` | security | – | yes |
| `typed` (proposed) | security | prod, traffic (d = 10, 50) | yes |

**Reference policies:** SOC playbook (`safeacd/playbook.py`), no defence, random.

**Budgets: "no more harm than the incumbent playbook".** d_prod = 10 and
d_traffic = 50 per 100-step episode. These are the playbook's mean costs
under the training distribution (7.9 and 41.7 on the evaluation seeds), rounded up. The hard
channels have d = 0.

**Shared hyper-parameters** (not tuned per method):
- **PPO:** 8 envs × 256 steps; 10 epochs; minibatch 256; lr 3e-4; γ 0.99;
  GAE λ 0.95; clip 0.2; entropy 0.01; grad-norm 0.5.
- **Network: entity-based (host-shared) actor-critic.**
  - One 2-layer tanh encoder (64 units) is applied to every host's 7
    observed and 7 static features.
  - A permutation-invariant context (mean- and max-pool, plus link state and
    time) feeds a per-host head (6 logits per host) and a global head (Sleep,
    Block/Unblock).
  - The critic has the same structure with separate weights and 1 + 4 value
    heads.
  - This follows Symes Thompson et al. (AutonomousCyber@CCS 2024).
- **Lagrangian:** multipliers updated once per rollout by projected dual
  ascent, λ ← max(0, λ + 0.05·clip((J − d)/max(d, 1), −1, 1)). The combined
  advantage is (A_r − Σλ_k A_k)/(1 + Σλ_k), as in OmniSafe.
- **Invalid-action masking** is applied to every method. Safety masking
  applies only to shielded ones.

**Pilot findings to report (all on seed 0, which is excluded from the main
grid).**
1. **Architecture.** A flat 256-256 MLP did not learn ticket-conditional
   behaviour within 0.8 M steps. It sprayed `Remove` over hosts, and the
   Typed agent never used `Restore`. The host-shared architecture learns the
   rule once for all hosts and fixed this for every method.
2. **Dual dynamics.** Plain gradient ascent on λ overshot to about 10 within
   20 updates and made agents passive. Adam on λ (as in OmniSafe) removed the
   overshoot, but its long second-moment memory made λ decay very slowly once
   costs were under budget. Typed stayed over-conservative at R ≈ −555.
   Clipped, budget-normalised ascent bounds the step size in both
   directions (cf. PID-Lagrangian, Stooke et al. 2020).
3. **Early segmentation phase.** Shielded agents first learn to block the
   enterprise→OT link, which costs about 4× the traffic budget. They switch
   to "image → reimage" only after λ_traffic grows. Report this as an
   observed cost of the shield during learning.

---

## 5. Training and evaluation protocol

**Training.**
- 1 M environment steps per run (10,000 episodes).
- Seeds 1–10 (seed 0 was used only for pilots). The first execution ran
  all 10 seeds, seed-major. `run_experiments.py` is resumable: it skips any
  run whose `model.pt` exists.
- The training attacker is a uniform B-line/Meander mixture.
- Main grid: 5 methods × 10 seeds = 50 runs.
- β-sweep for `shaped`: β ∈ {0.1, 0.3, 3, 10} × 5 seeds = 20 runs. β = 1
  is in the main grid.
- Each run logs per-update episode return, the four mean episode costs, λ,
  and *cumulative* training-time costs. These are needed for safe-exploration
  claims.

**Evaluation (deployment).**
- Each trained policy is evaluated with a stochastic policy. The
  constrained object is the stochastic policy.
- 100 episodes per attacker: B-line, Meander and Stealthy (held out).
- Evaluation episode *i* uses the environment seed 1,000,000 + i for *every*
  policy (common random numbers).
- For `ppo`, `shaped` and `lag`, the same weights are also evaluated with the
  shield attached at deployment (the "+S" variants, for RQ3).
- An audit monitor counts how often each policy *chose* an action the shield
  would have blocked. This is the would-be intervention rate.

---

## 6. Metrics and statistics

**Per episode:**
- security return;
- impact steps;
- the four costs;
- constraint satisfaction indicators (cost ≤ budget, per channel and all
  four jointly).

**Per training seed:** means over its evaluation episodes. *The seed is the
unit of analysis* for learned agents.

**Reported:**
- mean with a 95% percentile-bootstrap CI over seeds (10,000 resamples);
- IQM over seeds for the security return (Agarwal et al. 2021);
- P(all constraints satisfied).

**Pre-registered tests** (`scripts/analyze.py → results/eval/tests.csv`):
1. Typed vs Lag, Shaped-β1, Shield and PPO on {return, prod, traffic, hard
   violations, P(all satisfied)}: two-sided Mann–Whitney U on seed means.
2. Lag vs Lag+S (RQ3), with the same test.
3. Typed, Lag and Shaped vs the playbook: per-episode values (averaged over
   seeds) paired with the playbook on the *same* CRN episode, using a
   Wilcoxon signed-rank test.

All p-values are Holm–Bonferroni corrected together, at α = 0.05. Effect
sizes (mean differences with CIs) are reported next to p-values.

**Figures:**
1. Harm–security Pareto plot, marking hard violations.
2. Learning curves: return, downtime, traffic and cumulative hard violations
   against steps (mean and 10–90% band over seeds).
3. The β-sweep against Typed and Lag.

**Tables:** main (training distribution), OOD (Stealthy), and the RQ3
ablation.

---

## 7. How to reproduce (exact commands)

```bash
pip install -e .[dev]           # numpy, torch (CPU is fine), pandas, scipy, matplotlib, pyyaml
pytest -q                       # 24 verification tests
python scripts/calibrate.py     # reference-policy calibration table (§3.3)
python scripts/run_experiments.py --config configs/experiment.yaml --jobs 4   # resumable
python scripts/evaluate_all.py   --config configs/experiment.yaml --jobs 4
python scripts/analyze.py        --config configs/experiment.yaml
python scripts/sensitivity.py    --config configs/experiment.yaml --jobs 4
python scripts/shield_variants.py --config configs/experiment.yaml --jobs 4
cd paper && latexmk -pdf main.tex
```

**Compute.**
- One 1M-step run takes about 20–25 min on one CPU core (4 runs in
  parallel on a 4-core machine).
- Seeds 1–5 plus the β-sweep is 37 runs, about 3.5 h on 4 cores.
- The full protocol (seeds 1–10) is 62 runs, about 6 h. Evaluation takes
  about 20 min.
- No GPU is needed.

**Smoke test** (about 3 min):
`python scripts/run_experiments.py --steps 20000 --seeds 0 --jobs 4`, then
the evaluation and analysis steps with `--episodes 5`.

---

## 8. Threats to validity (write these into the paper)

- **Internal.**
  - The Lagrangian and the shield share one PPO implementation, which limits
    confounds.
  - Hyper-parameters were not tuned per method. Shaped could improve with
    tuning, and the β-sweep partly addresses this.
  - Design choices (architecture, dual step) were made on seed-0 pilots of
    Typed and Lag, and applied to all methods. Seed 0 is excluded from the
    reported results.
- **Construct.**
  - Costs are proxies: downtime in host-steps and traffic in flow-weights.
  - "Evidence" is binary per incident. Real forensics is graded (volatile vs
    persistent artefacts, RFC 3227). This is a modelling simplification.
- **External.**
  - It is one 12-host topology with scripted attackers. The results are
    about *mechanisms*, not about deployment-ready agents.
  - Generalisation to CybORG/CAGE-4 is future work. The cost channels are
    designed to be portable as wrappers.
- **Conclusion.**
  - 10 seeds, CRN, bootstrap CIs and Holm correction.
  - Hard-violation rates near zero need exact-binomial upper bounds
    (rule of three: 0 events in n episodes ⇒ 95% UB ≈ 3/n).

---

## 9. Timeline to the ANNSIM'27 deadline (18 Dec 2026)

| Dates (2026) | Milestone |
|---|---|
| 5–11 Oct | Freeze simulator + protocol (this document); fill the [verify] bibliography items; run the full grid. |
| 12–25 Oct | Analysis, figures and tables. Sensitivity analysis. Decide on the CAGE-2 wrapper (optional stretch). |
| 26 Oct – 15 Nov | Write the full draft (12 pp). Internal review by one safe-RL and one DFIR reader. |
| 16 Nov – 6 Dec | Revise, make the artefact README reproducible, and make a tagged release (Zenodo DOI). |
| 7–17 Dec | Format to the official SCS/ANNSIM'27 template, final checks, **submit by 18 Dec**. |
| Feb–Mar 2027 | Rebuttal/revisions; camera-ready by 9 Mar 2027. |

---

## 10. Change log
- 2026-10-04 (before the main grid):
  - Simulator fixes:
    - random initial foothold;
    - usable-foothold re-phishing;
    - incident-scoped evidence;
    - faster `valid_mask`/`feasible_exploits`, with identical trajectories
      verified by a hash.
  - Learner changes:
    - entity-based actor-critic (hidden 64);
    - clipped normalised dual ascent;
    - main-grid seeds 1–10 (seed 0 is reserved for pilots).
  - Pilots (seed 0): plain-SGD dual → Adam dual → clipped dual; flat MLP →
    entity network. See §4.
- 2026-10-04/05 (main grid):
  - An interim look at seeds 1–5 showed the pre-registered test cannot reach
    significance at n = 5. With 5 vs 5 seeds the minimum exact two-sided
    Mann–Whitney p is 0.0079, which cannot survive Holm correction over the
    test family.
  - We therefore ran the planned n = 10 seeds rather than change the test.
    No hypothesis, metric or test was altered.
  - Two container restarts interrupted in-flight runs. Interrupted runs were
    re-run from scratch, since resumption is per run and not mid-run.
    Completed runs are unaffected.
  - Added after the main grid, as clearly labelled exploratory analyses:
    - an image-freshness variant of H1 at deployment
      (`scripts/shield_variants.py`);
    - a P(no hard violation) metric with Clopper–Pearson bounds.

---

## 11. Outcome against the pre-registered hypotheses (10 seeds, 4–5 Oct 2026)

Numbers are from `results/eval/summary_train_dist.csv`, `tests.csv` and
`hard_violation_bounds.csv`. p-values are Holm-corrected.

| Hypothesis | Outcome | Evidence |
|---|---|---|
| H1a reward-only PPO beats the playbook on security but violates every constraint | **Supported** | R −1.5 vs −46.0; downtime 494, traffic 1181, evidence 4.3/ep, OT isolated 60/100 steps |
| H1b no β gives both safety and security | **Supported** | β ≤ 1: hard violations in ≥ 94% of episodes; β ≥ 3: safe but R −129.5 / −187.2 (Typed −61.0) |
| H2a Typed meets all constraints more often than Lag and Shaped | **Supported** | 69.5% vs 33.8% (p = 0.006) and 5.7% (p = 0.006) |
| H2b near-zero hard violations, train and deploy | **Supported, with a qualifier** | 47/2000 deployment episodes (97.6% violation-free), all monitor-relative H1 misses. Training-time cumulative hard violations are about 2 orders of magnitude below Lag |
| H2c Typed security not worse than Lag | **Supported** | −61.0 vs −60.5 (p = 1.0) |
| H3 deploy-only shield costs security | **Supported for Shaped** (−36 → −427, p = 0.006); **directional for Lag** | Lag −60.5 → −131.8, p = 0.18 (high seed variance) |
| H4 hard rules hold OOD, budgets degrade for all | **Partly supported** | Budgets: every method exceeds them, including the playbook. H2 holds exactly. H1 degrades to 90.1% violation-free (Lag 33.4%, Shaped 0.9%). A freshness window of ≤ 3 steps restores 98.0% |

**Honest negative.** The SOC playbook, whose mean harm defines the budgets,
has better security than Typed at equal harm (−46.0 vs −61.0, paired CRN
p < 1e−9). The paper frames its contribution as *how to constrain* a learned
defender, not as RL beating playbooks.
