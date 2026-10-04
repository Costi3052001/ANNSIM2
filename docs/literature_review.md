# Literature review and positioning (snapshot: 4 Oct 2026)

**Target venue.** ANNSIM'27 (SCS Annual Modeling and Simulation Conference),
Wrocław University of Science and Technology, Poland, 17–21 May 2027.
Paper submission deadline **18 Dec 2026**, notification 9 Feb 2027,
camera-ready 9 Mar 2027 ([scs.org/annsim](https://scs.org/annsim/)). ANNSIM has a dedicated track,
*Modeling and Simulation in Cyber Security (MSCS)*, which ran in ANNSIM'26 ([CFP](https://scs.org/wp-content/uploads/2025/10/ANNSIM-2026-Call-For-Papers-UCF.pdf)).
Full papers: up to 12 pages in the SCS template. **Check the page limit and the
template against the official ANNSIM'27 CFP before submission.**

> **How these sources were checked.** The build container could not reach
> arXiv, ACM, IEEE or scs.org directly. Every 2025–2026 item below was checked
> through web search (titles, abstracts, venues) on 4 Oct 2026. Items marked
> **[verify]** still need their author list or page numbers copied from the
> landing page before submission. The safe-RL foundations are standard, widely
> cited papers.

## 1. The gap in one paragraph

RL for autonomous cyber defence (ACD) has mostly optimised one scalar reward.
Operational harms such as restore penalties, or blocked users in CAGE-4, enter
that reward as hand-tuned penalties. Two 2025–2026 lines of work have exposed
the problem:

- **Reward design.** Bates, Hicks and Mavroudis show that dense, engineered
  rewards bias agents towards riskier policies.
- **Constrained MARL.** The *safety-contract* framework (ACD³-GAT, Jun 2026)
  moves operational costs into a CMDP. Its budgets cover MTTR, false-positive
  responses and firewall change-management disruption, enforced with
  Lagrangian multipliers.

**What no published RL-ACD work does (as of this snapshot):**

1. Treat **forensic-evidence destruction**, i.e. reimaging a compromised host
   before it is imaged, as a first-class, *irreversible* constraint.
2. Separate constraints by **type**, and give each type the mechanism that fits
   its semantics:
   - *budget* constraints (downtime, blocked traffic), where some harm is
     acceptable, go to Lagrangian relaxation;
   - *hard / irreversible* constraints (evidence, safety-critical OT assets),
     where any violation is unacceptable, go to a runtime shield built only
     from SOC-observable information.
3. Make that comparison in a **simulation-methodology** frame: separate cost
   channels, common-random-number evaluation, held-out attacker and
   monitor-relative guarantees.

That is the paper's niche.

## 2. Closest related work

| Work | Setting | Harm modelled | Mechanism | Evidence? | Typed? |
|---|---|---|---|---|---|
| Dutta, Al-Shaer & Chatterjee, *Constraints Satisfiability Driven RL for ACD*, AICA 2021 (arXiv:2104.08994) | custom | safety/security requirements as SMT constraints | SMT verification in the decision loop | no | no |
| Bates, Hicks & Mavroudis, *Less is more? Rewards in RL for Cyber Defence*, arXiv:2503.03245 (2025) | adapted cyber gym, 2–50 nodes | costly actions via reward | sparse vs dense rewards | no | no |
| Bates, Hicks & Mavroudis, *Beyond Rewards in RL for Cyber Defence*, arXiv:2602.04809 (2026) | Yawning Titan, CAGE-2 | risky/costly actions | reward structure + ground-truth eval | no | no |
| *Safety-Contract Graph MARL for Autonomous Network Security Response* (ACD³-GAT / C-MAPPO-GAT), arXiv:2606.13832 (Jun 2026) **[verify authors]** | CybORG, MARL | MTTR, false-positive response, firewall change disruption | Lagrangian + budget-aware counterfactual screening, CVaR | no | no (all budgets) |
| Hsain & Almuhammadi, *Shielded Analysis …*, arXiv:2606.13621 (Jun 2026) | two-player safety game | defender spec | shield synthesis as *design-time* defensibility analysis | no | no |
| Jamshidi, Shahabi, Khomh, Fung & Hamdaqa, *Multi-Agent LLM Governance for Safe Two-Timescale RL in SDN-IoT Defense*, arXiv:2604.01127 (2026) | SDN-IoT | control-plane stability, QoS | PPO + action masking + LLM-edited policy constitution | no | partially |
| Kiely et al., *CAGE Challenge 4*, AI Magazine 46(3), 2025, doi:10.1002/aaai.70021 | CybORG MARL | green-agent service disruption | penalty in general-sum reward | no | no |
| Tholl et al., *Towards Production-Worthy Simulation for ACO*, arXiv:2508.19278 (2025) | CAGE-2 extended (Patch/Isolate/Unisolate) | — | reward/feature redesign | no | no |
| Nyberg, Sommestad, Buhaiu, Loxdal, Johnson & Ekstedt, *A Cyber Range Evaluation of Autonomous Network Incident Response Agents*, arXiv:2609.16541 (2026) | emulated cyber range | availability costs | combined-cost RL vs heuristics | no | no |
| **This paper** | SafeACD-Sim (CAGE-2-like, 4 cost channels) | downtime, blocked traffic, **evidence**, OT-critical isolation | **typed:** Lagrangian for budgets + observation-based shield for hard rules | **yes** | **yes** |

### Supporting context
- **Environments and benchmarks.**
  - Standen et al., *CybORG: A Gym for the Development of Autonomous Cyber
    Agents*, IJCAI-21 ACD workshop (arXiv:2108.09118).
  - Microsoft *CyberBattleSim* (2021).
  - *Building Better Environments for Autonomous Cyber Defence*,
    arXiv:2604.08805 (2026) **[verify authors]**: a workshop of 25 experts
    gives best-practice guidelines for ACD environments and evaluation. We
    follow these guidelines and cite them.
- **Surveys.**
  - Vyas, Hannay, Bolton & Burnap, *Automated Cyber Defence: A Review*,
    arXiv:2303.04926 (2023).
  - Palmer, Parry, Harrold & Willis, *Deep RL for Autonomous Cyber Operations:
    A Survey*, arXiv:2310.07745 (2023/24).
  - Kott (ed.), *Autonomous Intelligent Cyber Defense Agent (AICA)*, Springer
    (2023).
- **Representative ACD agents.**
  - Foley, Hicks, Highnam & Mavroudis, *Autonomous Network Defence using RL*,
    ASIA CCS '22 (arXiv:2409.18197).
  - Singh et al., *Hierarchical Multi-agent RL for Cyber Network Defense*,
    RLC 2025 (arXiv:2410.17351) **[verify authors]**.
  - Hammar & Stadler, *Intrusion Prevention Through Optimal Stopping*, IEEE
    TNSM 2022.
- **Other explainable / constrained ACD (2026).** *Explainable Autonomous Cyber
  Defense using Adversarial MARL* (C-MADF), arXiv:2604.04442 / Expert Systems
  with Applications **[verify authors]**: causal constraints plus a "Council of
  Rivals" dual policy.

## 3. Safe-RL foundations we build on
- Altman, *Constrained Markov Decision Processes*, Chapman & Hall/CRC, 1999. CMDP formalism.
- Achiam, Held, Tamar & Abbeel, *Constrained Policy Optimization*, ICML 2017.
- Tessler, Mankowitz & Mannor, *Reward Constrained Policy Optimization*, ICLR 2019.
- Ray, Achiam & Amodei, *Benchmarking Safe Exploration in Deep RL*, OpenAI tech. report, 2019. PPO-Lagrangian.
- Stooke, Achiam & Abbeel, *Responsive Safety in RL by PID Lagrangian Methods*, ICML 2020. Lagrangian oscillation and overshoot, which we observed in pilots (Section 4).
- Alshiekh, Bloem, Ehlers, Könighofer, Niekum & Topcu, *Safe RL via Shielding*, AAAI 2018. Preemptive and post-posed shields.
- Huang & Ontañón, *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*, FLAIRS 2022. Masking keeps policy gradients valid.
- Grinsztajn, Ferret, Pietquin, Preux & Geist, *There Is No Turning Back: Reversibility-Aware RL*, NeurIPS 2021. Irreversible actions.
- García & Fernández, *A Comprehensive Survey on Safe RL*, JMLR 16, 2015; Gu et al., *A Review of Safe RL: Methods, Theories and Applications*, IEEE TPAMI 2024.
- Ji et al., *OmniSafe*, JMLR 25, 2024. Our dual update (Adam on λ) follows its PPO-Lag.

## 4. Forensics / incident-response grounding (why "evidence" is a hard constraint)
- NIST SP 800-86, *Guide to Integrating Forensic Techniques into Incident
  Response* (Kent, Chevalier, Grance & Dang, 2006).
- NIST SP 800-61r3, *Incident Response Recommendations and Considerations for
  Cybersecurity Risk Management: A CSF 2.0 Community Profile* (2025). Says
  response must preserve evidence while containing the incident.
- RFC 3227, *Guidelines for Evidence Collection and Archiving* (Brezinski &
  Killalea, 2002). Order of volatility.

The practitioner rule is "image before you reimage". Reimaging restores
operations but erases forensic artefacts for good, and that loss cannot be
traded back later. This is why the paper does not model evidence as a
penalty or a budget.

## 5. Evaluation-methodology references
- Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS
  2021: IQM, bootstrap CIs, multiple seeds.
- Henderson et al., *Deep RL that Matters*, AAAI 2018.
- Law, *Simulation Modeling and Analysis*, 5th ed., McGraw-Hill, 2015: common
  random numbers for paired comparison, the variance-reduction technique an
  ANNSIM audience expects.

## 6. Why this is a good fit for ANNSIM (and publishable)
- **Venue fit.**
  - The contribution is partly a *simulation model*: a documented,
    open-source, fast discrete-time simulator with explicit harm channels.
  - It is partly a *simulation-based evaluation methodology*: CRN paired
    tests, held-out attacker, monitor-relative guarantees.
  - Both are what the MSCS track asks for.
- **Novelty is narrow and defensible.** We do not claim a new RL algorithm.
  The paper makes three claims:
  1. Evidence destruction is a distinct, irreversible harm type.
  2. Mechanism should follow constraint type.
  3. A clear empirical demonstration of (1) and (2) with honest residuals.
- **Scope fits a conference.** One simulator, 5 learning conditions plus 3
  baselines, 10 seeds, 3 attackers. It runs on a 4-core laptop/CPU in a few
  hours.
- **Timely.** It extends three 2026 preprints (Bates et al.; ACD³-GAT;
  Hsain & Almuhammadi) in a direction none of them takes.
