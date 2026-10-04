# ANNSIM'27 submission checklist (deadline 18 Dec 2026)

## A. Before the final experiment run
- [ ] Re-read `docs/methodology.md` §2 (hypotheses). Do not edit them after
      seeing the 10-seed results; record any deviation in §10.
- [ ] `pytest -q` is green.
- [ ] `python scripts/calibrate.py` reproduces Table 2 of the paper exactly.
- [ ] Run the remaining seeds:
      `python scripts/run_experiments.py --jobs <cores>`. It resumes and
      adds seeds 6–10. Then run `evaluate_all.py`, `analyze.py` and
      `sensitivity.py`.

## B. Paper content
- [ ] Replace every `\todo{}` in `paper/main.tex`. `grep -n todo paper/main.tex`
      must return nothing.
- [ ] Every number in the text comes from `paper/results_macros.tex` (from
      `analyze.py`). Never type numbers by hand.
- [ ] Each RQ paragraph states the hypothesis, the effect size with CI, and
      the Holm-corrected p-value from `results/eval/tests.csv`.
- [ ] Zero hard-violation claims quote the Clopper–Pearson bound
      (`results/eval/hard_violation_bounds.csv`), e.g. "0/1000 episodes,
      95% UB 0.30%".
- [ ] Report the honest negatives:
  - the playbook vs Typed security gap;
  - the OOD degradation of budgets for all methods;
  - residual monitor-relative H1 violations, if any.
- [ ] Threats-to-validity paragraph matches methodology §8.

## C. Bibliography
- [ ] Fill the `TODO-VERIFY` entries in `paper/refs.bib`:
  - authors of arXiv:2606.13832 (ACD³-GAT);
  - authors of arXiv:2604.08805 (environment workshop);
  - the full CAGE-4 author list;
  - the NIST SP 800-61r3 authors.
- [ ] Re-check arXiv versions for any 2026 preprint that has since appeared at
      a venue, and cite the venue version.
- [ ] Search again (Google Scholar, arXiv cs.CR/cs.LG) in late November for
      new "safe RL + cyber defence + forensics/evidence" work. If something
      overlaps, add a sentence to Related Work explaining the difference.

## D. Format (SCS / ANNSIM)
- [ ] Download the official ANNSIM'27 LaTeX template from scs.org/annsim and
      port `main.tex` (title block, keywords, SCS copyright footer,
      reference style).
- [ ] Check the page limit (expected: 12 pages including references for full
      papers) and whether the review is double-blind.
- [ ] Figures are vector PDFs and readable in grayscale (markers and line
      styles differ, not only colour).
- [ ] Track: *Modeling and Simulation in Cyber Security*.

## E. Artefact
- [ ] Anonymise the repository for review (e.g. anonymous.4open.science).
- [ ] Tag a release and archive it on Zenodo (DOI) for camera-ready.
- [ ] Commit `results/eval/*.csv` and `results/runs/*/train_log.csv`. Model
      weights are optional: they are small, so a Zenodo upload is fine.

## F. Page budget (12 pp target)
| Section | Pages |
|---|---|
| Abstract + Introduction | 1.25 |
| Background & Related Work | 1.25 |
| Typed Safety Constraints (+ Fig. 1) | 1.0 |
| SafeACD-Sim model + V&V (Tables 1–2) | 2.0 |
| Experimental Design | 0.75 |
| Results (Table 3, Figs 2–3, Table 4 OOD/ablation) | 3.0 |
| Discussion, Threats, Conclusion | 1.0 |
| References | 1.5 |
