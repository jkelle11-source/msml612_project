# Milestone 10 — Final Report & Code Cleanup

> **GitHub Milestone**
> **Title:** `M10 — Final Report & Code Cleanup`
> **Timeframe:** Week 10 · Dec 2 – Dec 9
> **Depends on:** `M9`
> **Description:**
> Land the plane: a reproducible repo and the **Final Report (due Wed Dec 10)**. Jake makes the code genuinely reproducible — a README with one-command reproduction per variant (command → reported-result table), fixed/logged seeds, docstrings on every class/function, and the uncertainty fan plots (20 draws + 90% conformal band if it was completed). Josh + Sankha write the **Final Report** following the §9 structure. This is a shared, back-loaded week by design (§6); the reproducibility work and the writing proceed in parallel and meet at the results/figures the report cites.

## Assignment summary
- **Jake** — README with one-command reproduction per variant (command → reported-result table); fixed/logged seeds; docstrings on every class/function; uncertainty fan plots (20 draws + 90% conformal band if completed)
- **Josh + Sankha** — Write the **Final Report** (structure in §9)
- **All three** — Final review pass across report + repo

## Coordination notes
- **Reproducibility is a graded line** (§8, "clean reproducible code," 20 pts). The README's command→result table is the artifact that proves it: a grader runs one command per variant and sees the reported number. Fixed/logged seeds and docstrings are part of the same score.
- **The report structure is already fixed (§9)** — Introduction, Related Work, Method (Ext 1 honest framing, Ext 2 TSLO + true Time2Vec + why no attention bias, Ext 3 if done), Experiments (normalization, dual-mask, paired 5-seed, headline paired-Δ ± SE, ablation, per-variable, runtime, coverage), Discussion (main effects vs. exploratory interaction; nulls as findings), Limitations, Conclusion/future work.
- **Nulls stay stated as findings** (§4.5, §8). The Discussion frames flat contrasts as rigor, and Limitations names the real ones: patient-level exchangeability, grid vs. native-timestamp TSLO, reduced scale vs. published, single dataset.
- Cite the reused CSDI implementation and all references (§11) — this is its own graded line (10 pts).

---

## Issues

### `M10-1` — Reproducible repo: README, seeds, docstrings, fan plots
- **Assignee:** @jkelle11-source
- **Labels:** `infra`, `reproducibility`, `analysis`, `mvp`
- **Blocked by:** `M9-3`
- **Blocks:** `M10-2`

**Context.** Make the repo something a grader can *run*, not just read. Jake owns it as the infra/engine owner: one documented command per variant that reproduces its reported number, seeds fixed and logged everywhere, docstrings on every class/function, and the final uncertainty fan plots. This is the "clean reproducible code" score made concrete (§8).

**Deliverables**
- README with a **command → reported-result** table: one command per variant reproducing its headline number.
- Fixed and logged seeds throughout (data order, train mask, init); the `EvalMasker` seed documented as the frozen project seed.
- Docstrings on every class/function (single canonical class names, code matching the math — §8).
- Uncertainty fan plots (20 draws + 90% conformal band if `M9-3` completed conformal).

**Definition of Done**
- [ ] Each documented command reproduces its reported result within the stated tolerance.
- [ ] Seeds fixed/logged; a fresh clone reproduces a variant end-to-end.
- [ ] All classes/functions have docstrings; class names are canonical.
- [ ] Fan plots committed and referenced by the report.

---

### `M10-2` — ▶ DELIVERABLE: Final Report (due Wed Dec 10)
- **Assignee:** @JoshOlu, @SankhaS
- **Labels:** `writing`, `deliverable`
- **Blocked by:** `M9-1`, `M9-2`, `M10-1`
- **Blocks:** —

**Context.** The graded final report, following the fixed §9 structure. Josh + Sankha write it (Jake in support via the repo/reproducibility artifacts and figures), with all three doing a final review. The report's spine is the honest-performance story: reproduction + runtime as guaranteed wins, extension effects (including nulls) reported rigorously on top (§8).

**Deliverables**
- Final report following §9: Introduction, Related Work, Method (Ext 1/2/3), Experiments (normalization, dual-mask, paired 5-seed, headline paired-Δ ± SE, ablation, per-variable, runtime, coverage), Discussion, Limitations, Conclusion/future work.
- All figures/tables sourced from the frozen results (`M9-1`, `M9-2`, `M8-2`, `M9-3`).
- Full reference list (§11) and the reused-CSDI citation.

**Definition of Done**
- [ ] Final report submitted by **Wed Dec 10**.
- [ ] Structure matches §9; every results claim traces to a committed number/figure.
- [ ] Nulls stated as findings; limitations section complete.
- [ ] References complete; reused CSDI implementation cited.
- [ ] Reviewed by all three before submission.
