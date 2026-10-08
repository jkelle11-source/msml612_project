# Milestone 9 — Final Eval, Stretch Conformal & Presentation

> **GitHub Milestone**
> **Title:** `M9 — Final Eval, Stretch & Presentation`
> **Timeframe:** Week 9 · Nov 26 – Dec 1 *(Thanksgiving-light / buffer week, §6)*
> **Depends on:** `M8`
> **Description:**
> The consolidation week, timed light because of Thanksgiving (Nov 26). Sankha produces the **final best-config results table** per variant. Josh runs the **per-variable analysis** (all 35 variables: top-5/bottom-5 improvers, testing whether Ext-2 gains concentrate on high-autocorrelation variables). Jake, **if the conformal gate passed**, implements the **studentized patient-level conformal** intervals (§5.3); otherwise he produces uncertainty fan plots. All three build and rehearse the **Presentation (due Tue Dec 2)** and assign speaking parts. Keep the load genuinely light — this is the planned buffer, and it protects the final report week from spillover.

## Assignment summary
- **Sankha** — Final best-config eval per variant → final results table
- **Josh** — Per-variable MAE (all 35): top-5 / bottom-5 improvers; test whether Ext-2 gains concentrate on high-autocorrelation variables
- **Jake** — *(if gate passed)* Studentized patient-level conformal (§5.3): coverage, width, coverage-by-missingness-rate; *else* uncertainty fan plots
- **All three** — Build and rehearse the **Presentation**; assign speaking parts

## Coordination notes
- **This is the buffer week (§6).** Thanksgiving falls Nov 26; keep scope tight and use the slack to absorb any Week-7/8 spillover rather than adding new work.
- **Conformal must not repeat CSDI's sin** (§5.3). If attempted: split calibration/test **by patient**, aggregate the nonconformity score at the **patient** level, and use **studentized** scores (`s = |x_true − μ̂| / (σ̂ + ε)`) with adaptive intervals from the K=50 posterior draws. Report coverage (target ≥ 90%), mean width, and **coverage stratified by patient missingness rate**. Position-level coverage is a secondary check, not the guarantee.
- **The presentation leads with the guaranteed wins** (§8): benchmark reproduction and the MAE-vs-latency curve first, then each extension in its best honest regime, with the significance bar stated up front so nulls read as rigor.

---

## Issues

### `M9-1` — Final best-config results table
- **Assignee:** @SankhaS
- **Labels:** `eval`, `stats`, `analysis`, `mvp`
- **Blocked by:** `M7-2`, `M8-1`, `M8-2`
- **Blocks:** `M10-2`

**Context.** Consolidate the headline sweep, the ablation, and the runtime study into the single final results table the report and slides cite. Sankha owns it on the eval/stats track. This is where the confirmatory main effects and the exploratory interaction get their final, frozen numbers.

**Deliverables**
- Final table: all four variants (mean ± SE @10%, paired Δ ± SE), best-config selection justified, baselines included, block/random stratification retained.
- A one-paragraph results summary stating which effects are SE-separated and which are null.

**Definition of Done**
- [ ] Final numbers frozen and consistent with the raw results store.
- [ ] Best config per variant justified from the ablation.
- [ ] Effect verdicts (separated vs. null) stated explicitly.

---

### `M9-2` — Per-variable analysis (all 35 variables)
- **Assignee:** @JoshOlu
- **Labels:** `architecture`, `analysis`
- **Blocked by:** `M7-3`
- **Blocks:** `M10-2`

**Context.** The mechanistic story behind the aggregate numbers (§6, §8): per-variable MAE across all 35 variables, ranking the top-5/bottom-5 improvers, and testing the hypothesis that Ext-2 gains concentrate on **high-autocorrelation variables** (HR, SpO2) where recency matters most. Josh owns it, building on the Week-7 qualitative work.

**Deliverables**
- Per-variable MAE (all 35) for base vs. best variant, with top-5/bottom-5 improvers identified.
- An explicit check of whether Ext-2 improvements concentrate on high-autocorrelation variables.
- A short written interpretation for the report/slides.

**Definition of Done**
- [ ] Per-variable MAE computed for all 35 variables.
- [ ] Top/bottom improvers listed; the high-autocorrelation hypothesis addressed with evidence.
- [ ] Interpretation committed for reuse in `M10-2`.

---

### `M9-3` — Studentized patient-level conformal *(or uncertainty fan plots)*
- **Assignee:** @jkelle11-source
- **Labels:** `diffusion`, `conformal`, `stretch`, `analysis`
- **Blocked by:** `M8-3`
- **Blocks:** `M10-1`

**Context.** The stretch, gated by `M8-3`. If go: a post-hoc, no-retraining conformal wrapper done *correctly* — patient-level exchangeability and studentized scores that use the posterior spread CSDI discards (§5.3). If no-go: uncertainty fan plots instead, which are valuable regardless. Jake owns it (inference/calibration track).

**Deliverables**
- **If gate passed:** studentized patient-level split-conformal at 1−α = 0.90 — empirical coverage (target ≥ 90%), mean interval width, and coverage stratified by patient missingness rate; K=50 posterior draws.
- **If not:** uncertainty fan plots (posterior draws + observed) for a representative patient set.

**Definition of Done**
- [ ] Chosen path (conformal or fan plots) completed and its outputs committed.
- [ ] If conformal: coverage/width/stratified-coverage reported; patient-level split verified; position-level coverage reported only as a secondary check.
- [ ] Outputs ready for the final report's uncertainty section.

---

### `M9-4` — ▶ DELIVERABLE: Presentation (due Tue Dec 2)
- **Assignee:** @jkelle11-source, @JoshOlu, @SankhaS
- **Labels:** `writing`, `presentation`, `deliverable`
- **Blocked by:** `M9-1`
- **Blocks:** —

**Context.** The graded presentation. Lead with the guaranteed high-performance wins — benchmark reproduction and the MAE-vs-latency curve — then each extension in its best honest regime, with the significance bar stated up front (§8). Shared work with assigned speaking parts.

**Deliverables**
- Slide deck covering: problem/motivation, method (CSDI + the two extensions), reproduction + gate, headline paired-Δ results, ablation + runtime plot, uncertainty (conformal or fan plots), honest nulls and limitations.
- Assigned speaking parts and at least one full rehearsal.

**Definition of Done**
- [ ] Presentation delivered/submitted by **Tue Dec 2**.
- [ ] Deck leads with reproduction + runtime per the §8 strategy.
- [ ] Speaking parts assigned; rehearsed at least once.
