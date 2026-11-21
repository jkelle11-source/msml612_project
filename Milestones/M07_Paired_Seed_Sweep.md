# Milestone 7 — Full Paired 5-Seed Training

> **GitHub Milestone**
> **Title:** `M7 — Full Paired 5-Seed Training`
> **Timeframe:** Week 7 · Nov 12 – Nov 18
> **Depends on:** `M6`
> **Description:**
> Run the headline experiment. Jake executes the **full paired sweep** — all four variants × seeds 42–46 (20 runs, upstream's fixed 200 epochs each, final weights scored), plus base and Ext 1 in the block-aware training regime × seeds 42–46 (10 runs, §3.1) — on the harness proven in Week 6. Sankha populates the **headline table**: mean ± SE per variant and, crucially, the **paired Δ vs. base with its SE**, stratified by missingness type (block vs. random). Josh produces the qualitative story: reconstructed trajectories for 5 patients × 3 variables against mean/forward-fill. The statistical discipline is fixed (§3.5): main effects (Ext 1 vs. base, Ext 2 vs. base) are **confirmatory**; the interaction is **exploratory**, and any contrast whose paired-Δ error bar overlaps zero is reported **"no significant effect,"** never spun.

## Assignment summary
- **Jake** — Full training (upstream's fixed 200 epochs, final weights scored) for all four variants × seeds 42–46 (20 runs), plus base and Ext 1 in the block-aware regime × seeds 42–46 (10 runs): 30 in total; then base's final weights in the random regime scored at 50% and 90% (§6's robustness columns)
- **Sankha** — Populate the **headline table**: mean ± SE and **paired Δ vs. base with its SE** and 95% CI, each extension's gate beside its Δ (§3.5); stratify by missingness type (block vs. random). Main effects confirmatory, interaction exploratory
- **Josh** — Qualitative trajectories (5 patients × 3 variables: HR, serum sodium, GCS); temporal coherence vs. mean/forward-fill

## Coordination notes
- **GPU access is the schedule risk** (§8). 30 × 200-epoch runs need reliable single-GPU time: on the GPU `M6-2`'s timing test picked, Colab's G4 runtime (`NVIDIA A100-SXM4-80GB`, TF32 on; §3.3), about 15 h of training, 7 h of validation and 45 h of test scoring (§8 risk register). Every run trains and is scored on that GPU. When that GPU is not free, a run waits rather than taking another GPU, which would unpair it from its seed's other variants. If it's shared/queued, start Monday and checkpoint aggressively so a preemption doesn't cost a whole cell. Run the 20 random-regime runs first: if time runs out, the 10 block-aware runs drop to 3 seeds and the block contrast is labelled exploratory (§8 risk register). Base's 50%/90% scoring (10 passes, about 6.5 h) comes last and is the first cut: to seeds 42–44, as the gate, stated as such.
- **Report the paired Δ, not two independent means** (§3.3). The SE of the paired difference is what separates signal from noise here; expected gains (~0.01–0.02) are the same magnitude as CSDI's masking sensitivity, which is *why* masking is frozen and seeds are paired (§6).
- **Nulls are findings, stated as such** (§3.5, §7). An overlapping error bar is "no significant effect," reported plainly, with its 95% CI, so it states the largest gain it leaves possible, and with the extension's gate, so it shows whether the model used the extension (§3.5). This is the honesty bar the whole grade strategy leans on.
- Block-vs-random stratification matters: Ext 1's honest home is **block missingness** and Ext 2's is **high-autocorrelation variables** (§4.1, §4.2), so the table must let those regimes be read separately.
- **Ext 1's block contrast comes from the block-aware regime** (§3.1): base vs. Ext 1, both trained with `TrainMasker`'s `mix` strategy and scored on `BlockEvalMasker`. The random regime's block scores sit beside it as a carry-over check. Each regime is paired only within itself; no Δ crosses regimes.

---

## Issues

### `M7-1` — Full paired 5-seed sweep (30 runs)
- **Assignee:** @jkelle11-source
- **Labels:** `diffusion`, `training`, `protocol`, `mvp`, `critical-path`
- **Blocked by:** `M6-1`, `M6-2`
- **Blocks:** `M7-2`, `M8-1`, `M8-2`, `M8-4`, `M9-1`

**Context.** Execute the experiment the whole project has been building toward: four variants × seeds 42–46 under common random numbers, upstream's fixed 200 epochs with the final weights scored (§3.1), plus base and Ext 1 × seeds 42–46 in the block-aware training regime (§3.1), which gives Ext 1 a fair test in its block-missingness regime. Base's final weights in the random regime are also scored at CSDI's two robustness rates, 50% and 90%: the 50% and 90% columns of §6's reproduced-CSDI row, which no other issue measures (gap found at `M4-1`). Jake owns it as the training-engine/harness-execution owner. The output is the raw per-(variant, regime, seed) results that feed the headline table, the ablation, and the runtime analysis.

**Deliverables**
- 30 completed 200-epoch runs, checkpointed and tracked in wandb: 4 variants × 5 seeds in the random regime, then base and Ext 1 × 5 seeds in the block-aware regime.
- Per-run final MAE/RMSE @10% (standardized) under both holdouts (random `EvalMasker`, block `BlockEvalMasker`; §3.1) written to the structured results store, tagged by variant + regime + seed + holdout. Scores come from each run's final weights; validation MAE, tracked for the curves only, uses the holdout the regime trains for: `EvalMasker`'s in the random regime, `BlockEvalMasker`'s in the block-aware regime.
- Base (random regime) × seeds 42–46 also scored at 50% and 90%: `EvalMasker` at `protocol.robustness_fractions`, frozen sampler, final weights (`diffusion.score --fraction 0.5` and `--fraction 0.9`, `M4-1`). 10 passes of about 39 min each on the sweep's GPU, scored after the 30 runs, recorded in the results store with the fraction in the key, and their mean ± SE filling §6's reproduced-CSDI row at 50% and 90%. If compute runs short this is cut first, to seeds 42–44 as in the gate, and the cut is stated.
- A run manifest (config, regime, seed, GPU, wall-clock, checkpoint path) for reproducibility and the runtime study. For each run with an extension it also records the extension's final gate, read from the run's `model.pt` (`mask_encoder.gate`, `time_encoder.gate`), which `M7-2` reports beside the extension's Δ (§3.5).

**Definition of Done**
- [x] All 30 cells complete (or any incomplete cell is documented with cause + plan; a block-aware regime cut to 3 seeds is stated as exploratory).
- [ ] Results land in the structured store keyed by `(variant, regime, seed)`.
- [ ] Runs are resumable from checkpoints; the manifest reproduces any cell.
- [x] Pairing integrity confirmed (per-seed shared RNG across variants within each regime, per `M6-2`).
- [ ] Every run's manifest entry names the GPU `M6-2` picked, for training and for scoring (§3.3).
- [ ] Every extension run's manifest entry records its final gate (§3.5).
- [ ] Base (random regime) scored at 50% and 90% on 5 seeds, or on 42–44 with the cut stated; mean ± SE recorded for §6's reproduced-CSDI row.

---

### `M7-2` — Headline table: mean ± SE and paired Δ ± SE
- **Assignee:** @SankhaS
- **Labels:** `eval`, `stats`, `analysis`, `mvp`
- **Blocked by:** `M7-1`, `M3-3`
- **Blocks:** `M8-1`, `M8-4`, `M9-1`

**Context.** Turn the 30 runs into the defensible headline result (§3.3, §6). Compute each variant's mean ± SE and — the number that actually decides whether an extension "counts" — the **paired Δ vs. base with its SE**, stratified by block vs. random missingness. Sankha owns it on the eval/stats track. Main effects are confirmatory; the interaction is exploratory and will not be over-asserted at 5 seeds (§3.5).

**Deliverables**
- Headline table: base / +Ext1 / +Ext2 / +both, each mean ± SE @10%, plus baselines (mean, forward-fill, BRITS/fallback, and cited BRITS/V-RIN/CSDI numbers).
- A note under the table that our ± covers training randomness only: the test split is fixed, so every run scores the same 800 patients. The cited standard errors (CSDI, Table 3) also cover which patients were tested, because each of CSDI's five trials tested a different fifth of them (its Appendix E.2.1).
- **Paired Δ vs. base ± SE** for each extension and for the combined model, each with its 95% CI (t with n − 1 degrees of freedom: ±2.78 SE at 5 seeds, ±4.30 SE at 3; §3.5).
- Stratified view: every variant and baseline under both frozen holdouts, random (`EvalMasker`) and block (`BlockEvalMasker`, `M2-4`), at the same 10% rate, so Ext 1's and Ext 2's honest regimes read separately.
- **Ext 1's block contrast:** paired Δ ± SE of Ext 1 vs. base in the block-aware training regime, scored on the block holdout (§3.1, §4.1), with the random regime's block-holdout Δ beside it as a carry-over check. Whether it separates from zero is the trigger for Ext 1's attribution checks (`M8-4`), stated in one line as soon as it is known.
- A written verdict per contrast: SE-separated effect vs. "no significant effect," each null also stating the largest gain its 95% CI allows.
- Each extension's final gate, mean ± SE over seeds per regime, from `M7-1`'s manifest, beside its Δ (§3.5): a gate that moved away from 0 shows the model used the encoder.

**Definition of Done**
- [ ] Table reports mean ± SE for all four variants and all baselines in standardized units.
- [ ] Paired Δ ± SE computed correctly (paired, not independent-means) and spot-checked by hand on one contrast.
- [ ] Block/random stratification present: paired Δ ± SE reported under each holdout.
- [ ] Block-aware Ext 1 vs. base contrast reported; no Δ pairs runs across the two regimes.
- [ ] Whether that contrast separates from zero stated, so `M8-4` knows whether to run.
- [ ] Every null explicitly labeled "no significant effect," none dropped or spun.
- [ ] Each paired Δ carries its 95% CI, and each null states the largest gain its CI allows.
- [ ] Each extension's gate reported beside its Δ.

---

### `M7-3` — Qualitative trajectory analysis
- **Assignee:** @JoshOlu
- **Labels:** `architecture`, `analysis`
- **Blocked by:** `M7-1`
- **Blocks:** `M9-2`

**Context.** The interpretability complement to the numbers: reconstructed trajectories for 5 patients × 3 variables (HR, serum sodium, GCS), showing temporal coherence of the diffusion imputations against mean/forward-fill. Josh owns it (model/analysis track); it seeds the per-variable analysis in Week 9.

**Deliverables**
- Trajectory plots: 5 patients × 3 variables, imputed posterior (with a few draws) vs. mean and forward-fill, with the observed points marked.
- A short qualitative read: where diffusion tracks the true dynamics and where it doesn't.

**Definition of Done**
- [ ] Plots produced for the 5×3 set from the best variant's checkpoints.
- [ ] Observed vs. imputed vs. baselines clearly distinguished.
- [ ] Written observations committed for reuse in the final report.
