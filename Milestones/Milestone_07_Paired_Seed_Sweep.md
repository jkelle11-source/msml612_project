# Milestone 7 — Full Paired 5-Seed Training

> **GitHub Milestone**
> **Title:** `M7 — Full Paired 5-Seed Training`
> **Timeframe:** Week 7 · Nov 12 – Nov 18
> **Depends on:** `M6`
> **Description:**
> Run the headline experiment. Jake executes the **full paired sweep** — all four variants × seeds 42–46 (20 runs, 100 epochs each, early stop on val MAE, patience 10) — on the harness proven in Week 6. Sankha populates the **headline table**: mean ± SE per variant and, crucially, the **paired Δ vs. base with its SE**, stratified by missingness type (block vs. random). Josh produces the qualitative story: reconstructed trajectories for 5 patients × 3 variables against mean/forward-fill. The statistical discipline is fixed (§4.5): main effects (Ext 1 vs. base, Ext 2 vs. base) are **confirmatory**; the interaction is **exploratory**, and any contrast whose paired-Δ error bar overlaps zero is reported **"no significant effect,"** never spun.

## Assignment summary
- **Jake** — Full training (100 epochs, early stop on val MAE, patience 10) for all four variants × seeds 42–46 (20 runs)
- **Sankha** — Populate the **headline table**: mean ± SE and **paired Δ vs. base with its SE**; stratify by missingness type (block vs. random). Main effects confirmatory, interaction exploratory
- **Josh** — Qualitative trajectories (5 patients × 3 variables: HR, serum sodium, GCS); temporal coherence vs. mean/forward-fill

## Coordination notes
- **GPU access is the schedule risk** (§10). 20 × 100-epoch runs need reliable single-GPU time; if it's shared/queued, start Monday and checkpoint aggressively so a preemption doesn't cost a whole cell.
- **Report the paired Δ, not two independent means** (§4.3). The SE of the paired difference is what separates signal from noise here; expected gains (~0.01–0.02) are the same magnitude as CSDI's masking sensitivity, which is *why* masking is frozen and seeds are paired (§7).
- **Nulls are findings, stated as such** (§4.5, §8). An overlapping error bar is "no significant effect," reported plainly. This is the honesty bar the whole grade strategy leans on.
- Block-vs-random stratification matters: Ext 1's honest home is **block missingness** and Ext 2's is **high-autocorrelation variables** (§5.1, §5.2), so the table must let those regimes be read separately.

---

## Issues

### `M7-1` — Full paired 5-seed sweep (20 runs)
- **Assignee:** @jkelle11-source
- **Labels:** `diffusion`, `training`, `protocol`, `mvp`, `critical-path`
- **Blocked by:** `M6-1`, `M6-2`
- **Blocks:** `M7-2`, `M8-1`, `M8-2`, `M9-1`

**Context.** Execute the experiment the whole project has been building toward: four variants × seeds 42–46 under common random numbers, 100 epochs with early stopping (patience 10). Jake owns it as the training-engine/harness-execution owner. The output is the raw per-(variant, seed) results that feed the headline table, the ablation, and the runtime analysis.

**Deliverables**
- 20 completed runs (4 variants × 5 seeds), checkpointed and tracked in wandb, with early-stop on val MAE (patience 10).
- Per-run final MAE/RMSE @10% (standardized) written to the structured results store, tagged by variant + seed.
- A run manifest (config, seed, wall-clock, checkpoint path) for reproducibility and the runtime study.

**Definition of Done**
- [ ] All 20 cells complete (or any incomplete cell is documented with cause + plan).
- [ ] Results land in the structured store keyed by `(variant, seed)`.
- [ ] Runs are resumable from checkpoints; the manifest reproduces any cell.
- [ ] Pairing integrity confirmed (per-seed shared RNG across variants, per `M6-2`).

---

### `M7-2` — Headline table: mean ± SE and paired Δ ± SE
- **Assignee:** @SankhaS
- **Labels:** `eval`, `stats`, `analysis`, `mvp`
- **Blocked by:** `M7-1`, `M3-3`
- **Blocks:** `M8-1`, `M9-1`

**Context.** Turn the 20 runs into the defensible headline result (§4.3, §7). Compute each variant's mean ± SE and — the number that actually decides whether an extension "counts" — the **paired Δ vs. base with its SE**, stratified by block vs. random missingness. Sankha owns it on the eval/stats track. Main effects are confirmatory; the interaction is exploratory and will not be over-asserted at 5 seeds (§4.5).

**Deliverables**
- Headline table: base / +Ext1 / +Ext2 / +both, each mean ± SE @10%, plus baselines (mean, forward-fill, BRITS/fallback, and cited BRITS/V-RIN/CSDI numbers).
- **Paired Δ vs. base ± SE** for each extension and for the combined model.
- Stratified view: block vs. random missingness (Ext 1's and Ext 2's honest regimes readable separately).
- A written verdict per contrast: SE-separated effect vs. "no significant effect."

**Definition of Done**
- [ ] Table reports mean ± SE for all four variants and all baselines in standardized units.
- [ ] Paired Δ ± SE computed correctly (paired, not independent-means) and spot-checked by hand on one contrast.
- [ ] Block/random stratification present.
- [ ] Every null explicitly labeled "no significant effect," none dropped or spun.

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
