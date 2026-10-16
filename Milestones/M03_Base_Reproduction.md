# Milestone 3 — Base CSDI Reproduction

> **GitHub Milestone**
> **Title:** `M3 — Base CSDI Reproduction`
> **Timeframe:** Week 3 · Oct 15 – Oct 21
> **Depends on:** `M2`
> **Description:**
> Get the adapted base CSDI **training and converging below the forward-fill floor** on real PhysioNet data — the first real evidence the model learns. This is the project's critical path, so it is **paired: Josh leads (he owns the model) and Jake co-drives (he owns the training engine)**, because a stalled reproduction stalls everything measured against it (§3.4). Sankha builds the last baseline (BRITS, or a simplified bi-GRU fallback) and stands up live experiment tracking. The bar this week is *convergence and health* — beating forward-fill, clean gradients, no explosion or collapse in the first 500 steps — **not** the reproduction gate itself, which is Week 4.

## Assignment summary
- **Josh (lead) + Jake** — Drive the adapted base CSDI to train and converge below forward-fill; log grad norms; confirm no explosion/collapse in the first 500 steps
- **Sankha** — BRITS baseline (or simplified bi-GRU fallback, noted as such); wandb tracking live (loss/step, val MAE-RMSE/epoch, grad norms, config)

## Coordination notes
- **Pairing is deliberate, not doubling-up.** Josh owns the denoiser internals; Jake owns the loop/masker wiring/data feed. Reproduction bugs live at that seam (conditioning wiring, mask alignment, masker hand-off), so both owners on it in the same week resolves them fastest. Keep a shared debugging log.
- **"Below forward-fill" is the Week-3 success bar** — < 0.437 @10% standardized, forward-fill's validation MAE (`M2-3`). The **gate** (≤ ~0.24, ≥3 seeds) is Week 4; do not conflate them.
- BRITS is **cited, not re-derived**, for its published numbers (§6); our *run* of it is only to have a learned baseline in our own harness. If BRITS is fiddly, the simplified bi-GRU fallback is acceptable and must be labeled as such.
- Stand up wandb this week so the Week-4 gate runs and the Week-7 sweep are tracked from the start, not retrofitted.

---

## Issues

### `M3-1` — Base CSDI training & convergence below forward-fill
- **Assignee:** @JoshOlu, @jkelle11-source
- **Labels:** `architecture`, `diffusion`, `csdi`, `mvp`, `critical-path`
- **Blocked by:** `M2-1`, `M2-2`
- **Blocks:** `M4-1`

**Context.** The first real training of the adapted base CSDI on PhysioNet. Josh leads on the model, Jake co-drives on the loop and data feed. Success is a model that *learns* — validation MAE dropping below forward-fill with healthy optimization dynamics — which sets up the Week-4 push to the reproduction gate. Most bugs here are at the model↔loop seam (conditioning channel wiring, mask/δ alignment, masker hand-off), which is why the two owners share it.

**Expect loss spikes from the known bad values** (README, Dataset; found in `M2-3`). Our train-split scaling puts the training split's bad pH entries up to 66 standard deviations out and its bad Temp entries up to 33. Upstream's all-patient scaling puts a pH of 100 at 11. They are 17 of the 21 training values beyond 30; the other 4 are Urine. Whenever `TrainMasker` makes one of them a target, or it sits in the conditioning, the masked loss and grad norm can jump. Upstream trained through the same values without clipping. If the spikes destabilize training, revisit cleaning then, before the gate is measured (§3.4), not after.

**Deliverables**
- Base CSDI training run on the full Set A training split with the frozen protocol (z-scored inputs, `TrainMasker`, masked loss) and upstream's published diffusion configuration (quad schedule, 50 steps, 8 heads).
- Validation MAE (via `EvalMasker`, standardized) tracked per epoch and confirmed dropping **below forward-fill**.
- Gradient-norm logging; a documented check that the first 500 steps show no loss explosion or collapse.
- A shared debugging log capturing any seam issues (conditioning, alignment, masking) and their fixes.

**Definition of Done**
- [ ] Base CSDI trains without divergence; grad norms bounded through the first 500 steps.
- [ ] Val MAE @10% (standardized) is below forward-fill (< 0.437, its val MAE from `M2-3`) on ≥1 seed.
- [ ] Runs are launched from config + fixed seed and reproduce on a rerun.
- [ ] Debugging log committed so Week-4 hardening starts from known state.

---

### `M3-2` — Live experiment tracking (wandb)
- **Assignee:** @SankhaS
- **Labels:** `eval`, `infra`, `wandb`, `mvp`
- **Blocked by:** `M2-1`
- **Blocks:** `M3-3`, `M4-1`, `M7-1`

**Context.** Tracking has to exist *before* the gate runs and the seed sweep, or we lose the very curves that prove the gate passed and the sweep was healthy. Wire wandb to log the metrics that matter for the protocol (§3).

**Deliverables**
- wandb logging of loss/step, val MAE & RMSE/epoch, gradient norms, and the full run config (including seed and toggle flags).
- A project workspace/dashboard the whole team can view, with runs named by `variant × seed` and, from Week 6, the training regime (random or block-aware, §3.1).

**Definition of Done**
- [ ] A base-CSDI run streams loss, val MAE/RMSE, grad norms, and config to wandb.
- [ ] Run naming encodes variant + seed (+ training regime) so the Week-7 sweep is legible.
- [ ] All three can open the shared workspace.

---

### `M3-3` — BRITS baseline (or bi-GRU fallback)
- **Assignee:** @SankhaS
- **Labels:** `eval`, `baselines`, `mvp`
- **Blocked by:** `M2-3`, `M2-4` *(block holdout, for the block-holdout score only)*, `M3-2` *(wandb workspace, for logging the baselines only)*
- **Blocks:** `M7-2`

**Context.** A *learned* baseline stronger than mean/forward-fill, run inside our own harness so it shares our masking and metrics. BRITS is the target; a simplified bi-GRU imputer is an acceptable, clearly-labeled fallback if BRITS integration eats the week. Published BRITS/V-RIN numbers are cited separately (§6) and carry no SE of ours. Like every model we run ourselves, it is also scored on the block holdout (`BlockEvalMasker`, `M2-4`; §3.1), which costs one extra evaluation pass, not extra training.

**Deliverables**
- BRITS (or bi-GRU fallback) trained and evaluated with `EvalMasker` in standardized units.
- Its MAE @10% recorded to the results log (the README's baseline table and `results/baselines.json`) next to mean/forward-fill, under both holdouts (random `EvalMasker`, block `BlockEvalMasker`).
- All the baselines logged to wandb, in `M3-2`'s workspace: mean and forward-fill from `results/baselines.json` (`M2-3`, `M2-4`) and this learned baseline.
- A one-line note stating whether it's BRITS or the fallback, and why.

**Definition of Done**
- [ ] A learned baseline produces a logged MAE @10% in our harness, under both holdouts.
- [ ] Mean, forward-fill and the learned baseline are logged to wandb.
- [ ] Fallback (if used) is explicitly labeled in the results log and report notes.
- [ ] Number stored where the Week-7 headline table can cite it.
