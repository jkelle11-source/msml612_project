# Milestone 2 — Diffusion Process, Objective & Baselines Start

> **GitHub Milestone**
> **Title:** `M2 — Diffusion Process & Baselines`
> **Timeframe:** Week 2 · Oct 8 – Oct 14
> **Depends on:** `M1`
> **Description:**
> With the data, eval objects, and reused model in place, wire our protocol around the reused diffusion machinery and start the baseline suite. **The diffusion math is reused, not rewritten** (settled at the M1-2 review): upstream's quad noise schedule, forward noising, masked loss and reverse sampler stay as vendored, and only the masking is ours. Jake builds the training loop on top of it (`TrainMasker → reused forward noising → denoiser → masked loss → clip → step`); Josh adapts the reused model to our tensor shapes and conditioning and confirms the `mask_emb`/`time_enc` hooks accept extension inputs (still defaulting `None`); Sankha records the trivial baselines (mean, forward-fill) and shape/NaN-checks the reused reverse sampler through `EvalMasker`. Nothing here is tuned for accuracy yet — the goal is a training loop that steps cleanly and a reverse sampler that produces correctly-shaped, non-NaN output, so Week 3 can chase the reproduction number.

## Assignment summary
- **Jake** — Training loop on the reused diffusion process (`TrainMasker → reused forward noising → denoiser → masked loss → clip → step`), with `TrainMasker` replacing upstream's internal masking; numerical check of the reused schedule
- **Josh** — Adapt the reused model to our tensor shapes and conditioning; thread the `mask_emb`/`time_enc` hooks (both default `None`) through its loss and sampler paths
- **Sankha** — Mean + forward-fill baselines recorded; shape/NaN check of the reused reverse sampler, driven through `EvalMasker`

## Coordination notes
- **We reuse upstream's diffusion math; only the masking is ours** (settled at the M1-2 review; §0). The vendored `CSDI_base` keeps its quad β-schedule (1e-4 → 0.5, 50 steps), forward noising, masked noise-prediction loss, reverse sampler and 8 attention heads, so the gate run matches the configuration behind the published 0.217. Its internal train masking (`get_randmask`) is the one piece replaced: it draws from the global NumPy/torch RNGs, which would break seed pairing, so conditioning masks come from `TrainMasker` (training) and `EvalMasker` (evaluation, passed as upstream's `gt_mask`).
- The **training loop (Jake)** and the **model adaptation (Josh)** meet at one interface: the loop hands the reused model a batch plus the `TrainMasker` conditioning mask (with `mask_emb=None`, `time_enc=None`) and gets the masked loss back; noising, the denoiser call and the loss all happen inside the model. Freeze that call signature before either side is finished so the two can be built against a stub. Both issues touch `CSDI_base`'s entry points, so agree up front who edits the vendored file; every change to it carries an `ADAPTED` marker and a README line, as in `M1-2`.
- **Baselines are our floor, not our target.** Mean imputation (~0.72) and forward-fill (~0.42 @10%) are the numbers base CSDI must beat in Week 3 to prove it's learning at all (§6).
- Keep every number in **standardized units** (§3.2) from the very first baseline, or the Week-3 "below forward-fill" check is meaningless.

---

## Issues

### `M2-1` — Training loop on the reused diffusion process (`TrainMasker` wired in)
- **Assignee:** @jkelle11-source
- **Labels:** `diffusion`, `training`, `mvp`
- **Blocked by:** `M1-1`, `M1-2`
- **Blocks:** `M3-1`

**Context.** The loop that everything trains through. The forward process itself is **reused**: the vendored `CSDI_base` (`models/csdi/main_model.py`) already implements the quad β-schedule, forward noising and the masked noise-prediction loss at the settings behind the published 0.217, so we write no schedule or `q_sample` of our own (`M1-2` review decision; §0). What upstream does not give us is protocol-safe masking: its `get_randmask` draws from the global NumPy/torch RNGs, which would break seed pairing, so the loop takes its conditioning/target split from `TrainMasker` instead. The loop composes `TrainMasker → reused forward noising → denoiser → masked loss → gradient clip → optimizer step`. This is training-engine work on Jake's track, feeding directly into the Week-3 reproduction.

**Deliverables**
- Data feed: batches from `data.physionet.load_dataset` in the form the reused model consumes (`observed_data`, `observed_mask`, `gt_mask`, `timepoints`).
- Training loop wiring `TrainMasker` (in place of upstream's `get_randmask`), the reused forward noising and **masked** loss (with `mask_emb`/`time_enc` defaulting `None`), gradient clipping, and the optimizer step.
- A numerical check of the reused schedule: forward-noised samples have the closed-form marginal variance at sampled timesteps.
- Config surface for the diffusion settings (schedule type, β range, steps `T`; defaults are upstream's published values), learning rate, batch size, and clip norm.

**Definition of Done**
- [ ] Forward-noising variance matches the closed-form value for the reused schedule at sampled timesteps (numerical check).
- [ ] One training step runs end-to-end on a real batch without shape/NaN errors.
- [ ] Loss is computed only over `TrainMasker` targets (verified against the mask); upstream's `get_randmask` is not on the training path.
- [ ] Diffusion/optimizer settings are config-driven, not hardcoded, and the diffusion defaults equal upstream's published PhysioNet values.

---

### `M2-2` — Adapt the reused denoiser to our shapes + conditioning hooks
- **Assignee:** @JoshOlu
- **Labels:** `architecture`, `csdi`, `mvp`
- **Blocked by:** `M1-1`, `M1-2`
- **Blocks:** `M3-1`, `M4-2`, `M5-1`

**Context.** Adapt the forked CSDI denoiser to our `(B, 48, D)` tensors and conditioning inputs, and — the load-bearing part for the extensions — **confirm the input hook accepts `mask_emb` and `time_enc`, both default `None`**. Getting the hooks right now means Extension 1 (Josh, W4) and Extension 2 (Jake, W5) plug in without touching the denoiser core, which is exactly what keeps those two extensions parallelizable and low-friction. Because the diffusion process is reused as well (see `M2-1`), the denoiser is called from inside `CSDI_base`'s loss and sampler, so the hooks have to be threaded through both of those paths, not only added to the denoiser's own `forward`.

**Deliverables**
- Denoiser adapted to our tensor shapes, timestep embedding, and observation-mask conditioning channel.
- Input hook that **additively** accepts `mask_emb` and `time_enc` (each `(B, T, d_model)`), defaulting to `None` (no-op) when absent, threaded from the reused model's training-loss and sampler paths through to the denoiser.
- A short interface note documenting the exact call signature the training loop uses.

**Definition of Done**
- [ ] Denoiser runs forward+backward on a real `(B, 48, D)` batch.
- [ ] Passing a dummy nonzero `mask_emb`/`time_enc` changes the output; passing `None` reproduces the base exactly.
- [ ] Call signature documented and matches `M2-1`'s training loop.

---

### `M2-3` — Mean + forward-fill baselines; reverse-sampler shape/NaN check
- **Assignee:** @SankhaS
- **Labels:** `eval`, `baselines`, `diffusion`, `mvp`
- **Blocked by:** `M1-3`
- **Blocks:** `M3-3`

**Context.** The cheap deterministic baselines that define the performance floor (§6), plus an early correctness check on the reverse sampler. The sampler is **reused** (`CSDI_base.impute`, kept as vendored per the `M1-2` review decision) and the real denoiser has existed since `M1-2`, so no stub is needed: the check runs an untrained model on a real batch with `EvalMasker` supplying the conditioning mask, which catches shape/NaN and mask hand-off bugs before anything is trained. Mean and forward-fill are *our* runs, so they carry our own measurement conventions.

**Deliverables**
- Mean-imputation and forward-fill baselines, evaluated with `EvalMasker` in standardized units, recorded to the results log.
- Reused reverse sampler exercised end-to-end with an untrained model: `EvalMasker`'s conditioning mask passed as upstream's `gt_mask`, output shape correct, no NaNs, K-draw stacking works.

**Definition of Done**
- [ ] Mean ≈ 0.72 and forward-fill ≈ 0.42 @10% reproduced in standardized units and logged.
- [ ] Reused sampler returns samples in our `(K, B, 48, D)` layout (upstream's native layout is `(B, K, D, 48)`) with no NaNs.
- [ ] The scored target set (observed minus conditioning) is exactly `EvalMasker`'s held-out targets.
- [ ] Baseline numbers stored where the Week-7 headline table can cite them.
