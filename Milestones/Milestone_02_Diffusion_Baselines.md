# Milestone 2 — Diffusion Process, Objective & Baselines Start

> **GitHub Milestone**
> **Title:** `M2 — Diffusion Process & Baselines`
> **Timeframe:** Week 2 · Oct 8 – Oct 14
> **Depends on:** `M1`
> **Description:**
> With the data, eval objects, and reused model in place, wire the actual diffusion machinery and start the baseline suite. Jake builds the forward noising process and the training loop (`Masker → q_sample → denoiser → masked loss → clip → step`); Josh adapts the reused denoiser to our tensor shapes and conditioning and confirms the `mask_emb`/`time_enc` hooks accept extension inputs (still defaulting `None`); Sankha records the trivial baselines (mean, forward-fill) and shape/NaN-checks the reverse sampler against a stub. Nothing here is tuned for accuracy yet — the goal is a training loop that steps cleanly and a reverse sampler that produces correctly-shaped, non-NaN output, so Week 3 can chase the reproduction number.

## Assignment summary
- **Jake** — Cosine noising schedule + `q_sample`; wire the training loop (`Masker → q_sample → denoiser → masked loss → clip → step`)
- **Josh** — Adapt the reused denoiser to our tensor shapes and conditioning; confirm the input hook accepts `mask_emb`/`time_enc` (both default `None`)
- **Sankha** — Mean + forward-fill baselines recorded; DDPM reverse sampler shape/NaN check with a stub

## Coordination notes
- The **training loop (Jake)** and the **denoiser adaptation (Josh)** meet at one interface: the loop calls the denoiser with `(noisy_x, t, cond, mask_emb=None, time_enc=None)`. Freeze that call signature before either side is finished so the two can be built against a stub.
- **Baselines are our floor, not our target.** Mean imputation (~0.72) and forward-fill (~0.42 @10%) are the numbers base CSDI must beat in Week 3 to prove it's learning at all (§7).
- Keep every number in **standardized units** (§4.2) from the very first baseline, or the Week-3 "below forward-fill" check is meaningless.

---

## Issues

### `M2-1` — Cosine noising schedule, `q_sample`, and the training loop
- **Assignee:** @jkelle11-source
- **Labels:** `diffusion`, `training`, `mvp`
- **Blocked by:** `M1-1`
- **Blocks:** `M3-1`

**Context.** The forward process and the loop that everything trains through. A cosine schedule and `q_sample` implement the forward noising; the training loop composes the pieces (`TrainMasker → q_sample → denoiser → masked loss → gradient clip → optimizer step`). This is diffusion-engine work on Jake's track, feeding directly into the Week-3 reproduction.

**Deliverables**
- Cosine β-schedule and `q_sample(x0, t)` producing `x_t` with the correct marginal variance.
- Training loop wiring `TrainMasker`, `q_sample`, the denoiser call (with `mask_emb`/`time_enc` defaulting `None`), a **masked** loss over conditioning-held-out targets, gradient clipping, and the optimizer step.
- Config surface for schedule length `T`, learning rate, batch size, and clip norm.

**Definition of Done**
- [ ] `q_sample` variance matches the closed-form schedule at sampled timesteps (numerical check).
- [ ] One training step runs end-to-end on a real batch without shape/NaN errors.
- [ ] Loss is computed only over `TrainMasker` targets (verified against the mask).
- [ ] Schedule/optimizer settings are config-driven, not hardcoded.

---

### `M2-2` — Adapt the reused denoiser to our shapes + conditioning hooks
- **Assignee:** @JoshOlu
- **Labels:** `architecture`, `csdi`, `mvp`
- **Blocked by:** `M1-1`, `M1-2`
- **Blocks:** `M3-1`, `M4-2`, `M5-1`

**Context.** Adapt the forked CSDI denoiser to our `(B, 48, D)` tensors and conditioning inputs, and — the load-bearing part for the extensions — **confirm the input hook accepts `mask_emb` and `time_enc`, both default `None`**. Getting the hooks right now means Extension 1 (Josh, W4) and Extension 2 (Jake, W5) plug in without touching the denoiser core, which is exactly what keeps those two extensions parallelizable and low-friction.

**Deliverables**
- Denoiser adapted to our tensor shapes, timestep embedding, and observation-mask conditioning channel.
- Input hook that **additively** accepts `mask_emb` and `time_enc` (each `(B, T, d_model)`), defaulting to `None` (no-op) when absent.
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

**Context.** The cheap deterministic baselines that define the performance floor (§7), plus an early correctness check on the reverse sampler using a stub denoiser (so we catch shape/NaN bugs before the real model exists). Mean and forward-fill are *our* runs, so they carry our own measurement conventions.

**Deliverables**
- Mean-imputation and forward-fill baselines, evaluated with `EvalMasker` in standardized units, recorded to the results log.
- DDPM reverse sampler exercised end-to-end against a stub denoiser: output shape correct, no NaNs, K-draw stacking works.

**Definition of Done**
- [ ] Mean ≈ 0.72 and forward-fill ≈ 0.42 @10% reproduced in standardized units and logged.
- [ ] Reverse sampler produces `(K, B, 48, D)` samples with no NaNs from the stub.
- [ ] Baseline numbers stored where the Week-7 headline table can cite them.
