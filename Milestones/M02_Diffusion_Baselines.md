# Milestone 2 — Diffusion Process, Objective & Baselines Start

> **GitHub Milestone**
> **Title:** `M2 — Diffusion Process & Baselines`
> **Timeframe:** Week 2 · Oct 8 – Oct 14
> **Depends on:** `M1`
> **Description:**
> With the data, eval objects, and reused model in place, wire our protocol around the reused diffusion machinery and start the baseline suite. **The diffusion math is reused, not rewritten** (settled at the M1-2 review): upstream's quad noise schedule, forward noising, masked loss and reverse sampler stay as vendored, and only the masking is ours. Jake builds the training loop on top of it (`TrainMasker → reused forward noising → denoiser → masked loss → clip → step`); Josh adapts the reused model to our tensor shapes and conditioning and confirms the `mask_emb`/`time_enc` hooks accept extension inputs (still defaulting `None`); Sankha records the trivial baselines (mean, forward-fill) and shape/NaN-checks the reused reverse sampler through `EvalMasker`. Sankha then adds `BlockEvalMasker` (`M2-4`, added Oct 14), the frozen block-missingness holdout that Extension 1's primary regime is scored on (§3.1, §4.1); it is due Thu Oct 15, well before `M4-2` starts on Oct 22. Nothing here is tuned for accuracy yet — the goal is a training loop that steps cleanly and a reverse sampler that produces correctly-shaped, non-NaN output, so Week 3 can chase the reproduction number.

## Assignment summary
- **Jake** — Training loop on the reused diffusion process (`TrainMasker → reused forward noising → denoiser → masked loss → clip → step`), with `TrainMasker` replacing upstream's internal masking; numerical check of the reused schedule
- **Josh** — Adapt the reused model to our tensor shapes and conditioning; thread the `mask_emb`/`time_enc` hooks (both default `None`) through its loss and sampler paths
- **Sankha** — Mean + forward-fill baselines recorded; shape/NaN check of the reused reverse sampler, driven through `EvalMasker`. Then `BlockEvalMasker`, the frozen block-missingness holdout (`M2-4`, due Thu Oct 15)

## Coordination notes
- **We reuse upstream's diffusion math; only the masking is ours** (settled at the M1-2 review; §0). The vendored `CSDI_base` keeps its quad β-schedule (1e-4 → 0.5, 50 steps), forward noising, masked noise-prediction loss, reverse sampler and 8 attention heads, so the gate run matches the configuration behind the published 0.217. Its internal train masking (`get_randmask`) is the one piece replaced: it draws from the global NumPy/torch RNGs, which would break seed pairing, so conditioning masks come from `TrainMasker` (training) and `EvalMasker` (evaluation, passed as upstream's `gt_mask`).
- The **training loop (Jake)** and the **model adaptation (Josh)** meet at one interface: the loop hands the reused model a batch whose `gt_mask` is the `TrainMasker` conditioning mask and gets the masked loss back; noising, the denoiser call and the loss all happen inside the model. The call is frozen as `loss = model(batch, generator=streams.diffusion)`, so both sides can be built against a stub, and it is the same for every variant: the extension toggles are read once, by `models.build_model`, and the model computes `mask_emb`/`time_enc` itself, from the conditioning mask, on its loss and sampler paths. The loop never passes them (settled in `M2-1`; README, input-hook contract). Both issues touch `CSDI_base`'s entry points, so agree up front who edits the vendored file. `M2-1`'s additions (the `generator` argument and `target_strategy: "external"`) are already in it; every change to it carries an `ADAPTED` marker and a README line, as in `M1-2`.
- **Baselines are our floor, not our target.** Mean imputation (0.705 test, 0.696 val) and forward-fill (0.449 test, 0.437 val), @10% as measured in `M2-3`, are the numbers base CSDI must beat in Week 3 to prove it's learning at all (§6).
- Keep every number in **standardized units** (§3.2) from the very first baseline, or the Week-3 "below forward-fill" check is meaningless.
- **The block holdout was added late (Oct 14).** §4.1 makes block missingness Extension 1's primary regime, and `M4-2` and `M7-2` assume block masks exist, but no issue built them. `M2-4` closes that gap. It is frozen exactly like `EvalMasker`, with its own seed, and leaves `EvalMasker` untouched: `EvalMasker`'s golden fingerprint must not change. Training stays on `TrainMasker`'s `random` strategy for every variant; a second, block-aware training regime for base and Ext 1 is added later, in `M6-2` (§3.1).

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
- Config surface for the diffusion settings (schedule type, β range, steps `T`), learning rate, weight decay, LR schedule, epochs, batch size, and clip norm. Every default is upstream's published value; clipping is off by default, as upstream.
- `models.build_model(config, device)`: the one place the model is built from config. It takes the head count from `model.n_heads`, always sets `target_strategy: "external"`, and raises `NotImplementedError` if an extension toggle is on before that extension exists; `M4-2` and `M5-2` add their modules here.

**Definition of Done**
- [ ] Forward-noising variance matches the closed-form value for the reused schedule at sampled timesteps (numerical check).
- [ ] One training step runs end-to-end on a real batch without shape/NaN errors.
- [ ] Loss is computed only over `TrainMasker` targets (verified against the mask); upstream's `get_randmask` is not on the training path.
- [ ] Diffusion/optimizer settings are config-driven, not hardcoded, and their defaults equal upstream's published PhysioNet values.
- [ ] The model is built only through `build_model`, and an extension toggle that is on raises until that extension exists.

---

### `M2-2` — Adapt the reused denoiser to our shapes + conditioning hooks
- **Assignee:** @JoshOlu
- **Labels:** `architecture`, `csdi`, `mvp`
- **Blocked by:** `M1-1`, `M1-2`
- **Blocks:** `M3-1`, `M4-2`, `M5-1`

**Context.** Adapt the forked CSDI denoiser to our `(B, 48, D)` tensors and conditioning inputs, and — the load-bearing part for the extensions — **confirm the input hook accepts `mask_emb` and `time_enc`, both default `None`**. Getting the hooks right now means Extension 1 (Josh, W4) and Extension 2 (Jake, W5) plug in without touching the denoiser core, which is exactly what keeps those two extensions parallelizable and low-friction. Because the diffusion process is reused as well (see `M2-1`), the denoiser is called from inside `CSDI_base`'s loss and sampler, so the hooks have to be threaded through both of those paths, not only added to the denoiser's own `forward`. The training loop does not pass the hooks (settled in `M2-1`; see Coordination notes): the model will compute them itself from the conditioning mask, so they travel from inside `CSDI_base` to the denoiser, and `forward`/`evaluate` keep the signatures the loop and `EvalSampler` call.

**Deliverables**
- Denoiser adapted to our tensor shapes, timestep embedding, and observation-mask conditioning channel.
- Input hook that **additively** accepts `mask_emb` and `time_enc` (each `(B, T, D, d_model)`: one vector per hour and feature), defaulting to `None` (no-op) when absent, threaded from the reused model's training-loss and sampler paths through to the denoiser.
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
- Mean-imputation and forward-fill baselines, evaluated with `EvalMasker` in standardized units, recorded to the results log (the README's baseline table and `results/baselines.json`).
- Both also scored at CSDI's robustness rates, 50% and 90% (§6), each a frozen holdout from the same `EvalMasker` seed (`protocol.robustness_fractions`).
- Reused reverse sampler exercised end-to-end with an untrained model: `EvalMasker`'s conditioning mask passed as upstream's `gt_mask`, output shape correct, no NaNs, K-draw stacking works.

**Definition of Done**
- [ ] Mean and forward-fill @10% measured in standardized units and logged (test: 0.705 and 0.449; the plan's estimates were ~0.72 and ~0.42).
- [ ] Both measured and logged @50% and @90% as well.
- [ ] Reused sampler returns samples in our `(K, B, 48, D)` layout (upstream's native layout is `(B, K, D, 48)`) with no NaNs.
- [ ] The scored target set (observed minus conditioning) is exactly `EvalMasker`'s held-out targets.
- [ ] Baseline numbers stored where the Week-7 headline table can cite them.

---

### `M2-4` — `BlockEvalMasker`: frozen block-missingness holdout
- **Assignee:** @SankhaS
- **Labels:** `eval`, `protocol`, `mvp`
- **Blocked by:** `M1-3` *(`EvalMasker`, which it mirrors)*; `M2-3` *(the baseline functions; only for the block-holdout baseline numbers)*
- **Blocks:** `M3-3`, `M4-2`, `M6-2`, `M7-2`

**Context.** §4.1 makes block (cross-feature, contiguous) missingness Extension 1's primary regime, `M4-2`'s overfit check uses a block-masked subset, and `M7-2` splits the headline table into block and random. But the only holdout we have, `EvalMasker`, hides values uniformly at random, and no issue built a block one. This issue adds it as a second frozen holdout (§3.1). It mirrors `EvalMasker` in everything except *where* values are hidden: its own frozen seed, one generator per patient seeded from `(seed, record_id)`, the same `floor(fraction · n_observed)` count per patient, and the same call, `(mask, record_ids) → (cond_mask, target_mask)`, so `EvalSampler`, the reused sampler and the metrics take it unchanged. Hidden values come as contiguous 3-hour windows across all 35 features. Three hours is about the longest window that fits the 10% budget on our data: the median patient has 332 observed values, about 6 per hour, so 10% is roughly 5–6 hours of full blackout in total, and at 3 hours 95% of patients still get at least one complete window before the trim. Added Oct 14 and due Thu Oct 15, well before `M4-2` starts on Oct 22.

**Deliverables**
- `BlockEvalMasker` in `eval/masking.py`, exported from `eval`, with a `from_config()`. Per patient: tile the 48 hours into 16 fixed 3-hour slots, draw the slots in random order, and hide every observed value in each slot until the next slot would overshoot `floor(fraction · n_observed)`; trim that slot to a random subset of its observed values so the count is exact. A slot with nothing observed hides nothing. Write the slot selection as a function of `(mask, count, generator)`, so `TrainMasker`'s `mix` strategy (`M6-2`, block-aware training regime, §3.1) reuses it with its own generator and budget.
- Config: a new frozen seed under `protocol` (distinct from `eval_masker_seed` 2012, `eval_sampling_seed` 2021, `split_seed` 612 and the run seeds 42–46) and the window length (3 hours), both marked do-not-change like `eval_masker_seed`. The fraction is `eval_holdout_fraction`, shared with `EvalMasker`.
- Tests mirroring `tests/test_eval_masker.py`: byte-identical across run seeds and independent of patient order or subset; exactly `EvalMasker`'s count per patient; every target outside the one trimmed slot lies in a slot whose observed values are all hidden; targets are observed and `cond + target == mask` (no leakage); the input mask is not modified; a wrong grid layout raises; a golden SHA-256 fingerprint.
- Mean and forward-fill (from `M2-3`) scored on the block holdout of val and test, in standardized units, recorded next to their random-holdout numbers.
- README: the new keys under "Frozen protocol constants".

**Definition of Done**
- [ ] Each patient loses exactly `floor(0.10 · n_observed)` values, in contiguous 3-hour all-feature slots apart from the one trimmed slot (tests green).
- [ ] Targets are byte-identical across run seeds; golden fingerprint committed.
- [ ] `EvalMasker`'s golden fingerprint is unchanged.
- [ ] Seed and window length committed to `configs/config.yaml`.
- [ ] Mean and forward-fill block-holdout MAE recorded next to their random-holdout numbers.
- [ ] Merged by **Thu Oct 15**.
