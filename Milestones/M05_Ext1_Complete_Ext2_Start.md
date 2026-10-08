# Milestone 5 — Extension 1 Complete, Extension 2 Start

> **GitHub Milestone**
> **Title:** `M5 — Ext 1 Complete, Ext 2 Start`
> **Timeframe:** Week 5 · Oct 29 – Nov 4
> **Depends on:** `M4` *(gate must be passed; otherwise this week is baseline debugging per §4.4)*
> **Description:**
> With the gate passed, the extensions proceed **in parallel** — which is the whole reason Ext 2 sits on Jake's track, not Josh's. Jake owns the **TSLO δ lifecycle end-to-end**: he feeds δ through train and inference *and* builds the **`ContinuousTimeEncoding`** (Ext 2, §5.2), so the producer and consumer of δ are the same person and the only cross-track dependency is the already-frozen `time_enc` hook. Josh finishes Ext 1 polish (freed from the Week-4 crunch). Sankha runs preliminary single-seed base/Ext1/Ext2 runs for *health only*. The gate for both extensions this week is **mechanical** — gradients flow, loss drops, no NaNs — never effect detection, which waits for the 5-seed sweep.

## Assignment summary
- **Jake** — Feed TSLO δ through the pipeline at train and inference (confirm δ stays aligned to observations after masking); implement `ContinuousTimeEncoding` on δ (Ext 2, §5.2); validate on high-autocorrelation variables
- **Josh** — Ext 1 polish + hand-off note; support Jake integrating the `time_enc` hook
- **Sankha** — Preliminary single-seed runs of base / Ext 1 / Ext 2 for health only

## Coordination notes
- **δ alignment is the silent killer.** After `TrainMasker`/`EvalMasker` remove values, δ must still index the *true* last observation, not a masked-away one. A misalignment corrupts Ext 2 invisibly (§5.2 risk row, §7). Unit-test δ against observations *after* masking.
- **True Time2Vec, not the summed-linear version.** Ext 2 is *one* linear dimension + `d_k−1` sinusoidal dimensions **concatenated** (§5.2). The additive-input design means it touches the denoiser input like a positional encoding and **does not modify attention logits** — which is exactly why we need no custom-attention reimplementation and why the pairwise `TimeDeltaBias` was dropped.
- **Mechanical gate only this week.** "Gradients flow, loss drops, no NaNs" for both extensions. Any "it helps" reading is deferred to Week 7's paired 5-seed sweep.

---

## Issues

### `M5-1` — TSLO δ through train + inference, alignment-safe
- **Assignee:** @jkelle11-source
- **Labels:** `data`, `diffusion`, `tslo`, `ext2-tslo`, `mvp`
- **Blocked by:** `M4-1`
- **Blocks:** `M5-2`, `M6-1`

**Context.** Plumb the per-feature δ (computed back in `M1-1`) through the full train and inference path so the encoder in `M5-2` can consume it. The load-bearing correctness property is **alignment after masking**: δ must reflect the last *actually observed* value, unaffected by which values `TrainMasker`/`EvalMasker` hold out. Keeping this within Jake's track (same owner as δ computation) is the friction-minimizing choice.

**Deliverables**
- δ threaded through the dataloader → training loop → reverse sampler, batched consistently with `x` and the mask.
- A unit test asserting δ alignment holds *after* masking (δ points at the true last observation, not a masked one).
- Inference-path check that δ is available and correct when sampling.

**Definition of Done**
- [ ] δ flows through train and inference with correct shapes/batching.
- [ ] Alignment unit test passes on a fixture with known masking.
- [ ] Sampling with δ present runs without shape/NaN errors.

---

### `M5-2` — Extension 2: `ContinuousTimeEncoding` (true Time2Vec on δ)
- **Assignee:** @jkelle11-source
- **Labels:** `diffusion`, `ext2-tslo`, `mvp`
- **Blocked by:** `M5-1`, `M2-2`
- **Blocks:** `M6-1`

**Context.** The TSLO encoder (§5.2): embed δ with a **true Time2Vec** (linear dim + sinusoidal dims, concatenated), project to `d_model`, and **add** it to the denoiser input via the `time_enc` hook. Jake owns it because it's the δ consumer and he owns δ. Toggle: `--use_time_encoding`. Validate on high-autocorrelation variables (HR, SpO2), where recency should matter most — but only for mechanical health this week.

**Deliverables**
- `ContinuousTimeEncoding(d_k, n_features, d_model)`: `w0` linear (k=0) + `w` sinusoidal (k≥1), concatenated, projected `(n_features·d_k) → d_model`, output `(B, T, d_model)`.
- `--use_time_encoding` CLI toggle; default off reproduces base exactly; added to input, attention logits untouched.
- Mechanical-health validation run on high-autocorrelation variables (gradients flow, loss drops, no NaNs).

**Definition of Done**
- [ ] Time2Vec is the concatenated form (1 linear + `d_k−1` sinusoidal), not summed linears.
- [ ] `--use_time_encoding` on/off toggles cleanly; off == base; attention code unchanged.
- [ ] Nonzero gradients reach the encoder; loss drops without NaNs on a short run.

---

### `M5-3` — Ext 1 polish + preliminary single-seed health runs
- **Assignee:** @JoshOlu, @SankhaS
- **Labels:** `architecture`, `eval`, `ext1-mask-embedding`
- **Blocked by:** `M4-2`
- **Blocks:** `M6-1`

**Context.** Two lightweight, parallel tasks. Josh finalizes Ext 1 (cleanup, a short hand-off note so the combined model in W6 composes cleanly) now that the Week-4 crunch is behind him. Sankha runs base / Ext 1 / Ext 2 at a single seed for **health only** — confirming all three variants train and evaluate through the harness before the real sweep.

**Deliverables**
- **Josh:** Ext 1 finalized + a one-paragraph hand-off note (config flags, expected shapes) for the W6 combined model.
- **Sankha:** Single-seed runs of base, Ext 1, and Ext 2, each producing a logged (health-only) MAE @10%; no effect claims.

**Definition of Done**
- [ ] Ext 1 hand-off note committed; `--use_mask_embedding` confirmed composable.
- [ ] All three variants train + evaluate end-to-end at one seed and log a number.
- [ ] Results explicitly flagged "health only — not an effect measurement."
