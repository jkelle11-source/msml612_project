# Milestone 4 — Reproduction Gate, Extension 1 & Interim Report

> **GitHub Milestone**
> **Title:** `M4 — Gate, Extension 1 & Interim`
> **Timeframe:** Week 4 · Oct 22 – Oct 28
> **Depends on:** `M3`
> **Description:**
> The hardest, highest-stakes week and the second **paired** effort. Jake + Josh drive base CSDI to the **reproduction gate (§4.4): ≤ ~0.24 MAE @10%, averaged over ≥3 seeds** — the hard prerequisite before any extension delta is trusted. In parallel, Josh implements **Extension 1, the dual-axis `MissingnessEncoder`** (§5.1), wiring it through the `mask_emb` hook. All three draft the **Interim Report (due Wed Oct 29)**, which must report a **passed gate** — or say honestly that it didn't, in which case Week 5 becomes baseline debugging (§4.4). Ext 1 results in the interim are preliminary/mechanical-health only, **not** an effect claim.

## Assignment summary
- **Jake + Josh** — Drive base CSDI to the **reproduction gate** (≤ ~0.24 @10%, ≥3 seeds); ~6 hrs across the two, this is the fiddly dual-axis core everything is measured against
- **Josh** — Implement `MissingnessEncoder` (Ext 1, §5.1), wire via `mask_emb`; confirm nonzero gradients reach it; overfit sanity check
- **All three** — Draft the **Interim Report** (baselines, protocol, gate status, Ext 1 preliminary)

## Coordination notes
- **The gate is a hard threshold, not a vibe.** ≤ ~0.24 @10% over ≥3 seeds (42–44 suffice for the gate; the full 42–46 sweep is Week 7). The 0.22–0.24 band — not exactly the paper's 0.217 — is the target, because independent reimplementations land ~5% above the paper and say so (§4.4, §7). Until the gate passes, all extension runs are **debugging runs**.
- **Ext 1 this week is mechanical health only.** Nonzero gradients reach the encoder, and it overfits a tiny block-masked subset — that's it. Any "does it help?" claim waits for the paired 5-seed sweep (Week 7). Putting a premature effect claim in the interim is exactly the overclaim §5.1 warns against.
- **If the gate is not passed by Oct 29, the interim says so plainly** and Week 5 pivots to baseline debugging. Architecture work (Ext 2) freezes until the base is trustworthy.

---

## Issues

### `M4-1` — Reproduction gate: base CSDI ≤ ~0.24 @10%, ≥3 seeds
- **Assignee:** @jkelle11-source, @JoshOlu
- **Labels:** `diffusion`, `architecture`, `csdi`, `gate`, `critical-path`, `mvp`
- **Blocked by:** `M3-1`, `M3-2`
- **Blocks:** `M4-2`, `M4-3`, `M5-1`, `M5-2`

**Context.** The single most important engineering milestone: turn "it converges" into "it reproduces the benchmark." Paired again (Jake on the training/data engine, Josh on the model) because this is where the last reproduction bugs hide and where the ~6 focused hours of two-person debugging pay off. The gate is what makes every later "Δ vs. base" number trustworthy.

**Deliverables**
- Base CSDI trained to **≤ ~0.24 MAE @10% (standardized)** averaged over **≥3 seeds**, tracked in wandb.
- The gate result recorded in the results log with per-seed numbers and the mean.
- Any tuning that got us there (lr, schedule length, epochs, EMA, etc.) documented for reproducibility.

**Definition of Done**
- [ ] Mean MAE @10% over ≥3 seeds is within the 0.22–0.24 band (or the closest achieved value is documented honestly).
- [ ] Per-seed numbers + mean logged; runs reproducible from config.
- [ ] Gate status (PASS/FAIL) stated explicitly for the interim.
- [ ] If FAIL: a written debugging plan for Week 5 is committed (§4.4).

---

### `M4-2` — Extension 1: dual-axis `MissingnessEncoder`
- **Assignee:** @JoshOlu
- **Labels:** `architecture`, `ext1-mask-embedding`, `mvp`
- **Blocked by:** `M2-2`, `M4-1`
- **Blocks:** `M5-3`, `M6-1`

**Context.** Implement the **dual-axis** `MissingnessEncoder` (§5.1) — temporal *and* feature attention — that embeds the binary mask and adds it to the denoiser input via `mask_emb`. Dual-axis is the point: block missingness is a *cross-feature* structure a temporal-only encoder can't represent, and the honest claim is that a learned structural embedding adds signal *beyond* the raw mask channel CSDI already consumes. This is Josh's track (model internals). Toggle: `--use_mask_embedding`; trained end-to-end.

**Deliverables**
- `MissingnessEncoder(n_features, d_model, n_heads=2, n_layers=2)` with alternating temporal/feature `TransformerEncoderLayer`s, output `(B, T, d_model)` added to the denoiser input.
- `--use_mask_embedding` CLI toggle wired through the loop; default off reproduces base exactly.
- Gradient check (nonzero gradients reach the encoder) and an overfit sanity check (10-patient block-masked subset, MAE < 0.1 in 50 epochs).

**Definition of Done**
- [ ] Encoder attends over both time and features (shapes/transposes verified).
- [ ] `--use_mask_embedding` on/off toggles cleanly; off == base.
- [ ] Nonzero gradients reach the encoder during training.
- [ ] Overfit sanity check passes (MAE < 0.1 on the tiny block-masked subset).

---

### `M4-3` — ▶ DELIVERABLE: Interim Report (due Wed Oct 29)
- **Assignee:** @jkelle11-source, @JoshOlu, @SankhaS
- **Labels:** `writing`, `deliverable`
- **Blocked by:** `M4-1`
- **Blocks:** —

**Context.** The graded interim: baselines, the full frozen protocol, the **gate status**, and Ext 1 preliminary (mechanical-health, single-seed — *not* an effect claim). Its integrity requirement is honesty: if the gate passed, show it; if it didn't, say so and state the Week-5 debugging pivot (§4.4).

**Deliverables**
- Interim document: dataset + pipeline, normalization, dual-mask protocol (§4.1), baselines table, **gate status with per-seed numbers**, Ext 1 implementation + mechanical-health note.
- Explicit statement of whether the reproduction gate passed.

**Definition of Done**
- [ ] Interim submitted by **Wed Oct 29**.
- [ ] Gate status reported truthfully (pass, or fail + debugging plan).
- [ ] Ext 1 framed as preliminary/mechanical, with no effect claim.
- [ ] Reviewed by all three.
