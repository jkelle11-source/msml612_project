# Milestone 6 — Combined Model & Paired-Seed Harness

> **GitHub Milestone**
> **Title:** `M6 — Combined Model & Paired-Seed Harness`
> **Timeframe:** Week 6 · Nov 5 – Nov 11
> **Depends on:** `M5`
> **Description:**
> Assemble the four-variant matrix and the machinery that will train it. Josh composes the **combined Ext 1 + 2 model** and confirms all four toggle combinations (base / +Ext1 / +Ext2 / +both) run. Jake + Sankha stand up the **paired-seed harness** (§4.3) — one driver that runs all four variants at each of seeds 42–46 sharing data order, train-mask stream, and init RNG, so every variant-vs-base comparison is a *paired* difference. A 20-epoch dry-run confirms the pairing and logging are correct before the expensive full sweep in Week 7. This is the last setup week; nothing here chases accuracy, it chases *correct experimental plumbing*.

## Assignment summary
- **Josh** — Combined Ext 1 + 2 model; confirm both hooks compose and the model runs with all four toggle combinations
- **Jake + Sankha** — Stand up the **paired-seed harness** (§4.3): one driver runs all four variants at each of seeds 42–46 sharing data order and RNG; dry-run at 20 epochs to confirm pairing and logging

## Coordination notes
- **Common random numbers is the whole point of pairing** (§4.3). At a given seed, all four variants must share the *same* data order, the *same* `TrainMasker` stream, and the *same* init RNG — only the toggles differ. If the harness reseeds differently per variant, the pairing is broken and the SE-of-paired-Δ argument collapses.
- **The combined model is a composition test, not a tuning task.** The two hooks (`mask_emb`, `time_enc`) were built additive and independent precisely so `+both` is just both toggles on. Confirm it, don't re-engineer it.
- **Dry-run before the real sweep.** 20 epochs × 4 variants × a couple of seeds is enough to prove pairing + logging are correct. Discovering a pairing bug during the 20-run Week-7 sweep is the expensive failure this dry-run prevents.

---

## Issues

### `M6-1` — Combined Ext 1 + 2 model; all four toggle combinations
- **Assignee:** @JoshOlu
- **Labels:** `architecture`, `ext1-mask-embedding`, `ext2-tslo`, `mvp`
- **Blocked by:** `M4-2`, `M5-2`, `M5-3`
- **Blocks:** `M7-1`

**Context.** Compose Ext 1 (`mask_emb`) and Ext 2 (`time_enc`) into a single model driven by two independent CLI toggles, yielding the four variants the whole evaluation rests on. Because both encoders were designed as additive, independent inputs to the denoiser (§5.1, §5.2), this is primarily a verification task: confirm the four combinations each run and that `+both` composes cleanly.

**Deliverables**
- A single model/config path exposing `--use_mask_embedding` and `--use_time_encoding` independently.
- All four combinations (base / +Ext1 / +Ext2 / +both) confirmed to train and sample without shape/NaN errors.
- A short note documenting the exact flag combinations for the four variants.

**Definition of Done**
- [ ] All four toggle combinations run forward+backward and sample.
- [ ] `base` (both off) reproduces the gate model exactly.
- [ ] `+both` shows nonzero gradients reaching *both* encoders.
- [ ] Variant→flags mapping documented for the harness.

---

### `M6-2` — Paired-seed harness (common random numbers)
- **Assignee:** @jkelle11-source, @SankhaS
- **Labels:** `eval`, `protocol`, `infra`, `mvp`, `critical-path`
- **Blocked by:** `M3-2`, `M6-1`
- **Blocks:** `M7-1`, `M7-2`

**Context.** The harness that makes the statistics defensible (§4.3, the single most important statistical decision in the plan). One driver iterates seeds 42–46 and, at each seed, runs all four variants under **common random numbers** — identical data order, train-mask stream, and init RNG — so contrasts are paired differences with a small SE. Jake owns the run/RNG plumbing (his training-engine track); Sankha owns the results capture and the pairing assertions (his eval track).

**Deliverables**
- A single harness driver: `for seed in 42..46: for variant in {base, ext1, ext2, both}: train+eval` with shared RNG per seed.
- Guarantees that, within a seed, data order / `TrainMasker` stream / init RNG are identical across variants (only toggles differ).
- Structured results capture (per variant × seed: MAE/RMSE @10%, plus config) ready for the Week-7 headline table.
- A **20-epoch dry-run** over the four variants at ≥2 seeds proving pairing + logging.

**Definition of Done**
- [ ] Within a seed, the four variants demonstrably share data order and RNG (asserted in-harness or logged and diffed).
- [ ] Dry-run completes for all four variants at ≥2 seeds; results land in the structured log.
- [ ] Resuming/rerunning a `(variant, seed)` cell reproduces its numbers.
- [ ] Harness is config-driven and ready to scale to the full 20-run sweep.
