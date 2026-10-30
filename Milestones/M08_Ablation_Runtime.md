# Milestone 8 — Ablation & Runtime Analysis

> **GitHub Milestone**
> **Title:** `M8 — Ablation & Runtime`
> **Timeframe:** Week 8 · Nov 19 – Nov 25 *(Nov 26 is Thanksgiving; this week ends before the buffer)*
> **Depends on:** `M7`
> **Description:**
> Characterize the best variant and produce the runtime contribution that anchors the high-performance story (§7). The ablation is **not** a 3×3×3 grid — it's three **one-factor sensitivity studies** on the best variant plus the mask-encoder-size study, totaling ~35–40 runs, not 180 (§5). Sankha owns the ablation studies; Jake produces the **MAE-vs-inference-time plot** (runtime is explicitly named in the rubric's high-performance line, and this curve is a contribution most teams won't have). If Week 7's block-aware Ext 1 contrast separated from zero, Josh runs the two attribution checks §4.1 fixed for it (Ext 1-blind, and off-grid block scoring), on compute of his own alongside the ablation; otherwise they are recorded as not triggered. At end of week, the **conformal gate** is checked (§4.3): attempt the stretch only if the combined model is stable at test MAE ≤ ~0.24 @10%; otherwise cut cleanly and redirect Week-9 time to per-variable analysis and uncertainty fan plots.

## Assignment summary
- **Sankha** — The three one-factor sensitivity studies + the mask-encoder-layers study (see table); report cells whose paired-Δ bars overlap as **null**, not dropped
- **Jake** — Inference-latency profile per variant × `d_model`; the **MAE-vs-inference-time plot** (the runtime contribution, §7)
- **Josh** — *(only if `M7-2`'s block-aware Ext 1 contrast separates)* Ext 1's attribution checks (§4.1): Ext 1-blind at seeds 42–44, and off-grid block scoring of the block-aware base and Ext 1 runs
- **All three** — End-of-week **conformal gate check** (§4.3): decide go/no-go on the Extension 3 stretch

## Coordination notes
- **The ablation is bounded on purpose** (§5). Not a grid, not 36+ configs:

  | Study | Values | Variants | Seeds | Runs |
  |---|---|---|---|---|
  | Headline | — | all 4 | 5 | 20 *(from M7)* |
  | Block-aware regime (§3.1) | — | base, Ext 1 | 5 | 10 *(from M7)* |
  | `d_model` | 64 / 128 / 256 | best only | 3 | 9 |
  | Diffusion steps `T` (each a full retrain) | 50 / 100 / 200 | best only | 3 | 9\* |
  | Mask-encoder layers | 1 / 2 / 3 | the **two** with an encoder | 3 | 18 |
  | Ext 1 attribution checks, only if triggered (§4.1) | — | Ext 1-blind | 3 | 3† |

  \*If compute is tight, sweep `T` at a single `d_model` and **state it**. Total ≈ 35–40 runs.
  †Block-aware regime, plus 6 off-grid scoring passes of M7's block-aware runs (no retraining); about 37 h of compute, on Josh's machine, in parallel with the ablation.
- **Ext 1's attribution checks run only if triggered** (§4.1, fixed before any result). `M7-2` says whether Ext 1's block-aware contrast separates from zero; if it does not, there is no gain to attribute and `M8-4` closes with that one line.
- **`n_heads = 8` is fixed across the `d_model` sweep** (upstream CSDI's value; head_dim 8/16/32, all valid) so divisibility doesn't silently break the ablation (§3.6).
- **Runtime is a first-class result, not an afterthought** (§7). The MAE-vs-latency curve is one of the guaranteed high-performance stories even if every extension comes back null.
- **Conformal gate is a real decision with a real deadline** — end of this week (§4.3). If the combined model isn't stable at ≤ ~0.24 @10%, cut conformal cleanly; it does not weaken the core project.

---

## Issues

### `M8-1` — Ablation: `d_model`, diffusion steps `T`, mask-encoder layers
- **Assignee:** @SankhaS
- **Labels:** `eval`, `ablation`, `stats`, `mvp`
- **Blocked by:** `M7-1`, `M7-2`
- **Blocks:** `M9-1`

**Context.** Three bounded one-factor sensitivity studies on the best variant (plus the mask-encoder-layers study on the two encoder variants), run through the same paired harness. The design is deliberately ~35–40 runs, not a combinatorial grid (§5). Sankha owns it on the eval track; cells whose paired-Δ error bars overlap are reported **null**, not dropped.

**Deliverables**
- `d_model ∈ {64, 128, 256}` sweep on the best variant, 3 seeds, `n_heads = 8` fixed (§3.6).
- Diffusion-steps `T ∈ {50, 100, 200}` sweep (each a full retrain), best variant, 3 seeds — or at a single `d_model` if compute is tight, with that noted.
- Mask-encoder layers `∈ {1, 2, 3}` for the two encoder-bearing variants, 3 seeds, reported under the block holdout (`BlockEvalMasker`, Ext 1's primary regime) as well as the random one.
- Results folded into the results store with paired-Δ ± SE; null cells labeled.

**Definition of Done**
- [ ] All three studies complete at ~35–40 total runs (or the reduced `T` design is documented).
- [ ] `n_heads = 8` held constant across the `d_model` sweep.
- [ ] Each study reports paired-Δ ± SE; overlapping cells labeled null.
- [ ] Results reproducible from the run manifest.

---

### `M8-2` — Inference-latency profile + MAE-vs-inference-time plot
- **Assignee:** @jkelle11-source
- **Labels:** `diffusion`, `runtime`, `analysis`, `mvp`
- **Blocked by:** `M7-1`
- **Blocks:** `M9-1`

**Context.** The runtime contribution (§7) — explicitly rewarded by the rubric's high-performance line and rarely produced by other teams. Profile inference latency per variant × `d_model` and plot MAE against inference time, so the accuracy/compute trade-off is visible. Jake owns it (training/inference-engine track).

**Deliverables**
- Inference-latency measurements per variant × `d_model` (fixed hardware, warm-started, averaged over enough batches to be stable).
- **MAE-vs-inference-time plot** across variants/`d_model`, with points labeled.
- A short written read of the accuracy/latency trade-off.

**Definition of Done**
- [ ] Latency measured on fixed hardware with a documented protocol (warmup, repeats).
- [ ] MAE-vs-inference-time plot produced and committed.
- [ ] Trade-off interpretation written for the final report/presentation.

---

### `M8-3` — Conformal gate check (Extension 3 go/no-go)
- **Assignee:** @jkelle11-source, @JoshOlu, @SankhaS
- **Labels:** `eval`, `conformal`, `stretch`, `gate`
- **Blocked by:** `M7-2`
- **Blocks:** `M9-3`

**Context.** The end-of-week decision that governs Week 9 (§4.3). The studentized patient-level conformal stretch is attempted **only** if the combined model is stable with test MAE ≤ ~0.24 @10%. If not, cut cleanly and redirect that time to per-variable analysis and uncertainty fan plots — the stretch is explicitly optional and its removal does not weaken the core project.

**Deliverables**
- A documented go/no-go decision recorded in the repo, with the combined-model stability number it's based on.
- If go: a brief conformal implementation plan for Week 9 (patient-level split, studentized score, K=50 draws).
- If no-go: the redirect plan (per-variable analysis + fan plots) confirmed with owners.

**Definition of Done**
- [ ] Decision recorded with the supporting MAE number.
- [ ] Week-9 plan (conformal *or* redirect) written and owned.
- [ ] All three agree on the decision.

---

### `M8-4` — Ext 1 attribution checks (only if the block contrast separates)
- **Assignee:** @JoshOlu
- **Labels:** `eval`, `ablation`, `stats`, `ext1-mask-embedding`
- **Blocked by:** `M6-3`, `M7-1`, `M7-2`
- **Blocks:** `M9-1`

**Context.** The two checks §4.1 fixed before any result, built in `M6-3`. They run only if `M7-2` reports Ext 1's block-aware paired Δ vs. base separated from zero; otherwise this issue closes with that one line, since a null needs no control. **Ext 1-blind** asks whether the gain comes from the mask information or from the encoder's extra weights and learned position vector; the **off-grid block holdout** asks whether it survives blocks that do not sit on the training grid. Josh runs them: Week 8 is light on the Ext 1 track, and his compute runs in parallel with the ablation.

**Deliverables**
- If triggered: Ext 1-blind trained in the block-aware regime at seeds 42–44 through the paired harness, final weights scored on `BlockEvalMasker` (about 2.1 h of training and a 3.4 h scoring pass a run on an M4 Pro: about 16.5 h).
- If triggered: the block-aware base and Ext 1 runs at seeds 42–44 from `M7-1` scored on the off-grid holdout, no retraining (6 passes, about 20 h).
- Paired Δ ± SE over seeds 42–44 for Ext 1 vs. Ext 1-blind on the block holdout, and for Ext 1 vs. base on the off-grid holdout, with `M7-2`'s paired-Δ code, written into the results store.
- If not triggered: a one-line record of the `M7-2` Δ and SE that decided it.

**Definition of Done**
- [ ] Trigger decision recorded from `M7-2`'s numbers.
- [ ] If triggered: both contrasts reported as paired Δ ± SE; overlapping bars stated as no significant effect.
- [ ] Ext 1's claim worded per §4.1: "adds signal beyond the raw mask channel" only if Ext 1 beats Ext 1-blind, and "block missingness" in general only if the gain holds off the grid.
