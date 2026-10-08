# Milestone 8 — Ablation & Runtime Analysis

> **GitHub Milestone**
> **Title:** `M8 — Ablation & Runtime`
> **Timeframe:** Week 8 · Nov 19 – Nov 25 *(Nov 26 is Thanksgiving; this week ends before the buffer)*
> **Depends on:** `M7`
> **Description:**
> Characterize the best variant and produce the runtime contribution that anchors the high-performance story (§8). The ablation is **not** a 3×3×3 grid — it's three **one-factor sensitivity studies** on the best variant plus the mask-encoder-size study, totaling ~35–40 runs, not 180 (§6). Sankha owns the ablation studies; Jake produces the **MAE-vs-inference-time plot** (runtime is explicitly named in the rubric's high-performance line, and this curve is a contribution most teams won't have). At end of week, the **conformal gate** is checked (§5.3): attempt the stretch only if the combined model is stable at test MAE ≤ ~0.24 @10%; otherwise cut cleanly and redirect Week-9 time to per-variable analysis and uncertainty fan plots.

## Assignment summary
- **Sankha** — The three one-factor sensitivity studies + the mask-encoder-layers study (see table); report cells whose paired-Δ bars overlap as **null**, not dropped
- **Jake** — Inference-latency profile per variant × `d_model`; the **MAE-vs-inference-time plot** (the runtime contribution, §8)
- **All three** — End-of-week **conformal gate check** (§5.3): decide go/no-go on the Extension 3 stretch

## Coordination notes
- **The ablation is bounded on purpose** (§6). Not a grid, not 36+ configs:

  | Study | Values | Variants | Seeds | Runs |
  |---|---|---|---|---|
  | Headline | — | all 4 | 5 | 20 *(from M7)* |
  | `d_model` | 64 / 128 / 256 | best only | 3 | 9 |
  | Diffusion steps `T` (each a full retrain) | 50 / 100 / 200 | best only | 3 | 9\* |
  | Mask-encoder layers | 1 / 2 / 3 | the **two** with an encoder | 3 | 18 |

  \*If compute is tight, sweep `T` at a single `d_model` and **state it**. Total ≈ 35–40 runs.
- **`n_heads = 4` is fixed across the `d_model` sweep** (head_dim 16/32/64, all valid) so divisibility doesn't silently break the ablation (§4.6).
- **Runtime is a first-class result, not an afterthought** (§8). The MAE-vs-latency curve is one of the guaranteed high-performance stories even if every extension comes back null.
- **Conformal gate is a real decision with a real deadline** — end of this week (§5.3). If the combined model isn't stable at ≤ ~0.24 @10%, cut conformal cleanly; it does not weaken the core project.

---

## Issues

### `M8-1` — Ablation: `d_model`, diffusion steps `T`, mask-encoder layers
- **Assignee:** @SankhaS
- **Labels:** `eval`, `ablation`, `stats`, `mvp`
- **Blocked by:** `M7-1`, `M7-2`
- **Blocks:** `M9-1`

**Context.** Three bounded one-factor sensitivity studies on the best variant (plus the mask-encoder-layers study on the two encoder variants), run through the same paired harness. The design is deliberately ~35–40 runs, not a combinatorial grid (§6). Sankha owns it on the eval track; cells whose paired-Δ error bars overlap are reported **null**, not dropped.

**Deliverables**
- `d_model ∈ {64, 128, 256}` sweep on the best variant, 3 seeds, `n_heads = 4` fixed (§4.6).
- Diffusion-steps `T ∈ {50, 100, 200}` sweep (each a full retrain), best variant, 3 seeds — or at a single `d_model` if compute is tight, with that noted.
- Mask-encoder layers `∈ {1, 2, 3}` for the two encoder-bearing variants, 3 seeds.
- Results folded into the results store with paired-Δ ± SE; null cells labeled.

**Definition of Done**
- [ ] All three studies complete at ~35–40 total runs (or the reduced `T` design is documented).
- [ ] `n_heads = 4` held constant across the `d_model` sweep.
- [ ] Each study reports paired-Δ ± SE; overlapping cells labeled null.
- [ ] Results reproducible from the run manifest.

---

### `M8-2` — Inference-latency profile + MAE-vs-inference-time plot
- **Assignee:** @jkelle11-source
- **Labels:** `diffusion`, `runtime`, `analysis`, `mvp`
- **Blocked by:** `M7-1`
- **Blocks:** `M9-1`

**Context.** The runtime contribution (§8) — explicitly rewarded by the rubric's high-performance line and rarely produced by other teams. Profile inference latency per variant × `d_model` and plot MAE against inference time, so the accuracy/compute trade-off is visible. Jake owns it (training/inference-engine track).

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

**Context.** The end-of-week decision that governs Week 9 (§5.3). The studentized patient-level conformal stretch is attempted **only** if the combined model is stable with test MAE ≤ ~0.24 @10%. If not, cut cleanly and redirect that time to per-variable analysis and uncertainty fan plots — the stretch is explicitly optional and its removal does not weaken the core project.

**Deliverables**
- A documented go/no-go decision recorded in the repo, with the combined-model stability number it's based on.
- If go: a brief conformal implementation plan for Week 9 (patient-level split, studentized score, K=50 draws).
- If no-go: the redirect plan (per-variable analysis + fan plots) confirmed with owners.

**Definition of Done**
- [ ] Decision recorded with the supporting MAE number.
- [ ] Week-9 plan (conformal *or* redirect) written and owned.
- [ ] All three agree on the decision.
