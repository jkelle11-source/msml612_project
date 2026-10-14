# Project Plan: Missingness-Aware Diffusion Imputation for Clinical Time Series
### Extension of CSDI with a Learned Missingness Embedding & TSLO-Based Continuous-Time Encoding

**Course:** MSML612: Deep Learning — University of Maryland
**Duration:** ~10 weeks (Sep 30 – Dec 10, 2026) 
**Nominal effort:** ~3 hrs/student/week, front- and back-loaded around deliverables
**Dataset:** PhysioNet/CinC Challenge 2012

---

## 0. Decisions Locked In (read first)

These are settled and everything downstream assumes them:

1. **Reuse, don't reimplement, base CSDI.** We fork and adapt a published CSDI implementation (cited prominently) rather than re-deriving it from scratch. The rubric explicitly permits this. **Reuse covers the whole base model, not just the denoiser** (scope settled at the M1-2 review): the noise schedule (quad, β 1e-4 → 0.5, 50 steps), forward noising, masked noise-prediction loss, reverse sampler and attention head count (8) are upstream's published PhysioNet configuration, kept as-is so the reproduction gate (§3.4) is measured against the setup that produced 0.217. The one base component we replace is the masking (§3.1).
2. **Extension 2 uses TSLO (time-since-last-observation), not raw timestamps or a pairwise attention bias.** The time signal is *per-feature elapsed time since that feature was last actually measured*. This is genuinely irregular even on the 48-step grid, keeps our base comparable to the published benchmark, and makes Ext 2 non-trivial by construction.
3. **Paired-seed protocol.** All variants are trained on the *same* 5 seeds (42–46) — common random numbers — so every variant-vs-base comparison is a *paired* difference. The SE of the paired difference is what we test, not the SE of two independent means.
4. **Honesty-aware performance framing.** We lead with *benchmark reproduction* and *runtime analysis* as our guaranteed "high-performance" story, and report extension effects (including nulls) rigorously on top. See §7.

---

## 1. Overview & Contribution Narrative

We implement CSDI (Conditional Score-based Diffusion for Imputation) for multivariate clinical time-series imputation, reproduce its published benchmark, and extend it along two axes that address real limitations of the architecture — then, as a stretch goal, wrap it in calibrated uncertainty intervals.

- **Extension 1 — Learned Missingness Pattern Embedding.** CSDI already consumes the observation mask as a conditioning channel; we test whether a **learned, structural embedding** of the mask — one that attends across *both* time and features — adds signal beyond the raw mask channel, particularly under **block (cross-feature, contiguous) missingness**.
- **Extension 2 — TSLO-Based Continuous-Time Encoding.** CSDI's temporal positional encoding is index-based, implicitly assuming uniform spacing and uniform recency. We replace it with a Time2Vec-style encoding of **per-feature time-since-last-observation**, so the network weights a conditioning value by how stale it actually is.
- **Extension 3 (Stretch) — Conformal Prediction Intervals.** A post-hoc, patient-level, *studentized* split-conformal wrapper that turns the diffusion posterior into intervals with finite-sample coverage — using the sample spread CSDI normally discards.

**Why this is a defensible contribution (and where it is not):** each extension is toggleable and evaluated against our *own reproduced base* under a frozen protocol. We do not claim to beat published CSDI; we claim (a) faithful reproduction of the benchmark, (b) a rigorous, paired, multi-seed test of whether each extension helps, in the regime where it *should* help, and (c) a clinically meaningful uncertainty capability CSDI's evaluation omits.

**Why PhysioNet 2012.** ~4,000 ICU stays (Set A), 35 vitals/labs over 48 hours, ~80% missing, per-observation timestamps native to the archive (the raw signal Extension 2 needs), and an established CSDI benchmark to target. Freely available.

---

## 2. Milestone Calendar

| Date | Milestone | What must be true |
|---|---|---|
| **Wed Oct 8** | **Proposal due** | Scope, extensions, baselines, protocol, timeline written. Repackages §§1–8 of this plan. Environment + data pipeline underway. |
| **Wed Oct 29** | **Interim Report due** | Data pipeline + baselines done; **base CSDI reproduced and gate passed (≥3 seeds)**; Extension 1 implemented with preliminary mechanical-health results; full experimental protocol documented. |
| **Tue Dec 2** | **Presentation due** | Headline 5-seed table complete; ablation + runtime analysis done; slides built and rehearsed. |
| **Wed Dec 10** | **Final Report due** | Everything, plus stretch conformal (if gate passed), per-variable analysis, uncertainty plots, limitations. |

US Thanksgiving falls **Thu Nov 26**; the week of Nov 23–27 is treated as a light/buffer week (§5).

---

## 3. Frozen Experimental Protocol

These rules are non-negotiable and apply to **every** run — base and all extensions. They are what make the "Δ vs. base" numbers trustworthy; without them a 0.01–0.02 gain is unpublishable noise.

### 3.1 Two masks, not one (this is the subtle one)
CSDI's self-supervision requires *randomizing* which observed values are conditioning vs. target each iteration; the *evaluation* target set must be fixed. Freezing the training mask would depress our base MAE and poison the gate. We therefore implement **two** kinds of object, frozen evaluation holdouts and a fresh training split:

- **`EvalMasker`** — seeded **once**, holds out a fixed 10% of observed values as evaluation targets, identical across every model and every seed. This is the "frozen masking strategy."
- **`BlockEvalMasker`** — a second frozen holdout beside `EvalMasker`, for the block-missingness regime (§4.1). It is seeded the same way (its own frozen seed, one generator per patient) and hides exactly as many values per patient as `EvalMasker` does, `floor(0.10 · n_observed)`, but as contiguous **3-hour windows across all 35 features**: the 48 hours are tiled into 16 fixed 3-hour slots, the slots are drawn in random order, and every observed value in each slot is hidden until the next slot would overshoot the count; that slot is trimmed to a random subset of its observed values so the count is exact. The two holdouts therefore differ only in *where* values are hidden, not how many, so block and random results compare at one rate. Neither changes the other: `EvalMasker`'s targets stay exactly as frozen.
- **`TrainMasker`** — draws a *fresh* conditioning/target split each iteration; its stream is reproducible from the run seed but is **not** pinned to a single pattern. Its `random` strategy trains every variant; its `mix` strategy trains the block-aware regime below.

**Two training regimes.** The headline sweep trains all four variants with `TrainMasker`'s `random` strategy, upstream's published setup and the one the gate is measured against. Random training alone does not prepare a model for block missingness. It hides a whole 3-hour stretch of every feature only when most of the record is hidden: on our training split, the samples that contain such a gap keep a median 9% of their values visible, and no sample that keeps 60% or more contains one, whereas `BlockEvalMasker` leaves 90% visible around its gaps. So **base and Ext 1 are also trained in a block-aware regime**, at the same seeds 42–46 (10 more runs), with `TrainMasker`'s `mix` strategy. For each sample a fair coin, drawn from `TrainMasker`'s own generator, picks either the `random` split or block targets: `round(r · n_observed)` values with `r ~ U(0, 1)` as in `random`, hidden as 3-hour all-feature slots drawn and trimmed as in `BlockEvalMasker`. This is upstream's own answer to structured missingness (its `mix` strategy mixes random targets with a structured pattern), applied with our block shape. Each regime is paired only within itself; the two are never compared with each other. Because the block-aware regime trains on the same window shape the holdout uses, it is matched to our synthetic pattern, and the report says so (§8, Limitations).

**Scoring under both holdouts.** Every run's selected checkpoint is scored on the test split under both holdouts, which costs one extra evaluation pass per run. Early stopping and checkpoint selection use the validation MAE of the holdout the run is trained for: `EvalMasker`'s in the random regime, `BlockEvalMasker`'s in the block-aware regime. Ext 1's block-missingness contrast (§4.1) is read from the block-aware regime; the random regime's block scores are reported alongside, as a check of how well random-trained models carry over to blocks.

**How these map onto the reused code.** The maskers are the only part of base CSDI we do not reuse (§0). Upstream draws its training split inside the model (`get_randmask`) from the global NumPy/torch RNGs, so anything else that consumes random numbers — an extension's extra parameter initialisation, for example — would shift the mask stream and break the pairing in §3.3. `TrainMasker` therefore replaces it and draws from its **own** generator; it keeps upstream's `random` strategy (a per-sample target ratio drawn uniformly from [0, 1]), so the only thing that changes is where the random numbers come from. Adding `mix` leaves the `random` stream exactly as it was. `EvalMasker`'s conditioning mask is passed to the reused model as upstream's `gt_mask`, and so is `BlockEvalMasker`'s.

Unit test: `EvalMasker` and `BlockEvalMasker` targets are byte-identical across seeds; `BlockEvalMasker` hides exactly `EvalMasker`'s per-patient count, and every target outside the one trimmed slot lies in a 3-hour slot whose observed values are all hidden; `TrainMasker` targets differ across iterations; `TrainMasker`'s `random` output is byte-identical before and after `mix` is added, and `mix`'s block samples have the same slot structure as `BlockEvalMasker`'s; eval targets never enter any conditioning input (no leakage).

### 3.2 Normalization (load-bearing for every comparison)
Per-variable **z-scoring** using **training-split statistics only**, applied to val/test. All MAE/RMSE are reported in **standardized units**, so the published 0.217 is a valid comparison. We additionally report raw-unit per-variable MAE for clinical readability. Conformal residuals stay in standardized units so a single global quantile is coherent across variables.

### 3.3 Paired 5-seed evaluation
- Seeds **42–46** (field-standard set).
- **Common random numbers:** the same 5 seeds across all four variants, so each variant shares data order, train-mask stream, and init RNG with the others at the matching seed. The diffusion timestep and noise draws are paired as well, each stream drawn from its own generator rather than the global RNG (as `TrainMasker` is, §3.1); **dropout is the one stream that is not paired**, because a variant with extra dropout layers consumes more of the global RNG. Comparisons are **paired differences**; we report the mean paired Δ and the **SE of the paired Δ**, which is typically far smaller than the SE of either mean. The block-aware regime (§3.1) uses the same seeds and the same rules: its base and Ext 1 are paired with each other, never with the random regime's runs.
- **Frozen evaluation noise:** the reverse sampler's noise comes from one frozen seed (`eval_sampling_seed`), re-seeded at the start of every evaluation pass and run at a fixed batch size and patient order, so a checkpoint's score depends on its weights alone and every variant, seed and epoch is scored with identical noise.
- Every headline number is mean ± SE over the 5 seeds. A contrast whose paired-Δ error bar overlaps zero is reported as **"no significant effect,"** never hidden or spun.

### 3.4 Reproduction gate (hard, before any extension delta is trusted)
Our reproduced base CSDI must reach **≤ ~0.24 MAE @ 10% masked** (within ~10% of the paper's 0.217), averaged over **≥3 seeds**, before any extension result counts. Independent reimplementations land ~5% above the paper and say so in print, so the 0.22–0.24 band — not exactly 0.217 — is the target. Until the gate passes, extension runs are *debugging* runs. If the gate is not passed by the interim (Oct 29), architecture work freezes and the baseline is debugged.

### 3.5 Effect claims
- **Main effects** (Ext 1 vs. base; Ext 2 vs. base) are **confirmatory**, reported per-effect with paired SE.
- The **interaction** (Ext 1+2 vs. each alone) is **exploratory** — with four means inside a ~0.02–0.03 window, 5 seeds may not resolve it, and we will not assert an interaction the error bars can't support.

### 3.6 Head/`d_model` divisibility
Fix **`n_heads = 8`** — upstream CSDI's setting, so the gate run matches the published configuration — across the `d_model ∈ {64, 128, 256}` sweep (head_dim = 8/16/32, all valid). State the rule so the ablation doesn't silently break.

---

## 4. Novel Contributions: Technical Specification

### 4.1 Extension 1 — Learned Missingness Pattern Embedding

**CSDI consumes the missingness mask as a raw channel; we test whether a learned structural embedding of the mask adds signal beyond that.** 

**Architecture.** The binary mask $M \in \{0,1\}^{T \times D}$ passes through a small **dual-axis** encoder $\phi_\psi$ — temporal *and* feature attention, mirroring the main denoiser — because block missingness is a *cross-feature* structure a temporal-only encoder cannot represent. Like the denoiser, it keeps **one token per (hour, feature)**: feature attention needs features to attend over, and the output, `(B, T, D, d_model)`, gives every conditioning value its own embedding. A learned feature embedding tells the features apart, and a learned hour embedding gives temporal attention an order, without which it could not tell contiguous hours from scattered ones.

```python
class MissingnessEncoder(nn.Module):
    """Dual-axis embedding of the conditioning mask (temporal + feature attention)."""
    def __init__(self, n_features, n_timesteps, d_model, n_heads=2, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)               # one token per (hour, feature)
        self.feature_emb = nn.Embedding(n_features, d_model)  # which feature a token is
        self.time_emb = nn.Embedding(n_timesteps, d_model)    # which hour it is
        self.temporal = nn.ModuleList([      # dim_feedforward=64, as in the denoiser's layers
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=64, batch_first=True)
            for _ in range(n_layers)])
        self.feature = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=64, batch_first=True)
            for _ in range(n_layers)])

    def forward(self, mask):                 # mask: (B, T, D) float
        B, T, D = mask.shape
        x = (self.input_proj(mask.unsqueeze(-1))       # (B, T, D, d_model)
             + self.feature_emb.weight                  # (D, d_model)
             + self.time_emb.weight[:, None])           # (T, 1, d_model)
        for t_layer, f_layer in zip(self.temporal, self.feature):
            x = t_layer(x.transpose(1, 2).reshape(B * D, T, -1))          # attend over time
            x = x.reshape(B, D, T, -1).transpose(1, 2)
            x = f_layer(x.reshape(B * T, D, -1)).reshape(B, T, D, -1)     # attend over features
        return x                             # (B, T, D, d_model), added to denoiser input
```

- **Toggle:** `--use_mask_embedding`. Trained end-to-end (not pretrained).
- **Primary evaluation regime:** **block/structured missingness** — Ext 1's headline test, scored on `BlockEvalMasker`'s frozen holdout (§3.1): contiguous 3-hour windows hidden across all 35 features. That synthetic holdout *is* the block-missingness stress test: PhysioNet's own missingness cannot be controlled, so the block structure is imposed on held-out values. The contrast is base vs. Ext 1 in the block-aware training regime (§3.1), so the encoder has been trained on the structure it is scored on. If the embedding helps anywhere, it is here.
- **Named risk:** Ext 1 shows ≈0 gain because the base already sees the mask. Mitigated by the dual-axis encoder (richer than the raw channel) and the block-missingness primary regime, trained block-aware (§3.1); if still null, that is a legitimate, reportable finding, not a failure.

### 4.2 Extension 2 — TSLO-Based Continuous-Time Encoding

**The core idea.** For each feature $d$ at each step $t$, compute $\delta_{t,d}$ = hours since feature $d$ was last actually observed (GRU-D-style). On the 48-step hourly grid the *global* step index is uniform, but $\delta$ is **not** — it jumps whenever a feature goes unmeasured. This is the information Ext 2 seeks to exploit.

**Encoding.** $\delta$ is embedded with a **true Time2Vec** encoding — one linear dimension plus $d_k-1$ sinusoidal dimensions, *concatenated:*

$$\text{T2V}(\delta)_k = \begin{cases}\omega_0\delta + \phi_0 & k=0\\ \sin(\omega_k\delta + \phi_k) & 1 \le k \le d_k-1\end{cases}$$

Each (hour, feature) is encoded and projected to `d_model` on its own, so the value at that position is told its own staleness rather than a summary of every feature's. The projection is shared across features; the denoiser's feature embedding tells them apart.

```python
class ContinuousTimeEncoding(nn.Module):
    """True Time2Vec of per-feature TSLO; projected to d_model at each (hour, feature), added like a PE."""
    def __init__(self, d_k, d_model):
        super().__init__()
        self.w0 = nn.Linear(1, 1)             # linear component (k=0)
        self.w  = nn.Linear(1, d_k - 1)       # periodic components (k>=1)
        self.proj = nn.Linear(d_k, d_model)   # shared by every (hour, feature)

    def forward(self, delta):                 # delta: (B, T, D) TSLO in hours
        d = delta.unsqueeze(-1)               # (B, T, D, 1)
        t2v = torch.cat([self.w0(d), torch.sin(self.w(d))], dim=-1)  # (B,T,D,d_k)
        return self.proj(t2v)                                        # (B, T, D, d_model)
```

- **Toggle:** `--use_time_encoding`. The encoding is **added to the denoiser input** like a positional encoding — it does **not** touch the attention logits.
- **Primary evaluation regime:** high-autocorrelation variables (heart rate, SpO2), where recency matters most.

### 4.3 Extension 3 (Stretch) — Studentized, Patient-Level Conformal Intervals

Post-hoc, no retraining. Two key decisions ensure its validity:

1. **Patient-level exchangeability.** Split calibration/test **by patient**. The nonconformity score is aggregated at the **patient** level. Position-level coverage is reported afterward as a secondary empirical check, not as the guarantee.
2. **Studentized scores (use the spread CSDI discards).** With $K=50$ posterior draws, compute per-position mean $\hat\mu$ **and std $\hat\sigma$**, and score $s = |x_{\text{true}} - \hat\mu| / (\hat\sigma + \varepsilon)$. Intervals are **adaptive**: $[\hat\mu - \hat q\,\hat\sigma,\ \hat\mu + \hat q\,\hat\sigma]$. 

Report at $1-\alpha = 0.90$: empirical coverage (target ≥ 90%), mean interval width, and coverage stratified by patient missingness rate.

**Gate:** attempt only if the combined model (Ext 1+2) is stable with test MAE ≤ ~0.24 @10% by **end of Week 8 (Nov 25)**. Otherwise cut cleanly — it does not weaken the core project — and redirect that time to per-variable analysis and uncertainty fan plots.

---

## 5. Timeline (10 weeks, anchored to the four due dates)

Nominal cadence ~3 hrs/student/week. 

### Phase 1 — Foundations & Proposal (Weeks 1–2)

**Week 1 · Sep 30 – Oct 7 — Setup, data, proposal drafting**
- **All (day 1):** Register on PhysioNet (all three), pull Set A from the legacy archive in parallel; read CSDI §§3–4 together; B reads Time2Vec. Agree repo layout (`data/ models/ diffusion/ eval/`), fix and log the **`EvalMasker` seed**.
- **A:** Data loader → `(N, 48, D)`; **compute and store per-feature TSLO $\delta$**; per-variable **z-score** (train stats). uv env (py≥3.10, torch≥2.1, einops, wandb) — deps in `pyproject.toml`, pinned in `uv.lock`, reproduced via `uv sync`.
- **B:** Fork and stand up the reused CSDI repo; get it running on a toy batch; **cite the source in the README now**.
- **C:** `EvalMasker` + `TrainMasker` (§3.1) with the leakage/immutability unit tests; MAE/RMSE (masked, standardized).
- **All (writing):** Draft the **Proposal** from §§1–8.
- **▶ DELIVERABLE — Proposal due Wed Oct 8.**

**Week 2 · Oct 8 – Oct 14 — Diffusion process, objective, baselines start**
- **A:** Training loop on the **reused** diffusion process (`TrainMasker → reused forward noising → denoiser → masked loss → clip → step`), with `TrainMasker` replacing upstream's internal masking; numerically check the reused quad schedule's forward-noising variance. No schedule or `q_sample` of our own (§0). Then `BlockEvalMasker`, the frozen block-missingness holdout (§3.1), which must land before Week 4's Ext 1 overfit check uses it.
- **B:** Adapt the reused model to our tensor shapes and conditioning; thread the `mask_emb`/`time_enc` hooks (both default `None`) through its loss and sampler paths to the denoiser.
- **C:** Mean + forward-fill baselines recorded; shape/NaN check of the reused reverse sampler, driven through `EvalMasker`.

### Phase 2 — Base Reproduction & Extension 1 (Weeks 3–4)

**Week 3 · Oct 15 – Oct 21 — Base CSDI reproduction**
- **A + B:** Get the adapted base CSDI training and converging below forward-fill; log grad norms; no explosion/collapse in first 500 steps.
- **C:** BRITS baseline (or simplified bi-GRU fallback, noted); wandb tracking live (loss/step, val MAE-RMSE/epoch, grad norms, config).

**Week 4 · Oct 22 – Oct 28 — Gate + Extension 1 + interim drafting**
- **A + B (baseline hardening, ~6 hrs across the two):** Drive base CSDI to the **reproduction gate** (§3.4): ≤ ~0.24 @10%, ≥3 seeds. This is the fiddly dual-axis core everything is measured against; it ends on the gate, not on "converging."
- **C:** Implement `MissingnessEncoder` (§4.1), wire via `mask_emb`; confirm nonzero gradients reach it; overfit sanity check (10-patient subset scored on `BlockEvalMasker`'s targets, MAE < 0.1 in 50 epochs).
- **All (writing):** Draft **Interim Report** — baselines, protocol, gate status, Ext 1 preliminary (mechanical-health, single-seed; *not* an effect claim).
- **▶ DELIVERABLE — Interim Report due Wed Oct 29.** Must include a **passed gate**; if the gate is not passed, the interim says so honestly and Week 5 becomes baseline debugging (§3.4).

### Phase 3 — Extension 2 & Combined Model (Weeks 5–6)

**Week 5 · Oct 29 – Nov 4 — Extension 1 complete, Extension 2 start**
- **A:** Implement `ContinuousTimeEncoding` on TSLO (§4.2); add via `time_enc`; validate on high-autocorrelation variables. Gate here is **mechanical** (gradients flow, loss drops, no NaNs) — *not* effect detection (that waits for 5 seeds).
- **B:** Feed TSLO $\delta$ through the pipeline at train and inference; confirm $\delta$ stays aligned to observations after masking (a misalignment silently corrupts Ext 2).
- **C:** Preliminary single-seed runs of base / Ext1 / Ext2 for health only.

**Week 6 · Nov 5 – Nov 11 — Combined model, ready for the seed sweep**
- **A:** Combined Ext 1+2; confirm both hooks compose and the model runs with all four toggle combinations.
- **B + C:** Stand up the **paired-seed harness** (§3.3): one driver runs all four variants at each of seeds 42–46 sharing data order and RNG, plus base and Ext 1 in the block-aware regime (`TrainMasker`'s `mix` strategy, §3.1). Dry-run at 20 epochs to confirm pairing and logging.

### Phase 4 — Evaluation, Ablation, Runtime (Weeks 7–8)

**Week 7 · Nov 12 – Nov 18 — Full paired 5-seed training**
- **A:** Full training (100 epochs, early stop on val MAE, patience 10) for all four variants × seeds 42–46 (20 runs), plus base and Ext 1 in the block-aware regime × seeds 42–46 (10 runs): 30 in total.
- **B:** Populate the **headline table** — mean ± SE and **paired Δ vs. base with its SE**; stratify by missingness type, scoring every run under both frozen holdouts (random `EvalMasker`, block `BlockEvalMasker`, §3.1). Main effects confirmatory, interaction exploratory.
- **C:** Qualitative trajectories (5 patients × 3 variables: HR, serum sodium, GCS); temporal coherence vs. mean/forward-fill.

**Week 8 · Nov 19 – Nov 25 — Ablation + runtime**
Three **one-factor sensitivity studies** on the **best** variant, plus mask-encoder size only where it's defined:

| Study | Values | Variants | Seeds | Runs |
|---|---|---|---|---|
| Headline | — | all 4 | 5 | 20 |
| Block-aware regime (§3.1) | — | base, Ext 1 | 5 | 10 |
| `d_model` | 64 / 128 / 256 | best only | 3 | 9 |
| Diffusion steps `T` (each a full retrain) | 50 / 100 / 200 | best only | 3 | 9* |
| Mask-encoder layers | 1 / 2 / 3 | the **two** with an encoder | 3 | 18 |

\*If compute is tight, sweep `T` at a single `d_model` and state it. Total ≈ 35–40 runs, not 180. Cells whose paired-Δ error bars overlap are reported **null**, not dropped.
- **A:** Inference-latency profile per variant × `d_model`; **MAE-vs-inference-time plot** (the runtime contribution — see §7).
- ⚠️ **Conformal gate check** at end of week (§4.3).

### Phase 5 — Stretch, Presentation, Final Report (Weeks 9–10)

**Week 9 · Nov 26 – Dec 1 — (Thanksgiving-light) Final eval, stretch, slides**
- **A:** Final best-config eval per variant → final results table.
- **B:** Per-variable MAE (all 35): top-5 / bottom-5 improvers; hypothesize Ext-2 gains concentrate on high-autocorrelation variables.
- **C (if gate passed):** Studentized patient-level conformal (§4.3): coverage, width, coverage-by-missingness-rate. Else: uncertainty fan plots.
- **All:** Build and rehearse the **presentation**; assign speaking parts.
- **▶ DELIVERABLE — Presentation due Tue Dec 2.**

**Week 10 · Dec 2 – Dec 9 — Final report & code cleanup**
- **A:** README with one-command reproduction per variant (command→reported-result table); fixed/logged seeds; docstrings on every class/function; uncertainty fan plots (20 draws + 90% conformal band if completed).
- **B + C:** Write the **Final Report** (structure in §8).
- **▶ DELIVERABLE — Final Report due Wed Dec 10.**

---

## 6. Baselines & Targets (corrected scale)

CSDI is benchmarked at three masked-value rates, not one number. Headline rate is **10%** (matches our blackout protocol); 50/90% are robustness. Baseline-of-baselines numbers (BRITS, V-RIN) are **cited, not re-derived**, so they carry no SE of ours. The table is the random holdout; every model we run ourselves (mean, forward-fill, our BRITS, all CSDI variants) is also scored on the block holdout at the same 10% rate (§3.1), which has no published counterpart.

| Model | MAE @10% | MAE @50% | MAE @90% | Source |
|---|---|---|---|---|
| Mean imputation | ~0.72 | ~0.72 | ~0.72 | our run (floor) |
| Forward fill | ~0.42 | ~0.55 | ~0.72 | our run |
| BRITS | 0.284 | 0.368 | 0.517 | Tashiro 2021, Table 3 |
| V-RIN | 0.271 | 0.365 | 0.606 | Tashiro 2021, Table 3 |
| CSDI (paper) | 0.217 | 0.301 | 0.481 | Tashiro 2021 |
| CSDI (independent reimpl.) | 0.228 | 0.319 | 0.530 | CFMI 2025 |
| **CSDI (ours, reproduced)** | target 0.22–0.24 | 0.30–0.34 | 0.48–0.55 | our run — **the gate** |
| **+ Ext 1 (mask emb.)** | mean ± SE | — | — | our run, 5 paired seeds |
| **+ Ext 2 (TSLO enc.)** | mean ± SE | — | — | our run, 5 paired seeds |
| **+ Ext 1 & 2 (ours)** | mean ± SE | — | — | our run, 5 paired seeds |

**Performance targets** (headline 10%, all "ours" rows mean ± SE; an extension "counts" only if the **paired Δ** error bar separates from zero):

| Model | Target MAE @10% | Δ vs. our base |
|---|---|---|
| CSDI (ours, reproduced) | 0.22–0.24 | — (the gate) |
| + Ext 1 only | 0.21–0.23 | −0.01 to −0.02, SE-separated |
| + Ext 2 only | 0.21–0.23 | −0.01 to −0.02, SE-separated |
| **Full (Ext 1 + 2)** | **0.20–0.22** | **best, SE-separated** |

| Stretch | Target |
|---|---|
| Conformal coverage @ 90% | empirical ≥ 90% (patient-level) |
| Mean interval width | narrower than 2× full-model RMSE, adaptive by uncertainty |

Two honest notes for the report: (1) expected extension gains (~0.01–0.02) are the same magnitude as CSDI's documented masking sensitivity — which is *why* masking is frozen and seeds are paired; (2) independent reimplementations land ~5% above the paper and say so, so our base near 0.22–0.24 is the documented norm, not a failure.

---

## 7. Rubric Alignment & the Honesty/Performance Strategy

| Rubric line (pts) | How this plan scores it |
|---|---|
| Data prep/curation (10) | Parsing, TSLO, z-scoring, dual-mask + tests. Covered. |
| Difficulty of NN design (25) | Dual-axis denoiser + two architectural extensions. Effort redirected from reproduction to extensions via reuse. |
| Clean reproducible code (20) | CLI toggles, unit tests, fixed/logged seeds, command→result README, single canonical class names, code matching the math. |
| **High performance (25)** | **See strategy below.** |
| Presentation (10) | Proposal + interim + rehearsed slides now exist as tasks; report structure in §8. |
| References (10) | CSDI, Time2Vec, BRITS, DDPM, conformal, V-RIN, SADI, CFMI, "Beyond Random Missingness." Solid. |

**The high-performance strategy (deliberate, not spin).** This rubric line pays for *performance*, and rigorously reporting a null does not obviously earn it. We resolve the tension by guaranteeing at least one true positive performance story and framing the rest as method quality:
1. **Lead with benchmark reproduction** — hitting ~0.22 @10% *is* a strong, gradeable result on its own.
2. **Make the runtime/compute-quality plot prominent** — "running time" is explicitly named in this rubric line; the MAE-vs-latency curve is a contribution most teams won't have.
3. **Evaluate each extension in its best honest regime as a *primary* result** — block missingness for Ext 1, high-autocorrelation variables for Ext 2 — giving each its best legitimate shot at an SE-separated gain.
4. **State the significance bar up front**, so a flat pooled result reads as rigor, not failure.

Go in eyes-open: if every extension comes back null, the benchmark match + runtime analysis must carry the 25 points — and they can.

---

## 8. Deliverables, Report Structure, Risks

### Deliverables
Proposal (Oct 8) · Interim Report (Oct 29) · Presentation (Dec 2) · Final Report (Dec 10) · Modular PyTorch repo (four CLI-toggled variants, tests, reproducible scripts, cited CSDI source) · Baseline suite · Reproduced+gated CSDI · Ext 1 · Ext 2 · Full model · Restructured ablation + runtime plot · Conformal module (stretch).

### Final Report structure
1. Introduction — imputation in clinical TS; limits of deterministic methods; why diffusion.
2. Related Work — CSDI, BRITS, V-RIN, Time2Vec, conformal; our contribution distinguished from each.
3. Method — CSDI background; Ext 1 (honest framing, dual-axis encoder); Ext 2 (TSLO + true Time2Vec, why no attention bias); Ext 3 (studentized patient-level conformal, if done).
4. Experiments — dataset, **normalization**, dual-mask protocol (random and block holdouts, two training regimes), **paired 5-seed** design, baselines, headline table (paired Δ ± SE), ablation, per-variable, runtime, coverage.
5. Discussion — main effects vs. exploratory interaction; where/why extensions help; nulls stated as findings.
6. Limitations — patient-level exchangeability; grid vs. native-timestamp choice for TSLO; reduced scale vs. published; single dataset; block missingness is synthetic (one window shape imposed on held-out values), and the block-aware regime trains on that same shape.
7. Conclusion & future work (per-feature TSLO attention bias; native-timestamp irregular grid).

### Risk register
| Risk | Likelihood | Mitigation |
|---|---|---|
| Base CSDI misses the gate (>~0.24 @10%) | Medium | Reuse + adapt; 2 students in Week 4; ≥3-seed gate; freeze extensions until passed. |
| **Ext 1 ≈0 gain (base already sees mask)** | Medium | Dual-axis encoder; block-missingness as primary regime (frozen `BlockEvalMasker` holdout), with base and Ext 1 trained block-aware so the encoder sees the structure it is scored on (§3.1); null is a reportable finding. |
| Sweep compute (30 runs) exceeds GPU time | Medium | Block-aware runs go after the 20 headline runs; if time runs out, drop them to 3 seeds and label the block contrast exploratory, stated as such. |
| Extension gain within seed noise | Medium-High | **Paired** 5 seeds; report SE of the paired Δ; overlapping bars → null, not spun. |
| **Interaction not resolvable at 5 seeds** | Medium | Scope confirmatory claims to main effects; interaction exploratory. |
| Masking choice confounds the delta | Medium | Frozen holdouts (`EvalMasker`, `BlockEvalMasker`); `TrainMasker` randomized (§3.1). |
| Ablation labor under-budgeted | Medium | Restructured to ~35–40 runs; drop to 3 seeds/cell if tight and state it; runtime hours are a real line item. |
| Deliverable milestones missed | Medium | Proposal/interim/slides are explicit tasks with owners and dates in §5. |
| Conformal not reached | Medium | Explicit stretch; gated at end of Week 8; cut cleanly. |
| PhysioNet DUA wall | Low | Legacy archive Set A (open CSVs); if a DUA appears it's sign-and-download, not CITI. Register day 1, all three. |
| TSLO misaligned after masking | Medium | Week-5 alignment check; unit test $\delta$ against observations. |

---

## 9. Key References
- Tashiro et al. (2021). *CSDI.* NeurIPS. arXiv:2107.03502
- Kazemi et al. (2019). *Time2Vec.* arXiv:1907.05321
- Cao et al. (2018). *BRITS.* NeurIPS.
- Che et al. (2018). *GRU-D (Recurrent NN for multivariate TS with missing values).* Sci. Rep. — TSLO / decay motivation.
- Ho et al. (2020). *DDPM.* NeurIPS.
- Angelopoulos & Bates (2021). *A Gentle Introduction to Conformal Prediction.* arXiv:2107.07511
- CFMI (2025). arXiv:2506.09258 — independent CSDI reimplementation.
- *Beyond Random Missingness* (2025). arXiv:2405.17508 — masking sensitivity.
- SADI (2024). PMC11391213 — 5-seed (42–46) convention.
- PhysioNet/CinC Challenge 2012 archive; CSDI reference implementation (fork + cite in README).
