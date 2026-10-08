# Milestone 1 — Setup, Data & Proposal

> **GitHub Milestone**
> **Title:** `M1 — Setup, Data & Proposal`
> **Timeframe:** Week 1 · Sep 30 – Oct 7
> **Depends on:** none
> **Description:**
> Stand up the shared substrate and ship the **Proposal (due Wed Oct 8)**. Three things must exist by end of week: a working PhysioNet Set A data pipeline that emits `(N, 48, D)` tensors *plus the per-feature TSLO δ that Extension 2 will consume*, the reused CSDI implementation forked and running on a toy batch, and the two-mask evaluation objects (`EvalMasker`/`TrainMasker`) with their leakage tests. Everything downstream measures against decisions locked here — the repo layout (`data/ models/ diffusion/ eval/`), the **frozen `EvalMasker` seed**, and the per-variable z-scoring convention (§0, §4). This week is front-loaded by design (§6): pipeline, eval skeleton, and the proposal all land together.

## Assignment summary
- **Jake** — PhysioNet Set A loader → `(N, 48, D)`; per-feature **TSLO δ** computation + storage; per-variable **z-scoring** (train-split stats only); uv environment (`pyproject.toml` + `uv.lock`)
- **Josh** — Fork and stand up the reused CSDI implementation; run it on a toy batch; **cite the source in the README now**
- **Sankha** — `EvalMasker` + `TrainMasker` (§4.1) with leakage/immutability unit tests; masked, standardized MAE/RMSE metrics
- **All three** — Day-1 kickoff (PhysioNet registration ×3, read CSDI §§3–4, agree repo layout, fix + log the `EvalMasker` seed); draft the **Proposal**

## Coordination notes (agree at kickoff, before anyone writes code)
1. **We run three coherent tracks to minimize handoffs** — a data/diffusion track (Jake), an architecture track (Josh), and an evaluation/protocol track (Sankha) — but we **load-balance across weeks rather than lock fixed titles**. Ownership is stated per milestone and flexes to keep weekly effort even.
2. **Register on PhysioNet day 1 (all three).** The Challenge 2012 legacy archive Set A is open CSVs; if a DUA appears it's sign-and-download, not CITI (§7). Do it early so nobody is blocked at pull time.
3. **Freeze and log the `EvalMasker` seed this week.** It holds out a fixed 10% of observed values as evaluation targets, **identical across every model and every seed for the entire project** (§4.1). Changing it later silently invalidates every comparison, so it is committed and logged now.
4. **Agree the denoiser input-hook contract now** — the reused denoiser must accept `mask_emb` and `time_enc`, both defaulting to `None`. Freezing this interface lets the data, architecture, and eval tracks proceed in parallel without waiting on each other.

---

## Issues

### `M1-1` — PhysioNet pipeline: loader, TSLO δ, z-scoring, environment
- **Assignee:** @jkelle11-source
- **Labels:** `data`, `physionet`, `tslo`, `mvp`
- **Blocked by:** `M1-4` *(repo layout + seed agreed at kickoff)*
- **Blocks:** `M2-1`, `M2-2`, `M1-3`

**Context.** The data backbone the whole project trains on. Parse PhysioNet/CinC 2012 Set A (~4,000 ICU stays, 35 vitals/labs over 48 hours, ~80% missing) into a dense `(N, 48, D)` grid with an observation mask, and — critically — **compute and store the per-feature time-since-last-observation δ (GRU-D style)**, the raw irregularity signal Extension 2 exploits (§5.2). z-scoring uses **training-split statistics only**, applied to val/test, so reported MAE/RMSE are in standardized units comparable to the published 0.217 (§4.2). This is Jake's track because the δ lifecycle (compute here → plumb in W5 → encode in W5) stays within one owner to avoid handoff friction.

**Deliverables**
- Data loader producing `(N, 48, D)` tensors + binary observation mask `M ∈ {0,1}^{T×D}`, with a train/val/test split fixed and logged.
- Per-feature **TSLO δ** tensor `(N, 48, D)` (hours since each feature was last actually measured), stored alongside the data and aligned to the grid.
- Per-variable **z-scoring** computed on the **train split only** and applied to val/test; scaler persisted so inference is reproducible.
- uv-managed environment (`python ≥ 3.10`, `torch ≥ 2.1`, `einops`, `wandb`): dependencies declared in `pyproject.toml` and pinned in `uv.lock`. **All training runs on NVIDIA/CUDA.** `torch` resolves from a platform-marked PyTorch CUDA index on the Linux training box; macOS dev machines resolve CPU/MPS wheels from PyPI so local tests still run. Pin the exact `cuXXX` wheel to the training box's driver in M1-1.

**Definition of Done**
- [ ] Loader returns `(N, 48, D)` + mask for Set A; shapes and missingness rate (~80%) sanity-checked against the archive.
- [ ] TSLO δ verified against a hand-computed example on 2–3 patients; δ resets to 0 at each observation.
- [ ] z-score statistics are train-split-only (no val/test leakage) and the scaler round-trips (standardize → inverse ≈ identity).
- [ ] `uv sync` reproduces the env from `uv.lock` from scratch on a teammate's machine.

---

### `M1-2` — Fork and stand up the reused CSDI implementation
- **Assignee:** @JoshOlu
- **Labels:** `architecture`, `csdi`, `mvp`
- **Blocked by:** `M1-4`
- **Blocks:** `M2-2`

**Context.** We **reuse, don't reimplement, base CSDI** (§0, decision 1) — our engineering effort goes into the extensions. Fork a published CSDI implementation, get it running end-to-end on a toy batch, and **cite the source prominently in the README from day one** (this is the reuse the rubric permits, and it must be transparent — §8). Nailing this now de-risks the Week-3 reproduction: the model already runs before we point it at real data.

**Deliverables**
- Reused CSDI repo forked into our tree under `models/`, with the upstream commit hash recorded.
- Toy-batch smoke run: the denoiser trains for a few steps on synthetic tensors of our shape without shape or NaN errors.
- README section citing the reused implementation (repo, commit, paper) and stating exactly what we adapt vs. reuse.

**Definition of Done**
- [ ] `git log`/README records the exact upstream source + commit.
- [ ] A toy batch runs forward+backward through the denoiser without errors.
- [ ] README reuse citation is committed and visible on the repo root.

---

### `M1-3` — `EvalMasker` + `TrainMasker` + masked metrics
- **Assignee:** @SankhaS
- **Labels:** `eval`, `protocol`, `mvp`
- **Blocked by:** `M1-1` *(tensor/mask shape)*, `M1-4`
- **Blocks:** `M2-3`, `M3-3`

**Context.** The subtle heart of the protocol (§4.1). Two objects, not one: **`EvalMasker`** is seeded once and holds out a fixed 10% of observed values as evaluation targets, byte-identical across every model and seed; **`TrainMasker`** draws a fresh conditioning/target split each iteration (reproducible from the run seed, not pinned). Freezing the *training* mask would depress base MAE and poison the gate, so the split is deliberate. Metrics are masked and reported in **standardized units** (§4.2).

**Deliverables**
- `EvalMasker` (fixed 10% holdout, seeded from the logged project seed) and `TrainMasker` (fresh per-iteration split, seeded from the run seed).
- Masked MAE/RMSE metrics computed only over held-out targets, in standardized units, with an optional raw-unit per-variable readout for clinical readability.
- Unit tests: `EvalMasker` targets byte-identical across seeds; `TrainMasker` targets differ across iterations; **eval targets never enter any conditioning input** (no leakage).

**Definition of Done**
- [ ] `EvalMasker` output is byte-identical across two different run seeds (test green).
- [ ] `TrainMasker` output differs across iterations but is reproducible from a fixed seed (test green).
- [ ] Leakage test proves no eval target appears in any conditioning tensor.
- [ ] MAE/RMSE match a hand-computed value on a tiny fixture.

---

### `M1-4` — Kickoff: repo layout, PhysioNet accounts, frozen `EvalMasker` seed
- **Assignee:** @jkelle11-source, @JoshOlu, @SankhaS
- **Labels:** `infra`, `coordination`, `mvp`
- **Blocked by:** none
- **Blocks:** `M1-1`, `M1-2`, `M1-3`, `M1-5`

**Context.** The day-1 shared setup that unblocks the three parallel tracks. Small but load-bearing: the repo layout, the input-hook contract, and — most importantly — the **frozen `EvalMasker` seed** that pins every comparison for the rest of the project.

**Deliverables**
- All three registered on PhysioNet; Set A pull confirmed reachable.
- Repo layout created (`data/ models/ diffusion/ eval/`) and documented in the README.
- Denoiser input-hook contract agreed and written down: `mask_emb` and `time_enc`, both default `None`.
- The project-wide `EvalMasker` seed chosen, committed to config, and logged.
- CSDI §§3–4 read together; a shared one-paragraph summary of the score-based imputation objective in `Docs/`.

**Definition of Done**
- [ ] All three PhysioNet accounts confirmed; Set A downloadable.
- [ ] Repo skeleton merged; README documents layout + hook contract.
- [ ] `EvalMasker` seed committed to a config file and referenced by `M1-3`.
- [ ] Shared CSDI-objective note committed.

---

### `M1-5` — ▶ DELIVERABLE: Proposal (due Wed Oct 8)
- **Assignee:** @jkelle11-source, @JoshOlu, @SankhaS
- **Labels:** `writing`, `deliverable`
- **Blocked by:** `M1-4`
- **Blocks:** —

**Context.** The graded Proposal repackages §§1–9 of the project plan: scope, the two extensions (+ stretch conformal), baselines, the frozen protocol, and the milestone-anchored timeline. Shared writing — no single person owns the report (§3). It can be drafted in parallel with the pipeline work since the plan already contains the substance.

**Deliverables**
- Proposal document covering: contribution narrative (§1), milestone calendar (§2), roles/resource plan (§3, framed as load-balanced tracks), frozen protocol (§4), extension specs (§5), timeline (§6), baselines/targets (§7).
- Explicit statement of the honesty-aware performance framing (§8) and the code-reuse decision with citation (§0).

**Definition of Done**
- [ ] Proposal submitted by **Wed Oct 8**.
- [ ] All three co-authored / reviewed it (each section has a reviewer other than its author).
- [ ] Reuse-of-CSDI decision and citation stated in the proposal.
