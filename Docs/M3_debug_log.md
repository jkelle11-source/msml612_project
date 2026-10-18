# M3 debugging log: base CSDI reproduction

Shared log for M3-1 (Josh leads on the model, Jake co-drives on the loop and data feed), so
Week-4 hardening starts from known state. Add an entry for every run worth remembering and every
seam issue (conditioning, mask alignment, masker hand-off), with what was found and what fixed it.
Newest entries go at the bottom of each section.

## Known state

| | |
| --- | --- |
| Code | `diffusion/run.py` on top of `diffusion.train.train` (M2-1) and the vendored CSDI with the M2-2 hooks |
| Config | `configs/config.yaml`: upstream's diffusion setup (quad schedule, β 1e-4 → 0.5, 50 steps, 8 heads, 64 channels, 4 layers), Adam 1e-3, batch 16, no clipping |
| Data | Set A, 2,800 train / 400 val / 800 test patients, train-split z-scoring |
| Bar | forward fill's 10% val MAE, **0.4366** (`results/baselines.json`) |

## Environment and cost

**2026-10-17, Josh. Timing on an Apple M-series laptop (MPS), full model, batch 16.**

- Training: 282 ms per step on MPS, 3.2 s on CPU, so about 50 s per epoch (175 steps) on MPS
  and about 2.7 h for upstream's 200 epochs. CPU is about 11× slower and not practical.
- Sampling: one draw of 50 reverse steps over the 400 val patients takes about 130 s on MPS,
  whether in batches of 16 or 64 (batch size does not speed it up on MPS). The frozen evaluation
  (100 draws) would take about 3.6 h on MPS, so it runs on Colab.
- Decision: validation while training uses 5 draws every 10 epochs (`validation:` in the
  config), about 11 min per pass on MPS. It is labeled tracking-only. A noisier median raises
  MAE slightly, so a tracked MAE below forward fill is conservative.

**2026-10-17, Josh. Timing on Colab (T4 GPU, CUDA), full model, 2-epoch test run.** Training
takes about 23.5 s per epoch (29 s for the first, with warm-up), so about 78 min for 200 epochs,
about twice as fast as MPS. One tracking validation pass (5 draws, 400 val patients, batches of
64) takes 281 s. For the 200-epoch run, validating every 25 epochs (8 passes) puts the total at
about 1 h 55 min, against about 2 h 50 min at every 10.

**2026-10-17, Josh. CUDA reruns differ slightly, as expected.** Two identical 2-epoch runs on
Colab matched on epoch 1 (loss 0.3985, max gradient norm 7.56) but not exactly after: epoch 2's
max gradient norm was 1.75 vs 1.72, and val MAE 0.5536 vs 0.5529. That is the CUDA
nondeterminism noted in the README. MPS and CPU reruns are exact.

**2026-10-17, Josh. MPS is deterministic for this model.** Two runs of the same 20 real-data
steps (seed 42) gave identical losses and bit-identical weights. CPU was already tested
(`tests/test_train.py`). CUDA reruns may still differ slightly (README, Training).

## Seam checks

Covered by tests before the first real run, so a bad number is not one of these:

- **Conditioning wiring:** the denoiser's values channel is exactly `observed_data * cond_mask`
  and its mask channel exactly `cond_mask` (`tests/test_csdi_training_path.py`).
- **Masker hand-off:** the loss scores exactly `TrainMasker`'s targets, step by step, in the real
  loop (`tests/test_train.py`); upstream's own masking never runs.
- **Eval hand-off:** validation conditions on `EvalMasker`'s `cond_mask` and scores only its
  targets, and no scored value reaches the model (`tests/test_run.py`).
- **Axis alignment:** our `(B, T, D)` batch is permuted to upstream's `(B, D, T)` in
  `process_data`, and `timepoints` is the hour index 0 to 47 in both training and evaluation
  (`tests/test_diffusion_data.py`).
- **δ alignment:** not on the base path. It matters from Ext 2 (M5), which must recompute TSLO
  from `cond_mask` (README, input-hook contract).

## Runs

<!-- One entry per run: date, who, command, device, result, anything odd. -->

### 2026-10-17, Josh. Base, seed 42, 20 epochs, MPS

```text
uv run python -m diffusion.run --seed 42 --device mps --epochs 20 --out outputs/base_seed42_e20
```

Upstream's diffusion and optimizer settings; only `train.epochs` is cut, from 200 to 20, which
moves the learning-rate drops to epochs 15 and 18. The point was M3-1's two checks, not the gate.

**First 500 steps: healthy** (`health_report`). Every loss and gradient norm was finite. The loss
fell from 0.506 (mean of the first 50 steps) to 0.313 (last 50), starting at 1.01 on step 0. The
median gradient norm was 0.76, p99 4.7, and the maximum 7.36, at step 16, during the initial drop
from loss 1.0. No step exceeded 10× the median. **None of the known bad values caused a visible
spike in this window**, though they will be sampled as targets at some point in a 200-epoch run.

**Epochs 1 to 10:** mean loss 0.398, 0.335, 0.320, 0.312, 0.301, 0.273, 0.278, 0.270, 0.283,
0.268. Max gradient norm per epoch fell from 7.4 to about 1.3 to 1.8.

**Val at epoch 10: MAE 0.3339, RMSE 0.6943** (5 draws, tracking). That is **below forward fill
(0.4366)**, so M3-1's bar is met on seed 42. For reference, upstream reports 0.217 at 10% after
200 epochs, and the gate is ≤ 0.24.

**Epochs 11 to 20:** mean loss 0.268, 0.265, 0.259, 0.264, 0.251, then 0.253, 0.258, 0.253 at
lr 1e-4, then 0.247, 0.248 at lr 1e-5. Max gradient norm per epoch was 1.1 to 1.7 before the
first drop and 0.55 to 0.86 after it.

**Val at epoch 20: MAE 0.2910, RMSE 0.5282** (5 draws, tracking), about a third below forward
fill (0.4366 / 0.7409). It was still falling, from 0.334 at epoch 10.

**Reruns reproduce exactly.** Relaunched from the run's own saved `config.yaml` on MPS and stopped
after 3 epochs, the rerun's 525 history records (loss, gradient norm, learning rate) were
identical to the original's.

M3-1 Definition of Done, from this run:

- [x] Trains without divergence; grad norms bounded through the first 500 steps.
- [x] Val MAE @10% below forward fill on ≥ 1 seed (0.2910 < 0.4366, seed 42).
- [x] Launched from config + fixed seed; a rerun reproduces it.

### 2026-10-17, Josh. Base, seed 42, 200 epochs, Colab (T4, CUDA)

```text
uv run python -m diffusion.run --seed 42 --device cuda --out /content/drive/MyDrive/msml612/base_seed42
```

Upstream's full configuration, 200 epochs, with `validation.every_n_epochs` set to 25 (8 tracking
passes) to keep the session about 2 h. Outputs are in Josh's Google Drive (`msml612/base_seed42/`),
not in the repo.

**First 500 steps: healthy.** Loss 0.507 (first 50 steps) to 0.316 (last 50); median gradient
norm 0.81, max 7.56 at step 16, as on MPS. No step above 10× the median.

**Val at epoch 200: MAE 0.2483** (5 draws, tracking), the best of the 8 passes. Below forward
fill (0.4366) and close to the gate (≤ 0.24), but **this is not the gate number**: the gate uses
the frozen 100-draw evaluation, whose median is less noisy and usually scores slightly lower. That
has to be measured, not assumed.

**Validation curve** (MAE / RMSE, 5 draws):

| Epoch | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAE | 0.2959 | 0.2781 | 0.2655 | 0.2574 | 0.2622 | 0.2579 | 0.2486 | 0.2483 |
| RMSE | 0.7983 | 0.5456 | 0.5015 | 0.4921 | 0.5032 | 0.4938 | 0.4873 | 0.4862 |

It **plateaued**: about 0.257 to 0.262 from epoch 100 to 150, then a step down after each
learning-rate drop (epochs 150 and 180), with only 0.0003 gained from 175 to 200. More epochs at
this setting are unlikely to close much of the remaining gap to upstream's 0.217.

**Gradient norms over all 35,000 steps:** median 0.277. All 16 steps above 10× the median are
steps 2 to 54, the initial descent from loss 1.0 (largest 7.56 at step 16). After step 500 the
largest was 2.09, at step 820. **The known bad values (pH, Temp, Urine) caused no visible spike
in the whole run**, so there is no stability reason to clean them before the gate (M3-1's
"revisit cleaning if spikes destabilize training").

## Starting state for Week 4 (gate)

- One 200-epoch run exists (seed 42, Colab): tracked val MAE 0.2483 at epoch 200. Next is scoring
  its `model.pt` with the frozen sampler (`EvalSampler.from_config()`, 100 draws). On a T4 that
  is about 100 × 56 s ≈ 1.5 h per split, going by the 281 s measured for 5 draws.
- The gate needs ≥ 3 seeds (43 and 44 next), each about 2 h on Colab at every-25 validation.
- The known bad values did not destabilize training over 200 epochs (seed 42), so cleaning them
  is not needed for stability. Whether they affect the *scores* is a separate question (README,
  Baselines: a visible bad value can be carried into a scored one).
- The tracked curve plateaued near 0.25, so if the frozen evaluation lands above 0.24, the gap is
  more likely closed by looking at differences from upstream's setup than by training longer.
