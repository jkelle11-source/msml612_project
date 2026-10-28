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

**2026-10-20, Jake (M3-1 review). Deterministic kernels, checked on MPS.** `diffusion.train.train`
now starts with `use_deterministic_kernels()` (README, Training), so CUDA reruns should match bit
for bit too. On MPS with it on, a 3-epoch run of seed 42 (525 steps) and a rerun from its saved
`config.yaml` gave identical history, curve and weights, and the same weights as a run made before
the change: it changes nothing on MPS. Still to check on Colab: two 2-epoch runs of seed 42 should
now match bit for bit, and their seconds per epoch, against 23.5 s, show what it costs.

**2026-10-21, Jake. Deterministic kernels, checked on CUDA.** Two 2-epoch runs of seed 42 on a
Colab T4 (`--device cuda --epochs 2`, separate output folders) matched bit for bit: history, curve
(timings aside), summary and weights. CUDA reruns now reproduce exactly, so M3-1's rerun item holds
on CPU, MPS and CUDA. Epochs took 32.3 and 25.7 s in one run and 28.9 and 26.7 s in the other (the
first includes warm-up), against 29 and 23.5 s without deterministic kernels (2026-10-17, another
session): the steady epoch is about 11% slower, about 26 s, so a 200-epoch run trains in about
87 min instead of 78. That rests on one steady epoch per run, so treat it as an estimate.

**2026-10-22, Jake (M3-2 review). wandb tracking leaves a CUDA run unchanged.** Two 3-epoch runs
of seed 42 on a Colab T4 (`--epochs 3 --validate-every 3`, PR #44 with its review fixes), one with
`--wandb disabled` and one tracked online (`--tag check`), matched bit for bit: `history.jsonl` and
`model.pt`. So tracking can't break seed pairing on CUDA either; on the CPU with wandb offline it
is tested (`tests/test_run.py`). Both runs: epoch losses 0.3985, 0.3355 and 0.3272 (max gradient
norms 7.56, 1.91 and 1.20), healthy over the first 500 steps (one spike, 7.56 at step 16, in the
initial descent), val MAE 0.4850 and RMSE 0.7643. That is above forward fill, as expected after
3 epochs, the third at lr 1e-5 (below 5 epochs both drops land in the same epoch). The tracked
run is [`DATA612/msml612-csdi/85b9o6ls`](https://wandb.ai/DATA612/msml612-csdi/runs/85b9o6ls)
(`base_random_seed42`, tagged `check`), and its `summary.json` names it in `wandb_run`. Epochs
took 29–32 s, then 26–27 s; the validation pass took 300 s (281 s on 2026-10-17). Setting up a
fresh runtime: PhysioNet served Set A at about 12 KB/s that day, about 9 min for the 6.6 MB
archive. Uploading `data/processed/physionet_set_a.npz` from a machine that has it, and checking
its MD5, is quicker.

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
uv run python -m diffusion.run --seed 42 --device cuda --validate-every 25 --out /content/drive/MyDrive/msml612/base_seed42
```

Upstream's full configuration, 200 epochs, with `validation.every_n_epochs` set to 25 (8 tracking
passes) to keep the session about 2 h. `--validate-every` did not exist yet: the 25 came from the
config the run read, and the run's saved `config.yaml` records it. Validation cannot shift
training's random streams, so the cadence does not change what is trained. Outputs are in Josh's Google Drive
(`msml612/base_seed42/`), not in the repo.

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
  its final weights (`model.pt`) with the frozen sampler (`EvalSampler.from_config()`, 100 draws).
  Going by the 281 s measured for 5 draws over the 400 val patients at batch 64, that is about
  1.6 h on a T4 for the val split and 3.1 h for the 800 test patients. The frozen batch size is
  16, not yet timed on a T4, and may be slower.
- The gate needs ≥ 3 seeds (43 and 44 next), each about 2 h of training on Colab at every-25
  validation, plus its test scoring.
- Seed 42's 200-epoch run predates deterministic kernels, so a rerun will not reproduce its
  weights exactly. Rerunning it with 43 and 44 would make all three gate seeds reproducible, for
  about 2 h more on Colab.
- No early stopping or checkpoint selection (M3-1 review, PROJECT_PLAN §3.1): every run trains
  the fixed 200 epochs, and its final weights are scored.
- Runs stream to wandb ([DATA612/msml612-csdi](https://wandb.ai/DATA612/msml612-csdi), M3-2).
  Tag the gate runs `--tag gate`, and on Colab point `--out` at Drive so a dropped session keeps
  `model.pt`. Each run's `summary.json` names its wandb run in `wandb_run`, for attaching the
  test-split scores to it.
- The known bad values did not destabilize training over 200 epochs (seed 42), so cleaning them
  is not needed for stability. Whether they affect the *scores* is a separate question (README,
  Baselines: a visible bad value can be carried into a scored one).
- The tracked curve plateaued near 0.25, so if the frozen evaluation lands above 0.24, the gap is
  more likely closed by looking at differences from upstream's setup than by training longer.

## Week 4: reproduction gate (M4-1)

### 2026-10-27 to 28, Jake. Base, seeds 42–44, 200 epochs, MPS, frozen test scoring

```text
uv run python -m diffusion.run --seed 42 --device mps --validate-every 25 --tag gate     # then 43, 44
uv run python -m diffusion.score outputs/base_random_seed42 --device mps                 # then 43, 44
uv run python -m diffusion.gate outputs/base_random_seed42 outputs/base_random_seed43 outputs/base_random_seed44
```

Upstream's configuration with nothing changed: `results/gate.json` lists no departures from
`configs/config.yaml`. Validating every 25 epochs, as Josh's seed-42 run did, is tracking only.
Seed 42 was retrained rather than reusing Josh's Oct 17 run, which predates deterministic kernels,
so all three reproduce from their saved config. Device: Jake's Apple M4 Pro laptop (MPS). Outputs
are in its `outputs/`; the wandb runs are tagged `gate`:
[`bqoimtqw`](https://wandb.ai/DATA612/msml612-csdi/runs/bqoimtqw) (42),
[`xki8hsx9`](https://wandb.ai/DATA612/msml612-csdi/runs/xki8hsx9) (43) and
[`c754hh0k`](https://wandb.ai/DATA612/msml612-csdi/runs/c754hh0k) (44), each with its test MAE and
RMSE in its summary (`random/0.1/test/mae`).

**Training: healthy on all three seeds.** Every 500-step health report passed with no spike
above 10× the median; the largest gradient norms, 7.36, 7.13 and 7.85, came at steps 16, 6 and 6,
in the initial descent. Tracked val MAE (5 draws):

| Epoch | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed 42 | 0.3049 | 0.2698 | 0.2626 | 0.2537 | 0.2660 | 0.2594 | 0.2469 | 0.2464 |
| seed 43 | 0.2924 | 0.2651 | 0.2568 | 0.2695 | 0.2611 | 0.2563 | 0.2471 | 0.2436 |
| seed 44 | 0.2884 | 0.2764 | 0.2685 | 0.2632 | 0.2528 | 0.2545 | 0.2462 | 0.2471 |

The same shape as Josh's CUDA run of seed 42 (0.2483 at epoch 200): a plateau up to epoch 150,
then a step down after each learning-rate drop.

**Reruns reproduce exactly.** Seed 42 relaunched from its saved `config.yaml` on MPS and stopped
after 2 epochs gave 350 history records identical to the run's.

**Frozen test scores** (random 10% holdout, 800 patients, 26,027 targets; 100 draws, batch 16,
`eval_sampling_seed` 2021):

| Seed | Test MAE | Test RMSE | Tracked val MAE |
| --- | --- | --- | --- |
| 42 | 0.2288 | 1.1842 | 0.2464 |
| 43 | 0.2178 | 0.4933 | 0.2436 |
| 44 | 0.2185 | 0.4880 | 0.2471 |
| mean ± SE | **0.2217 ± 0.0036** | 0.7218 ± 0.2312 | |

**Gate: PASS** (≤ 0.24 over ≥ 3 seeds), inside the 0.22–0.24 band, 2% above the paper's 0.217
and just below the independent reimplementation's 0.228. The frozen scores come in 0.02–0.03 below the tracked validation. More draws
make the median less noisy, which lowers MAE, but the split differs too (test, not val), and the
two effects are not separated here.

**A visible bad pH is carried along pH, not across features.** The PaCO2/PaO2 check (M4-1):
CSDI's feature attention does not do what BRITS's feature regression does. The one PaCO2 or PaO2
target at an hour with a visible bad pH, record 140525's PaO2 of 54 next to its pH of 735 at
hour 22, is guessed at 106, 111 and 107 mmHg by seeds 42, 43 and 44 (misses of 0.60, 0.66 and
0.62 standardized units), where BRITS guessed 5,204. PaCO2's largest misses (2.34, 2.30, 1.93) and
PaO2's (3.14, 3.05, 2.98) are in patients who show no bad value. Temporal attention does carry a
bad pH along pH itself, the way forward fill does, by a seed-dependent amount:

| Held-out pH | Visible bad pH | Seed 42 | Seed 43 | Seed 44 |
| --- | --- | --- | --- | --- |
| record 140525, hour 25: 7.33 | 734 and 735 at hours 19 and 22 | 220.5 (miss 152.8) | 8.78 (1.0) | 7.46 (0.1) |
| record 141680, hour 38: 7.46 | 187 at hour 8 | 125.9 (84.8) | 12.87 (3.9) | 9.35 (1.4) |

Misses in standardized units. In seed 42 those two values are 84% of the squared error: without
the 12 test patients who show a bad value, its RMSE would be 0.4795 instead of 1.1842 and its
MAE 0.2198 instead of 0.2288. Seeds 43's and 44's barely move without them (MAE 0.2178 and
0.2186, RMSE 0.4936 and 0.4887), so the mean MAE without those patients would be 0.2187.

**Cost.** Seed 42 trained alone on the GPU: 30.2 s per epoch, 101 min for 200 epochs, plus about
337 s per tracked validation pass (45 min for 8). Seed 44 was scored alone: 243 s per batch of 16,
3.37 h for the test split's 50 batches, against about 3.1 h estimated for a T4 at batch 64 (M3-1);
batch 16 on a T4 is still untimed. Everything else ran two at a time on the one GPU, with a
training epoch at about 52 s and a scoring batch at about 418 s: about 15% more work done per hour
than one job at a time. The three seeds took 16.2 h of wall-clock time, 19:27 on Oct 27 to 11:39
on Oct 28.
