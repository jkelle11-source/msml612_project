# Missingness-Aware Diffusion Imputation for Clinical Time Series

An extension of **CSDI** (Conditional Score-based Diffusion for Imputation) for
multivariate clinical time-series imputation on the **PhysioNet/CinC Challenge 2012**
dataset, with a learned missingness embedding (Ext 1) and a TSLO-based continuous-time
encoding (Ext 2), plus a stretch conformal-interval wrapper (Ext 3).

**Course:** MSML612 — Deep Learning, University of Maryland.
See [`Docs/PROJECT_PLAN.md`](Docs/PROJECT_PLAN.md) for the authoritative plan and
[`Milestones/`](Milestones/) for the milestone/issue breakdown.

> **Reuse decision (PROJECT_PLAN §0):** we **reuse, don't reimplement**, the base CSDI
> denoiser — our engineering effort goes into the extensions, the protocol, and the
> evaluation. The reused source is cited below.

---

## Repository layout

The layout is a locked-in decision (PROJECT_PLAN §0; agreed at the M1-4 kickoff). Each
top-level source directory is a Python package:

```text
data/          # PhysioNet Set A pipeline: loader -> (N,48,D) + mask, TSLO delta,
               #   z-scoring (train-split stats only).                      [M1-1]
  raw/         #   raw Set A download target        (git-ignored contents)
  processed/   #   processed tensors/scaler/splits  (git-ignored contents)
models/        # reused CSDI model (denoiser + diffusion process), build_model
               #   (config -> model), and our extensions (MissingnessEncoder,
               #   ContinuousTimeEncoding).                                 [M1-2]
               #   BRITS, the learned baseline (brits.py).                  [M3-3]
               #   Ext 1's MissingnessEncoder (missingness.py).             [M4-2]
               #   Ext 2's δ from the conditioning mask (time_encoding.py). [M5-1]
               #   Ext 2's ContinuousTimeEncoding (time_encoding.py).       [M5-2]
diffusion/     # training data feed, loop and random streams around the
               #   reused diffusion process; no schedule or q_sample of our
               #   own.                                                     [M2-1]
               #   Run entry point (run.py).                                [M3-1]
               #   Frozen scoring of a run (score.py), the gate (gate.py).  [M4-1]
               #   Ext 1's overfit sanity check (overfit.py).               [M4-2]
               #   Ext 2's mechanical health check (ext2_health.py).        [M5-2]
               #   Single-seed health record from wandb (health.py).        [M5-3]
               #   Pairing hashes, paired-GPU guard (run.py, score.py).     [M6-2]
               #   The sweep's checked results store (results.py).          [M6-2]
               #   The GPU timing test (timing.py).                         [M6-2]
               #   The sweep's driver (sweep.py); resuming a run (run.py).  [M6-2]
               #   The sweep's run manifest (manifest.py).                  [M7-1]
               #   Trajectories and their figure (trajectories*.py).        [M7-3]
eval/          # EvalMasker / TrainMasker + masked, standardized metrics;
               #   EvalSampler (frozen-noise evaluation pass).              [M1-3]
               #   Mean and forward-fill baselines.                         [M2-3]
results/       # recorded results (baselines.json).                         [M2-3]
               #   BRITS's record (learned_baseline.json).                  [M3-3]
               #   The reproduction gate's record (gate.json).              [M4-1]
               #   Ext 1's overfit check (ext1_overfit.json).               [M4-2]
               #   Ext 2's health check (ext2_health.json).                 [M5-2]
               #   Single-seed health runs (health_seed42.json).            [M5-3]
               #   The sweep's runs (sweep.json, sweep_manifest.json).      [M7-1]
               #   Trajectory figure and its read (trajectories/).          [M7-3]
configs/       # frozen protocol config (config.yaml).
Docs/          # project plan + shared notes (e.g. CSDI_Objective.md).
Milestones/    # per-milestone GitHub issue breakdown (M00-M10).
```

Processed tensors, the fitted scaler, model checkpoints, and `wandb/` outputs are
git-ignored; only code and small configs are committed.

---

## Frozen protocol constants

All locked protocol values live in [`configs/config.yaml`](configs/config.yaml). The
load-bearing one is the **`EvalMasker` seed** (`protocol.eval_masker_seed`), which pins
the fixed 10% evaluation-target holdout identically across every model and every seed for
the entire project (PROJECT_PLAN §3.1, §4.1). **It must never change** — doing so silently
invalidates every comparison. `eval.EvalMasker` (M1-3) reads it from this config.

The same seed also pins two more holdouts at CSDI's robustness rates (PROJECT_PLAN §6):
`protocol.robustness_fractions` (`[0.50, 0.90]`) holds out 50% and 90% of observed values
instead of 10%. The headline rate stays `protocol.eval_holdout_fraction` (10%). The baselines
are scored at all three (see [Baselines](#baselines)).

The block-missingness holdout, `eval.BlockEvalMasker` (M2-4; PROJECT_PLAN §3.1, §4.1), has two
frozen keys of its own. **Neither may change**, for the same reason as `eval_masker_seed`:

- `protocol.block_eval_masker_seed` (8080) seeds it the way `eval_masker_seed` seeds
  `EvalMasker`, with one generator per patient from `(seed, record_id)`.
- `protocol.block_window_hours` (3) is the length of each hidden window. The 48 hours are tiled
  into 16 fixed 3-hour slots.

It has no fraction of its own: it reads `protocol.eval_holdout_fraction`, so every patient loses
exactly as many values as under `EvalMasker`. The difference is where they are hidden. The slots
are visited in a random order and each is hidden whole, across all 35 features, until the next
one would overshoot the count; that one is trimmed to a random subset of its observed values.
On Set A, 94% of patients lose at least one complete window. The slot walk is
`eval.masking.block_targets(mask, count, generator, window)`, which M6-2's block-aware training
(`TrainMasker`'s `mix`) reuses with its own generator. `EvalMasker`'s holdout is unchanged by
it. Tests: `tests/test_block_eval_masker.py`, including a golden fingerprint of the holdout.

The off-grid block holdout, `eval.OffGridBlockEvalMasker` (M6-3; PROJECT_PLAN §4.1's grid
check), has two more frozen keys: `protocol.offgrid_block_eval_masker_seed` (9090) and
`protocol.offgrid_block_offset_hours` (1). It is the block holdout with its windows shifted off
the 16-slot grid (see [Ext 1's attribution checks](#ext-1s-attribution-checks-m6-3)).

The evaluation sampler is frozen the same way: `eval.EvalSampler` reads
`protocol.eval_sampling_seed`, `protocol.eval_n_samples` and `protocol.eval_batch_size`
(see [Random streams](#random-streams-seed-pairing)).

`protocol.paired_gpu` (M6-2) names the one GPU model every paired run trains and is scored on
(PROJECT_PLAN §3.3): `NVIDIA A100-SXM4-80GB`, Colab's G4 runtime, picked by M6-2's timing test
([The timing test](#the-timing-test-m6-2)). A paired run refuses to start on any other GPU (see
[Paired sweep checks](#paired-sweep-checks-m6-2)).

`protocol.tf32` (M6-2) says whether CUDA computes float32 convolutions and matrix multiplies in
TF32. It is `true`, as the timing test chose: on the paired GPU it takes nearly a quarter off the
Week-7 sweep (88.0 h to 67.5 h). Training and scoring both apply it, so it is the same for every
run. On a T4, MPS or the CPU, which have no TF32, it changes nothing.

The model and training settings live in the same file, under upstream's key names (M2-1):
`diffusion` (the reused noise schedule and denoiser size), the side-information keys in
`model`, and `train` (Adam, weight decay, learning rate and its schedule, epochs, batch size,
gradient clipping). Their defaults are upstream's published PhysioNet values, the
configuration the gate is measured against (PROJECT_PLAN §3.4). Unlike the protocol
constants they may be tuned, but every departure from upstream is documented (M4-1).
`train.clip_grad_norm` is `null`, meaning no clipping, as upstream. Two of upstream's keys are
deliberately absent: the head count comes from `model.n_heads` alone, and `target_strategy` is
always `"external"`, set by `models.build_model`, because upstream's `"random"` draws its masks
from the global RNGs.

Departures from upstream so far:

- **No learning-rate drop before epoch 1** (M3-2 review). A drop comes at
  `max(1, int(fraction × train.epochs))` for each `train.lr_milestones` fraction; upstream's
  `int(0.75 × epochs)` and `int(0.9 × epochs)` are both 0 for a 1-epoch run, so `MultiStepLR`
  applies both drops before the first step and the run trains at 1% of the learning rate. Only
  1-epoch runs change: from 2 epochs on the milestones are upstream's, and a 200-epoch run, the
  gate and the sweep included, still drops at epochs 150 and 180.

---

## Denoiser input-hook contract

Frozen at the M1-4 kickoff so the data, architecture, and evaluation tracks can proceed in
parallel without waiting on each other. The reused CSDI denoiser's `forward` **must** accept
two optional hooks, both defaulting to `None`:

| Hook | Produced by | Shape | How it enters the denoiser |
| --- | --- | --- | --- |
| `mask_emb` | `MissingnessEncoder` (Ext 1, `--use_mask_embedding`) | `(B, T, D, d_model)` | **added to the denoiser input** |
| `time_enc` | `ContinuousTimeEncoding` (Ext 2, `--use_time_encoding`) | `(B, T, D, d_model)` | **added to the denoiser input** |

```python
# models/csdi/diff_models.py (M2-2)
def forward(self, x, cond_info, diffusion_step, *, mask_emb=None, time_enc=None):
    # When a hook is None the denoiser behaves exactly as the reproduced base CSDI,
    # so base / Ext1 / Ext2 / Ext1+2 are the four toggle combinations of one model.
    ...
```

Rules:

- Both hooks default to `None`; with both `None` the model **is** the reproduced base CSDI.
- Each hook is **added to the denoiser input** like a positional encoding — it does **not**
  modify the attention logits (PROJECT_PLAN §4.2).
- Each hook gives every (hour, feature) its own `d_model` vector, so the value at that hour
  and feature gets its own embedding. Both extensions encode per-value quantities (which
  values are hidden, how stale each feature is), and a vector shared by every feature at an
  hour could not say which feature it describes.

**Who computes the hooks** (settled in M2-1). The extensions belong to the model, not to the
training loop:

- `models.build_model(config, device)` is the one place the model is built. It reads the
  extension toggles (`model.use_mask_embedding`, `model.use_time_encoding`) once and builds
  each enabled extension's module into the model, after the parts every variant shares and
  each from the init-RNG state right after them (see
  [Random streams](#random-streams-seed-pairing)). Both extensions exist since M5-2, so every
  toggle combination builds; until then, an extension's toggle raised `NotImplementedError`
  instead of silently training the base model.
- The model computes `mask_emb` and `time_enc` itself, on both its training-loss path and its
  sampler path, and hands them to the denoiser.
- Every variant is therefore trained and scored through the same calls, and neither passes a
  hook: `loss = model(batch, generator=streams.diffusion)` in the training loop, and
  `eval.EvalSampler` (which calls `model.evaluate`) at evaluation. A batch carries upstream's
  four keys and nothing else: `observed_data`, `observed_mask`, `gt_mask` (the conditioning
  mask) and `timepoints`.
- **Both hooks are computed from the conditioning mask** (`gt_mask`, which `CSDI_base` calls
  `cond_mask`), never from `observed_mask` or the dataset's `delta_obs`. Those two still
  include the held-out targets, so an extension that read them would know where the targets
  are, which is the leak PROJECT_PLAN §3.1 rules out. For Extension 2 this means TSLO δ is
  recomputed from the conditioning mask for every batch, not read from `delta_obs` (M5-1):
  `CSDI_base.get_time_enc` runs `models.time_encoding.tslo`, which is
  `data.physionet.compute_tslo`'s rule in torch, on `cond_mask`, on the batch's device. A
  held-out target therefore counts as unobserved: δ keeps counting from the last value the
  model can see.

**Where the hooks travel** (M2-2). The frozen calls above stay as they are; the hooks run
inside the model, from the two places `CSDI_base` calls the denoiser:

```text
training    model(batch, generator=...)              # frozen, M2-1
              -> calc_loss(..., mask_emb=None, time_enc=None)        # one denoiser call
              -> diffmodel(x, side_info, t, mask_emb=..., time_enc=...)
validation  model(batch, is_train=0, generator=...)
              -> calc_loss_valid(..., mask_emb, time_enc)            # calc_loss at every step
evaluation  model.evaluate(batch, n_samples, generator=...)   # frozen, via EvalSampler
              -> impute(..., mask_emb=None, time_enc=None)           # every reverse step
              -> diffmodel(x, side_info, t, mask_emb=..., time_enc=...)
```

An extension adds its module in `build_model`, computes its hook from `cond_mask` inside
`forward` and `evaluate`, and hands it to `calc_loss` or `impute`. Extension 1 does this
since M4-2 (see [Extension 1](#extension-1-missingnessencoder-m4-2)), and Extension 2 since
M5-2 (see [Extension 2](#extension-2-continuoustimeencoding-m5-2)): M5-1 put its path in
place, `forward` and `evaluate` computing δ from `cond_mask` and handing it to
`self.time_encoder`, and M5-2 builds that encoder. Each hook is computed once per call and the
same tensor is used at every reverse step, since it depends on the conditioning mask and not on
the diffusion step.

Shapes, since the axes swap on the way in:

- Our data is `(B, T, D)`. `CSDI_base.process_data` turns it into `(B, D, T)`, and the
  denoiser's stream is `(B, channels, D, T)`.
- A hook is `(B, T, D, channels)`, in our axis order: one vector per hour and feature.
  `channels` is upstream's name for `d_model` (`diffusion.channels` in the config, 64).
- The denoiser reorders it to `(B, channels, D, T)` and adds it right after its first
  projection, so each vector lands on its own hour and feature. A hook of any other shape
  raises `ValueError`, which catches one built `(B, D, T, channels)` by mistake, or a single
  vector per hour.

---

## Reused CSDI implementation

The base CSDI model in [`models/csdi/`](models/csdi/) is **not our work**. It is the authors'
official implementation, vendored into this repository (M1-2):

| | |
| --- | --- |
| Repository | <https://github.com/ermongroup/CSDI> |
| Pinned commit | [`7f24a436f08d98853a6b43d4f7f04e5a65ecdf27`](https://github.com/ermongroup/CSDI/commit/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27) (2024-03-15, upstream `main`) |
| Paper | Tashiro, Song, Song, Ermon (2021). *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation.* NeurIPS. [arXiv:2107.03502](https://arxiv.org/abs/2107.03502) |
| License | MIT, Copyright (c) 2021 Yusuke Tashiro. Kept at [`models/csdi/LICENSE`](models/csdi/LICENSE) |

**Reused unchanged** (copied from the pinned commit):

- `models/csdi/diff_models.py`: the denoiser `diff_CSDI`, its residual blocks with temporal
  and feature attention, and the diffusion-step embedding.
- `models/csdi/main_model.py`: `CSDI_base` and `CSDI_Physio`, covering the noise schedule,
  side information (time embedding, feature embedding, conditioning mask), the masked
  noise-prediction loss, and the reverse sampler.

**Adapted so far.** Every changed line is marked `ADAPTED` in the source, with the issue
that changed it.

M1-2, two import lines; no model logic is changed:

- `main_model.py` imports `diff_models` relatively, because upstream is a flat script
  directory and ours is a package.
- `diff_models.py` imports `linear_attention_transformer` only when `is_linear=True`. The
  PhysioNet configuration never sets it, so the package is not one of our dependencies.

M2-1, two opt-in additions to `main_model.py`. With upstream's own settings
(`target_strategy: random`, no generator) the model behaves exactly as before:

- `CSDI_base.forward` accepts a new `target_strategy: "external"`. Training then conditions
  on the mask the caller puts in `batch["gt_mask"]` (from `eval.TrainMasker`), the way
  evaluation already does, and upstream's `get_randmask` is never called.
- `forward`, `calc_loss`, `calc_loss_valid`, `evaluate` and `impute` take an optional
  `generator`, a CPU `torch.Generator`. The diffusion timestep and every noise draw, in
  training and in the reverse sampler, come from it instead of the global RNG, through one
  added module-level helper, `randn_like` (see [Random streams](#random-streams-seed-pairing)).

**Not taken from upstream:** its data loading, training and evaluation scripts, and configs
(`dataset_*.py`, `exe_*.py`, `utils.py`, `config/`). Ours are in `data/`, `eval/` and
`configs/`.

M2-2, the input hooks from the contract above. With both `None` (the only value the frozen
calls pass) the model computes exactly what it did before:

- `diff_CSDI.forward` takes keyword-only `mask_emb` and `time_enc` and adds each to its
  input stream after the first projection, each vector at its own hour and feature.
- `calc_loss`, `calc_loss_valid` and `impute` take the same two arguments and pass them to
  every denoiser call they make. `forward` and `evaluate` are unchanged.

M4-2, Extension 1's slot in `main_model.py`. With `mask_encoder` left `None` (the default,
and what `build_model` leaves when `model.use_mask_embedding` is off) the model computes
exactly what it did before:

- `CSDI_base.__init__` sets `self.mask_encoder = None`, after everything upstream builds.
- `CSDI_base.get_mask_emb(cond_mask)` returns `None` without an encoder, and otherwise the
  encoder's `mask_emb` from the conditioning mask, reordered to our `(B, T, D)`.
- `forward` and `evaluate` call it once and pass `mask_emb` to `calc_loss`/`calc_loss_valid`
  and `impute`.

M5-1, Extension 2's slot, built the same way. With `time_encoder` left `None` (the default,
and what `build_model` leaves when `model.use_time_encoding` is off) the model computes exactly
what it did before, and computes no δ:

- `main_model.py` imports `tslo` from `models.time_encoding`.
- `CSDI_base.__init__` sets `self.time_encoder = None`, after `mask_encoder`.
- `CSDI_base.get_time_enc(cond_mask)` returns `None` without an encoder, and otherwise the
  encoder's `time_enc` from δ, which `tslo` computes from the conditioning mask reordered to
  our `(B, T, D)`.
- `forward` and `evaluate` call it once and pass `time_enc` to `calc_loss`/`calc_loss_valid`
  and `impute`.

**Replaced** (M2-1): upstream's internal train masking. `get_randmask` draws from the global
NumPy/torch RNGs, which would break seed pairing, so `models.build_model` sets
`target_strategy: "external"` and the training loop (see [Training](#training)) supplies
`eval.TrainMasker`'s conditioning mask. The schedule, loss and sampler stay upstream's; we write
none of our own (PROJECT_PLAN §0, decision 1).

Any later change to the vendored files gets the same `ADAPTED` marker and a line in this
section.

`tests/test_csdi_smoke.py` trains the vendored model for a few steps on a synthetic batch
of our shape, `(B, 48, 35)` at about 80% missing, and checks that the loss, gradients and
sampled imputations are finite. Run it with `uv run pytest tests/test_csdi_smoke.py`.
`tests/test_csdi_training_path.py` covers the two M2-1 additions on the training path: the
loss is scored on exactly `TrainMasker`'s targets, the denoiser sees only conditioning
values, and the timestep and noise come from the generator passed in.
`tests/test_eval_sampler.py` covers the sampler side. `tests/test_denoiser_hooks.py` covers
M2-2: with no hooks the loss and samples equal the frozen calls' exactly; a nonzero hook
changes the training loss, the validation loss and the samples; every denoiser call on those
paths receives the hooks; each vector of a hook lands at its own hour and feature; a wrongly
shaped hook is refused; and a real Set A batch trains through both hooks with gradients reaching them. `tests/test_forward_noising.py` checks the
reused schedule numerically (M2-1): the model's ᾱ_t equals the quad schedule worked out by hand
from `configs/config.yaml`, and the values it noises at every step have the closed-form mean
√ᾱ_t · x₀ and variance 1 − ᾱ_t. Mind upstream's names when reading `CSDI_base`: `alpha` is the
cumulative ᾱ_t and `alpha_hat` the per-step 1 − β_t, the reverse of the DDPM paper's.

`tests/test_time_encoding.py` covers M5-1. `tslo` matches `compute_tslo`, keeps the mask's
device and dtype, and leaves the mask alone. With a stand-in encoder in the slot, on the
training, validation and sampler paths: the encoder is given `compute_tslo` of the
conditioning mask, once per call; changing `observed_mask` leaves δ unchanged; a held-out
target counts as unobserved where `delta_obs` reads 0; every denoiser call receives the
resulting `time_enc`; `EvalSampler` hands the model δ of `EvalMasker`'s and
`BlockEvalMasker`'s conditioning masks; sampling with δ present is finite; and the base model
never computes δ.

---

## Random streams (seed pairing)

Variants are compared as paired differences on a shared run seed (PROJECT_PLAN §3.3). That
only works if every variant draws the same random numbers for the same purpose, and the
global RNG cannot promise it: one extra draw, such as an extension's layer init, shifts
every draw after it. `diffusion.streams.RunStreams(run_seed)` therefore gives each source of
training randomness its own stream, seeded from `(run_seed, stream, epoch)`:

| Stream | Used for | In the training loop |
| --- | --- | --- |
| `streams.data_order` | shuffling | `DataLoader(..., shuffle=True, generator=streams.data_order)` |
| `streams.train_masker` | conditioning/target split of each batch, in the regime's strategy ([Training regimes](#training-regimes-m6-2)) | `cond, target = streams.train_masker(mask)`; `cond` goes in `batch["gt_mask"]` |
| `streams.diffusion` | diffusion timestep and noise | `model(batch, generator=streams.diffusion)` |
| global RNG | model init, dropout | `streams.seed_model_init()`, `streams.start_epoch(epoch)` |

Rules:

- Call `streams.seed_model_init()` right before building the model, and build the parts every
  variant shares before any extension module, so shared weights start identical.
- **Each extension starts from the same draws in every variant that has it** (M5-2).
  `models.build_model` saves the init RNG's state right after the shared parts
  (`torch.get_rng_state()`) and restores it before building each extension, so the time encoder
  starts from the same weights in Ext 2 and Ext 1+2, and the mask encoder from the same weights
  in Ext 1 and Ext 1+2 as it did in M4-2. Without the restore, whichever encoder is built second
  would be dealt later draws: at seed 42, Ext 1+2's time encoder would start with
  `w0` = −0.9452 instead of Ext 2's −0.5433.
- Call `streams.start_epoch(epoch)` at the top of every epoch, after the model is built. It
  re-seeds all four streams in place, so a DataLoader built once keeps working, and the train
  masker keeps the strategy it was built with (a masker rebuilt as `TrainMasker(seed)` would be
  `random`, the default, from then on).
- **Dropout is paired only between variants with the same dropout layers.** It draws from the
  global RNG, and a variant with extra dropout layers consumes more of it. Ext 1's encoder has
  dropout layers, so dropout is not paired between base and Ext 1 (or Ext 2 and Ext 1+2). Ext 2's
  encoder has none, and `start_epoch` re-seeds the global RNG, so base and Ext 2 (and Ext 1 and
  Ext 1+2) draw identical dropout masks (M5-2, tested). Ext 1-blind (M6-3) has exactly Ext 1's
  layers, so Ext 1 and Ext 1-blind draw identical dropout masks too, and are paired in every
  stream (tested). Shuffling, masks, timesteps and noise stay identical regardless.
- **Resuming needs no RNG state.** Because every stream is a function of the run seed and the
  epoch, a checkpoint taken at an epoch boundary needs only the model, the optimizer (plus
  the learning-rate scheduler) and the epoch number; the resumed run then ends
  with the same weights as an uninterrupted one. A run is resumed from its last checkpoint,
  taken after its last validated epoch, and redoes the epochs since. `run(..., resume=True)` does
  this (M6-2,
  [Launching a run](#launching-a-run-m3-1)).
- **Evaluation noise is frozen.** `eval.EvalSampler` runs one evaluation pass of the reused
  sampler: `samples = EvalSampler.from_config()(model, values, mask, cond_mask)`, shape
  `(K, N, 48, D)`. It re-seeds a private generator with `protocol.eval_sampling_seed` at the
  start of every pass, so a checkpoint's samples depend on its weights alone, every variant,
  run seed and epoch is scored with the same noise, and a validation pass during training
  draws nothing from the global RNG. Patients go through in the order given,
  `protocol.eval_batch_size` at a time; the noise a patient receives depends on its position
  in the pass, which is why the batch size is frozen with the seed. The model is put in eval
  mode for the pass and handed back as it was.

---

## Training

`diffusion.train.train(config, dataset, run_seed, device, on_epoch_end=None)` trains one model
and returns `(model, history)` (M2-1):

```python
from pathlib import Path

import yaml
from data.physionet import load_dataset
from diffusion.train import train

config = yaml.safe_load(Path("configs/config.yaml").read_text())
dataset = load_dataset("data/processed/physionet_set_a.npz")
model, history = train(config, dataset, run_seed=42, device="cuda")
```

What one run does:

- Builds the model with `models.build_model(config, device)`, right after
  `streams.seed_model_init()`. `build_model` is the only place a model is built. It converts
  our config to upstream's layout (`models.build.config_to_dict`), and while an extension does
  not exist yet, turning on its toggle raises (see the input-hook contract).
- Feeds the training split only, shuffled on `streams.data_order`
  (`diffusion.data.train_loader`), as upstream's four batch keys (`diffusion.data.to_batch`).
- Each step: `TrainMasker` split → `model(batch, generator=streams.diffusion)`, which is
  upstream's forward noising and masked loss → backward → `clip_grad_norm_` → Adam step.
  Clipping is off by default (`train.clip_grad_norm: null`, as upstream), but the gradient norm
  is still computed and recorded.
- Steps the learning-rate schedule once per epoch (×0.1 at 75% and 90% of `train.epochs`,
  rounded down but never before epoch 1: a [departure from upstream](#frozen-protocol-constants)
  that changes 1-epoch runs only; below 5 epochs both drops land in the same epoch), then calls
  `on_epoch_end(epoch, model)` if one is given. M3 attaches validation there. Nothing the
  hook does can shift training, because `start_epoch` re-seeds every stream at the top of the
  next epoch and `train` puts the model back in train mode.
- `history` has one record per step, `{"epoch", "loss", "grad_norm", "lr"}`, as plain numbers.

On the CPU, the same config, dataset and run seed give bit-identical weights and history (tested).
On an Apple M-series GPU (MPS) they did too: two runs of 525 real-data steps (3 epochs) matched
bit for bit, weights included (M3-1). Some CUDA kernels are nondeterministic by default (they add
floating-point numbers in whatever order their threads finish), so `train` first calls
`diffusion.train.use_deterministic_kernels()`: `torch.use_deterministic_algorithms(True)`, with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` unless that is already set. cuBLAS reads that variable when
CUDA first runs a matrix multiply, so in a notebook that has already used CUDA, restart the
runtime first. The switch is process-wide, so a run's validation passes are covered; a separate
scoring script must call it before any CUDA work. With it, CUDA reruns match bit for bit too: two
2-epoch runs on a Colab T4 did ([`Docs/M3_debug_log.md`](Docs/M3_debug_log.md)). `on_step_end(record)`, if
given, is called with a copy of each history record as it is made (M3-1); the run log uses it,
and wandb (M3-2) does. `on_checkpoint(state)` is called at the end of every epoch with the epoch
and the model's, optimizer's and schedule's states (`diffusion.run` writes one to disk after each
validated epoch but the last), and `resume=state` goes on from one (M6-2,
[Launching a run](#launching-a-run-m3-1)).

### Training regimes (M6-2)

`train.regime` in the config picks how a run trains and what its validation tracks (PROJECT_PLAN
§3.1). `diffusion.train.REGIMES` maps each regime to its settings, and `diffusion.run --regime`
sets it:

| `train.regime` | `TrainMasker` strategy | Validation holdout | Trains | Run name |
| --- | --- | --- | --- | --- |
| `random` (the config's) | `random` | `EvalMasker`'s | all four variants: the headline sweep and the gate | `base_random_seed42` |
| `block-aware` | `mix` | `BlockEvalMasker`'s | base and Ext 1, and Ext 1-blind if M8-4 is triggered | `ext1_block-aware_seed42` |

Test scoring is the same in both: every run is scored under both holdouts
([Scoring a run](#scoring-a-run-m4-1)). Each regime is paired only within itself, and the store
keeps them apart ([Paired sweep checks](#paired-sweep-checks-m6-2)). At a seed, a block-aware run
draws the same data order, diffusion draws and starting weights as a random-regime run of the
same variant; only its train masks differ (tested). A config without `train.regime`, such as one
saved before M6-2, is `random`, which is how those runs trained. An unknown regime is refused
before anything runs, from Python and from the command line.

`eval.TrainMasker(seed, strategy="random", window=None)` draws a fresh split every call from its
own generator:

- `random` (upstream's): per patient, a ratio `r ~ U(0, 1)`, and `round(r · n_observed)` of their
  observed values hidden, picked uniformly.
- `mix`: the same split, then a fair coin per patient keeps it (heads) or hides the same
  `round(r · n_observed)` values in whole `window`-hour slots across all features, apart from one
  trimmed slot (tails). That is `block_targets`' slot walk on `BlockEvalMasker`'s 16-slot grid, with
  a NumPy generator seeded from the masker's own. `RunStreams` passes `protocol.block_window_hours`
  (3) as the window. Over 1,600 test patients, 50.4% came up tails.

Each call reads, in this order, `B` ratios and `B × T·D` scores (all that `random` reads), then,
for `mix` only, `B` coins and one seed per tails patient. The order is part of the protocol:
`random`'s output is byte-identical to before `mix` existed, and `mix`'s stream is pinned, each by
a SHA-256 fingerprint of five calls. Because `mix` trains on the holdout's own slot grid, the
block-aware regime is matched to our synthetic block pattern (PROJECT_PLAN §8, Limitations; §4.1's
grid check).

Tests:

- `tests/test_train_masker.py`: both fingerprints (`mix`'s worked out by replaying the documented
  reads, not by running `TrainMasker`), a hand-checked example, heads keeping `random`'s split and
  tails its count, tails in whole slots bar one, a fair coin read from the masks, pairing under
  other random draws, valid splits (a patient with nothing observed too), GPU matching CPU, bad
  arguments, and dry runs in both strategies on the built dataset and Set A.
- `tests/test_streams.py`: `RunStreams` builds its masker with the strategy given, and a `mix`
  masker is still `mix` after `start_epoch`, at epochs 0 to 199.
- `tests/test_run.py`: names, the default for an old config, the validation holdout built from
  the run's own config, a block-aware run training on `mix` masks every epoch and validating on
  the block holdout, the two regimes' hashes, an unknown regime refused before `--overwrite`
  deletes anything, and `--regime` recorded so a rerun of the saved config keeps it.
- `tests/test_health.py`: a block-aware health record names the block holdout.

### Launching a run (M3-1)

```text
uv run python -m diffusion.run --seed 42
```

`diffusion/run.py` trains one model from `configs/config.yaml` and a run seed, scores the
validation split as it goes, and writes to `outputs/<run name>/` (git-ignored), e.g.
`outputs/base_random_seed42/`, the run's name on wandb
([Experiment tracking](#experiment-tracking-m3-2)). Options: `--device` (`cuda`, `mps`, `cpu`,
default the first available), `--epochs` to override `train.epochs`, `--validate-every` to
override `validation.every_n_epochs`, `--variant` to set the extension toggles
([The four variants](#the-four-variants-m6-1)), `--regime` to set `train.regime`
([Training regimes](#training-regimes-m6-2)), `--config`, `--dataset`, `--out`, `--overwrite`,
`--resume` (below),
`--wandb`, `--tag`, `--paired` (a sweep run: refused on any GPU but `protocol.paired_gpu`,
[Paired sweep checks](#paired-sweep-checks-m6-2)). The output folder must be empty or new, so two runs' files never mix; `--overwrite`
first deletes the files listed below from it, and nothing else.
Everything is written while the run is going, so a stopped run keeps what it has done:

| File | Contents |
| --- | --- |
| `config.yaml` | the resolved config, `--epochs`, `--validate-every`, `--regime` and the variant flags included, `tracking` left out (where a run is logged isn't part of it). `--config` on it reruns the run. |
| `history.jsonl` | one line per training step: `epoch`, `loss`, `grad_norm`, `lr` |
| `curve.json` | one row per epoch: mean loss, largest gradient norm, training seconds, the epoch's `data_order_sha256`, `train_mask_sha256` and `diffusion_sha256` (M6-2), and `val_mae` / `val_rmse` / `val_seconds` when scored |
| `model.pt` | weights after the latest scored epoch. The last epoch is always scored, so a finished run's are its final weights, the ones that are scored (no best-epoch checkpoint; PROJECT_PLAN §3.1). |
| `summary.json` | the 500-step health report, the final val MAE, forward fill's val MAE, `below_forward_fill`, the `validation` settings (split, holdout, draws, batch size), the `gpu` it trained on and its `pairing` hashes (M6-2, [Paired sweep checks](#paired-sweep-checks-m6-2)), `model_sha256`, the SHA-256 of the final `model.pt` (M6-2, checked by [`diffusion.score`](#scoring-a-run-m4-1)), and `wandb_run`, the run's path on wandb (`entity/project/id`; null when untracked) |
| `checkpoint.pt` | the model, optimizer and learning-rate schedule, the epoch, the number of steps so far, the run's wandb path, and the device and GPU it was written on (M6-2), rewritten whole after every validated epoch but the last |

**Resuming a run (M6-2).** `checkpoint.pt` is written beside itself and then renamed, so a
session lost mid-save leaves the last one whole. It is written after every validated epoch (every
`validation.every_n_epochs`, 10, when `model.pt` is saved too), a tenth of the Google Drive uploads
of one every epoch, so a lost session redoes at most 10 epochs, about 1–2 minutes on the G4. It is
not written after the last epoch: the summary follows that one, so a run stopped between the two
redoes its last 10 epochs, and the final `model.pt` whose hash the summary records is always written
in the same session as the summary.

`--resume` (`run(..., resume=True)`) continues the unfinished run in the output folder from its
checkpoint. The model is still built from the run seed and hashed as built (`on_start`), then given
the checkpoint; the steps and epochs after the checkpoint are dropped and redone; and the run goes
on as the same wandb run. Every stream is a function of the run seed and the epoch
([Random streams](#random-streams-seed-pairing)), so a resumed run ends with the same files,
numbers and weights as one never stopped. The tests check that for a session lost right after a
checkpoint, between an epoch's files and its checkpoint, and in the middle of an epoch.

A folder with no usable checkpoint is started again, which is always correct: one with no checkpoint
yet; one whose checkpoint was written on another device or GPU, since float rounding differs from
one to another, so a run trained on two is neither one never stopped nor paired, though its summary
would name only the GPU it finished on; and one whose `history.jsonl` and `curve.json` do not reach
its checkpoint's epoch. Google Drive uploads each file in the background, so a session lost on it
can leave a newer `checkpoint.pt` than those two, and a run resumed from it would lack the steps
and epochs in between. A last `history.jsonl` line cut short (a session lost while writing it) is
ignored. The tests check the checkpoints' epochs and step counts, and that a `curve.json` a row
short, missing or cut short, a `history.jsonl` an epoch short, or a checkpoint from a T4, from
`mps` or from before checkpoints recorded the device, each start the run again and end as one never
stopped, while a line cut short after the checkpoint resumes. A
finished run is refused, and so is a run started with another config: give `--resume` the flags the
run started with. [`diffusion.sweep`](#running-the-paired-sweep-m6-2) resumes every run it finds
unfinished.

**Validation while training is for tracking, not reporting.** Every `validation.every_n_epochs`
epochs (and on the last), the run scores the val split's targets at 10% on the holdout its
regime trains for, `EvalMasker`'s in the random regime and `BlockEvalMasker`'s in the block-aware
one ([Training regimes](#training-regimes-m6-2)), exactly as the baselines are scored, using the median of `validation.n_samples` draws (default 5). The
frozen protocol uses 100 draws (`eval.EvalSampler.from_config()`), but one draw over the 400 val
patients takes about 130 s on MPS, so 100 would take hours. Fewer draws make the median
noisier, which raises MAE slightly, so a tracked MAE below forward fill is a conservative claim.

**The health report** (`diffusion.run.health_report`) checks the first 500 steps. They pass if
all 500 were run (`complete`; a shorter run is not reported healthy), every loss and gradient
norm is finite, the loss over the last tenth is below the first tenth
(learning), it has not collapsed toward 0 (below 1e-3, which would mean targets are leaked or
skipped), and the gradient norm has not exploded (the last tenth's median more than 10× the
overall median). Isolated spikes above 10× the median are listed, not failed, since the known bad
values in Set A cause some (see [Dataset](#dataset)). The thresholds are ours.

Run results and seam issues are recorded in [`Docs/M3_debug_log.md`](Docs/M3_debug_log.md).

### Experiment tracking (M3-2)

Every run also streams to wandb while it trains, so all three of us can watch any run from any
machine: team **`DATA612`**, project **`msml612-csdi`**
([wandb.ai/DATA612/msml612-csdi](https://wandb.ai/DATA612/msml612-csdi)), set under `tracking:`
in [`configs/config.yaml`](configs/config.yaml). `DATA612` is the wandb *team*; the organization
around it, `DATA612-org`, can't hold runs. Ask Sankha for an invite to the team.

Log in once per machine with `uv run wandb login` (the key is at
[wandb.ai/authorize](https://wandb.ai/authorize)). On Colab, add the key as a Colab secret named
`WANDB_API_KEY` instead and pass it to the shell cell that launches the run:

```text
from google.colab import userdata
import os; os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
!uv run python -m diffusion.run --seed 42
```

`--wandb` picks the mode: `online`, `offline` (files only, under `wandb/`; upload later with
`uv run wandb sync wandb/offline-run-*`) or `disabled`. Without it, wandb follows the
`WANDB_MODE` environment variable if set, else goes online, so a real run is tracked unless
told otherwise. `run()` called from Python follows the same rule unless given `wandb_mode`, so
a harness looping over it (M6-2) is tracked too. `tests/conftest.py` sets
`WANDB_MODE=disabled`, so the tests never reach wandb's servers.

Runs are named `<variant>_<regime>_seed<seed>`, e.g. `base_random_seed42` or
`ext1+2_random_seed46`, the same as their output folder, and tagged with the variant, regime and
seed. The regime is the config's `train.regime`, `random` or `block-aware`
([Training regimes](#training-regimes-m6-2)), e.g. `ext1_block-aware_seed42`. Every run of a variant, regime
and seed gets the same name, so say what a run is for with `--tag` (repeatable; `tags=` on
`run()`), e.g. `--tag gate`, `--tag sweep`, `--tag dry-run` or `--tag debug`, and filter on it
in the workspace. What a run sends, all of it also in its `outputs/` files:

| wandb key | Sent | From |
| --- | --- | --- |
| `train/loss`, `train/grad_norm`, `train/lr`, `epoch` | every step (step = steps since the start, 175 per epoch) | `history.jsonl` |
| `epoch/loss_mean`, `epoch/grad_norm_max`, `epoch/seconds` | at the end of every epoch | `curve.json` |
| `val/mae`, `val/rmse`, `val/seconds` | on every scored epoch | `curve.json` |
| config (without `tracking`), plus `seed`, `variant`, `regime`, `device`, `gpu`, `out_dir` | once, at the start | `config.yaml`, and the output folder's path |
| summary: `health.*`, `final_val_mae`, `forward_fill_val_mae`, `below_forward_fill`, `gpu`, `pairing.*`, `wandb_run` | at the end | `summary.json` |

Run names aren't unique (a rerun gets the same one), so the two copies of a run point at each
other instead: `wandb_run` in `summary.json` is the run's path on wandb, and `out_dir` in its
wandb config is the folder its files are in. A later step, such as the gate's test-split scoring
(M4-1), finds the run from `wandb_run`. Two tracked reruns differ only in `wandb_run`.

A run that raises once wandb has started (out of memory, Ctrl-C) is closed on wandb as failed
rather than left "running". A run whose machine goes away (a lost Colab session) can't close
itself: wandb marks it crashed once it stops hearing from it. Nothing is written to the output
folder until wandb has started, so a run that can't start it (not logged in, no Colab secret)
leaves the folder empty, and the same command works again once you're logged in, without
`--overwrite`.

Tracking doesn't change training: a tracked run's history and weights are bit-identical to an
untracked one's, on the CPU with wandb offline (`tests/test_run.py`) and on a Colab T4 with wandb
online (M3-2's check run, [`Docs/M3_debug_log.md`](Docs/M3_debug_log.md)).

Any `--epochs` works for a quick check, 1 included (see [Training](#training)). The record that
logging works is M3-2's check run, `base_random_seed42` tagged `check`
([run 85b9o6ls](https://wandb.ai/DATA612/msml612-csdi/runs/85b9o6ls)): seed 42 for 3 epochs on a
Colab T4. The earlier `m3-2_smoke_check` (tagged `smoke`) trained its one epoch at lr 1e-5,
before the learning-rate fix, so its val MAE means nothing.

Tests:

- `tests/test_build_model.py`: config → upstream's dict, defaults equal upstream's published
  values, extension toggles raise, no hidden seeding, weights on the requested device.
- `tests/test_diffusion_data.py`: training patients only and each exactly once, values and masks
  shuffled together, the shuffle a function of run seed and epoch, `timepoints` as in
  `EvalSampler`.
- `tests/test_train.py`: one step on a real batch, upstream's masking never runs, the loss
  scores exactly `TrainMasker`'s targets, every step replayed from scratch at `lr = 0`,
  same-seed reproducibility, clipping, the learning-rate schedule (no drop before a 1-epoch run
  has trained), the epoch hook, deterministic
  kernels switched on (an existing `CUBLAS_WORKSPACE_CONFIG` kept).
- `tests/test_run.py` (M3-1): the health report on made-up histories (healthy, one spike,
  explosion, flat, collapsed, NaN, too short); validation scoring exactly `EvalMasker`'s targets
  without showing them to the model; a run's output files; validation every n epochs and on the
  last; the command line rerunning a run from its saved config with identical results;
  `--validate-every` recorded in the saved config; a folder that is not empty refused, and
  `--overwrite` deleting only a run's own files; the step hook, which gets a copy of each record.
  For wandb (M3-2), with a fake wandb that records what it's sent: run names from the toggles and
  seed; one log per step matching `history.jsonl`; epoch rows, with val scores only on scored
  epochs; the config, seed, toggles and output folder; `summary.json` as the summary, with the
  run's wandb path; a run that raises closed as failed; a wandb start that fails (no login)
  writing nothing, so the same command reruns without `--overwrite`; a run that fails before
  training (no folder, no masker) never left open on wandb or started twice; Python calls and
  the command line both following `WANDB_MODE`; `--tag` and `tags=` added to the run's tags; the
  default output folder named like the wandb run; the `tracking` block left out of the saved
  config and the wandb config. With real wandb offline: a tracked
  run's history and weights bit-identical to an untracked one's, and its summary too except
  `wandb_run`, which names the offline run (null untracked).
- `tests/test_forward_noising.py`: the schedule check described under
  [Reused CSDI implementation](#reused-csdi-implementation).

### Scoring a run (M4-1)

```text
uv run python -m diffusion.score outputs/base_random_seed42
```

`diffusion/score.py` gives a finished run the protocol's number; the validation it tracks while
training (5 draws) is only for watching it. It rebuilds the model from the run's saved
`config.yaml`, loads its `model.pt` (the final weights, PROJECT_PLAN §3.1) and scores the test
split's held-out values with `eval.EvalSampler.from_config()`: 100 draws per patient with the
frozen `eval_sampling_seed`, 16 patients at a time, the median of the draws scored in standardized
units exactly as the baselines are. A run without `summary.json` has not finished, so its
`model.pt` is an earlier epoch's, and it is refused. So is a `model.pt` that is not the final
weights `summary.json` records (`model_sha256`, M6-2): on Google Drive, a session lost just after a
run finished can leave its summary beside an earlier epoch's weights, which would otherwise be
scored as the final ones. A run from before the summary recorded them is not checked.

The holdout is the frozen random one at 10% unless told otherwise: `--fraction 0.5` or `0.9` for
CSDI's robustness rates, `--holdout block` for `BlockEvalMasker`'s (10% only), `--split val` for
the validation split. Other options: `--device`, `--config` (where the frozen protocol is read
from), `--dataset`, `--wandb`, `--paired` (a sweep run's scoring: refused on any GPU but
`protocol.paired_gpu`). After every batch it prints the running MAE and the time left.
Two files go into the run folder, named for what was scored:

| File | Contents |
| --- | --- |
| `scores_test_random_0.1.json` | MAE, RMSE and per-variable MAE (standardized), target count, the holdout's and sampler's settings, device, seconds, and `model_sha256`, the hash of the weights scored |
| `median_test_random_0.1.npy` | the median imputation, `(N, 48, 35)` float32, in `dataset["test_idx"]` order, for checks that need more than the pooled scores (the gate's PaCO2/PaO2 check, M9-2) |

The MAE and RMSE are also added to the run's wandb summary (its `wandb_run`) under the baselines'
keys, e.g. `random/0.1/test/mae`, so CSDI runs line up with the baselines in the runs table.
Beside them go what the sweep's results store collects and checks (M6-2,
[Paired sweep checks](#paired-sweep-checks-m6-2)): the GPU (`random/0.1/test/gpu`), whether the pass
ran in TF32 (`.../tf32`, from the frozen `protocol.tf32`, which the pass applies), the sampler's
and holdout's settings (`.../sampler.eval_n_samples`, `.../holdout_settings.eval_masker_seed`, ...)
and the per-variable MAE (`.../per_variable_mae.HR`, ...).
That needs wandb online: `--wandb disabled`, or `WANDB_MODE` set to anything but `online`, skips
it. Retraining a run with `--overwrite` leaves its score files behind; their `model_sha256` no
longer matches `model.pt`, and the gate refuses them until the run is scored again.

A scoring pass draws all its noise from `EvalSampler`'s frozen generator, through
`EvalSampler.batches`, the same pass one batch at a time, so rescoring the same weights gives
the same numbers bit for bit.

### The reproduction gate (M4-1)

```text
uv run python -m diffusion.gate outputs/base_random_seed42 outputs/base_random_seed43 outputs/base_random_seed44
```

`diffusion/gate.py` reads the runs' random 10% test scores and writes
[`results/gate.json`](results/gate.json): each seed's MAE and RMSE, their mean ± SE (sample
standard deviation over √n, as in BRITS's record) and the status, **PASS** when at least
`protocol.gate_min_seeds` (3) seeds average at most `protocol.gate_mae_at_10pct` (0.24), FAIL
when they average more, INCOMPLETE with fewer seeds. It refuses runs that would make the number
mean something else: a variant other than base, a seed given twice, configs that differ between
the runs (the tracking-only `validation` block aside), scores not taken with the frozen holdout
and sampler, scores of other weights than the run's `model.pt`, or a median that does not give
the recorded MAE. Every departure of the runs' config from the repo's (upstream's published
setup) is listed in the record under `config_departures`.

The record also carries the PaCO2/PaO2 check (`bad_values` per seed; see
[Learned baseline](#learned-baseline-brits-m3-3) for why): PaCO2's and PaO2's MAE and largest
miss, every PaCO2 or PaO2 target at an hour where the patient's pH is visible and bad, the
pooled MAE and RMSE with and without the patients who show a bad value, and the five largest
misses with their share of the squared error and any visible bad value of their own feature.
A patient *shows* a bad value when one of
`data.physionet.known_bad_values` (the [Dataset](#dataset) section's list, as raw-unit ranges in
`KNOWN_BAD_RANGES`) is visible to the model.

**Result: PASS.** Base CSDI on upstream's configuration with nothing changed
(`config_departures` is empty), seeds 42–44, 200 epochs each on an Apple M4 Pro (MPS), final
weights scored on the random 10% test holdout (800 patients, 26,027 targets). The runs, their
training curves and their cost are in [`Docs/M3_debug_log.md`](Docs/M3_debug_log.md):

| Seed | Test MAE | Test RMSE | wandb run |
| --- | --- | --- | --- |
| 42 | 0.2288 | 1.1842 ‡ | [`bqoimtqw`](https://wandb.ai/DATA612/msml612-csdi/runs/bqoimtqw) |
| 43 | 0.2178 | 0.4933 | [`xki8hsx9`](https://wandb.ai/DATA612/msml612-csdi/runs/xki8hsx9) |
| 44 | 0.2185 | 0.4880 | [`c754hh0k`](https://wandb.ai/DATA612/msml612-csdi/runs/c754hh0k) |
| **mean ± SE** | **0.2217 ± 0.0036** | 0.7218 ± 0.2312 | |

That is inside the 0.22–0.24 band (PROJECT_PLAN §3.4), 2% above the paper's 0.217 and just below
the independent reimplementation's 0.228 (CFMI 2025). On the same split and holdout BRITS scores
0.2667 ± 0.0010 and forward fill 0.4487. As with BRITS, the SE reflects training randomness only,
since the test split is fixed.

‡ **A visible bad pH is carried along pH, not across features.** M4-1 asked whether CSDI's
feature attention drags same-hour PaCO2 and PaO2 guesses off a visible bad pH, as BRITS's feature
regression does. It does not. On this holdout the one such target, record 140525's PaO2 of 54 at
hour 22 next to its pH of 735, is guessed at 106, 111 and 107 mmHg by the three seeds, misses of
0.60–0.66 standardized units (BRITS guesses 5,204 mmHg, a miss of 60). PaCO2's and PaO2's largest
misses (1.9–2.3 and 3.0–3.1) are in patients who show no bad value. What CSDI does instead,
through its temporal attention, is carry a bad pH along pH, as forward fill does, by an amount
that depends on the seed:

| Held-out pH | Visible bad pH | Seed 42 | Seed 43 | Seed 44 |
| --- | --- | --- | --- | --- |
| record 140525, hour 25: 7.33 | 734 and 735 at hours 19 and 22 | 220.5 (miss 152.8) | 8.78 (1.0) | 7.46 (0.1) |
| record 141680, hour 38: 7.46 | 187 at hour 8 | 125.9 (84.8) | 12.87 (3.9) | 9.35 (1.4) |

Misses are in standardized units. In seed 42 those two values are 84% of the squared error.
Without the 12 test patients who show a bad value, seed 42's RMSE would be 0.4795 and its MAE
0.2198, and the three-seed mean MAE would be 0.2187 instead of 0.2217. So MAE, the headline
metric, hardly moves, and seed 42's RMSE is mostly two values. pH stays on M9-2's list of
per-variable MAEs not to trust (see [Dataset](#dataset)).

Tests:

- `tests/test_score.py`: a stand-in sampler whose draws miss each held-out value by known amounts
  gives exactly MAE 0.5 (the lower middle of four draws, not their mean) on exactly the targets;
  the test split by default, in its order; the frozen holdouts' conditioning masks, the
  robustness rates and the block holdout, and nothing outside the protocol; the frozen sampler
  (2021, 100 draws, 16); the run's own weights; a run without `summary.json` refused; held-out
  values never reaching the guess; rescoring identical; the files and their names; the running
  MAE; the wandb keys, and nothing sent unless online.
- `tests/test_gate.py`: mean ± SE and PASS/FAIL/INCOMPLETE worked by hand, the threshold itself
  passing; every refusal above; departures listed, the tracking-only blocks skipped; the
  PaCO2/PaO2 check on a hand-built pair of patients, its largest misses including a bad pH
  carried along pH and never an unscored box; the recorded `results/gate.json` on the frozen
  sampler and holdout, with `baselines.json`'s patient and target counts, adding up.
- `tests/test_known_bad_values.py`: values outside each range flagged and the bounds not, raw
  units judged rather than standardized, unmeasured boxes never flagged, Set A's 10 pH, 73 Temp,
  1 K and 2 PaO2.
- `tests/test_eval_sampler.py`: `batches` byte-identical to `__call__`, and an abandoned pass
  handing the model back.

---

## Extension 1: MissingnessEncoder (M4-2)

```text
uv run python -m diffusion.run --seed 42 --use_mask_embedding   # or model.use_mask_embedding: true
uv run python -m diffusion.overfit --device cpu             # the overfit sanity check
```

[`models/missingness.py`](models/missingness.py) is PROJECT_PLAN §4.1's dual-axis encoder,
`MissingnessEncoder(n_features, n_timesteps, d_model, n_heads=2, n_layers=2)`. Every (hour,
feature) is one token: its conditioning-mask bit projected to `d_model`, plus a learned feature
embedding and a learned hour embedding. Each of the two layer pairs attends over the 48 hours
within every feature, then over the 35 features within every hour, so the cross-feature
structure of block missingness is visible to it. Its output, `(B, T, D, d_model)`, scaled by a
learned gate that starts at 0, is the `mask_emb` hook, added to the denoiser input (see
[Denoiser input-hook contract](#denoiser-input-hook-contract)).

- **Toggle:** `model.use_mask_embedding`. `models.build_model` builds the encoder into the
  model at the denoiser's width (`d_model = diffusion.channels`), after the shared parts, so a
  run seed's shared weights start identical in base and Ext 1. The training loop and
  `EvalSampler` are unchanged; the model computes `mask_emb` itself, once per call, on its
  training, validation and sampler paths. Off, there is no encoder and the model is the base.
- **Conditioning mask only.** The encoder reads `gt_mask` (`cond_mask`), never
  `observed_mask`, which still marks the held-out targets (PROJECT_PLAN §3.1).
- **Starts as base, trained end-to-end** with the rest of the model (M4-2 review). The output is
  multiplied by a learned scalar `gate` that starts at 0, so at the start Ext 1 computes exactly
  what base does, and the encoder's output enters only as far as training finds it useful.
  Without it, the untrained encoder's output (RMS 1.0 per value on a real batch) outweighs the
  denoiser input it is added to (RMS 0.62) from the first step; on the overfit check below,
  that version started worse than base (first-epoch loss 0.672 against 0.634; gated, 0.639). Upstream
  zero-initialises the denoiser's last layer, so step 1 trains only that layer; step 2 moves the
  gate, whose gradient does not depend on its own value; from step 3, every encoder weight trains.
- **Size and cost.** 106,305 more weights than base's 414,065 (+26%, at `channels: 64`). A
  training step at batch 16 takes about 24% longer (244 against 197 ms, Apple M4 Pro, MPS);
  scoring does not, since `mask_emb` is computed once per sampler call, not per reverse step.
- Its dropout layers (PyTorch's default 0.1) draw from the global RNG, so as with any extra
  dropout, dropout is not paired between base and Ext 1 (see
  [Random streams](#random-streams-seed-pairing)).

**Overfit sanity check: PASS.** `diffusion/overfit.py` trains Ext 1 on the first 10 train
patients with `BlockEvalMasker`'s split as the conditioning mask, so the loss is taken on the
very targets it then scores with the frozen sampler (2021, 100 draws, median). Recorded in
[`results/ext1_overfit.json`](results/ext1_overfit.json): seed 42, CPU (Apple M4 Pro, torch
2.14.1), 50 epochs, **block MAE 0.0733** (RMSE 0.1114) on 353 targets, against a bar of 0.1 and
0.6895 for guessing zero; loss 0.639 in the first epoch, 0.105 in the last; encoder gradient
norm 0.098 on the last step, and the gate at −0.149 from its start at 0 (its sign means
nothing: the layers behind it can flip theirs). It shows the model with the encoder can fit,
nothing about how Ext 1 imputes values it has not seen; whether Ext 1 helps waits for the
paired 5-seed sweep (PROJECT_PLAN §4.1).

Base, put through the same protocol and recorded beside it as `base_reference`, reaches 0.1109
(FAIL), though it passed the reproduction gate on unseen patients (M4-1). The gap is not
evidence for Ext 1. With each patient's conditioning mask fixed, the encoder's output acts as a
fingerprint of patient and cell that a model can memorise answers against, and Ext 1 has 26%
more weights to memorise with; neither carries over to unseen patients. What it does show is
that the bar sits where the protocol, not only the model, decides PASS or FAIL.

One departure from the issue's "50 epochs": 10 patients are one batch, so an epoch of the
training loop is one optimizer step, and 50 steps leave a diffusion model far from fitting.
Each epoch here is 20 steps on that batch (1,000 in all), with the config's lr, weight decay
and learning-rate drops at 0.75 and 0.9 of the epochs. The protocol was settled on the encoder
before it had a gate (exploratory runs, not recorded: about 0.55 after 50 steps, 0.10–0.19
after 1,000 without the drops); that encoder reached 0.0751 on Josh's machine and 0.0738 on
this one, and the gated encoder was run on the same protocol unchanged. Results reproduce bit
for bit on one machine; another CPU's float rounding drifts over 1,000 steps, which is why the
record names the machine and torch version.

**Hand-off to the combined model (M5-3).** What `M6-1` needs to compose Ext 1 with Ext 2 into
the four variants:

- **Finalized.** `models/missingness.py` is unchanged since M4-2. M5-2's change to
  `models.build_model` (each encoder built from the init-RNG state right after the shared parts)
  leaves Ext 1's starting weights where M4-2 had them, and Ext 1 at `c41b890` retrains bit for
  bit identically to Ext 1 at `dd787a5` over 500 steps. Its 200-epoch health run is healthy, with
  its gate opened to −0.159 ([Health runs](#health-runs-m5-3)).
- **Flags.** `--use_mask_embedding` on `diffusion.run`, or `model.use_mask_embedding: true`
  (default `false`), is Ext 1's only switch, independent of `--use_time_encoding`. With both on,
  the variant is `ext1+2` and a run is named `ext1+2_<regime>_seed<seed>`. Ext 1 has no config
  key of its own: `d_model` is `diffusion.channels` (64; the `d_model` sweep resizes both
  encoders with it), `n_features` and `n_timesteps` come from `data` (35 and 48), and its 2 heads
  and 2 layer pairs are `MissingnessEncoder`'s defaults. M8-1's mask-encoder-layers study will
  need a config key for the layers.
- **Shapes.** `CSDI_base.get_mask_emb` gives the encoder the conditioning mask, permuted from
  upstream's `(B, K, L)` to our `(B, T, D)` = `(B, 48, 35)`. It returns `mask_emb`,
  `(B, T, D, d_model)` = `(B, 48, 35, 64)`, already scaled by its gate (`mask_encoder.gate` in
  `model.pt`). The denoiser checks that shape and adds `mask_emb`, permuted to
  `(B, channels, K, L)`, to its projected input; `time_enc` has the same shape and is added the
  same way after it. The two are separate additions from the same conditioning mask, and neither
  encoder reads the other's output. Each is computed once per model call: once per training
  step, and once per sampler call, reused at every reverse step.
- **Composes with Ext 2** (tests): every toggle combination builds (`tests/test_build_model.py`);
  both flags together train through `diffusion.run` and save both encoders (`tests/test_run.py`);
  the two hooks add together in the denoiser (`tests/test_denoiser_hooks.py`); at a run seed the
  mask encoder starts from the same weights in Ext 1 and Ext 1+2, and the shared weights from
  base's; and Ext 1 changes a working Ext 2 exactly as it changes base: with its gate at 0,
  Ext 1+2 computes exactly Ext 2, and with it open, something else, on the training, validation
  and sampler paths (`tests/test_time_encoding.py`). As between base and Ext 1, dropout is not
  paired between Ext 2 and Ext 1+2: Ext 1's dropout draws from the global RNG
  ([Random streams](#random-streams-seed-pairing)).

Tests:

- `tests/test_missingness_encoder.py`: the output's shape; the layers matching the same layers
  run one feature's hours and one hour's features at a time (the transposes); one mask value
  reaching other hours of its feature and other features at its hour; on both the training and
  the sampler path, the encoder given the conditioning mask in our axis order, and `mask_emb`
  (and on the sampler path, the samples) unchanged when `observed_mask` changes and changed
  when the conditioning mask does; the same `mask_emb` reaching every denoiser call on the
  training, validation and sampler paths, computed once per call; off building no encoder; on
  building it at the denoiser's width; shared weights identical to base; with the encoder
  removed, base's loss and samples exactly; the gate starting at 0, so Ext 1's loss and samples
  are base's exactly; the gate training first and the weights behind it after; nonzero
  gradients on every encoder weight; the unchanged training loop training it.
- `tests/test_overfit.py`: the check on the fake patients: the right patients and targets, the
  record, gradients reaching the encoder and its gate opening, the base reference run the same
  way without an encoder, the holdout following the config it is given, and the same seed
  giving the same record.

---

## Extension 2: ContinuousTimeEncoding (M5-2)

```text
uv run python -m diffusion.run --seed 42 --use_time_encoding   # or model.use_time_encoding: true
uv run python -m diffusion.ext2_health                         # the mechanical health check
```

[`models/time_encoding.py`](models/time_encoding.py) is PROJECT_PLAN §4.2's true Time2Vec of
TSLO δ, `ContinuousTimeEncoding(d_k, d_model)`. At every (hour, feature), δ (the hours since that
feature was last visible to the model) becomes one linear feature, `w0(δ)`, and `d_k − 1` sine
features, `sin(w(δ))`, concatenated (not summed: a sum of linear terms is still one linear term)
and projected to `d_model` by one projection shared by every (hour, feature). Its output,
`(B, T, D, d_model)`, scaled by a learned gate that starts at 0, is the `time_enc` hook, added to
the denoiser input (see [Denoiser input-hook contract](#denoiser-input-hook-contract)).

- **Toggle:** `model.use_time_encoding`. `models.build_model` builds the encoder at the
  denoiser's width (`d_model = diffusion.channels`) with `d_k = model.time_encoding_dk` (16),
  after the shared parts and from the init-RNG state right after them, so at a run seed it
  starts from the same weights in Ext 2 and Ext 1+2 (see
  [Random streams](#random-streams-seed-pairing)). The training loop and `EvalSampler` are
  unchanged: the model computes δ with `tslo(cond_mask)`, and the encoding, once per call on its
  training, validation and sampler paths (M5-1). Off, there is no encoder and no δ is computed.
- **δ from the conditioning mask only** (M5-1), so a held-out target counts as unobserved; never
  the dataset's `delta_obs`, which still counts the held-out targets as observed.
- **Raw hours, gate from 0** (both settled in M5-2). Raw hours keep the waves informative from
  the start: on δ / 48, with the layers' starting frequencies (at most 1 per hour), each wave is
  nearly flat over its range. Untrained, though, the encoder's output (RMS 1.26 per value at
  `d_k` 16, from 0.42 to 3.18 across five init seeds, on 64 training patients) outweighs the
  denoiser input it is added to (0.67) by an amount that depends on the seed, mostly through the
  linear feature, which grows with δ. The gate makes Ext 2 start out computing exactly what base
  does. As with Ext 1, step 1 trains only upstream's zero-initialised last layer, step 2 moves the
  gate, and from step 3 every encoder weight trains.
- **Size and cost.** 1,121 more weights than base's 414,065 (+0.3%, at `channels: 64`). On the
  health check below, a training epoch at batch 16 took 31 s against base's 30 s (Apple M4 Pro,
  MPS), and a 5-draw validation pass 339 s against 335 s.
- No dropout layers, so dropout stays paired between base and Ext 2 (see
  [Random streams](#random-streams-seed-pairing)).

**Mechanical health check: PASS.** `diffusion/ext2_health.py` trains Ext 2 at seed 42 through
the unchanged `diffusion.run.run` for the fewest epochs that reach 500 steps (3 at batch 16, so
525 steps), untracked, so the run's 500-step health report is complete. It then scores the
validation split's HR and SysABP targets alone, with the run's own masker and frozen 5-draw
sampler; SysABP stands in for SpO2, which is not one of our 35 features (PROJECT_PLAN §4.2).
Recorded in [`results/ext2_health.json`](results/ext2_health.json) (MPS, Apple M4 Pro, torch
2.14.1):

| | Ext 2 | Base, same short run |
| --- | --- | --- |
| 500-step health | healthy | healthy |
| Loss, first → last tenth of the 500 steps | 0.506 → 0.320 | 0.506 → 0.322 |
| Gate (starts at 0) | −0.044 | — |
| Encoder gradient norm, last step | 0.032 | — |
| Val MAE @ 10% (5 draws) | 0.4731 | 0.4824 |
| HR val MAE | 0.4729 (8.45 bpm) | 0.4684 (8.37 bpm) |
| SysABP val MAE | 0.4670 (11.27 mmHg) | 0.4711 (11.37 mmHg) |

PASS means the run is healthy, the gate has opened and gradients reach the encoder; nothing
more. The two runs start almost identically (first-tenth loss 0.50617 against 0.50614) and share
their three early gradient spikes (steps 5, 14 and 16), which therefore come from the data and
the shared start, not from Ext 2. After 525 steps both sit above forward fill's 0.437 (the gate
run needed 200 epochs to reach 0.246), and Ext 2 is ahead on SysABP and behind on HR. That is one
seed at three epochs, so not an effect: whether Ext 2 helps waits for the paired 5-seed sweep.
The config's two learning-rate drops both fall at epoch 2 of 3. Base is recorded beside it as
`base_reference`, for scale.

Ext 2's full 200-epoch health run is healthy, with its gate opened to −0.053
([Health runs](#health-runs-m5-3)). For composing it with Ext 1 in `M6-1`, see
[Hand-off to the combined model](#extension-1-missingnessencoder-m4-2) in Extension 1's section,
which covers both hooks.

Tests:

- `tests/test_time_encoding.py`, the M5-2 part: lesson 3's hand-worked weights (δ = 1 gives
  `[0.1, 1.5]`, δ = 3 gives `[0.3, 0]`); the layers' shapes (1,121 weights at `d_k` 16,
  `d_model` 64); the features equal to one linear feature then `d_k − 1` waves, concatenated,
  worked out element by element; the linear feature growing past 1 while the waves stay in
  [−1, 1]; the gate a parameter that starts at 0, so the output is exactly 0 although the
  projection has a bias; each value its own vector from shared weights; at gate 0 only the gate
  having a gradient, and every weight once it has moved; off building no encoder, on building it
  at the config's `d_k` and the denoiser's width; at seed 42, every variant's shared weights equal
  to base's, the mask encoder identical in Ext 1 and Ext 1+2 and to M4-2's, and the time encoder
  identical in Ext 2 and Ext 1+2 (`w0` −0.5433); with the gate closed, Ext 2's loss, validation
  loss and samples exactly those of the variant without it, and with it open, different; the gate
  training first; nonzero gradients on every encoder weight after two steps; the unchanged
  training loop training it; and a training step drawing no more from the global RNG than base's.
- `tests/test_build_model.py`: every toggle combination builds, with exactly the encoders its
  toggles ask for.
- `tests/test_run.py`: `--use_time_encoding`, alone and with `--use_mask_embedding`, recorded in
  the saved config, with the encoders' weights saved.
- `tests/test_ext2_health.py`: `epochs_for` at its boundaries; `status` passing only a healthy
  run whose gate has opened and whose encoder has gradients; `variable_mae` on a stand-in model
  that guesses 0; and the whole check on the fake patients: Ext 2 then base through `run()`,
  untracked, with Ext 1 turned off; a run under 500 steps not passed; the recorded gate equal to
  the one in `model.pt`; `"all"` reproducing each run's own val MAE; and a rerun replacing its
  folders and giving the same record.

---

## Health runs (M5-3)

```text
uv run python -m diffusion.health bqoimtqw t504tr0f 8m16obhn \
    --note "base=the M4-1 gate run, reused: base at dd787a5 and at c41b890 retrains bit for bit identically to it over the first 500 steps on the same M4 Pro (M5-3 review)" \
    --note "ext1=rerun on the M4 Pro so every run is on one machine; replaces Sankha's run gwx2us22 on an Apple M4 (healthy, final val MAE 0.2456, gate -0.156 as he read it), which diverges from this one from step 8 (M5-3)"
```

**Health only, not an effect measurement.** Before the sweep, each variant trains its 200
epochs at seed 42 through `diffusion.run`, validated as it goes, to show that it trains and
evaluates end to end. `diffusion/health.py` reads the runs from wandb, so a teammate's run is
recorded without its output folder, and writes
[`results/health_seed42.json`](results/health_seed42.json). For each variant the record has:

- **Where the run came from:** its wandb path, tags, commit, device and machine.
- **The 500-step health report:** recomputed from the logged steps and checked against the one
  in the run's summary.
- **Validation:** the final val MAE @10% against forward fill's, and the val curve.
- **Timings:** training and validation seconds, the median epoch and the slowest one; and which
  runs shared a machine, since only those compare.
- **Each extension encoder's gate:** read from the run's `model.pt` on wandb (or a local one
  given with `--weights`). A gate starts at 0, so one that has moved shows the encoder in use.

It refuses runs that are unfinished, at different seeds or regimes, or validated with different
settings; runs whose summary does not match their logged steps; and weights for a run without an
encoder, or without its encoder's gate.

All three runs are on one Apple M4 Pro (20-core GPU), validated every 25 epochs:

| Variant | Run | 500-step health | Gate | Training | Validation pass | Final val MAE @10% (5 draws) |
|---|---|---|---|---|---|---|
| base | `bqoimtqw`, the M4-1 gate run, reused | healthy | — | 6,048 s | 337 s | 0.2464 |
| Ext 1 | `t504tr0f` | healthy | −0.159 | 7,570 s (+25%) | 338 s | 0.2433 |
| Ext 2 | `8m16obhn` | healthy | −0.053 | 6,254 s (+3.4%) | 341 s | 0.2469 |

Validation here is tracking only: 5 draws on the val split ([Launching a run](#launching-a-run-m3-1)),
against forward fill's 0.4366. Each number is one seed, so they say nothing about whether either
extension helps (PROJECT_PLAN §3.3).

- **The runs stand for the current code.** They trained at three commits: base (the M4-1 gate
  run) at `9314475`, Ext 2 at `c41b890`, after M5-2 had changed `models/build.py` and
  `models/csdi/main_model.py`, and Ext 1 at `a6e4deb`, whose training code is `c41b890`'s. Base
  at `c41b890` retrains bit for bit identically to the gate run over the first 500 steps on this
  M4 Pro, as it does at `dd787a5`: every loss, gradient norm and learning rate (M5-3 review). The
  runs' configs differ only in the extension toggles and in `model.time_encoding_dk`, which runs
  from before M5-2 do not have.
- **Both gates have moved from 0**, so both encoders are in use: Ext 1's to −0.159
  (`mask_encoder_gate`) and Ext 2's to −0.053 (`time_encoder_gate`), read from the `model.pt`
  uploaded to each wandb run.
- **What the encoders cost** (one run each, one machine). Ext 1 trains 25% longer than base
  (a median epoch of 37.8 s against 30.1 s), in line with PROJECT_PLAN §8's 24% per step; Ext 2,
  3.4% longer (31.2 s), in line with M5-2's short check. A validation pass costs about the same
  for all three, 337–341 s.
- **Ext 1 was rerun on this machine.** Sankha's run (`gwx2us22`, at `dd787a5`) was on an Apple
  M4 with an 8-core GPU, validated every 10 epochs: also healthy, with final val MAE 0.2456 and
  gate −0.156 as he read it from its `model.pt`, which is not on wandb. It took 5.0 h to train
  and 868 s a validation pass, 2.6 times as long as here, so that was the machine, not the
  encoder. The two Ext 1 runs share their code and seed but first differ at step 8 by float
  rounding: training is bit-identical only on one kind of chip, which is why every paired run
  goes on one GPU model (PROJECT_PLAN §3.3). The rerun itself repeats, bit for bit over its first
  500 steps, a separate 500-step run of Ext 1 at `c41b890` on this M4 Pro, which in turn matches
  Ext 1 at `dd787a5`.

Tests:

- `tests/test_health.py`: the record of hand-built runs, worked out by hand (health report,
  val curve, timings, machine, config differences); which runs share a machine; wandb's row order
  and a nested or flattened summary making no difference; a float wandb rounded in its last digit
  not counting as a mismatch; each refusal; each encoder's gate read from a run's weights (Ext 1's,
  Ext 2's, both in Ext 1+2, null with no file), a run from before M5-2 reading as without Ext 2,
  and weights refused for a run without an encoder or without its encoder's gate; the command line
  resolving bare ids to the tracking project.

---

## The four variants (M6-1)

One model, two independent toggles. A variant is nothing but its toggles: the shared parts are
built and seeded the same way in all four, and each encoder is built from the same init-RNG
state in every variant that has it ([Random streams](#random-streams-seed-pairing)).

| Variant | `--variant` | Toggle flags | `model.use_mask_embedding` | `model.use_time_encoding` | Run name |
| --- | --- | --- | --- | --- | --- |
| `base` | `--variant base` | *(none)* | `false` | `false` | `base_<regime>_seed<seed>` |
| `ext1` | `--variant ext1` | `--use_mask_embedding` | `true` | `false` | `ext1_<regime>_seed<seed>` |
| `ext2` | `--variant ext2` | `--use_time_encoding` | `false` | `true` | `ext2_<regime>_seed<seed>` |
| `ext1+2` | `--variant ext1+2` | `--use_mask_embedding --use_time_encoding` | `true` | `true` | `ext1+2_<regime>_seed<seed>` |

```text
uv run python -m diffusion.run --seed 42 --variant ext1+2      # ext1+2, seed 42
```

`diffusion.run.VARIANTS` is the one definition of a variant: its toggles, every one written on or
off. There are three ways to set a run to a variant, and they give the same run:

- **`--variant <name>`** on `diffusion.run` sets every toggle to the variant's, on or off,
  whatever the config held, so `--variant base` on a saved Ext 1 config trains base. It is
  refused alongside a toggle flag.
- **`with_variant(config, name)`** does the same for a harness that calls `run()` itself: it
  returns a copy of the config with every toggle written, and leaves the caller's alone.
- **The toggle flags**, `VARIANT_FLAGS[name]`, only switch toggles on, so they give the variant
  only on a config whose toggles are all off, as the repo's are.

`variant_name(config)` reads a variant back from its toggles. `VARIANTS` lists every variant a
run can be, not the sweep: the harness (M6-2) names the variants it trains, so M6-3's
`ext1blind`, which goes into `VARIANTS` with its toggle written in every variant, is not swept by
being there.

- **`base` is the gate model, exactly.** With both toggles off the model has neither encoder and
  only upstream's weights. On the Apple M4 Pro (20-core GPU, MPS, torch 2.14.1) that trained the
  gate runs, base at seed 42 on this branch retrains the M4-1 gate run (`bqoimtqw`, trained at
  `9314475`) bit for bit: every one of its 35,000 steps' losses, gradient norms and learning
  rates, and its final weights, all 139 tensors (M6-1 review). Samples from equal weights match
  too, at `9314475` and on this branch (2 draws on 32 val patients, the frozen seed). Nothing since
  the gate has changed what base computes: M4-2, M5-1 and M5-2 only added code that runs when a
  toggle is on. `tests/test_variants.py` holds base to numbers worked out at `9314475`, so a
  later change to base fails the suite.
- **`ext1+2` composes cleanly.** Its two hooks are separate additions to the denoiser input,
  computed from the same conditioning mask, and neither encoder reads the other (see
  [Hand-off to the combined model](#extension-1-missingnessencoder-m4-2)). A 3-epoch run at
  seed 42 on the same M4 Pro, through `diffusion.run` with both on (not logged to wandb):
  healthy over its first 500 steps (loss 0.506 in the first tenth, 0.316 in the last; one
  gradient spike, at step 16, where base's and Ext 2's short runs also spike, besides steps 5
  and 14), finite throughout, and both gates opened from 0 (`mask_encoder.gate` −0.052,
  `time_encoder.gate` −0.035). Its val MAE after 3 epochs, 0.4728 (5 draws), is above forward
  fill's 0.4366, as Ext 2's (0.4731) and base's (0.4824) are after the same short run
  ([Extension 2](#extension-2-continuoustimeencoding-m5-2)): 525 steps is a plumbing check, not a
  result. Josh's run of it on his own M4 Pro gave 0.4723 and gates −0.051 and −0.036; runs
  repeat bit for bit only on one machine, which is why every health run is on this one (M5-3).

Tests:

- `tests/test_variants.py`: `VARIANTS` and `VARIANT_FLAGS` hold exactly the four variants'
  toggles and flags; `with_variant` writes every toggle, from a config with all of them off or
  all on, and leaves the caller's config alone; each variant's toggle flags, and `--variant` on
  a config with every toggle on, train that variant and save exactly its encoders; `--variant`
  with a toggle flag refused before anything runs; every variant, at the gate's width and depth
  on 4 Set A patients, runs forward and backward with finite gradients on every weight and
  samples finite `(K, N, 48, 35)` draws through `EvalSampler`; with both on, one backward pass
  gives nonzero gradients on every weight of both encoders and both gates have opened; and base
  at seed 42 on the fake patients (CPU, width 16, 2 layers, 8 steps) gives, to 1e-4, the losses,
  gradient norms, weights and samples worked out at `9314475`. It runs 2 layers because a
  1-layer model hides a change to the skip connections' 1/√layers.
- Composition itself is covered where each extension landed: `tests/test_build_model.py`,
  `tests/test_time_encoding.py` (pairing, each gate at 0 giving the variant without it on every
  path) and `tests/test_run.py` (both flags through `diffusion.run`).

---

## Running the paired sweep (M6-2)

`diffusion/sweep.py` is the driver that trains and scores the paired sweep (PROJECT_PLAN §3.3).
A cell is a regime, a variant and a seed. The config's `sweep` block names each regime's variants
([Training regimes](#training-regimes-m6-2)), and the seeds are `protocol.run_seeds`:

```yaml
sweep:
  regimes:
    random: [base, ext1, ext2, ext1+2]     # 20 cells at seeds 42-46: the headline sweep
    block-aware: [base, ext1]              # 10 cells
  holdouts: [random, block]                # every run's test scorings, at 10%
```

The block says which runs to launch, not what one trains, so `diffusion.run` leaves it out of
every run's saved and wandb config, as it does `tracking`: a variant added to it later (M8-4's
`ext1blind`) changes no run already trained. Cells run regime by regime (random first), then seed
by seed, then variant by variant, so a sweep cut short leaves whole seeds behind it, and the
block-aware runs come after the 20 headline ones (PROJECT_PLAN §8). `--regime`, `--variants` and
`--seeds` run some of the cells. `--variants` alone runs each variant only in the regimes that plan
it, so `--variants ext2` never trains a block-aware ext2; with `--regime` it takes any variant
(M8-4's `ext1blind`), and a variant no regime plans needs one. For each cell, in
`<out>/<run name>`, the driver:

1. trains it with `run(..., paired=True, resume=True)`, tagged with `--tag`. A run a lost session
   stopped goes on from its last checkpoint ([Resuming a run](#launching-a-run-m3-1)); a finished
   one is kept, once its saved config is checked to be the cell's and, in a paired sweep, the GPU
   it trained on the paired one (an `--unpaired` session's run in the folder is refused: move it
   out, and the cell is trained again);
2. checks its pairing hashes against those of the regime's other variants at its seed already in
   `<out>`, with the store's own check (`diffusion.results.unpaired`), and stops at the first cell
   that unpairs, not when the store is written after the last;
3. scores its final weights on test under both holdouts with `score(..., paired=True)`, unless
   `<out>` already holds a scoring of those weights done the frozen way, with the protocol's
   sampler, holdout and TF32 setting and on the paired GPU, as the store checks (a scoring by hand
   is done again, not kept), and with its median imputation beside it (`span_mae` reads it), and
   adds every scoring to the run's wandb summary (`attach_to_wandb`), where the store reads it.

So the same command, run again in a new session, finishes what the last one started, redoing at
most 10 epochs or the scoring pass a session was lost in. `<out>/sweep.json` records the tag and
git commit the folder was started with, and another is refused: the store refuses a regime trained
from two commits, so a `git pull` between sessions is caught before it costs a run. The commit is
all the store checks, so a paired sweep also refuses code with changes the commit does not hold
(tracked files edited, deleted or staged; untracked files such as data and outputs are fine):
commit them or undo them first. The driver refuses to start on any GPU but `protocol.paired_gpu`;
`--unpaired` lifts that and the commit check, for a check on another machine, whose runs the store
refuses.

On Colab's G4 runtime, after the setup in [Environment](#environment), put `<out>` on Google
Drive, since a session's own disk goes with it. The setup clones `main`, so a later session checks
out the commit the folder was started from (`sweep.json`) before it goes on; otherwise a merge into
`main` between sessions has the driver refuse to start. Drive uploads files in the background, so
flush it before you end a session on purpose (a session that is lost cannot, and the driver redoes
what it lost):

```text
from google.colab import drive, userdata
import json, os, pathlib, subprocess
drive.mount("/content/drive")
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
OUT = "/content/drive/MyDrive/msml612/dry-run"
started = pathlib.Path(OUT) / "sweep.json"
if started.exists():        # a later session: the folder's commit, not main's newest
    subprocess.run(["git", "checkout", json.loads(started.read_text())["commit"]], check=True)
!uv run python -m diffusion.sweep --tag dry-run --epochs 20 --seeds 42 43 --out {OUT}
drive.flush_and_unmount()   # before ending the session yourself
```

| What | `uv run python -m diffusion.sweep ...` | Cells | Hours on the G4 |
| --- | --- | --- | --- |
| the dry run (M6-2) | `--tag dry-run --epochs 20 --seeds 42 43 --out <Drive>/dry-run` | 12 | about 16.4 |
| its rerun check | `--tag rerun --epochs 20 --regime random --variants base --seeds 42 --out <Drive>/rerun` | 1 | about 1.4 |
| the sweep (M7-1) | `--tag sweep --out <Drive>/sweep` | 30 | about 67.5 |
| M8-4's control, if triggered | `--tag sweep --out <Drive>/sweep --regime block-aware --variants ext1blind --seeds 42 43 44 --holdouts block` | 3 | about 4.5 |
| M8-4's grid check, if triggered | `--tag sweep --out <Drive>/sweep --regime block-aware --variants base ext1 --seeds 42 43 44 --holdouts offgrid` | 6 scorings, no training | about 3.9 |

The hours are the timing test's ([The timing test](#the-timing-test-m6-2)). The sweep's 67.5 h
includes base's ten 50% and 90% scoring passes (about 6.5 h), which run after the driver, and the
finished sweep measured 67.5 h ([The run manifest](#the-run-manifest-m7-1)). Then
`uv run python -m diffusion.results --tag dry-run --seeds 42 43` collects and checks the runs
(below), and
`--compare` checks the rerun:

```text
uv run python -m diffusion.sweep --regime random --variants base --seeds 42 --out <Drive>/dry-run --compare <Drive>/rerun
```

It trains nothing. For each cell finished in both folders it checks that the two runs left the same
numbers: every training step, every epoch's row but its seconds, the summary but its wandb run, the
final weights, and each scoring's MAE, RMSE, per-variable MAE and target count. It exits non-zero
if any differ, or if no cell is in both. Base's 50% and 90% scorings (M7-1) are not the driver's:
`diffusion.score <run folder> --fraction 0.5 --paired` scores one.

Tests (`tests/test_sweep.py`): the cells and their order (30 for the sweep, 12 for the dry run, 3
for M8-4), and `--variants` alone only in the regimes that plan each; every cell trained, scored
under both holdouts with the final weights and attached, and tagged; the same command again
training and scoring nothing more; a sweep a lost session stopped, in training or in scoring,
ending with the same numbers as one never stopped; a cell that unpairs its seed stopping the sweep
before it is scored; a scoring of other weights done again, one whose median imputation is missing,
and one with fewer draws, another holdout, the other TF32 setting or no settings recorded, ending
as the frozen scoring; a paired sweep keeping only a scoring on the paired GPU; a finished run with
another config, another tag or another commit refused, and in a paired sweep one trained on another
GPU, until it is moved out; a paired sweep only on the picked GPU, scoring paired, and never from code with
uncommitted changes (`uncommitted` against a scratch git repository: changed and deleted tracked
files named, untracked ones not); and `--compare` naming what differs.

## Paired sweep checks (M6-2)

The parts of M6-2's harness that show its runs are paired and keep their numbers: pairing
hashes, the paired GPU, and the results store. The driver, `diffusion.sweep`
([Running the paired sweep](#running-the-paired-sweep-m6-2)), calls `run()` and `score()` with
`paired=True` (`--paired` on either command line), and `diffusion.score.attach_to_wandb` after
every `score()`, since the store reads wandb.

**Pairing hashes.** Every run hashes what its random streams gave it, as training draws it, so
a seed's variants are shown to be paired rather than assumed to be (PROJECT_PLAN §3.3):

| Hash | Of | Where |
| --- | --- | --- |
| `data_order_sha256` | the epoch's batches in order (values and observation mask) | each `curve.json` row |
| `train_mask_sha256` | the epoch's train masks (`TrainMasker`'s conditioning masks) | each `curve.json` row |
| `diffusion_sha256` | the diffusion generator's state after each step's timesteps and noise | each `curve.json` row |
| `pairing.data_order`, `pairing.train_mask`, `pairing.diffusion` | one SHA-256 over the run's per-epoch hashes, in order | `summary.json`, wandb summary |
| `pairing.init_shared` | the starting weights of the parts every variant shares | `summary.json`, wandb summary |
| `pairing.init_mask_encoder`, `pairing.init_time_encoder` | each encoder's starting weights, in the variants that have it | `summary.json`, wandb summary |

`train()` has two hooks for these, and hands both copies, so a hook cannot change training
(tested): `on_start` gets the weights once, right after the model is built from the run seed,
and `on_batch` gets each step's batch, mask and train mask, and the diffusion generator's state
once the model has drawn that step's timesteps and noise. The generator is re-seeded every epoch,
so equal states at every step mean equal draws; an extension that drew once more from it would
change every hash after. A hashed run's history equals an unhashed one's (tested). Dropout is
not hashed: it is unpaired between some variants by design
([Random streams](#random-streams-seed-pairing)). A run resumed from a checkpoint keeps its
pairing: it loads the checkpoint after `on_start` has hashed the weights as built, and rebuilds
each stream's chain from the per-epoch hashes in its `curve.json`.

**The paired GPU.** `protocol.paired_gpu` names the one GPU model every paired run trains and
is scored on, as `torch.cuda.get_device_name()` names it: `NVIDIA A100-SXM4-80GB`, picked by
M6-2's timing test ([Environment](#environment)). `run(..., paired=True)` and
`score(..., paired=True)` (`--paired` on the command line) refuse to start on any other GPU, before
writing or sending anything,
and refuse to start at all while it is `null`. Every run records the GPU it trained on (`gpu`
in `summary.json` and in its wandb config and summary; null off CUDA), and every scoring the GPU
it was scored on (`gpu` in its scores file, `<holdout>/<fraction>/test/gpu` in the run's wandb
summary). Runs without `paired` (health checks, debugging) still go anywhere.

**The results store.** `diffusion/results.py` collects the runs from wandb, so runs from
different Colab sessions need no shared folder:

```text
uv run python -m diffusion.results --tag dry-run --seeds 42 43    # results/dry-run.json
uv run python -m diffusion.results --tag sweep                    # results/sweep.json, as M7-1's runs land
```

It reads every finished run with the tag and writes `results/<tag>.json`:
`runs[regime][variant][seed]` holds each run's wandb path, commit, training GPU, and the MAE,
RMSE, scoring GPU and per-variable MAE of every holdout scored (`random/0.1`, `block/0.1`, base's
`random/0.5` and `random/0.9` once M7-1 scores them, and any other holdout `diffusion.score`
knows, such as M6-3's off-grid one); `configs[regime]` the config the regime's runs share,
toggles aside; `gpu` the paired GPU; `unscored` the runs still missing a 10% holdout, whose
scores are null meanwhile, so the store can be written while the sweep runs; and `missing` the
cells `sweep.regimes` plans at `protocol.run_seeds` with no finished run yet (one still
training, one wandb lists as crashed or failed, or a seed the sweep never reached), so M7-2's
table never drops a cell unnoticed. `--seeds` checks fewer seeds, `42 43` for the dry run. The
per-variable MAE is there for Ext 2's high-autocorrelation variables (HR, SysABP; PROJECT_PLAN
§4.2) and M9-2, so no run folder has to outlive its Colab session. It writes nothing, and lists
every reason, unless:

- within each regime and seed, every variant's data order, train masks, diffusion draws and
  shared starting weights hash the same, and each encoder's starting weights hash the same in
  every variant that has it;
- every run of a regime has the same config but for the toggles (validation settings aside:
  they do not touch training), and comes from the same git commit (as wandb recorded it), so
  run each regime from one pinned commit;
- every run trained, and every holdout was scored, on `protocol.paired_gpu`, one GPU for all;
- every holdout was scored with the frozen sampler and holdout of `configs/config.yaml`
  (`eval_sampling_seed`, `eval_n_samples`, `eval_batch_size`, and the holdout's seed and window),
  as the gate requires of its runs, and with its `protocol.tf32`, which every run trained with
  too; a holdout `diffusion.score` does not know is refused too;
- no cell has two runs. A rerun gets the same name on wandb; untag the one to leave out, or tag
  a rerun apart (`rerun`) when checking that a cell reproduces.

Tests:

- `tests/test_train.py`: `on_start` gets the weights the run seed builds, once, before any step;
  hooks that edit what they are given leave training unchanged.
- `tests/test_run.py`: the per-epoch hashes are of exactly the batches and masks training drew,
  and of the diffusion draws replayed independently on the epoch's generator, and the summary's
  of their chain; the starting weights are hashed before the first update; at one seed all four
  variants hash the same streams and shared weights, Ext 1's encoder the same in ext1 and ext1+2
  and Ext 2's in ext2 and ext1+2, and another seed hashes differently; hashing leaves the history
  unchanged; a paired run is refused with no GPU picked and on the wrong GPU, from Python and
  from the command line, writing and sending nothing, and starts on the picked one, recording it.
- `tests/test_score.py`: the scores record the GPU and send it to wandb (null for older scores),
  with the sampler's and holdout's settings and every variable's MAE; a paired scoring is refused
  before it scores, from Python and from the command line, and runs on the picked GPU; a `model.pt`
  that is not the final weights its `summary.json` records is refused before any draw, writing
  nothing, and a run from before the summary recorded them is scored unchecked.
- `tests/test_results.py`: the store's layout, configs, per-variable MAE, unscored holdouts,
  missing planned runs (a seed with no run at all among them), robustness rates and a holdout of
  another name; each way runs can be unpaired refused on its own (diffusion draws, a sampler,
  holdout or TF32 setting other than the protocol's in scoring or in training, a config from
  before TF32 was recorded counting as off, a second commit in a regime among them); validation
  settings, output folders and each regime's commit allowed to differ; a pairing block wandb
  flattened read the same; and real runs through `diffusion.run`, `diffusion.score` and
  `attach_to_wandb` stored, with one relabeled to another seed caught by its data-order hash.

---

## The run manifest (M7-1)

The finished sweep has two records. `results/sweep.json`, the results store above, holds each
run's scores from wandb. `results/sweep_manifest.json` holds what the store does not: each run's
config, GPUs, time, weights and gates, read from the sweep's folder. Copy that folder from Drive
(`MyDrive/msml612/sweep`) to `outputs/sweep`, then:

```text
uv run python -m diffusion.manifest --tag sweep     # writes results/sweep_manifest.json
```

It is keyed as the store is, `runs[regime][variant][seed]`, so M7-2 can join the two files. It
writes nothing, and lists every problem, unless all 30 runs are finished, match their cells' configs
and were trained and scored on the paired GPU.

| Field | What it holds |
| --- | --- |
| `config` | the run's `config.yaml` in full |
| `gpu` | the GPU it trained on and each scoring's |
| `seconds` | training, validation and each scoring; epochs lost with a crashed session are not counted |
| `weights` | `model.pt`'s path and SHA-256 (the weights are in the Drive folder, not in Git or wandb) |
| `gates` | each extension's final gate (PROJECT_PLAN §3.5); base has none |
| `rerun` | the command that trains and scores the cell again |

**Rerunning a cell.** On the paired GPU, run the cell's `rerun` with `<another folder>` replaced by
a new folder. It checks out the sweep's commit, trains the cell, and scores it as the sweep did,
including the 50% and 90% scorings of base's random-regime cells. Then compare the two folders,
which exits non-zero if any number differs:

```text
uv run python -m diffusion.sweep --regime random --variants base --seeds 42 --out outputs/sweep --compare <another folder>
```

**What the sweep cost**, measured on the A100 with TF32 on:

| Part | Passes | Hours |
| --- | --- | --- |
| Training, 30 runs of 200 epochs | | 15.2 |
| Validation | | 7.1 |
| Test scoring at 10%, both holdouts | 60 | 38.9 |
| Base's scoring at 50% and 90% | 10 | 6.3 |
| **Total** | | **67.5** |

Tests (`tests/test_manifest.py`): an entry worked out by hand, each refusal, a rerun's extra scoring
steps run through `diffusion.score`'s real command line, and one real sweep run when its folder is here.

## Ext 1's attribution checks (M6-3)

PROJECT_PLAN §4.1 fixed two checks on Ext 1 before any result. They run in M8-4 only if M7-2 finds
that Ext 1's block-aware contrast with base separates from zero. Going from base to Ext 1 adds the
mask information, but it also adds 106,305 weights and a learned vector for each (hour, feature)
in the denoiser input. On top of that, `BlockEvalMasker` and the `mix` strategy put every block on
the same 16 fixed 3-hour slots, which the encoder's hour embedding can learn. The two checks are
built here so that a triggered M8-4 only costs compute.

**Ext 1-blind** (`ext1blind`) is Ext 1 with `model.mask_embedding_blind: true`. Its encoder is given
an all-ones mask in place of the conditioning mask, on the training, validation and sampler paths
alike. Its `mask_emb` is therefore the same for every mask. It keeps every weight (the hour and
feature embeddings included) and loses only the mask information. Upstream's side information
still carries the conditioning mask, as it does in every variant.

| Variant | `--variant` | Toggle flags | Toggles on | Run name |
| --- | --- | --- | --- | --- |
| `ext1blind` | `--variant ext1blind` | `--use_mask_embedding --mask_embedding_blind` | `use_mask_embedding`, `mask_embedding_blind` | `ext1blind_<regime>_seed<seed>` |

- The toggle is read once, by `models.build_model`, which passes `blind` to `MissingnessEncoder`.
  It is valid only with `use_mask_embedding`: on its own it is refused by `build_model`,
  `variant_name`, `diffusion.run`'s command line and `run()`, before anything runs (`run()` refuses
  it before `overwrite` deletes an earlier run's files).
- `blind` adds no weight and draws nothing from the RNG, so at any run seed Ext 1-blind starts
  from exactly Ext 1's weights and its encoder hashes the same. That is the pairing
  `diffusion.results` checks.
- It also has exactly Ext 1's dropout layers, so at a seed the two draw identical dropout masks:
  Ext 1 and Ext 1-blind are paired in every stream, dropout included, which base and Ext 1 are not
  ([Random streams](#random-streams-seed-pairing)). Their paired Δ carries no dropout noise.
- A config saved before M6-3 has no `mask_embedding_blind` key, and reads as not blind.
- With the toggle off, Ext 1 is unchanged. Ext 1 at seed 42 on the fake patients gives bit-identical
  losses, gradient norms, weights, samples and MAE at `6b8a298` (before M6-3) and on this branch.
  A test pins them, as M6-1's test pins base.
- `ext1blind` is in `VARIANTS`, which lists every variant a run can be, not the sweep. The M7 sweep
  does not train it: M8-4 trains it in the block-aware regime at seeds 42–44, only if triggered.

**The off-grid block holdout** (`diffusion.score --holdout offgrid`, scored as `offgrid/0.1`)
is `eval.OffGridBlockEvalMasker`. It has `BlockEvalMasker`'s per-patient count (the 10% that
`EvalMasker` takes) and its 3-hour windows, but the windows are shifted
`protocol.offgrid_block_offset_hours` (1) later:

- There are 15 windows, at hours 1–3, 4–6, …, 43–45, so none starts on a multiple of 3. Hours 0,
  46 and 47 are never hidden.
- It is seeded and frozen like `BlockEvalMasker`, from a seed of its own
  (`protocol.offgrid_block_eval_masker_seed`, 9090), with one generator per patient from
  `(seed, record_id)`.
- The walk is `block_targets(..., offset=1)`. At the default offset of 0, `block_targets` is the
  grid, draw for draw, so `BlockEvalMasker`'s targets and its golden fingerprint are unchanged.
- On Set A every patient fits: each keeps at least 61% of their observed values inside hours
  1–45, against the 10% hidden. 93% lose at least one whole window (94% on the grid).
- It is for scoring only. M8-4 scores M7's block-aware base and Ext 1 runs at seeds 42–44 on it,
  with no retraining. No baseline is scored on it: the check is Ext 1 against base, paired.
- It cannot hide hours 0, 46 or 47 (a window over hour 0 would start on the grid), and hour 0 is
  base's hardest hour: on the gate runs' random holdout its MAE is about 0.40, against about 0.20
  for a typical hour (seeds 42–44; hours 46 and 47 are about average). The block holdout puts 1.5%
  of its test targets at hour 0 and 3.5% at hours 46–47. So the off-grid Δ is compared with the
  block holdout's Δ over the same hours, 1–45 (`OffGridBlockEvalMasker.hours`), as well as with its
  Δ over all 48. A gain that holds on the grid over hours 1–45 and vanishes off it is specific to
  the grid; one that is smaller over hours 1–45 than over all 48 sits partly at the record's edges.
- `diffusion.score.span_mae(run_dir, dataset, holdout, hours)` gives a holdout's MAE over some hours
  alone, from the median imputation `diffusion.score` saved beside the run
  (`median_test_block_0.1.npy`), so it costs no scoring pass. It refuses a scoring that did not use
  the frozen holdout and sampler, and a `model.pt` that is no longer the weights scored.

M8-4 runs both checks through M6-2's driver, in the sweep's folder under the sweep's tag
([Running the paired sweep](#running-the-paired-sweep-m6-2)). The first line below trains Ext
1-blind in the block-aware regime at seeds 42–44 and scores it on the block holdout; the second
scores M7's block-aware base and Ext 1 runs at those seeds off the grid, training nothing (6
passes, each checked against its run's final weights, resumable and added to wandb); and the Python
gives the block holdout's MAE over hours 1–45 for one run:

```text
uv run python -m diffusion.sweep --tag sweep --out <sweep folder> --regime block-aware --variants ext1blind --seeds 42 43 44 --holdouts block
uv run python -m diffusion.sweep --tag sweep --out <sweep folder> --regime block-aware --variants base ext1 --seeds 42 43 44 --holdouts offgrid
```

```python
from data.physionet import load_dataset
from diffusion.run import DATASET_PATH
from diffusion.score import span_mae
from eval import OffGridBlockEvalMasker

span_mae("<sweep folder>/ext1_block-aware_seed42", load_dataset(DATASET_PATH), "block", OffGridBlockEvalMasker.from_config().hours)
```

What a triggered M8-4 needs besides compute, because of how `diffusion.results` pairs runs:

- **The same commit as M7's block-aware runs.** The store refuses a regime whose runs come from
  different commits, so Ext 1-blind is trained from the commit M7's runs were (check it out), or
  the whole block-aware regime is refused. In the sweep's folder, `diffusion.sweep` refuses any other
  commit itself (`sweep.json`).
- **A tag the store collects with them.** The store reads runs by wandb tag: give Ext 1-blind the
  sweep's tag (the driver refuses another in the sweep's folder), or add M8-4's tag to M7's
  block-aware runs at seeds 42–44. Ext 1-blind is scored on the block holdout only, so the store
  lists its `random/0.1` as unscored: expected, not a refusal.
- **M7's block-aware run folders at seeds 42–44, kept.** Off-grid scoring needs their `model.pt`, and
  `span_mae` their `median_test_block_0.1.npy`; the store holds neither.
- **The sweep's folder itself, shared.** Ext 1-blind trains into it beside M7's block-aware runs,
  so the driver checks its pairing against Ext 1's and base's as each cell ends, and the off-grid
  pass reads their weights from it. In a folder of its own, Ext 1-blind would be checked against
  nothing until the store is written. The folder is on the Google Drive of whoever ran M7: share
  it with whoever runs M8-4 as an editor, who adds it to their own Drive as a shortcut in My Drive
  so that Colab's `drive.mount` shows it, and passes that path as `--out`.

The short run (M6-3) trained Ext 1-blind at seed 42 for 1 epoch on Set A through `diffusion.run`
(MPS), then scored it through `diffusion.score` on the val split's block and off-grid holdouts,
with 2 draws instead of the frozen 100 (a scratch config). Training took 175 steps, all finite,
with no spikes. The loss fell from 0.68 in the first tenth to 0.32 in the last. Both holdouts hid
the same 12,886 values on the 400 val patients, as they should: the block holdout's MAE was 0.795
and the off-grid holdout's 0.799. Each pass took about 4.7 minutes. One epoch is a plumbing check,
not a result: the MAE is far above forward fill's 0.437.

Tests:

- `tests/test_ext1_blind.py`: the variant's toggles, name, run name and flags; blind without
  Ext 1 refused, by `run()` before it deletes anything; an older saved config not blind; on each
  of the training, validation and sampler paths, `mask_emb` identical for five different
  conditioning masks and equal to Ext 1's for an all-ones mask, while Ext 1's changes, and the same
  on the training path with dropout on; the global RNG state, so the dropout masks, equal to
  Ext 1's after every training step (and not base's); mask_emb reaching the denoiser once per call;
  starting weights and weight hashes equal to Ext 1's at seeds 42 and 43; the same first two losses
  as Ext 1 and different ones after the gate opens; Ext 1 with the toggle off pinned to the
  fingerprint from `6b8a298`; and a tiny run through `diffusion.run --variant ext1blind`, scored
  through `diffusion.score` on the block and off-grid holdouts.
- `tests/test_offgrid_block_eval_masker.py`: the frozen keys; `block_targets` at offset 0 equal
  to the grid, draw for draw; a hand-checked toy; per-patient counts equal to `BlockEvalMasker`'s;
  no target at hours 0, 46 or 47, and every window hidden whole or not at all apart from one
  trimmed window, with every window starting off a multiple of 3; the targets straddling the
  grid's slots; identical targets across run seeds, patient order and subsets; a golden
  fingerprint; an offset of 0 (the grid) refused; `hours` the span its windows cover; and Set A.
- `tests/test_score.py`: `--holdout offgrid` scores the off-grid targets under their own file
  names and settings, and only at 10%; `span_mae` over all hours equal to the scoring's MAE, over
  hours 1–45 equal to a hand computation (each box missing by its hour), the off-grid holdout's
  over its own hours equal to all of it, and a scoring with another holdout or sampler, or
  weights changed since, refused.
- `tests/test_variants.py`: `ext1blind` is in `VARIANTS`, trains through its flags and
  `--variant`, and runs forward, backward and sampling on Set A like the other four.

## Reconstructed trajectories (M7-3)

Five test patients by HR, serum sodium and GCS, imputed by Ext 1+2 (PROJECT_PLAN §3.5: the lowest
mean test MAE, not separated from base) against mean imputation and forward fill, on the frozen 10%
random holdout. The figure is
[`results/trajectories/trajectories.png`](results/trajectories/trajectories.png) (and `.pdf`). The
written read, for the final report, is
[`results/trajectories/observations.md`](results/trajectories/observations.md).

The sweep's weights are not in Git or wandb. Copy the variant's five run folders from Drive
(`MyDrive/msml612/sweep`) to `outputs/sweep`, keeping their names, then run the steps:

```text
uv run python -m diffusion.trajectories                       # results/trajectories/trajectories.npz and .json
uv run --group plots python -m diffusion.trajectories_plot    # results/trajectories/trajectories.png and .pdf
uv run python -m diffusion.trajectories --draw-check          # results/trajectories/draw_check.json
```

- **`diffusion.trajectories`** checks every seed's `model.pt` against its SHA-256 in
  `results/sweep_manifest.json`, and refuses to go on if one is missing or differs.
  - **Patients:** chosen by a fixed rule: among the test patients with at least one held-out value
    and at least two visible values in each variable (137 of 800), five evenly spaced in the test
    split's order.
  - **Draws:** 100 draws for those five from one seed's weights (`--seed 42`), with the frozen sampling
    seed. Each seed's protocol median (`median_test_random_0.1.npy`) is kept beside them.
  - **Test-split MAE:** per variable in clinical units over all 800 patients, for each seed, the two
    baselines, and linear interpolation between the visible values. Interpolation is not a protocol
    baseline and is not plotted: it is the simplest guess that reads both sides of a gap.
  - **Curves:** every curve keeps the visible values and takes the model's draws only in between. The
    sampler's output at a visible value is never scored and is not an imputation: there, it is off by
    hundreds to thousands of units.
  - **Variant:** `--variant` (default `ext1+2`, §3.5's pick) names the variant.
- **`diffusion.trajectories --draw-check`** writes `results/trajectories/draw_check.json` and leaves
  the five patients' data alone. It draws for all 137 eligible patients and records how many held-out
  values the 5–95% band covers, and how far the draws move over 1 to 12 hours against how far the
  charted values really move. It took about 31 minutes on an M4 Pro.
- **`diffusion.trajectories_plot`** needs only numpy and matplotlib, which the optional `plots`
  dependency group installs (`uv run --group plots ...`). Visible values are filled dots and held-out
  values open rings. The diffusion median is solid blue, with its 5–95% band and three single draws.
  Forward fill is orange steps, and mean imputation dashed aqua. Each method has a line style of its
  own as well as a colour.

Tests (`tests/test_trajectories.py`): weights that differ from the manifest's, or are missing,
refused; the patient rule, linear interpolation, band coverage, lag changes and draw correlation
worked out by hand; z-scores undone; and, on fake patients charted every hour, tiny ext1+2 runs at
two seeds through every step. They recompute each curve, panel score, test score and band count
independently, and check that the draws come from the chosen seed's weights with the frozen noise.
The figure test checks which data each curve and marker draws, and is skipped where matplotlib is
missing (`uv run --group plots pytest` includes it).

---

## Environment

The environment is managed with [uv](https://docs.astral.sh/uv/): dependencies are declared
in `pyproject.toml`, pinned in `uv.lock`, and reproduced from scratch with `uv sync --locked`
(M1-1). The project targets **Python 3.13** (`.python-version`), the version Google Colab
ships. Runtime dependencies are `torch`, `einops`, `wandb`, `numpy`, `pandas` and `pyyaml`,
with `pytest` in the dev group.

**Training runs on Google Colab's NVIDIA GPUs.** On Linux, `torch` is pinned to the PyTorch
`cu130` index (CUDA 13.0, matching Colab's driver), so the lock resolves `torch 2.14.1+cu130`
there. macOS and Windows resolve the plain PyPI wheel (CPU/MPS), so local tests still run with
no extra config.

**Every run in a paired comparison trains and is scored on one GPU model: Colab's G4 runtime,
an `NVIDIA A100-SXM4-80GB`, with TF32 on** (PROJECT_PLAN §3.3), picked by M6-2's timing test
([below](#the-timing-test-m6-2)). The rule was settled at the M5-3 review. It covers the M6 dry
run, the Week-7 sweep, and the Week-8 ablation and attribution checks: every variant, regime and
seed. Deterministic kernels ([Training](#training)) make reruns bit-identical only on one kind of
GPU. On another, float rounding differs from the first optimizer step: base at seed 42, with the
same code and learning rate, gave the same loss on a T4 and on an M4 Pro at step 0, and losses
0.002 apart by step 10. So a seed whose base ran on one GPU and whose Ext 1 ran on another is not
paired. When the chosen GPU is not free, wait for one; never take another GPU. That includes
Colab's A100 runtime: its 40 GB card is the same chip, but nothing has checked that it rounds the
same, so the guard treats it as another GPU.

Until the test, the T4 was the default: it is on every Colab tier, the plan's budget was timed on
it, and it has no TF32. It is also Colab's slowest GPU: the test puts the Week-7 sweep at 322 h
on it, against 67.5 h on the pick. An A100, L4 or H100 runs the denoiser's convolutions in TF32
unless told not to (PyTorch's default: `torch.backends.cudnn.conv.fp32_precision` is `"tf32"` in
torch 2.14.1). `protocol.tf32` tells it, the same for every run, in training and scoring; the
pick turns it on ([The timing test](#the-timing-test-m6-2) says why). Josh, without student Colab
Pro, can get only the T4, so M8-4's runs need an account with Colab Pro.

Check the GPU before launching:

```text
!nvidia-smi --query-gpu=name --format=csv,noheader
```

It should print `NVIDIA A100-SXM4-80GB`, the name the guard compares, not the runtime menu's
"G4". wandb records each run's GPU in
its metadata, and `diffusion.health` copies it into its record. Since M6-2 every run and every
scoring also records its GPU itself, and a paired run refuses to start on any GPU but
`protocol.paired_gpu` ([Paired sweep checks](#paired-sweep-checks-m6-2)). Health checks and debugging can run on any machine, but their
timings compare only with runs on the same machine
([Health runs](#health-runs-m5-3)).

Colab notebook cells run Colab's own Python, not this environment, so launch project code
with `uv run` from a shell cell:

```text
!git clone https://github.com/jkelle11-source/msml612_project.git
%cd msml612_project
!uv sync --locked
!uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The last line should print `2.14.1+cu130 13.0 True`. Run the tests anywhere with
`uv run pytest`; the tests that need the raw Set A files skip themselves when the data has
not been downloaded.

### The timing test (M6-2)

On each GPU everyone who runs a paired run can get, after the setup above:

```text
!uv run python -m diffusion.timing           # results/timing/<GPU>.json, e.g. tesla_t4.json
!uv run python -m diffusion.timing --tf32    # an L4 or A100 only: the same with TF32 on
```

`diffusion/timing.py` trains base and Ext 1+2 for 2 epochs at seed 42 through `diffusion.run`, as
a sweep run trains (deterministic kernels, pairing hashes, validation), with wandb off, into
`outputs/timing/`. It reads each run's second epoch and its validation pass from `curve.json`,
then times the first 2 of the test split's 50 batches of a frozen scoring pass (100 draws, batch
16) on that run's weights. The second epoch and the second batch are the clean ones: the first
epoch also builds the model, and the first batch warms the GPU up. The validation pass counts
because a 200-epoch run makes 20 of them. Nothing is scored or sent to wandb.

Base and Ext 1+2 are the sweep's cheapest and dearest variants. Ext 2's encoder adds about
nothing to base, and Ext 1's is most of what Ext 1+2 adds, so the estimates cost ext2's runs at
base's timings and ext1's (and ext1blind's) at Ext 1+2's (a slight overestimate). The cells costed
are the config's `sweep.regimes`, the driver's ([Running the paired sweep](#running-the-paired-sweep-m6-2)).
The record holds the GPU's exact name, the TF32 setting, the timings, and two estimates, with the
counts they assume:

- the Week-7 sweep: 30 runs of 200 epochs with 20 validation passes each, every run scored under
  both holdouts, and base's 50% and 90% passes at 5 seeds: 70 scoring passes in all;
- the dry run: the same 12 cells at 20 epochs and 2 seeds, scored as the sweep is, since the
  results store refuses any scoring but the frozen one, so its 24 passes cost a sweep pass each.

To pick:

- Compare the candidates' sweep hours, with TF32 off and, on an L4 or A100, on.
- Note how often Colab gave you each GPU, and its compute units per hour (Colab's Resources
  panel). Colab Pro's compute units are a monthly allowance per account, so compare the sweep's
  total units (hours × units per hour) with what the accounts running it have left, not only
  the rate.
- Keep the T4 unless another is clearly faster, reliably free and affordable in units.
- Write the pick into `protocol.paired_gpu` exactly as its record's `gpu` says (`Tesla T4`,
  `NVIDIA L4`, `NVIDIA A100-SXM4-80GB`), not as Colab's runtime menu names it. The guard
  compares the string, so the 40 GB and 80 GB A100s are two GPUs to it. Set `protocol.tf32` to
  the setting the pick was chosen with.
- Fill in the table below, commit the records in `results/timing/`, and write the pick into
  PROJECT_PLAN §3.3, all before the dry run.

The records in `results/timing/`, timed on Colab Pro in November 2026 (the sweep's hours split
into the scoring part and the rest):

| GPU (its record's `gpu`) | Colab runtime | TF32 | Sweep h | of which scoring h | Dry run h |
| --- | --- | --- | --- | --- | --- |
| `Tesla T4` | T4 | none | 321.9 | 222.7 | 80.4 |
| `NVIDIA L4` | L4 | off | 136.5 | 87.7 | 32.0 |
| `NVIDIA L4` | L4 | on | 116.9 | 71.2 | 26.3 |
| `NVIDIA A100-SXM4-40GB` | A100 | off | 88.5 | 49.7 | 18.6 |
| `NVIDIA A100-SXM4-40GB` | A100 | on | 65.9 | 44.1 | 16.0 |
| `NVIDIA A100-SXM4-80GB` | G4 | off | 88.0 | 49.7 | 18.6 |
| **`NVIDIA A100-SXM4-80GB`** | **G4** | **on** | **67.5** | **45.3** | **16.4** |

Compute units: the hourly rates were not read from the Resources panel. Each runtime's
session, both timing runs plus setup, cost about 3 units for the T4 and L4 together, 5 for the
A100 runtime, and 2.5 for G4. The A100 and G4 runtimes were each free on the first try.

**The pick: Colab's G4 runtime, `NVIDIA A100-SXM4-80GB`, with TF32 on**
(`protocol.paired_gpu: "NVIDIA A100-SXM4-80GB"`, `protocol.tf32: true`).

- The A100s are the fastest. The 40 GB and 80 GB cards are the same chip and ran the sweep in the
  same time (88.5 h and 88.0 h with TF32 off): the extra memory does nothing for this model.
  G4's session cost half the A100 runtime's units for the same timed work (about 12 minutes of
  timing each).
- TF32 takes another 20.5 h off (88.0 h to 67.5 h). It roughly halves training (28.7 h to
  15.2 h) but takes only about a tenth off scoring (49.7 h to 45.3 h). Scoring is two-thirds of
  the sweep on this GPU, and a bigger GPU barely speeds it up: the scorer runs each patient's 100
  draws one after another at batch 16, calls too small to fill an A100.
- Runs stay paired with TF32 on: every run computes the same way on the same GPU with
  deterministic kernels, and M6-2's rerun check (a rerun cell reproduces its numbers) runs with
  TF32 on. The cost
  is precision: TF32 rounds the inputs of convolutions and matrix multiplies to a 10-bit
  mantissa, as PyTorch already does by default for convolutions on these GPUs. Every Δ is taken
  against the sweep's own base, trained the same way.
- The T4 would take 322 h, more than the plan's estimate of about 270 h, which came from scoring
  timed at batch 64.

Tests:

- `tests/test_timing.py`: both estimates against a hand count; a timing run trains 2 epochs of
  each variant, draws exactly 2 batches of each one's scoring pass, writes no scores, and
  records its timings from the runs' `curve.json`; with `--tf32` the runs and the scoring
  batches compute in TF32; a record is named for its GPU and TF32 setting.
- `tests/test_train.py` and `tests/test_score.py`: training and scoring compute float32 as
  `protocol.tf32` says (a config without it: float32), and a scoring records it.

---

## Dataset

**Source and license.** [PhysioNet/CinC Challenge 2012, v1.0.0](https://physionet.org/content/challenge-2012/1.0.0/)
(Silva, Moody, Mark and Celi), under the
[Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/). The five
patients' values in `results/trajectories/` are derived from it. Cite the challenge paper (Silva,
Moody, Scott, Celi and Mark, *Predicting In-Hospital Mortality of Patients in ICU: The
PhysioNet/Computing in Cardiology Challenge 2012*, Computing in Cardiology, 2012) and PhysioNet's
standard citation (Pollard et al., *PhysioNet as a global platform for biomedical research*, Nature
Health 1(8), 792–795, 2026, [doi:10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z)).

The raw Set A files and the processed dataset are git-ignored, so each machine builds its own
copy (M1-1). Download PhysioNet/CinC Challenge 2012 Set A and extract it so the record files
sit at `data/raw/set-a/<RecordID>.txt`, then run from the repo root:

```text
uv run python -m data.physionet
```

This writes `data/processed/physionet_set_a.npz` (about 10 MB) in a few seconds. Load it with
`data.physionet.load_dataset`, which returns a dictionary of ten arrays: `values`, `mask`,
`delta_obs`, `record_ids`, `features`, `train_idx`, `val_idx`, `test_idx`, `mean` and `std`.
`values` is z-scored with training-split statistics and is 0 where `mask` is 0. The split is
70/10/20 by patient, fixed by `data.split_seed` in [`configs/config.yaml`](configs/config.yaml).
`delta_obs` is computed from the full observation mask; `data.physionet.compute_tslo`
recomputes it for any other mask. `delta_obs` never reaches the model, because it still
counts the held-out targets as observed: Extension 2's δ is recomputed from the conditioning
mask (see the [input-hook contract](#denoiser-input-hook-contract)).

**Known bad values, kept as recorded.** Raw Set A contains data-entry errors, and
`data.physionet` keeps them, as upstream CSDI's loader does, so our numbers rest on the same
data as the published 0.217 (decided at the M2-3 review):

- pH: 10 of 23,437 values lie outside 6.5–8.0. They are 733, 734 and 735 (a dropped decimal
  point); 94, 95, 96 and 100 (apparently SaO2 logged under pH); 187, which is record 141680's
  PaO2 (at hour 8 its pH and PaO2 are recorded in each other's place); and 1 and 3.
- Temp: 73 of 71,213 values lie below 25 °C, most of them −17.8.
- K: one value of 22.9; the next highest is 15.6.
- PaO2: 2 of 22,397 values lie below 20 mmHg: a 0, and the 7.47 that is record 141680's pH. The
  next lowest is 20. Both are in test patients.

What this does to our numbers:

- **pH and Temp count for less than they should in pooled MAE.** The bad values widen the
  training-split scaling: pH's std is 1.395 instead of 0.077 without them, and Temp's 1.675
  instead of 0.843. Every pH error therefore shrinks about 18× in standardized units and every
  Temp error about 2×. Upstream's published numbers come from the same uncleaned data. Upstream
  scales with all patients, where pH's std is 8.385.
- **pH's and Temp's per-variable MAE (M9-2) can't be trusted,** in raw or standardized units.
  One bad value, scored or carried into a scored value, can dominate it, and every model sees the
  two features compressed. Forward fill's pH MAE on the 10% test holdout, for example, is 1.20
  standardized units, almost all of it one carried-forward pH of 735.
- **RMSE is inflated wherever a bad value is scored or carried forward** (see the footnote under
  [Baselines](#baselines)). MAE is the headline metric.
- **A visible bad pH can spread into other variables' guesses** in a model that predicts across
  features at the same hour. BRITS misses PaCO2 and PaO2 values next to a pH of 735 or 187 by up
  to 6,796 standardized units (the ‡ footnote under
  [Learned baseline: BRITS](#learned-baseline-brits-m3-3)). CSDI's feature attention does not
  (M4-1), but its temporal attention can carry a visible bad pH into the same patient's held-out
  pH values: seed 42 misses a pH of 7.33 by 153 standardized units three hours after a visible
  735 (see [The reproduction gate](#the-reproduction-gate-m4-1)).

---

## Baselines

Mean imputation and forward fill (`eval/baselines.py`, M2-3) are the floor every model we
train must beat (PROJECT_PLAN §6). Both are scored like any model, on the frozen holdouts'
held-out values (`EvalMasker`'s random holdout and `BlockEvalMasker`'s block holdout, M2-4), in
standardized units. Import them from `eval.baselines`: the `eval` package leaves
that module unimported, as `data` leaves `data.physionet`, so `python -m eval.baselines` runs it
only once.

- `mean_imputation(values)` guesses each variable's training-split mean, which is 0 in
  standardized units, everywhere. It never looks at the patient.
- `forward_fill(values, cond_mask)` carries each variable's last visible value forward. It
  reads only boxes where `cond_mask == 1`, so held-out values can't leak in. **The rule for the
  hours before a variable's first visible value is backward fill:** they take that first value.
  Filling them with the training mean instead scores worse (MAE 0.460 on val and 0.470 on test at
  10%). A variable the patient never shows gets 0.

MAE and RMSE in standardized units; val has 400 patients and test 800:

| Holdout | Split | Mean imp. MAE | Mean imp. RMSE | Forward fill MAE | Forward fill RMSE |
| --- | --- | --- | --- | --- | --- |
| Random 10% (headline) | val | 0.6956 | 0.9537 | 0.4366 | 0.7409 |
| Random 10% (headline) | test | 0.7053 | 0.9906 | 0.4487 | 3.3251 * |
| Random 50% | val | 0.7003 | 0.9651 | 0.4844 | 0.8197 |
| Random 50% | test | 0.7132 | 2.2870 * | 0.4881 | 2.6419 * |
| Random 90% | val | 0.7000 | 0.9641 | 0.5923 | 0.9084 |
| Random 90% | test | 0.7108 | 2.1178 * | 0.5908 | 2.0913 * |
| Block 10% | val | 0.6895 | 0.9530 | 0.4749 | 0.8105 |
| Block 10% | test | 0.7338 † | 3.3892 † | 0.4786 † | 3.3191 † |

\* Inflated by the [known bad values](#dataset). On the random holdout at 10% none of them is
scored, but a pH of 735 is visible in one test patient and forward fill carries it into a scored
pH value. That one error is 521.5 standardized units; without it, forward fill's test RMSE would
be 0.7798 instead of 3.3251 and its MAE 0.4287 instead of 0.4487. At 50% and 90% some bad values
are scored themselves (11 and 14 on test, 3 and 4 on val), which inflates both baselines' test
RMSE.

† Inflated by the known bad values, MAE as well as RMSE. On the block holdout four of them are
test targets themselves: a pH of 733, a pH of 1, a Temp of 0 and a PaO2 of 7.47. The pH of 733
dominates: it is about 520 standardized units from any plausible pH, so every model misses it by
about that much, which adds about 0.02 to every model's block test MAE. Differences between
models barely change, but block test RMSE is mostly that one value. Without it, mean
imputation's test MAE and RMSE would be 0.7139 and 1.0477, and forward fill's 0.4587 and 0.7910.
No bad value is a block val target.

Forward fill's 10% validation MAE, **0.437**, is M3-1's "below forward fill" bar, because M3-1
tracks validation MAE. Its block validation MAE, **0.475**, is the matching bar for runs validated
on the block holdout (M6-2's block-aware regime, PROJECT_PLAN §3.1). Forward fill does worse on
blocks because a whole 3-hour window of every feature is hidden at once, so the value it carries
in is older. The 50% and 90% rows are CSDI's robustness rates, each its own frozen holdout, and
the block rows hide the same number of values per patient as the random 10% rows (see
[Frozen protocol constants](#frozen-protocol-constants)).

The same numbers are recorded, unrounded, in [`results/baselines.json`](results/baselines.json),
keyed `random` → `fractions` → `"0.1"` / `"0.5"` / `"0.9"` → `val` / `test`, and
`block` → `fractions` → `"0.1"` → `val` / `test`. Each holdout's frozen seed is stored next to its
fractions (`eval_masker_seed`; `block_eval_masker_seed` and `block_window_hours`). To regenerate
the file, run from the repo root:

```text
uv run python -m eval.baselines
```

`tests/test_baselines.py` fails when the recorded file no longer matches a fresh run, so rerun
it, and update this table, whenever the baselines, the data pipeline or the protocol change.
Add `--wandb online` to also log each baseline as a wandb run (`mean_imputation`,
`forward_fill`; job type `baseline`; M3-3), with its scores as the run summary under
`<holdout>/<fraction>/<split>/<metric>` (e.g. `random/0.1/test/mae`), plus `val/mae` and
`val/rmse` for the random 10% val split, the key CSDI runs track. Logging is opt-in, so
regenerating the file doesn't post duplicate runs.

`score_holdout` and `baseline_results` take an optional `guessers` dict,
`name → guess(values, cond_mask)`, defaulting to the two baselines (`BASELINES`). Any model
scored through them gets exactly this layout and these holdouts; the learned baseline below is.

### Learned baseline: BRITS (M3-3)

**This is BRITS, not the bi-GRU fallback.** The fallback was only for the case where BRITS
couldn't be brought into our harness within the week (M3-3), and it could.
[`models/brits.py`](models/brits.py) implements Cao et al. (2018) in plain PyTorch, matching the
original code ([caow13/BRITS](https://github.com/caow13/BRITS)) where it differs from the
paper, because CSDI's published BRITS numbers come from that code. There are three differences:

- the original code has **no sigmoid** on the mixing weight β, where the paper's eq. 8 has one;
- the imputation loss is **summed over the 48 hours and weighted 0.3**, its README's recommended
  `impute_weight`;
- the decay layers are initialized in ±1/√outputs.

**The one place we follow the paper instead: the time gaps.** The original's `input_process.py`
gives each hour the paper's gap for the hour after it (it reads that hour's own mask and starts at
1, not 0), and gives the backward direction the forward direction's gaps, because it reverses the
masks twice. That second part is a known bug,
[caow13/BRITS#12](https://github.com/caow13/BRITS/issues/12), open since 2021. We use the paper's
eq. 2 in both directions rather than copy it. It changes little: retrained with the original's
gaps, bug included, seed 42 scores 0.2708 instead of 0.2670 on the random 10% test targets, 0.4182
instead of 0.4326 at 50%, and 0.4676 instead of 0.4744 on the block holdout.

**One difference we can't match:** the original also trains a mortality classifier alongside
(`label_weight` 1.0), and that needs PhysioNet's outcome labels, which our pipeline doesn't load.
Ours is imputation only.

**How it works.** Each direction reads a patient's 48 hours with an LSTM and, at every hour,
guesses every variable *before* reading it: from its memory, decayed by how long ago each
variable was last seen, and from the other variables at that hour (a feature regression that
never sees the variable itself), mixed by learned weights. The two directions' guesses are
averaged, with a consistency loss between them.

**What it sees.** Training patients: their full mask. The loss is on visible values only, and
each is guessed before it's read, so copying is impossible. Val and test patients: only
`values` where the holdout's `cond_mask` is 1; held-out boxes are zeroed before the model runs.
The time gaps it decays by are computed from that same mask. The dataset's `delta_obs` is never
used, because it comes from the full mask and would tell the model when the held-out values
were measured.

**Training.**

- **Settings:** hidden size 64, Adam at 1e-3 and consistency weight 0.1, as in the original
  code. Batch 64 is from its README's example; its code's default is 32.
- **Device:** CPU, which is about twice as fast as MPS for its small sequential steps.
- **Which weights are kept:** after every epoch the val split's random 10% targets are scored,
  and the weights with the lowest val MAE are kept. This is ours: the original trains a fixed
  1000 epochs and keeps the last weights. Training stops after 20 epochs without improvement,
  with the original's 1000 epochs as the cap. The test split never chooses anything. BRITS is
  the one model picked this way; CSDI runs keep their final weights (PROJECT_PLAN §3.1).
- **Why not the original's rule:** kept to 1000 epochs, BRITS overfits on our split. On seed
  42, val MAE is lowest at epoch 321 (0.2715) and climbs to 0.2905 by epoch 999. Test MAE at
  random 10% rises from 0.2670 to 0.2984, above the published 0.284, and block test MAE from
  0.4744 to 0.5013, worse than forward fill's 0.4786. The original trains on every patient,
  with the held-out values removed, and scores those same patients; we train on the training
  split and score unseen test patients, as CSDI does.
- **What that means for the table below:** the val rows are the numbers the weights were
  picked on, so they flatter BRITS a little; the test rows don't. The block rows use weights
  picked on the random holdout, the same carry-over check as CSDI's random-regime block scores.
- **The 50% and 90% rows:** the original code removes the held-out values from its training
  inputs too, so run at 50% or 90% it trains at that rate. Ours trains once, on all of each
  training patient's values, and that one model is scored at all three rates.
- **Seeds:** 42–46, one model each, reproducible bit for bit.

To train, score and record all five, run from the repo root:

```text
uv run python -m models.brits --seeds 42 43 44 45 46
```

On an Apple M-series CPU a seed takes 5–10 minutes (best epochs 147–321). Each seed is a wandb
run, `brits_seed<seed>` (job type `baseline`), with `brits/loss` and `brits/val_mae` per epoch,
against `epoch`, and its scores as the summary, in the keys above. The per-epoch keys are kept
under `brits/` so they never share a panel with CSDI's `epoch/loss_mean`, a different loss
logged against training steps. The record is
[`results/learned_baseline.json`](results/learned_baseline.json): the model's name, `fallback`,
the settings, each seed's best epoch and scores (`per_seed`, in `baselines.json`'s layout under
`brits`), and their mean ± SE (`mean_se`; SE is the sample standard deviation over the 5 seeds
over √5). It is kept apart from `baselines.json` because that file is regenerated from scratch
by its test in seconds, and BRITS takes about an hour to train.

BRITS's MAE and RMSE, mean ± SE over seeds 42–46, standardized units, with forward fill's MAE
from the table above and BRITS as published (cited, CSDI's paper, PROJECT_PLAN §6):

| Holdout | Split | BRITS MAE | BRITS RMSE | Forward fill MAE | Published BRITS MAE |
| --- | --- | --- | --- | --- | --- |
| Random 10% (headline) | val | 0.2705 ± 0.0008 | 0.5465 ± 0.0013 | 0.4366 | |
| Random 10% (headline) | test | **0.2667 ± 0.0010** | 0.6855 ± 0.0419 | 0.4487 | 0.284 |
| Random 50% | val | 0.3790 ± 0.0006 | 0.6669 ± 0.0017 | 0.4844 | |
| Random 50% | test | 0.4206 ± 0.0037 ‡ | 15.2511 ± 1.1929 * ‡ | 0.4881 | 0.368 |
| Random 90% | val | 0.5386 ± 0.0018 | 0.8149 ± 0.0016 | 0.5923 | |
| Random 90% | test | 0.5382 ± 0.0020 | 2.0481 ± 0.0007 * | 0.5908 | 0.517 |
| Block 10% | val | 0.3942 ± 0.0009 | 0.6635 ± 0.0015 | 0.4749 | |
| Block 10% | test | 0.4692 ± 0.0023 † ‡ | 9.3472 ± 0.4342 † ‡ | 0.4786 | — |

Our ± is smaller than the published one for a reason that has nothing to do with the model:
CSDI's five runs each used a different fifth of the patients as test data, so their SE includes
which patients were tested. Ours uses one fixed test split, so it only reflects training
randomness.

\* and † as under the table above: bad values scored themselves.

‡ **A visible bad pH spreads into other variables' guesses.** Forward fill only carries a value
along its own variable. BRITS's feature regression predicts each variable from the others at the
same hour, and it has learned the Henderson–Hasselbalch link between pH, PaCO2 and HCO3: in seed
42's model the PaCO2 row weights pH by −8.5 forwards and −9.4 backwards, and the next largest
weight is HCO3's, +0.8. The pH weight is that large because the bad values widen pH's scaling
(Dataset, above): a normal swing of 0.1 in pH is only 0.07 standardized units. So a visible pH
hundreds of standardized units out drags a scored PaCO2 value at that hour with it, and PaO2 too:

- Random 50% test: record 140525 shows a pH of 735 (521 standardized units) at hour 22, where its
  PaCO2 of 45 is held out. Seed 42 misses that PaCO2 by 6,796 standardized units. That one value
  is almost all of the RMSE. Leaving out the test patients who show a bad value, seed 42's MAE
  would be 0.3767 instead of 0.4326 (close to the published 0.368), and its RMSE 1.63 instead
  of 18.85. What's left of the RMSE is the bad pH values scored themselves (\*).
- Block test: record 141680 shows a pH of 187 (129 standardized units; really its PaO2, see
  Dataset) at hour 8, where its PaCO2
  of 29 is in a held-out window. Seed 42 misses it by 1,595. Without the patients who show a bad
  value, seed 42's block test MAE would be 0.4125 instead of 0.4744, and its RMSE 3.32 instead of
  10.42, the rest being the pH of 733 scored itself (†).
- Random 10% test: the same pH of 735 sits next to record 140525's held-out PaO2 of 54 at hour
  22. Seed 42 guesses 5,204 mmHg, a miss of 60 standardized units and 27% of its squared error
  on these targets. MAE hardly moves: without the patients who show a bad value, seed 42's test
  MAE would be 0.2635 instead of 0.2670, so the headline row is barely touched.

**The paper's sigmoid on β wouldn't prevent this.** β only mixes the feature regression's guess
with the history guess, so a sigmoid that keeps it between 0 and 1 still lets a huge
feature-regression guess through. Retrained with only the sigmoid added, seed 42 misses the two
PaCO2 values above by 5,080 and 1,254 instead of 6,796 and 1,595, and the PaO2 at 10% by 109
instead of 60. The first version of this model (commit `eaf18d3`) had the sigmoid, averaged the
loss over the hours and didn't weight it by 0.3. It missed the two PaCO2 values by 3,936 and 975,
and scored 0.4162 ± 0.0027 at random 50% and 0.4602 ± 0.0038 on the block holdout. We report the
original code's version because CSDI's published numbers come from it.

Any model that predicts across features at the same hour can be exposed the same way, so PaCO2
and PaO2 join pH and Temp in M9-2's list of per-variable MAEs to flag. CSDI's feature attention
turned out not to be (M4-1, [The reproduction gate](#the-reproduction-gate-m4-1)); it carries a
bad pH along pH instead.

Tests:

- `tests/test_baselines.py`: hand-worked fixtures; forward fill checked against a plain-loop
  reference; no leakage from held-out values; the scored values exactly `EvalMasker`'s held-out
  targets; each rate holding out its own fraction; pinned Set A test MAEs at 10%, 50% and 90%;
  the recorded file matching a fresh run; importing `eval` leaving `eval.baselines` unimported.
  For the block holdout (M2-4): its entry sitting next to `random` with its seed and window, at
  10% only; the same target count as the random 10% holdout; scores taken on exactly
  `BlockEvalMasker`'s targets; pinned Set A test MAEs; the four bad values it scores on test.
  For M3-3: `guessers` defaulting to the two baselines; a guesser shown only the conditioning
  mask and scored into the same layout; each baseline logged to wandb (fake wandb) with its
  flattened scores as the summary.
- `tests/test_brits.py` (M3-3): the time gaps worked by hand, forwards and backwards, and taken
  from the mask BRITS is given; output shape, visible values returned unchanged; hidden values
  never read (×100, +5, NaN); a variable never guessed from itself; patients independent;
  training lowering the loss and keeping the best val weights; stopping after `patience`
  epochs and returning the best epoch's weights, not the last; same seed same weights,
  different seed different; scores in `baselines.json`'s
  layout; mean ± SE by hand; the command line writing the record and logging each seed with no
  per-epoch key that would overwrite a summary key or share a key with CSDI's; the recorded
  BRITS beating forward fill on val at 10%, 50% and 90%; the recorded BRITS scored on the same
  split and frozen holdouts as `baselines.json` (seeds, window, fractions, patient and target
  counts), with its mean ± SE matching its per-seed scores.
- `tests/test_sampler_real_batch.py`: the reused reverse sampler on real Set A patients (M2-3).
  An untrained model runs upstream's full 50-step schedule, with `EvalMasker`'s conditioning
  mask passed as `gt_mask`. The samples come back as `(K, B, 48, D)` with no NaNs, the K draws
  differ, and the held-out values can't reach them.

---

## References

Primary references are listed in [`Docs/PROJECT_PLAN.md`](Docs/PROJECT_PLAN.md) §9
(CSDI, Time2Vec, BRITS, GRU-D, DDPM, conformal prediction, and the dataset).
