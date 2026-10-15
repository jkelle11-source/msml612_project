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
diffusion/     # training data feed, loop and random streams around the
               #   reused diffusion process; no schedule or q_sample of our
               #   own.                                                     [M2-1]
eval/          # EvalMasker / TrainMasker + masked, standardized metrics;
               #   EvalSampler (frozen-noise evaluation pass).              [M1-3]
               #   Mean and forward-fill baselines.                         [M2-3]
results/       # recorded results (baselines.json).                         [M2-3]
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

The evaluation sampler is frozen the same way: `eval.EvalSampler` reads
`protocol.eval_sampling_seed`, `protocol.eval_n_samples` and `protocol.eval_batch_size`
(see [Random streams](#random-streams-seed-pairing)).

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
  each enabled extension's module into the model, after the parts every variant shares (see
  [Random streams](#random-streams-seed-pairing)). Until an extension exists, turning its
  toggle on raises `NotImplementedError` instead of silently training the base model.
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
  are, which is the leak PROJECT_PLAN §3.1 rules out. For Extension 2 this means TSLO is
  recomputed from the conditioning mask for every batch, not read from `delta_obs`.

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

Nothing computes a hook yet, so `forward` and `evaluate` pass none and the model is the base.
An extension (M4-2, M5-2) adds its module in `build_model`, computes its hook from
`cond_mask` inside `forward` and `evaluate`, and hands it to `calc_loss` or `impute`; their
signatures already accept it. The same hook is used at every reverse step, since it depends
on the conditioning mask and not on the diffusion step.

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
| `streams.train_masker` | conditioning/target split of each batch | `cond, target = streams.train_masker(mask)`; `cond` goes in `batch["gt_mask"]` |
| `streams.diffusion` | diffusion timestep and noise | `model(batch, generator=streams.diffusion)` |
| global RNG | model init, dropout | `streams.seed_model_init()`, `streams.start_epoch(epoch)` |

Rules:

- Call `streams.seed_model_init()` right before building the model, and build the parts every
  variant shares before any extension module, so shared weights start identical.
- Call `streams.start_epoch(epoch)` at the top of every epoch, after the model is built. It
  re-seeds all four streams in place, so a DataLoader built once keeps working.
- **Dropout is not paired across variants.** It draws from the global RNG, and a variant with
  extra dropout layers consumes more of it. Shuffling, masks, timesteps and noise stay
  identical regardless.
- **Resuming needs no RNG state.** Because every stream is a function of the run seed and the
  epoch, a checkpoint taken at an epoch boundary needs only the model, the optimizer (plus
  any scheduler and early-stopping counters) and the epoch number; the resumed run then ends
  with the same weights as an uninterrupted one. A run resumed mid-epoch does not.
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
- Steps the learning-rate schedule once per epoch (×0.1 at 75% and 90% of `train.epochs`), then
  calls `on_epoch_end(epoch, model)` if one is given. M3 attaches validation there. Nothing the
  hook does can shift training, because `start_epoch` re-seeds every stream at the top of the
  next epoch and `train` puts the model back in train mode.
- `history` has one record per step, `{"epoch", "loss", "grad_norm", "lr"}`, as plain numbers.

On the CPU, the same config, dataset and run seed give bit-identical weights and history (tested).
Some GPU kernels are nondeterministic (the backward pass of an embedding layer, for one), so GPU
reruns may differ slightly. Still to come in M3: a command-line entry point, validation MAE,
checkpointing, and wandb logging.

Tests:

- `tests/test_build_model.py`: config → upstream's dict, defaults equal upstream's published
  values, extension toggles raise, no hidden seeding, weights on the requested device.
- `tests/test_diffusion_data.py`: training patients only and each exactly once, values and masks
  shuffled together, the shuffle a function of run seed and epoch, `timepoints` as in
  `EvalSampler`.
- `tests/test_train.py`: one step on a real batch, upstream's masking never runs, the loss
  scores exactly `TrainMasker`'s targets, every step replayed from scratch at `lr = 0`,
  same-seed reproducibility, clipping, the learning-rate schedule, the epoch hook.
- `tests/test_forward_noising.py`: the schedule check described under
  [Reused CSDI implementation](#reused-csdi-implementation).

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

---

## Dataset

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
recomputes it for any other mask.

**Known bad values, kept as recorded.** Raw Set A contains data-entry errors, and
`data.physionet` keeps them, as upstream CSDI's loader does, so our numbers rest on the same
data as the published 0.217 (decided at the M2-3 review):

- pH: 10 of 23,437 values lie outside 6.5–8.0. They are 733, 734 and 735 (a dropped decimal
  point); 94, 95, 96 and 100 (apparently SaO2 logged under pH); and 187, 1 and 3.
- Temp: 73 of 71,213 values lie below 25 °C, most of them −17.8.
- K: one value of 22.9; the next highest is 15.6.

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

---

## Baselines

Mean imputation and forward fill (`eval/baselines.py`, M2-3) are the floor every model we
train must beat (PROJECT_PLAN §6). Both are scored like any model, on `EvalMasker`'s held-out
values, in standardized units:

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
| 10% (headline) | val | 0.6956 | 0.9537 | 0.4366 | 0.7409 |
| 10% (headline) | test | 0.7053 | 0.9906 | 0.4487 | 3.3251 * |
| 50% | val | 0.7003 | 0.9651 | 0.4844 | 0.8197 |
| 50% | test | 0.7132 | 2.2870 * | 0.4881 | 2.6419 * |
| 90% | val | 0.7000 | 0.9641 | 0.5923 | 0.9084 |
| 90% | test | 0.7108 | 2.1178 * | 0.5908 | 2.0913 * |

\* Inflated by the [known bad values](#dataset). At 10% none of them is scored, but a pH of 735
is visible in one test patient and forward fill carries it into a scored pH value. That one
error is 521.5 standardized units; without it, forward fill's test RMSE would be 0.7798
instead of 3.3251 and its MAE 0.4287 instead of 0.4487. At 50% and 90% some bad values are
scored themselves (9 and 12 on test, 3 and 4 on val), which inflates both baselines' test RMSE.

Forward fill's 10% validation MAE, **0.437**, is M3-1's "below forward fill" bar, because M3-1
tracks validation MAE. The 50% and 90% rows are CSDI's robustness rates, each its own frozen
holdout (see [Frozen protocol constants](#frozen-protocol-constants)).

The same numbers are recorded, unrounded, in [`results/baselines.json`](results/baselines.json),
keyed `random` → `fractions` → `"0.1"` / `"0.5"` / `"0.9"` → `val` / `test`. The block holdout
(M2-4) is recorded next to `random`, and M3-3 logs the baselines to wandb. To regenerate the
file, run from the repo root:

```text
uv run python -m eval.baselines
```

`tests/test_baselines.py` fails when the recorded file no longer matches a fresh run, so rerun
it, and update this table, whenever the baselines, the data pipeline or the protocol change.

Tests:

- `tests/test_baselines.py`: hand-worked fixtures; forward fill checked against a plain-loop
  reference; no leakage from held-out values; the scored values exactly `EvalMasker`'s held-out
  targets; each rate holding out its own fraction; pinned Set A test MAEs at 10%, 50% and 90%;
  the recorded file matching a fresh run.
- `tests/test_sampler_real_batch.py`: the reused reverse sampler on real Set A patients (M2-3).
  An untrained model runs upstream's full 50-step schedule, with `EvalMasker`'s conditioning
  mask passed as `gt_mask`. The samples come back as `(K, B, 48, D)` with no NaNs, the K draws
  differ, and the held-out values can't reach them.

---

## References

Primary references are listed in [`Docs/PROJECT_PLAN.md`](Docs/PROJECT_PLAN.md) §9
(CSDI, Time2Vec, BRITS, GRU-D, DDPM, conformal prediction, and the dataset).
