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
  the learning-rate scheduler) and the epoch number; the resumed run then ends
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
and wandb (M3-2) does.

### Launching a run (M3-1)

```text
uv run python -m diffusion.run --seed 42
```

`diffusion/run.py` trains one model from `configs/config.yaml` and a run seed, scores the
validation split as it goes, and writes to `outputs/<run name>/` (git-ignored), e.g.
`outputs/base_random_seed42/`, the run's name on wandb
([Experiment tracking](#experiment-tracking-m3-2)). Options: `--device` (`cuda`, `mps`, `cpu`,
default the first available), `--epochs` to override `train.epochs`, `--validate-every` to
override `validation.every_n_epochs`, `--config`, `--dataset`, `--out`, `--overwrite`, `--wandb`,
`--tag`. The output folder must be empty or new, so two runs' files never mix; `--overwrite`
first deletes the files listed below from it, and nothing else.
Everything is written while the run is going, so a stopped run keeps what it has done:

| File | Contents |
| --- | --- |
| `config.yaml` | the resolved config, `--epochs` and `--validate-every` included, `tracking` left out (where a run is logged isn't part of it). `--config` on it reruns the run. |
| `history.jsonl` | one line per training step: `epoch`, `loss`, `grad_norm`, `lr` |
| `curve.json` | one row per epoch: mean loss, largest gradient norm, training seconds, and `val_mae` / `val_rmse` / `val_seconds` when scored |
| `model.pt` | weights after the latest scored epoch. The last epoch is always scored, so a finished run's are its final weights, the ones that are scored (no best-epoch checkpoint; PROJECT_PLAN §3.1). |
| `summary.json` | the 500-step health report, the final val MAE, forward fill's val MAE, `below_forward_fill`, and `wandb_run`, the run's path on wandb (`entity/project/id`; null when untracked) |

**Validation while training is for tracking, not reporting.** Every `validation.every_n_epochs`
epochs (and on the last), the run scores the val split's `EvalMasker` targets at 10%, exactly
as the baselines are scored, using the median of `validation.n_samples` draws (default 5). The
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
seed. The regime is `random` until M6-2 adds the block-aware one. Every run of a variant, regime
and seed gets the same name, so say what a run is for with `--tag` (repeatable; `tags=` on
`run()`), e.g. `--tag gate`, `--tag sweep`, `--tag dry-run` or `--tag debug`, and filter on it
in the workspace. What a run sends, all of it also in its `outputs/` files:

| wandb key | Sent | From |
| --- | --- | --- |
| `train/loss`, `train/grad_norm`, `train/lr`, `epoch` | every step (step = steps since the start, 175 per epoch) | `history.jsonl` |
| `epoch/loss_mean`, `epoch/grad_norm_max`, `epoch/seconds` | at the end of every epoch | `curve.json` |
| `val/mae`, `val/rmse`, `val/seconds` | on every scored epoch | `curve.json` |
| config (without `tracking`), plus `seed`, `variant`, `regime`, `device`, `out_dir` | once, at the start | `config.yaml`, and the output folder's path |
| summary: `health.*`, `final_val_mae`, `forward_fill_val_mae`, `below_forward_fill`, `wandb_run` | at the end | `summary.json` |

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

Tracking doesn't change training, as far as tested: on the CPU, a run tracked by wandb offline
has history and weights bit-identical to an untracked one's (`tests/test_run.py`). Still to
confirm on CUDA with wandb online, where the real runs are: M3-2's tracked check run on Colab.

Any `--epochs` works for a quick check, 1 included (see [Training](#training)). M3-2's first
check run (`m3-2_smoke_check`, tagged `smoke`) predates that fix: it trained its one epoch at
lr 1e-5, so its val MAE means nothing. It is kept as the record that logging works.

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
are scored themselves (9 and 12 on test, 3 and 4 on val), which inflates both baselines' test
RMSE.

† Inflated by the known bad values, MAE as well as RMSE. On the block holdout three of them are
test targets themselves: a pH of 733, a pH of 1 and a Temp of 0. The pH of 733 dominates: it is
about 520 standardized units from any plausible pH, so every model misses it by about that much,
which adds about 0.02 to every model's block test MAE. Differences between models barely change, but
block test RMSE is mostly that one value. Without it, mean imputation's test MAE and RMSE would be
0.7139 and 1.0477, and forward fill's 0.4587 and 0.7910. No bad value is a block val target.

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
fractions (`eval_masker_seed`; `block_eval_masker_seed` and `block_window_hours`). M3-3 logs the
baselines to wandb. To regenerate the file, run from the repo root:

```text
uv run python -m eval.baselines
```

`tests/test_baselines.py` fails when the recorded file no longer matches a fresh run, so rerun
it, and update this table, whenever the baselines, the data pipeline or the protocol change.

Tests:

- `tests/test_baselines.py`: hand-worked fixtures; forward fill checked against a plain-loop
  reference; no leakage from held-out values; the scored values exactly `EvalMasker`'s held-out
  targets; each rate holding out its own fraction; pinned Set A test MAEs at 10%, 50% and 90%;
  the recorded file matching a fresh run; importing `eval` leaving `eval.baselines` unimported.
  For the block holdout (M2-4): its entry sitting next to `random` with its seed and window, at
  10% only; the same target count as the random 10% holdout; scores taken on exactly
  `BlockEvalMasker`'s targets; pinned Set A test MAEs; the three bad values it scores on test.
- `tests/test_sampler_real_batch.py`: the reused reverse sampler on real Set A patients (M2-3).
  An untrained model runs upstream's full 50-step schedule, with `EvalMasker`'s conditioning
  mask passed as `gt_mask`. The samples come back as `(K, B, 48, D)` with no NaNs, the K draws
  differ, and the held-out values can't reach them.

---

## References

Primary references are listed in [`Docs/PROJECT_PLAN.md`](Docs/PROJECT_PLAN.md) §9
(CSDI, Time2Vec, BRITS, GRU-D, DDPM, conformal prediction, and the dataset).
