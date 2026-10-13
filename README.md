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
models/        # reused CSDI denoiser + extensions (MissingnessEncoder,
               #   ContinuousTimeEncoding).                                 [M1-2]
diffusion/     # noising schedule, q_sample, masked objective, DDPM sampler.
eval/          # EvalMasker / TrainMasker + masked, standardized metrics.   [M1-3]
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

---

## Denoiser input-hook contract

Frozen at the M1-4 kickoff so the data, architecture, and evaluation tracks can proceed in
parallel without waiting on each other. The reused CSDI denoiser's `forward` **must** accept
two optional hooks, both defaulting to `None`:

| Hook | Produced by | Shape | How it enters the denoiser |
| --- | --- | --- | --- |
| `mask_emb` | `MissingnessEncoder` (Ext 1, `--use_mask_embedding`) | `(B, T, d_model)` | **added to the denoiser input** |
| `time_enc` | `ContinuousTimeEncoding` (Ext 2, `--use_time_encoding`) | `(B, T, d_model)` | **added to the denoiser input** |

```python
def forward(self, x, cond, diffusion_step, *, mask_emb=None, time_enc=None):
    # When a hook is None the denoiser behaves exactly as the reproduced base CSDI,
    # so base / Ext1 / Ext2 / Ext1+2 are the four toggle combinations of one model.
    ...
```

Rules:

- Both hooks default to `None`; with both `None` the model **is** the reproduced base CSDI.
- Each hook is **added to the denoiser input** like a positional encoding — it does **not**
  modify the attention logits (PROJECT_PLAN §4.2).
- Both must project to `d_model` so they are broadcast-compatible with the input stream.

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

**Adapted so far.** Two import lines, each marked `ADAPTED (M1-2)` in the source; no model
logic is changed:

- `main_model.py` imports `diff_models` relatively, because upstream is a flat script
  directory and ours is a package.
- `diff_models.py` imports `linear_attention_transformer` only when `is_linear=True`. The
  PhysioNet configuration never sets it, so the package is not one of our dependencies.

**Not taken from upstream:** its data loading, training and evaluation scripts, and configs
(`dataset_*.py`, `exe_*.py`, `utils.py`, `config/`). Ours are in `data/`, `eval/` and
`configs/`.

**Still to adapt** (M2-2): the `mask_emb` / `time_enc` input hooks from the contract above.
The vendored `diff_CSDI.forward(x, cond_info, diffusion_step)` does not accept them yet. Any
later change to the vendored files gets the same `ADAPTED` marker and a line in this section.

`tests/test_csdi_smoke.py` trains the vendored model for a few steps on a synthetic batch
of our shape, `(B, 48, 35)` at about 80% missing, and checks that the loss, gradients and
sampled imputations are finite. Run it with `uv run pytest tests/test_csdi_smoke.py`.

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

---

## References

Primary references are listed in [`Docs/PROJECT_PLAN.md`](Docs/PROJECT_PLAN.md) §9
(CSDI, Time2Vec, BRITS, GRU-D, DDPM, conformal prediction, and the dataset).
