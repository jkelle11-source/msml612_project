# CSDI — Score-Based Imputation Objective (shared note)

> **Status:** M1-4 deliverable (shared one-paragraph summary from reading CSDI §§3–4
> together). Drafted for team review — Josh/Sankha, please refine against the paper.
> **Reference:** Tashiro, Song, Song, Ermon (2021), *CSDI: Conditional Score-based
> Diffusion Models for Probabilistic Time Series Imputation*, NeurIPS — arXiv:2107.03502.

CSDI casts imputation as **conditional generation**: it learns the distribution of the
missing (target) values *conditioned on* the observed values, rather than regressing a
single point estimate. It does this with a DDPM-style diffusion process applied **only to
the target values**, while the observed values are kept noise-free and fed in as
conditioning. Training is **self-supervised**: each iteration randomly partitions the
*observed* entries into a conditioning subset and an imputation-target subset, adds
Gaussian noise to the targets at a randomly sampled diffusion step `t`, and trains a
denoiser `ε_θ` to predict that added noise, conditioned on the observed values, the
observation mask, and `t`. The loss is the standard denoising (noise-prediction) MSE,
evaluated **only at the target positions** — so CSDI never trains on, or leaks, the values
it must impute. The denoiser itself is a transformer with **two-dimensional attention**
(one axis over time, one over features), letting it exploit correlations across both
timesteps and variables. At inference, the learned reverse process starts from Gaussian
noise at the missing positions and denoises step by step, conditioned on the observed
values, to **sample** an imputation; running the reverse process multiple times yields a
*posterior* of plausible imputations whose spread is a usable uncertainty signal (the
spread Extension 3's conformal wrapper reuses).

## How this maps onto our project

- **Two-mask protocol (§3.1):** CSDI's self-supervised conditioning/target split is exactly
  our `TrainMasker` (fresh split each iteration). Our `EvalMasker` adds a *separate*, frozen
  10% target holdout for scoring, which CSDI's own setup does not pin.
- **The extensions hook the denoiser *input*, not the objective:** Ext 1 (`mask_emb`) and
  Ext 2 (`time_enc`) are added to the denoiser input; the noise-prediction objective above
  is unchanged, which is what keeps base / Ext1 / Ext2 / Ext1+2 directly comparable.
- **Two-dimensional attention** is the structure Ext 1's dual-axis `MissingnessEncoder`
  deliberately mirrors (temporal + feature), because block missingness is cross-feature.
