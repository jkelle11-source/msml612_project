"""Evaluation and protocol objects (M1-3, owner: Sankha).

The subtle heart of the frozen protocol (PROJECT_PLAN §3.1, §4.1):

- ``EvalMasker`` — seeded **once** from the frozen project seed in ``configs/config.yaml``
  (``eval_masker_seed``); holds out a fixed 10% of observed values as evaluation targets,
  byte-identical across every model and every seed for the whole project.
- ``TrainMasker`` — draws a *fresh* conditioning/target split each iteration, reproducible
  from the per-run seed but deliberately **not** pinned to one pattern.

Also houses masked MAE/RMSE metrics reported in standardized units (§3.2), computed only
over held-out eval targets, with an optional raw-unit per-variable readout, and
``EvalSampler``, which runs the reused sampler for one evaluation pass with its noise
re-seeded from a frozen seed (``eval_sampling_seed``, §3.3).

Baselines (M2-3): ``mean_imputation`` and ``forward_fill``, scored with the same metrics.
"""

from .baselines import forward_fill, mean_imputation
from .masking import EvalMasker, TrainMasker
from .metrics import masked_errors, masked_mae, masked_rmse, median_imputation, per_variable_mae
from .sampling import EvalSampler

__all__ = [
    "EvalMasker",
    "EvalSampler",
    "TrainMasker",
    "forward_fill",
    "masked_errors",
    "masked_mae",
    "masked_rmse",
    "mean_imputation",
    "median_imputation",
    "per_variable_mae",
]
