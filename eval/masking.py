from pathlib import Path

import numpy as np
import torch
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

class EvalMasker:
    """Fixed evaluation holdout (PROJECT_PLAN §3.1).

    For each patient, hides ``floor(fraction * n_observed)`` of their observed values as
    evaluation targets. Each patient gets its own generator seeded from
    ``(seed, record_id)``, so a patient's targets depend only on the frozen seed and their
    record id -- never on run seeds, patient order, batching, or which split is passed in.
    The pick is made over the patient's flattened ``(T, D)`` grid, so that layout is part of
    the frozen protocol too: ``grid`` is ``(T, D)`` and any other mask shape is rejected.
    """

    def __init__(self, seed, fraction, grid=(48, 35)):
        self.seed = seed
        self.fraction = fraction
        self.grid = tuple(grid)

    @classmethod
    def from_config(cls, path=CONFIG_PATH):
        config = yaml.safe_load(Path(path).read_text())
        protocol, data = config["protocol"], config["data"]
        grid = (data["n_timesteps"], data["n_features"])
        return cls(protocol["eval_masker_seed"], protocol["eval_holdout_fraction"], grid)

    def __call__(self, mask, record_ids):
        """Split ``mask`` (N, T, D) into ``(cond_mask, target_mask)``, same shape and dtype.

        ``target_mask`` marks the held-out values to score; ``cond_mask`` is everything
        else that was observed, and is all the model may see.
        """
        if mask.ndim != 3 or mask.shape[1:] != self.grid:
            raise ValueError(f"mask must be (N, {self.grid[0]}, {self.grid[1]}), got {mask.shape}")
        if len(mask) != len(record_ids):
            raise ValueError(f"{len(mask)} patients in mask but {len(record_ids)} record ids")
        flat_mask = mask.reshape(len(mask), -1)
        # Written into a buffer of our own: reshaping an array laid out like ``mask`` can
        # return a copy (e.g. Fortran order), and writes to a copy would be silently lost.
        flat_target = np.zeros(flat_mask.shape, dtype=mask.dtype)
        for i, record_id in enumerate(record_ids):
            observed = np.flatnonzero(flat_mask[i])
            k = int(len(observed) * self.fraction)
            rng = np.random.default_rng([self.seed, int(record_id)])
            flat_target[i, rng.choice(observed, k, replace=False)] = 1
        target = flat_target.reshape(mask.shape)
        return mask - target, target

class TrainMasker:
    """Fresh conditioning/target split every training step (PROJECT_PLAN §3.1; CSDI "random").

    For each patient, draws a ratio ``r ~ U(0, 1)`` and hides ``round(r * n_observed)`` of
    their observed values as targets. Uses a private CPU generator seeded from the run seed,
    so the mask stream is reproducible, is the same on CPU and GPU, and is not shifted by any
    other random draws (e.g. Ext 1's extra layers) -- which keeps paired seeds paired.
    Only ever removes values from ``mask``: pass ``EvalMasker``'s ``cond_mask`` for val/test.
    """

    def __init__(self, seed):
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, mask):
        """Split ``mask`` (B, T, D) tensor into ``(cond_mask, target_mask)`` on ``mask.device``."""
        observed = mask.cpu().reshape(len(mask), -1)
        ratio = torch.rand(len(mask), 1, generator=self.generator)
        n_hide = (observed.sum(1, keepdim=True) * ratio).round()
        scores = torch.rand(observed.shape, generator=self.generator) * observed
        rank = scores.argsort(1, descending=True).argsort(1)    # 0 = highest score
        target = ((rank < n_hide) & (observed == 1)).to(mask.dtype)
        target = target.reshape(mask.shape).to(mask.device)
        return mask - target, target
