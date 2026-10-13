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
    """

    def __init__(self, seed, fraction):
        self.seed = seed
        self.fraction = fraction

    @classmethod
    def from_config(cls, path=CONFIG_PATH):
        protocol = yaml.safe_load(Path(path).read_text())["protocol"]
        return cls(protocol["eval_masker_seed"], protocol["eval_holdout_fraction"])

    def __call__(self, mask, record_ids):
        """Split ``mask`` (N, T, D) into ``(cond_mask, target_mask)``, same shape and dtype.

        ``target_mask`` marks the held-out values to score; ``cond_mask`` is everything
        else that was observed, and is all the model may see.
        """
        if len(mask) != len(record_ids):
            raise ValueError(f"{len(mask)} patients in mask but {len(record_ids)} record ids")
        target = np.zeros_like(mask)
        flat_mask, flat_target = mask.reshape(len(mask), -1), target.reshape(len(mask), -1)
        for i, record_id in enumerate(record_ids):
            observed = np.flatnonzero(flat_mask[i])
            k = int(len(observed) * self.fraction)
            rng = np.random.default_rng([self.seed, int(record_id)])
            flat_target[i, rng.choice(observed, k, replace=False)] = 1
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
