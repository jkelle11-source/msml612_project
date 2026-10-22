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

def block_targets(mask, count, generator, window):
    """Pick ``count`` of one patient's observed values as targets, in whole time slots (§3.1).

    ``mask`` is one patient's ``(T, D)`` observation mask. Its ``T`` hours are tiled into
    ``T // window`` fixed slots of ``window`` hours across all ``D`` features, and the slots
    are visited in an order drawn from ``generator``. Every observed value in each slot is
    hidden until the next slot would overshoot ``count``; that slot is trimmed to a random
    subset of its observed values, so exactly ``count`` are hidden, and the walk stops. A slot
    with nothing observed hides nothing. Returns a fresh ``(T, D)`` array with ``mask``'s dtype.

    All randomness comes from ``generator`` (a ``np.random.Generator``): one permutation of the
    slots, then at most one ``choice`` in the trimmed slot. That draw order is part of the
    frozen block holdout. Shared by ``BlockEvalMasker`` (``count = floor(fraction *
    n_observed)``) and, from M6-2, ``TrainMasker``'s ``mix`` strategy.
    """
    T, D = mask.shape
    slots, remainder = T // window, T % window
    if remainder:
        raise ValueError(f"window {window} must divide the {T} hours of the grid")
    flat_mask = mask.reshape(-1)
    n_observed = np.count_nonzero(flat_mask)
    if count > n_observed:
        raise ValueError(f"count ({count}) greater than {n_observed} observed values")
    shuffled_slots = generator.permutation(slots)
    target = np.zeros(T * D, dtype=mask.dtype)
    left = count
    for s in shuffled_slots:
        if left == 0:
            break
        start, end = s * (window * D), (s + 1) * (window * D)
        observed = np.flatnonzero(flat_mask[start:end]) + start
        if len(observed) <= left:
            target[observed] = 1
            left -= len(observed)
        else:
            chosen = generator.choice(observed, left, replace=False)
            target[chosen] = 1
            break
    return target.reshape(mask.shape)

class BlockEvalMasker:
    """Fixed block-missingness evaluation holdout (PROJECT_PLAN §3.1, §4.1).

    For each patient, hides exactly ``floor(fraction * n_observed)`` of their observed values
    as evaluation targets, the same count as ``EvalMasker``, but in whole ``window``-hour slots
    across all features, apart from one trimmed slot (see ``block_targets``). Each patient gets
    its own generator seeded from ``(seed, record_id)``, so a patient's targets depend only on
    the frozen seed and their record id -- never on run seeds, patient order, batching, or
    which split is passed in. The slots are cut from the patient's ``(T, D)`` grid, so that
    layout is part of the frozen protocol too: ``grid`` is ``(T, D)`` and any other mask shape
    is rejected.
    """

    def __init__(self, seed, fraction, window, grid=(48, 35)):
        self.seed = seed
        self.fraction = fraction
        self.window = window
        self.grid = tuple(grid)

    @classmethod
    def from_config(cls, path=CONFIG_PATH):
        config = yaml.safe_load(Path(path).read_text())
        protocol, data = config["protocol"], config["data"]
        grid = (data["n_timesteps"], data["n_features"])
        return cls(protocol["block_eval_masker_seed"], protocol["eval_holdout_fraction"],
                   protocol["block_window_hours"], grid)

    def __call__(self, mask, record_ids):
        """Split ``mask`` (N, T, D) into ``(cond_mask, target_mask)``, same shape and dtype.

        ``target_mask`` marks the held-out values to score; ``cond_mask`` is everything
        else that was observed, and is all the model may see.
        """
        if mask.ndim != 3 or mask.shape[1:] != self.grid:
            raise ValueError(f"mask must be (N, {self.grid[0]}, {self.grid[1]}), got {mask.shape}")
        if len(mask) != len(record_ids):
            raise ValueError(f"{len(mask)} patients in mask but {len(record_ids)} record ids")
        target = np.zeros(mask.shape, dtype=mask.dtype)
        for i, record_id in enumerate(record_ids):
            obs_count = np.count_nonzero(mask[i])
            count = int(obs_count * self.fraction)
            rng = np.random.default_rng([self.seed, int(record_id)])
            target[i] = block_targets(mask[i], count, rng, self.window)
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
