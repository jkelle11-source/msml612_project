from pathlib import Path

import numpy as np
import yaml

from .masking import CONFIG_PATH, EvalMasker
from .metrics import masked_mae, masked_rmse

_CONFIG = yaml.safe_load(Path(CONFIG_PATH).read_text())
GRID = (_CONFIG["data"]["n_timesteps"], _CONFIG["data"]["n_features"])
RESULTS_PATH = Path(CONFIG_PATH).resolve().parents[1] / "results" / "baselines.json"

def check_grid(x):
    """Return ``x`` (NumPy array or CPU torch tensor) as an array, rejecting anything not ``(N, 48, D)``.

    Upstream's ``(N, D, 48)`` layout is rejected too: a baseline run on it would fill along the
    wrong axis and still return a plausible-looking number.
    """
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != GRID:
        raise ValueError(f"expected (N, {GRID[0]}, {GRID[1]}), got {x.shape}")
    return x

def mean_imputation(values):
    """Mean-imputation baseline (M2-3): guess every variable's training-split mean, everywhere.

    ``values`` are z-scored with training-split statistics, so the training mean is exactly 0
    and the guess is a fresh ``(N, 48, D)`` float32 grid of zeros. It never looks at the patient.
    Score it like any model: ``masked_mae(mean_imputation(values), values, target)``.
    """
    return np.zeros(check_grid(values).shape, dtype=np.float32)

def forward_fill(values, cond_mask):
    """Forward-fill baseline (M2-3): carry each variable's last visible value forward in time.

    Only boxes where ``cond_mask == 1`` are ever read, so the held-out targets cannot leak into
    the guess. Hours before a variable's first visible value take that first value (backward
    fill), and a variable the patient never shows gets the training mean, 0. Returns a fresh
    ``(N, 48, D)`` float32 grid: ``masked_mae(forward_fill(values, cond), values, target)``.
    """
    values, cond = check_grid(values), check_grid(cond_mask) == 1
    if cond.shape != values.shape:
        raise ValueError(f"cond_mask {cond.shape} does not match values {values.shape}")
    hours = np.arange(values.shape[1])[None, :, None]
    last = np.maximum.accumulate(np.where(cond, hours, -1), axis=1)              # latest visible hour so far
    first = np.where(cond.any(axis=1, keepdims=True), cond.argmax(axis=1, keepdims=True), -1)
    source = np.where(last >= 0, last, first)                                    # -1: never visible
    filled = np.take_along_axis(values, np.maximum(source, 0), axis=1)
    return np.where(source >= 0, filled, 0).astype(np.float32)

def baseline_results(dataset):
    """Both baselines' MAE/RMSE on ``EvalMasker``'s val and test targets: the record in ``RESULTS_PATH``.

    Keyed by holdout, so the block holdout (``M2-4``) is recorded next to the random one.
    """
    random = {"eval_masker_seed": _CONFIG["protocol"]["eval_masker_seed"]}
    for split in ("val", "test"):
        idx = dataset[f"{split}_idx"]
        values, mask = dataset["values"][idx], dataset["mask"][idx]
        cond, target = EvalMasker.from_config()(mask, dataset["record_ids"][idx])
        row = {"n_patients": len(idx), "n_targets": int(target.sum())}
        for name, guess in (("mean_imputation", mean_imputation(values)), ("forward_fill", forward_fill(values, cond))):
            row[name] = {"mae": round(masked_mae(guess, values, target), 6),
                         "rmse": round(masked_rmse(guess, values, target), 6)}
        random[split] = row
    return {
        "units": "standardized (train-split z-scores)",
        "split_seed": _CONFIG["data"]["split_seed"],
        "eval_holdout_fraction": _CONFIG["protocol"]["eval_holdout_fraction"],
        "random": random,
    }

if __name__ == "__main__":
    import json
    from data.physionet import load_dataset

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results = baseline_results(load_dataset("data/processed/physionet_set_a.npz"))
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
