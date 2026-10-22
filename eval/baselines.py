from pathlib import Path

import numpy as np
import yaml

from .masking import CONFIG_PATH, BlockEvalMasker, EvalMasker
from .metrics import masked_mae, masked_rmse

_CONFIG = yaml.safe_load(Path(CONFIG_PATH).read_text())
GRID = (_CONFIG["data"]["n_timesteps"], _CONFIG["data"]["n_features"])
ROOT = Path(CONFIG_PATH).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "physionet_set_a.npz"
RESULTS_PATH = ROOT / "results" / "baselines.json"

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
    the guess. The rule for the hours before a variable's first visible value is backward fill:
    they take that first value (filling them with the training mean instead scores worse on
    Set A). A variable the patient never shows gets the training mean, 0. Returns a fresh
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

# The two baselines as guessers, ``guess(values, cond_mask)``: what ``score_holdout`` scores by default.
BASELINES = {
    "mean_imputation": lambda values, cond_mask: mean_imputation(values),
    "forward_fill": forward_fill,
}

def score_holdout(dataset, masker, guessers=None):
    """Each guesser's val and test MAE/RMSE on ``masker``'s held-out targets, keyed by split.

    ``guessers`` maps a name to ``guess(values, cond_mask) -> (N, 48, D)``; the default is the two
    baselines. A guesser is only ever shown ``cond_mask``; the held-out values stay in ``values``
    for scoring, so it must not read them (each guesser's tests check that it doesn't).
    """
    rows = {}
    for split in ("val", "test"):
        idx = dataset[f"{split}_idx"]
        values, mask = dataset["values"][idx], dataset["mask"][idx]
        cond, target = masker(mask, dataset["record_ids"][idx])
        row = {"n_patients": len(idx), "n_targets": int(target.sum())}
        for name, guesser in (guessers or BASELINES).items():
            guess = guesser(values, cond)
            row[name] = {"mae": round(masked_mae(guess, values, target), 6),
                         "rmse": round(masked_rmse(guess, values, target), 6)}
        rows[split] = row
    return rows

def baseline_results(dataset, guessers=None):
    """Each guesser's MAE/RMSE on the frozen holdouts' val and test targets: the record in ``RESULTS_PATH``.

    Keyed by holdout. The random holdout (``EvalMasker``) is scored at the headline rate
    (``eval_holdout_fraction``, 10%) and at CSDI's robustness rates (``robustness_fractions``,
    50% and 90%), all with the frozen ``eval_masker_seed``, under
    ``random["fractions"][str(fraction)]``. The block holdout (``BlockEvalMasker``, M2-4) is
    scored at the headline rate only, with its own frozen seed and window, under
    ``block["fractions"][str(fraction)]``. ``guessers`` defaults to the two baselines; the
    learned baseline (``models.brits``, M3-3) is scored through here too, so its record has
    this same layout.
    """
    protocol = _CONFIG["protocol"]
    random = {"eval_masker_seed": protocol["eval_masker_seed"], "fractions": {}}
    block = {"block_eval_masker_seed": protocol["block_eval_masker_seed"],
             "block_window_hours": protocol["block_window_hours"],
             "fractions": {}}
    for fraction in (protocol["eval_holdout_fraction"], *protocol["robustness_fractions"]):
        masker = EvalMasker(protocol["eval_masker_seed"], fraction, GRID)
        random["fractions"][str(fraction)] = score_holdout(dataset, masker, guessers)
    block_masker = BlockEvalMasker.from_config()
    block["fractions"][str(block_masker.fraction)] = score_holdout(dataset, block_masker, guessers)
    return {
        "units": "standardized (train-split z-scores)",
        "split_seed": _CONFIG["data"]["split_seed"],
        "eval_holdout_fraction": protocol["eval_holdout_fraction"],
        "random": random,
        "block": block,
    }

def wandb_summary(results, name):
    """One guesser's scores in ``results``, flattened for a wandb run summary (M3-3).

    Keys are ``<holdout>/<fraction>/<split>/<metric>``, e.g. ``random/0.1/test/mae``, plus
    ``val/mae`` and ``val/rmse`` for the random headline holdout's val split: the key CSDI runs
    log their tracked validation under, so a baseline lines up with them in the runs table.
    """
    summary = {f"{holdout}/{fraction}/{split}/{metric}": value
               for holdout in ("random", "block")
               for fraction, rows in results[holdout]["fractions"].items()
               for split, row in rows.items()
               for metric, value in row[name].items()}
    headline = results["random"]["fractions"][str(results["eval_holdout_fraction"])]["val"][name]
    return summary | {f"val/{metric}": value for metric, value in headline.items()}

def log_to_wandb(results, mode=None):
    """One wandb run per baseline in ``results`` (job type ``baseline``), its scores as the summary (M3-3)."""
    import wandb

    tracking = _CONFIG["tracking"]
    row = results["random"]["fractions"][str(results["eval_holdout_fraction"])]["val"]
    for name in (key for key, scores in row.items() if isinstance(scores, dict)):
        with wandb.init(entity=tracking["wandb_entity"], project=tracking["wandb_project"], mode=mode,
                        name=name, job_type="baseline", tags=["baseline", name],
                        config={k: results[k] for k in ("units", "split_seed", "eval_holdout_fraction")}) as run:
            run.summary.update(wandb_summary(results, name))

if __name__ == "__main__":
    import argparse
    import json
    from data.physionet import load_dataset

    parser = argparse.ArgumentParser(description="Score mean imputation and forward fill; write RESULTS_PATH.")
    parser.add_argument("--wandb", choices=("online", "offline"),
                        help="also log each baseline as a wandb run (M3-3); off unless given")
    args = parser.parse_args()
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results = baseline_results(load_dataset(DATASET_PATH))
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if args.wandb:
        log_to_wandb(results, args.wandb)
