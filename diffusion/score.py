"""Score a finished run's final weights with the frozen sampler (M4-1).

    uv run python -m diffusion.score outputs/base_random_seed42

This is the protocol's number for a run; validation while training (5 draws) is tracking only.
The model is rebuilt from the run's saved ``config.yaml`` and given its ``model.pt``, its final
weights (PROJECT_PLAN §3.1). A run without ``summary.json`` has not finished, so its ``model.pt``
is an earlier epoch's, and it is refused. The holdout and the sampler are the frozen protocol's,
from ``configs/config.yaml``:

- the holdout: ``EvalMasker``'s random holdout at 10% (``--fraction 0.5`` or ``0.9`` for CSDI's
  robustness rates), or ``BlockEvalMasker``'s block holdout (``--holdout block``, 10% only);
- the sampler: ``EvalSampler.from_config()``, 100 draws per patient with the frozen
  ``eval_sampling_seed``, 16 patients at a time.

The median of the draws is scored on the held-out values only, in standardized units, as the
baselines are (``eval.baselines``). The split is the test split unless ``--split val``. The
running MAE is printed after every batch, since a pass over the test split takes hours.

Two files are written to the run folder, named for what was scored (e.g. ``test_random_0.1``):

- ``scores_test_random_0.1.json``: MAE, RMSE, per-variable MAE, the target count, the holdout's
  and sampler's settings, the device and seconds taken, and ``model_sha256``, the hash of the
  weights scored. ``diffusion.gate`` checks it against ``model.pt``, so retrained weights are
  never reported with an earlier run's scores.
- ``median_test_random_0.1.npy``: the median imputation, ``(N, 48, 35)`` float32, in the split's
  patient order (``dataset["test_idx"]``), for checks that need more than the pooled scores (the
  gate's PaCO2/PaO2 check, M9-2's per-variable analysis).

The scores are also added to the run's wandb summary (``wandb_run`` in its ``summary.json``) under
the baselines' keys, e.g. ``random/0.1/test/mae``, so the runs table lines CSDI up with them.
``--wandb disabled`` skips that, as does ``WANDB_MODE`` set to anything but ``online``.
"""

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from data.physionet import load_dataset
from diffusion.run import CONFIG_PATH, DATASET_PATH, pick_device
from diffusion.train import use_deterministic_kernels
from eval import BlockEvalMasker, EvalMasker, EvalSampler, masked_errors, masked_mae, masked_rmse, \
    median_imputation, per_variable_mae
from models.build import build_model

def scores_name(split, holdout, fraction):
    """``test_random_0.1``: how a scoring's files are named."""
    return f"{split}_{holdout}_{fraction}"

def holdout_masker(protocol, grid, holdout, fraction):
    """The frozen holdout to score, and its settings as ``results/baselines.json`` records them."""
    if holdout == "random":
        rates = (protocol["eval_holdout_fraction"], *protocol["robustness_fractions"])
        if fraction not in rates:
            raise ValueError(f"the random holdout is frozen at {rates}, not {fraction}")
        return (EvalMasker(protocol["eval_masker_seed"], fraction, grid),
                {"eval_masker_seed": protocol["eval_masker_seed"]})
    if holdout == "block":
        if fraction != protocol["eval_holdout_fraction"]:
            raise ValueError(f"the block holdout is frozen at {protocol['eval_holdout_fraction']}, not {fraction}")
        return (BlockEvalMasker(protocol["block_eval_masker_seed"], fraction, protocol["block_window_hours"], grid),
                {"block_eval_masker_seed": protocol["block_eval_masker_seed"],
                 "block_window_hours": protocol["block_window_hours"]})
    raise ValueError(f"holdout must be 'random' or 'block', not {holdout!r}")

def score(run_dir, dataset, device, holdout="random", fraction=None, split="test",
          config_path=CONFIG_PATH, log=print):
    """Score ``run_dir``'s final weights on ``split``'s frozen holdout; writes and returns its scores.

    ``fraction`` defaults to ``protocol.eval_holdout_fraction`` (10%). The holdout and sampler come
    from ``config_path``'s frozen ``protocol`` block; the model from the run's own saved config.
    """
    use_deterministic_kernels()                 # before any CUDA work (README, Training)
    run_dir = Path(run_dir)
    if not (run_dir / "summary.json").exists():
        raise FileNotFoundError(f"{run_dir} has no summary.json: the run has not finished, "
                                "so its model.pt is not its final weights")
    summary = json.loads((run_dir / "summary.json").read_text())
    run_config = yaml.safe_load((run_dir / "config.yaml").read_text())
    protocol = yaml.safe_load(Path(config_path).read_text())["protocol"]
    fraction = protocol["eval_holdout_fraction"] if fraction is None else fraction
    grid = (run_config["data"]["n_timesteps"], run_config["data"]["n_features"])
    masker, holdout_settings = holdout_masker(protocol, grid, holdout, fraction)
    sampler = EvalSampler(protocol["eval_sampling_seed"], protocol["eval_n_samples"], protocol["eval_batch_size"])

    weights = (run_dir / "model.pt").read_bytes()
    model = build_model(run_config, device)
    model.load_state_dict(torch.load(io.BytesIO(weights), map_location=device))

    idx = dataset[f"{split}_idx"]
    values, mask = dataset["values"][idx], dataset["mask"][idx]
    cond, target = masker(mask, dataset["record_ids"][idx])
    median = np.zeros(values.shape, dtype=np.float32)
    n_batches = -(-len(idx) // sampler.batch_size)
    abs_error, n_scored, started = 0.0, 0, time.time()
    for i, (rows, samples) in enumerate(sampler.batches(model, values, mask, cond), start=1):
        median[rows] = median_imputation(samples)
        if target[rows].any():
            abs_error += float(np.abs(masked_errors(median[rows], values[rows], target[rows])).sum())
            n_scored += int(target[rows].sum())
        elapsed = time.time() - started
        log(f"batch {i}/{n_batches}  running MAE {abs_error / max(n_scored, 1):.4f}  "
            f"[{elapsed:.0f}s, about {elapsed / i * (n_batches - i):.0f}s left]")

    per_variable = per_variable_mae(median, values, target)
    scores = {
        "run": run_dir.name,
        "variant": summary["variant"],
        "seed": summary["seed"],
        "split": split,
        "holdout": holdout,
        "fraction": fraction,
        "units": "standardized (train-split z-scores)",
        "n_patients": len(idx),
        "n_targets": int(target.sum()),
        "mae": masked_mae(median, values, target),
        "rmse": masked_rmse(median, values, target),
        "per_variable_mae": {str(name): None if np.isnan(v) else float(v)       # None: nothing held out
                             for name, v in zip(dataset["features"], per_variable)},
        "holdout_settings": holdout_settings,
        "sampler": {"eval_sampling_seed": sampler.seed, "eval_n_samples": sampler.n_samples,
                    "eval_batch_size": sampler.batch_size},
        "device": device,
        "seconds": round(time.time() - started, 1),
        "model_sha256": hashlib.sha256(weights).hexdigest(),
        "wandb_run": summary.get("wandb_run"),
    }
    name = scores_name(split, holdout, fraction)
    np.save(run_dir / f"median_{name}.npy", median)
    (run_dir / f"scores_{name}.json").write_text(json.dumps(scores, indent=1) + "\n")
    return scores

def attach_to_wandb(scores, mode=None):
    """Add ``scores``' MAE and RMSE to its run's wandb summary; returns whether it did.

    Keys are the baselines' (``eval.baselines.wandb_summary``): ``<holdout>/<fraction>/<split>/<metric>``.
    Skipped for an untracked run, and unless ``mode`` (default ``WANDB_MODE``, else online) is online.
    """
    mode = mode or os.environ.get("WANDB_MODE") or "online"
    if mode != "online" or scores["wandb_run"] is None:
        return False
    prefix = f"{scores['holdout']}/{scores['fraction']}/{scores['split']}"
    wandb.Api().run(scores["wandb_run"]).summary.update({f"{prefix}/mae": scores["mae"],
                                                         f"{prefix}/rmse": scores["rmse"]})
    return True

def main(argv=None):
    parser = argparse.ArgumentParser(description="Score a finished run's final weights with the frozen sampler (M4-1).")
    parser.add_argument("run_dir", type=Path, help="the run's output folder, e.g. outputs/base_random_seed42")
    parser.add_argument("--holdout", choices=("random", "block"), default="random")
    parser.add_argument("--fraction", type=float,
                        help="the random holdout's rate: 0.1 (default), or 0.5 or 0.9, the robustness rates")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="where the frozen protocol is read from")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto (the first available)")
    parser.add_argument("--wandb", choices=("online", "disabled"),
                        help="add the scores to the run's wandb summary (default: WANDB_MODE if set, else online)")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    print(f"scoring {args.run_dir} on {device}: {args.split} split, {args.holdout} holdout", flush=True)
    scores = score(args.run_dir, load_dataset(args.dataset), device, args.holdout, args.fraction, args.split,
                   config_path=args.config, log=lambda line: print(line, flush=True))
    print(json.dumps({key: value for key, value in scores.items() if key != "per_variable_mae"}, indent=1))
    if attach_to_wandb(scores, args.wandb):
        print(f"added to the wandb summary of {scores['wandb_run']}", flush=True)

if __name__ == "__main__":
    main()
