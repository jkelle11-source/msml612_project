"""Score a finished run's final weights with the frozen sampler (M4-1).

    uv run python -m diffusion.score outputs/base_random_seed42

This is the protocol's number for a run; validation while training (5 draws) is tracking only.
The model is rebuilt from the run's saved ``config.yaml`` and given its ``model.pt``, its final
weights (PROJECT_PLAN §3.1). A run without ``summary.json`` has not finished, so its ``model.pt``
is an earlier epoch's, and it is refused. The holdout and the sampler are the frozen protocol's,
from ``configs/config.yaml``:

- the holdout: ``EvalMasker``'s random holdout at 10% (``--fraction 0.5`` or ``0.9`` for CSDI's
  robustness rates), ``BlockEvalMasker``'s block holdout (``--holdout block``, 10% only), or
  ``OffGridBlockEvalMasker``'s, the block holdout off its grid (``--holdout offgrid``, 10% only;
  §4.1's grid check, M6-3, which also reads the block holdout's MAE over the off-grid holdout's
  hours from a saved scoring, ``span_mae``);
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
``--wandb disabled`` skips that, as does ``WANDB_MODE`` set to anything but ``online``. Beside the
MAE and RMSE go what ``diffusion.results`` needs to collect and check them (M6-2): the GPU they were
scored on (``random/0.1/test/gpu``), whether the pass ran in TF32 (``.../tf32``, from the frozen
``protocol.tf32``, which the pass applies), the sampler's and holdout's settings
(``random/0.1/test/sampler.eval_n_samples``, ``.../holdout_settings.eval_masker_seed``) and the
per-variable MAE (``.../per_variable_mae.HR``). A paired run's scoring (``score(..., paired=True)``
or ``--paired``) starts only on ``protocol.paired_gpu``.
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
from diffusion.run import CONFIG_PATH, DATASET_PATH, gpu_name, pick_device, require_paired_gpu
from diffusion.train import use_deterministic_kernels, use_tf32
from eval import BlockEvalMasker, EvalMasker, EvalSampler, OffGridBlockEvalMasker, masked_errors, masked_mae, \
    masked_rmse, median_imputation, per_variable_mae
from models.build import build_model

HOLDOUTS = ("random", "block", "offgrid")

def scores_name(split, holdout, fraction):
    """``test_random_0.1``: how a scoring's files are named."""
    return f"{split}_{holdout}_{fraction}"

def holdout_settings(protocol, holdout):
    """``holdout``'s frozen settings, as its scores and ``results/baselines.json`` record them.

    ``diffusion.results`` checks every scoring it stores against these (M6-2).
    """
    if holdout == "random":
        return {"eval_masker_seed": protocol["eval_masker_seed"]}
    if holdout == "block":
        return {"block_eval_masker_seed": protocol["block_eval_masker_seed"],
                "block_window_hours": protocol["block_window_hours"]}
    if holdout == "offgrid":
        return {"offgrid_block_eval_masker_seed": protocol["offgrid_block_eval_masker_seed"],
                "block_window_hours": protocol["block_window_hours"],
                "offgrid_block_offset_hours": protocol["offgrid_block_offset_hours"]}
    raise ValueError(f"holdout must be one of {HOLDOUTS}, not {holdout!r}")

def holdout_masker(protocol, grid, holdout, fraction):
    """The frozen holdout to score, and its settings as ``results/baselines.json`` records them."""
    settings = holdout_settings(protocol, holdout)
    if holdout == "random":
        rates = (protocol["eval_holdout_fraction"], *protocol["robustness_fractions"])
        if fraction not in rates:
            raise ValueError(f"the random holdout is frozen at {rates}, not {fraction}")
        return EvalMasker(protocol["eval_masker_seed"], fraction, grid), settings
    if fraction != protocol["eval_holdout_fraction"]:
        raise ValueError(f"the {holdout} holdout is frozen at {protocol['eval_holdout_fraction']}, not {fraction}")
    if holdout == "offgrid":
        return OffGridBlockEvalMasker(protocol["offgrid_block_eval_masker_seed"], fraction, protocol["block_window_hours"],
                                      protocol["offgrid_block_offset_hours"], grid), settings
    return BlockEvalMasker(protocol["block_eval_masker_seed"], fraction, protocol["block_window_hours"], grid), settings

def score(run_dir, dataset, device, holdout="random", fraction=None, split="test",
          config_path=CONFIG_PATH, log=print, paired=False):
    """Score ``run_dir``'s final weights on ``split``'s frozen holdout; writes and returns its scores.

    ``fraction`` defaults to ``protocol.eval_holdout_fraction`` (10%). The holdout and sampler come
    from ``config_path``'s frozen ``protocol`` block; the model from the run's own saved config.
    With ``paired`` (a sweep run, M6-2) it refuses any GPU but that block's ``paired_gpu``.
    """
    use_deterministic_kernels()                 # before any CUDA work (README, Training)
    protocol = yaml.safe_load(Path(config_path).read_text())["protocol"]
    if paired:
        require_paired_gpu(protocol, device)
    tf32 = protocol.get("tf32", False)          # the frozen protocol's, as the sampler's settings are
    use_tf32(tf32)
    run_dir = Path(run_dir)
    if not (run_dir / "summary.json").exists():
        raise FileNotFoundError(f"{run_dir} has no summary.json: the run has not finished, "
                                "so its model.pt is not its final weights")
    summary = json.loads((run_dir / "summary.json").read_text())
    run_config = yaml.safe_load((run_dir / "config.yaml").read_text())
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
        "gpu": gpu_name(device),
        "tf32": tf32,
        "seconds": round(time.time() - started, 1),
        "model_sha256": hashlib.sha256(weights).hexdigest(),
        "wandb_run": summary.get("wandb_run"),
    }
    name = scores_name(split, holdout, fraction)
    np.save(run_dir / f"median_{name}.npy", median)
    (run_dir / f"scores_{name}.json").write_text(json.dumps(scores, indent=1) + "\n")
    return scores

def span_mae(run_dir, dataset, holdout, hours, fraction=None, split="test", config_path=CONFIG_PATH):
    """MAE of ``run_dir``'s ``holdout`` scoring over the targets in ``hours`` alone; no new sampling.

    For PROJECT_PLAN §4.1's grid check (M6-3, run in M8-4): the off-grid holdout never hides hours
    0, 46 and 47, and hour 0 is base's hardest hour, so its Δ is set beside the block holdout's over
    the same hours, ``OffGridBlockEvalMasker.from_config().hours`` (1-45). Reads the median imputation
    ``score`` saved beside the run (``median_<split>_<holdout>_<fraction>.npy``), rebuilds the frozen
    holdout's targets, keeps those in ``hours`` and returns ``{"mae", "n_targets"}``, standardized.
    Refused if that scoring did not use ``config_path``'s frozen holdout and sampler (a quick
    scoring's draws are not the protocol's), or if ``model.pt`` is no longer the weights it scored.
    """
    run_dir = Path(run_dir)
    protocol = yaml.safe_load(Path(config_path).read_text())["protocol"]
    fraction = protocol["eval_holdout_fraction"] if fraction is None else fraction
    name = scores_name(split, holdout, fraction)
    scores = json.loads((run_dir / f"scores_{name}.json").read_text())
    run_config = yaml.safe_load((run_dir / "config.yaml").read_text())
    grid = (run_config["data"]["n_timesteps"], run_config["data"]["n_features"])
    masker, settings = holdout_masker(protocol, grid, holdout, fraction)
    sampler = {key: protocol[key] for key in ("eval_sampling_seed", "eval_n_samples", "eval_batch_size")}
    if scores["holdout_settings"] != settings or scores["sampler"] != sampler:
        raise ValueError(f"{run_dir.name}'s {name} scoring did not use the frozen holdout and sampler of {config_path}")
    if scores["model_sha256"] != hashlib.sha256((run_dir / "model.pt").read_bytes()).hexdigest():
        raise ValueError(f"{run_dir.name}'s model.pt is not the weights its {name} scoring imputed with")

    idx = dataset[f"{split}_idx"]
    values, mask = dataset["values"][idx], dataset["mask"][idx]
    _, target = masker(mask, dataset["record_ids"][idx])
    in_span = np.zeros(grid[0], dtype=target.dtype)
    in_span[list(hours)] = 1
    target = target * in_span[None, :, None]
    median = np.load(run_dir / f"median_{name}.npy")
    return {"mae": masked_mae(median, values, target), "n_targets": int(target.sum())}

def attach_to_wandb(scores, mode=None):
    """Add ``scores`` to its run's wandb summary; returns whether it did.

    Keys are the baselines' (``eval.baselines.wandb_summary``): ``<holdout>/<fraction>/<split>/<metric>``
    for the MAE and RMSE. Beside them, for ``diffusion.results`` (M6-2): ``.../gpu`` (null off CUDA,
    and for scores from before M6-2, which did not record it), ``.../tf32`` (whether the pass ran in
    TF32, ``protocol.tf32``; null for scores from before it was recorded), and the sampler's and holdout's
    settings and the per-variable MAE as ``.../sampler.<key>``, ``.../holdout_settings.<key>`` and
    ``.../per_variable_mae.<variable>``, written flat as wandb keeps a nested block anyway.
    Skipped for an untracked run, and unless ``mode`` (default ``WANDB_MODE``, else online) is online.
    """
    mode = mode or os.environ.get("WANDB_MODE") or "online"
    if mode != "online" or scores["wandb_run"] is None:
        return False
    prefix = f"{scores['holdout']}/{scores['fraction']}/{scores['split']}"
    update = {f"{prefix}/mae": scores["mae"], f"{prefix}/rmse": scores["rmse"], f"{prefix}/gpu": scores.get("gpu"),
              f"{prefix}/tf32": scores.get("tf32")}
    for block in ("sampler", "holdout_settings", "per_variable_mae"):
        update |= {f"{prefix}/{block}.{key}": value for key, value in scores.get(block, {}).items()}
    wandb.Api().run(scores["wandb_run"]).summary.update(update)
    return True

def main(argv=None):
    parser = argparse.ArgumentParser(description="Score a finished run's final weights with the frozen sampler (M4-1).")
    parser.add_argument("run_dir", type=Path, help="the run's output folder, e.g. outputs/base_random_seed42")
    parser.add_argument("--holdout", choices=HOLDOUTS, default="random")
    parser.add_argument("--fraction", type=float,
                        help="the random holdout's rate: 0.1 (default), or 0.5 or 0.9, the robustness rates")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="where the frozen protocol is read from")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto (the first available)")
    parser.add_argument("--wandb", choices=("online", "disabled"),
                        help="add the scores to the run's wandb summary (default: WANDB_MODE if set, else online)")
    parser.add_argument("--paired", action="store_true",
                        help="a paired run's scoring (M6-2): refuse to start on any GPU but protocol.paired_gpu")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    print(f"scoring {args.run_dir} on {device}: {args.split} split, {args.holdout} holdout", flush=True)
    scores = score(args.run_dir, load_dataset(args.dataset), device, args.holdout, args.fraction, args.split,
                   config_path=args.config, log=lambda line: print(line, flush=True), paired=args.paired)
    print(json.dumps({key: value for key, value in scores.items() if key != "per_variable_mae"}, indent=1))
    if attach_to_wandb(scores, args.wandb):
        print(f"added to the wandb summary of {scores['wandb_run']}", flush=True)

if __name__ == "__main__":
    main()
