"""Extension 2's mechanical health check (M5-2): does the model with the ContinuousTimeEncoding train?

    uv run python -m diffusion.ext2_health

Trains Ext 2 at one seed through the unchanged ``diffusion.run.run`` for the fewest epochs that
reach ``HEALTH_STEPS`` optimizer steps (3 epochs at batch 16), so the run's own 500-step health
report is complete: every loss and gradient norm finite, the loss dropping, no collapse toward 0,
no exploding gradients. It also records how far the encoder's gate (which starts at 0) has opened
and the encoder's gradient norm on the last step, both of which must be nonzero for the encoder to
be in use, and passes only if all three hold. That is the issue's mechanical gate, nothing more.

The issue asks for the check on high-autocorrelation variables, where recency should matter most
(PROJECT_PLAN §4.2), so the record adds validation MAE on ``VARIABLES`` alone, scored exactly as the
run's own validation (``EvalMasker``'s targets, the validation sampler's frozen draws, median).
§4.2 names heart rate and SpO2; SpO2 is not one of our 35 features (SaO2, a sparse lab, is), so
SysABP stands in: after HR it is the densest vital apart from Urine, and changes slowly hour to hour
(settled in M5-2).

The base model goes through the same short run and is recorded beside it as ``base_reference``,
for scale only: three epochs say nothing about whether Ext 2 helps, which waits for the paired
5-seed sweep. The runs are not tracked on wandb: ``run`` names them after variant and seed, and
``ext2_random_seed42`` is the name of the full single-seed health run (M5-3).

Writes ``results/ext2_health.json``; the run folders go to ``outputs/ext2_health/``.
"""

import argparse
import copy
import json
import math
import platform
import time
from pathlib import Path

import torch
import yaml

from data.physionet import load_dataset
from diffusion.overfit import grad_norm
from diffusion.run import CONFIG_PATH, DATASET_PATH, OUTPUTS, ROOT, pick_device, run
from eval import EvalMasker, EvalSampler, masked_mae, median_imputation, per_variable_mae

HEALTH_PATH = ROOT / "results" / "ext2_health.json"
RUNS_DIR = OUTPUTS / "ext2_health"
HEALTH_STEPS = 500          # diffusion.run.health_report's window
VARIABLES = ("HR", "SysABP")

def epochs_for(config, dataset, n_steps=HEALTH_STEPS):
    """The fewest epochs whose steps reach ``n_steps``; the loader keeps the last, short batch."""
    steps_per_epoch = math.ceil(len(dataset["train_idx"]) / config["train"]["batch_size"])
    return math.ceil(n_steps / steps_per_epoch)

def short_run(config, dataset, ext2, seed, device, out_dir, n_steps=HEALTH_STEPS, log=print):
    """Ext 2 (base with ``ext2=False``) through ``diffusion.run.run`` for ``epochs_for`` epochs."""
    config = copy.deepcopy(config)
    config["model"].update(use_mask_embedding=False, use_time_encoding=ext2)
    config["train"]["epochs"] = epochs_for(config, dataset, n_steps)
    model, summary = run(config, dataset, seed, device, out_dir, log=log, overwrite=True, wandb_mode="disabled")
    return model, summary, config

def variable_mae(model, dataset, config, names=VARIABLES, split="val"):
    """Validation MAE on ``names`` alone, standardized and in raw units, scored as ``run``'s validation is.

    The same masker and sampler as ``diffusion.run.run``, whose sampler re-seeds itself each pass,
    so ``"all"``, the MAE over every target, reproduces the run's final val MAE.
    """
    protocol, data, validation = config["protocol"], config["data"], config["validation"]
    masker = EvalMasker(protocol["eval_masker_seed"], protocol["eval_holdout_fraction"],
                        (data["n_timesteps"], data["n_features"]))
    sampler = EvalSampler(protocol["eval_sampling_seed"], validation["n_samples"], validation["batch_size"])
    idx = dataset[f"{split}_idx"]
    values, mask = dataset["values"][idx], dataset["mask"][idx]
    cond, target = masker(mask, dataset["record_ids"][idx])
    guess = median_imputation(sampler(model, values, mask, cond))
    mae = per_variable_mae(guess, values, target)
    raw = per_variable_mae(guess, values, target, std=dataset["std"])
    features = list(dataset["features"])
    scores = {}
    for name in names:
        j = features.index(name)
        scores[name] = {"mae": round(float(mae[j]), 6), "mae_raw": round(float(raw[j]), 4),
                        "n_targets": int(target[..., j].sum())}
    scores["all"] = {"mae": round(masked_mae(guess, values, target), 6), "n_targets": int(target.sum())}
    return scores

def status(health, gate, encoder_grad_norm, scores):
    """PASS when the run is healthy and the encoder is in use: gate off 0, gradients reaching it."""
    finite = all(math.isfinite(s["mae"]) for s in scores.values())
    return "PASS" if health["healthy"] and gate != 0 and encoder_grad_norm > 0 and finite else "FAIL"

def health_check(config, dataset, seed=42, device="cpu", runs_dir=RUNS_DIR, n_steps=HEALTH_STEPS, log=print):
    """Ext 2's short run, scored, with base's beside it; returns the record."""
    started = time.time()
    model, summary, run_config = short_run(config, dataset, True, seed, device, runs_dir / "ext2", n_steps, log)
    encoder = model.time_encoder
    gate, encoder_grad = encoder.gate.item(), grad_norm(encoder)    # the last step's, still on the weights
    scores = variable_mae(model, dataset, run_config)
    train_config = run_config["train"]
    record = {
        "check": "M5-2 mechanical health check, Extension 2 (ContinuousTimeEncoding)",
        "note": "health only, not an effect measurement; whether Ext 2 helps waits for the paired 5-seed sweep",
        "status": status(summary["health"], gate, encoder_grad, scores),
        "variant": summary["variant"],
        "seed": seed,
        "epochs": train_config["epochs"],
        "steps": train_config["epochs"] * math.ceil(len(dataset["train_idx"]) / train_config["batch_size"]),
        "lr_drops_at_epochs": [max(1, int(f * train_config["epochs"])) for f in train_config["lr_milestones"]],
        "health": summary["health"],
        "final_val_mae": summary["final_val_mae"],
        "variable_mae": scores,
        "variables_note": "SysABP stands in for SpO2, which is not one of the 35 features",
        "validation": summary["validation"],
        "time_encoder": {
            "d_k": run_config["model"]["time_encoding_dk"],
            "n_parameters": sum(p.numel() for p in encoder.parameters()),
            "gate_last_step": gate,                                 # starts at 0
            "grad_norm_last_step": encoder_grad,
        },
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "device": device,
        "machine": platform.platform(),
        "torch": torch.__version__,
    }
    record["seconds"] = round(time.time() - started, 1)            # Ext 2's run and scoring

    started = time.time()
    base, base_summary, base_config = short_run(config, dataset, False, seed, device, runs_dir / "base", n_steps, log)
    record["base_reference"] = {
        "note": "the same short run without the encoder, for scale; not an effect of Ext 2",
        "variant": base_summary["variant"],
        "health": base_summary["health"],
        "final_val_mae": base_summary["final_val_mae"],
        "variable_mae": variable_mae(base, dataset, base_config),
        "n_parameters": sum(p.numel() for p in base.parameters()),
        "seconds": round(time.time() - started, 1),
    }
    return record

def main(argv=None):
    parser = argparse.ArgumentParser(description="Extension 2's mechanical health check (M5-2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--runs-dir", default=RUNS_DIR, type=Path, help="where the two run folders go")
    parser.add_argument("--steps", type=int, default=HEALTH_STEPS,
                        help="steps to reach; below 500 the health report is incomplete and the check FAILs")
    parser.add_argument("--out", default=HEALTH_PATH)
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    dataset, device = load_dataset(args.dataset), pick_device(args.device)
    record = health_check(config, dataset, args.seed, device, args.runs_dir, args.steps)
    with open(args.out, "w") as f:
        f.write(json.dumps(record, indent=1) + "\n")
    base = record["base_reference"]
    hr, sys_abp = (record["variable_mae"][name]["mae"] for name in VARIABLES)
    print(f"{record['status']}: healthy {record['health']['healthy']}, gate {record['time_encoder']['gate_last_step']:.3g}, "
          f"encoder grad norm {record['time_encoder']['grad_norm_last_step']:.3g}; val MAE {record['final_val_mae']:.4f} "
          f"(HR {hr:.4f}, SysABP {sys_abp:.4f}); base on the same run {base['final_val_mae']:.4f}  -> {args.out}")

if __name__ == "__main__":
    main()
