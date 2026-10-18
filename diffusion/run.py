"""Launch one training run from the config and a seed (M3-1).

    uv run python -m diffusion.run --seed 42

Trains with ``diffusion.train.train``, scores the validation split every
``validation.every_n_epochs`` epochs, and writes everything to ``outputs/<run name>/``
(git-ignored) as it goes, so a run that is stopped keeps what it has done:

- ``config.yaml``: the resolved config the run used. ``--config`` on it reruns the run.
- ``history.jsonl``: one record per training step, ``{"epoch", "loss", "grad_norm", "lr"}``.
- ``curve.json``: one row per epoch: mean loss, largest gradient norm, seconds, and val
  MAE/RMSE on the epochs that were scored.
- ``model.pt`` (weights after the latest scored epoch) and ``best.pt`` (lowest val MAE).
- ``summary.json``: the 500-step health report, the best val MAE, and whether it is below
  forward fill's.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data.physionet import load_dataset
from diffusion.train import train
from eval import EvalMasker, EvalSampler, masked_mae, masked_rmse, median_imputation

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
DATASET_PATH = ROOT / "data" / "processed" / "physionet_set_a.npz"
BASELINES_PATH = ROOT / "results" / "baselines.json"
OUTPUTS = ROOT / "outputs"

# Health-report thresholds (ours, M3-1). A spike is a step whose gradient norm is this many
# times the median over the steps checked; an explosion is the last tenth of those steps
# sitting that far above the median, i.e. a spike training never comes back from.
SPIKE_FACTOR = 10.0
COLLAPSE_LOSS = 1.0e-3      # noise-prediction MSE this low means the targets are being leaked or skipped

def pick_device(name="auto"):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def variant_name(config):
    """``base``, ``ext1``, ``ext2`` or ``ext1+2``, from the extension toggles."""
    on = [name for name, toggle in (("ext1", "use_mask_embedding"), ("ext2", "use_time_encoding"))
          if config["model"][toggle]]
    return "+".join(on).replace("ext1+ext2", "ext1+2") or "base"

def forward_fill_val_mae(path=BASELINES_PATH):
    """Forward fill's 10% validation MAE from the recorded baselines (M2-3): M3-1's bar."""
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())["random"]["fractions"]["0.1"]["val"]["forward_fill"]["mae"]

def validate(model, dataset, sampler, masker, split="val"):
    """MAE/RMSE of the median of the sampler's draws on ``masker``'s held-out targets, standardized.

    Scored the way the baselines are (``eval.baselines``): the model sees only ``cond_mask``, and
    only the held-out ``target`` boxes are scored.
    """
    idx = dataset[f"{split}_idx"]
    values, mask = dataset["values"][idx], dataset["mask"][idx]
    cond, target = masker(mask, dataset["record_ids"][idx])
    guess = median_imputation(sampler(model, values, mask, cond))
    return {"mae": masked_mae(guess, values, target), "rmse": masked_rmse(guess, values, target)}

def health_report(history, n_steps=500):
    """Optimisation health over the first ``n_steps`` steps (M3-1 Definition of Done).

    Healthy means every loss and gradient norm is finite, the loss over the last tenth of the
    steps is below the first tenth (it is learning), it has not collapsed toward zero, and the
    gradient norm has not exploded (see ``SPIKE_FACTOR``). Isolated spikes are listed, not
    failed: the known bad values in Set A (README, Dataset) produce some.
    """
    steps = history[:n_steps]
    loss = np.array([r["loss"] for r in steps], dtype=np.float64)
    norm = np.array([r["grad_norm"] for r in steps], dtype=np.float64)
    k = max(1, len(steps) // 10)
    median = float(np.median(norm))
    report = {
        "n_steps": len(steps),
        "finite": bool(np.isfinite(loss).all() and np.isfinite(norm).all()),
        "loss_first_tenth": float(loss[:k].mean()),
        "loss_last_tenth": float(loss[-k:].mean()),
        "grad_norm_median": median,
        "grad_norm_max": float(norm.max()),
        "grad_norm_max_step": int(norm.argmax()),
        "spikes": [{"step": int(i), "loss": float(loss[i]), "grad_norm": float(norm[i])}
                   for i in np.flatnonzero(norm > SPIKE_FACTOR * median)],
    }
    report["learning"] = report["loss_last_tenth"] < report["loss_first_tenth"]
    report["collapsed"] = report["loss_last_tenth"] < COLLAPSE_LOSS
    report["exploded"] = bool(np.median(norm[-k:]) > SPIKE_FACTOR * median)
    report["healthy"] = report["finite"] and report["learning"] and not report["collapsed"] and not report["exploded"]
    return report

def run(config, dataset, seed, device, out_dir, log=print):
    """Train one model and score it on the validation split as it goes; returns ``(model, summary)``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (out_dir / "history.jsonl").write_text("")

    protocol, data, validation = config["protocol"], config["data"], config["validation"]
    masker = EvalMasker(protocol["eval_masker_seed"], protocol["eval_holdout_fraction"],
                        (data["n_timesteps"], data["n_features"]))
    sampler = EvalSampler(protocol["eval_sampling_seed"], validation["n_samples"], validation["batch_size"])
    epochs, bar = config["train"]["epochs"], forward_fill_val_mae()
    history, curve, clock = [], [], {"start": time.time()}

    def on_step_end(record):
        history.append(record)

    def on_epoch_end(epoch, model):
        rows = [r for r in history if r["epoch"] == epoch]
        with open(out_dir / "history.jsonl", "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        row = {"epoch": epoch,
               "loss": float(np.mean([r["loss"] for r in rows])),
               "grad_norm_max": max(r["grad_norm"] for r in rows),
               "lr": rows[-1]["lr"],
               "seconds": round(time.time() - clock["start"], 1)}
        line = f"epoch {epoch + 1}/{epochs}  loss {row['loss']:.4f}  max grad norm {row['grad_norm_max']:.2f}"
        if (epoch + 1) % validation["every_n_epochs"] == 0 or epoch + 1 == epochs:
            started = time.time()
            row.update({f"val_{k}": v for k, v in validate(model, dataset, sampler, masker).items()})
            row["val_seconds"] = round(time.time() - started, 1)
            torch.save(model.state_dict(), out_dir / "model.pt")
            if row["val_mae"] <= min((r["val_mae"] for r in curve if "val_mae" in r), default=np.inf):
                torch.save(model.state_dict(), out_dir / "best.pt")
            line += f"  val MAE {row['val_mae']:.4f}  val RMSE {row['val_rmse']:.4f}  [val {row['val_seconds']:.0f}s]"
            if bar is not None:
                line += f"  (forward fill {bar:.4f})"
        curve.append(row)
        (out_dir / "curve.json").write_text(json.dumps(curve, indent=1) + "\n")
        log(f"{line}  [{row['seconds']:.0f}s]")
        clock["start"] = time.time()

    model, _ = train(config, dataset, seed, device, on_epoch_end=on_epoch_end, on_step_end=on_step_end)

    scored = [r for r in curve if "val_mae" in r]
    best = min(scored, key=lambda r: r["val_mae"])
    summary = {
        "variant": variant_name(config),
        "seed": seed,
        "device": device,
        "validation": {"split": "val", "n_samples": validation["n_samples"], "batch_size": validation["batch_size"]},
        "health": health_report(history),
        "best_val_mae": best["val_mae"],
        "best_epoch": best["epoch"],
        "final_val_mae": scored[-1]["val_mae"],
        "forward_fill_val_mae": bar,
        "below_forward_fill": None if bar is None else best["val_mae"] < bar,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return model, summary

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train one CSDI variant from the config and a seed (M3-1).")
    parser.add_argument("--seed", type=int, required=True, help="run seed, e.g. 42 (protocol.run_seeds)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto (the first available)")
    parser.add_argument("--epochs", type=int, help="override train.epochs; recorded in the saved config")
    parser.add_argument("--out", type=Path, help="output folder (default outputs/<variant>_seed<seed>)")
    args = parser.parse_args(argv)

    config = copy.deepcopy(yaml.safe_load(args.config.read_text()))
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    device = pick_device(args.device)
    out_dir = args.out or OUTPUTS / f"{variant_name(config)}_seed{args.seed}"
    print(f"{variant_name(config)} seed {args.seed} on {device}, {config['train']['epochs']} epochs -> {out_dir}", flush=True)
    _, summary = run(config, load_dataset(args.dataset), args.seed, device, out_dir,
                     log=lambda line: print(line, flush=True))
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
