"""Launch one training run from the config and a seed (M3-1).

    uv run python -m diffusion.run --seed 42

Trains with ``diffusion.train.train``, scores the validation split every
``validation.every_n_epochs`` epochs, and writes everything to ``outputs/<run name>/``
(git-ignored) as it goes, so a run that is stopped keeps what it has done:

- ``config.yaml``: the resolved config the run used. ``--config`` on it reruns the run.
- ``history.jsonl``: one record per training step, ``{"epoch", "loss", "grad_norm", "lr"}``.
- ``curve.json``: one row per epoch: mean loss, largest gradient norm, seconds, and val
  MAE/RMSE on the epochs that were scored.
- ``model.pt``: the weights after the latest scored epoch. The last epoch is always scored, so a
  finished run leaves its final weights here, the ones that are scored (PROJECT_PLAN §3.1).
- ``summary.json``: the 500-step health report, the final val MAE, and whether it is below
  forward fill's.

The run refuses an output folder that is not empty, so two runs' files never mix; with
``--overwrite`` it first deletes the files a run writes (``RUN_FILES``), and nothing else.

The same numbers stream to wandb as the run goes (M3-2), to the team and project in the
config's ``tracking`` block, under the name ``run_name()`` gives: every step's loss, gradient
norm and learning rate, every epoch's ``curve.json`` row, the config, and ``summary.json`` as
the run's summary. ``--wandb`` picks the mode (online, offline, disabled); without it wandb
follows ``WANDB_MODE`` if that is set, else goes online. ``run()`` called from Python is
untracked unless given ``wandb_mode``.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from data.physionet import load_dataset
from diffusion.train import train
from eval import EvalMasker, EvalSampler, masked_mae, masked_rmse, median_imputation

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
DATASET_PATH = ROOT / "data" / "processed" / "physionet_set_a.npz"
BASELINES_PATH = ROOT / "results" / "baselines.json"
OUTPUTS = ROOT / "outputs"
RUN_FILES = ("config.yaml", "history.jsonl", "curve.json", "model.pt", "summary.json")

# Where runs are logged (M3-2), read from the repo's config rather than the run's, so a
# config saved before M3-2 (no ``tracking`` block) still reruns.
TRACKING = yaml.safe_load(CONFIG_PATH.read_text())["tracking"]
# ponytail: every run trains in the random regime until M6-2 adds TrainMasker's block-aware
# ``mix`` strategy; then read the regime from the config here.
REGIME = "random"

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

def run_name(config, seed):
    """``<variant>_<regime>_seed<seed>``, e.g. ``base_random_seed42``: the run's name on wandb (M3-2)."""
    return f"{variant_name(config)}_{REGIME}_seed{seed}"

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
    report["complete"] = report["n_steps"] == n_steps
    report["healthy"] = (report["finite"]
                         and report["learning"]
                         and not report["collapsed"]
                         and not report["exploded"]
                         and report["complete"])
    return report

def run(config, dataset, seed, device, out_dir, log=print, overwrite=False, wandb_mode="disabled"):
    """Train one model and score it on the validation split as it goes; returns ``(model, summary)``.

    ``out_dir`` must be empty or new. With ``overwrite`` an earlier run's files are deleted first,
    so a run that stops early cannot leave them next to its own. ``wandb_mode`` is passed to
    ``wandb.init``: ``"online"``, ``"offline"``, ``"disabled"`` (the default, so calls from
    Python and the tests are untracked), or ``None`` to let ``WANDB_MODE`` decide.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to replace the run in it")
        for name in RUN_FILES:      # only a run's own files: a mistaken --out must not wipe a folder
            (out_dir / name).unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (out_dir / "history.jsonl").write_text("")

    protocol, data, validation = config["protocol"], config["data"], config["validation"]
    masker = EvalMasker(protocol["eval_masker_seed"], protocol["eval_holdout_fraction"],
                        (data["n_timesteps"], data["n_features"]))
    sampler = EvalSampler(protocol["eval_sampling_seed"], validation["n_samples"], validation["batch_size"])
    epochs, bar = config["train"]["epochs"], forward_fill_val_mae()
    history, curve, clock = [], [], {"start": time.time()}
    variant = variant_name(config)
    tracker = wandb.init(entity=TRACKING["wandb_entity"], project=TRACKING["wandb_project"],
                         mode=wandb_mode, name=run_name(config, seed), job_type="train",
                         tags=[variant, REGIME, f"seed{seed}"],
                         config={**config, "seed": seed, "variant": variant, "regime": REGIME, "device": device})

    def on_step_end(record):
        history.append(record)
        tracker.log({"train/loss": record["loss"], "train/grad_norm": record["grad_norm"],
                     "train/lr": record["lr"], "epoch": record["epoch"]}, step=len(history) - 1)

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
            line += f"  val MAE {row['val_mae']:.4f}  val RMSE {row['val_rmse']:.4f}  [val {row['val_seconds']:.0f}s]"
            if bar is not None:
                line += f"  (forward fill {bar:.4f})"
        curve.append(row)
        (out_dir / "curve.json").write_text(json.dumps(curve, indent=1) + "\n")
        tracker.log({"epoch/loss_mean": row["loss"], "epoch/grad_norm_max": row["grad_norm_max"],
                     "epoch/seconds": row["seconds"],
                     **{f"val/{k[4:]}": v for k, v in row.items() if k.startswith("val_")}},
                    step=len(history) - 1)          # the epoch's last step
        log(f"{line}  [{row['seconds']:.0f}s]")
        clock["start"] = time.time()

    with tracker:                                   # a run that raises is closed on wandb as failed
        model, _ = train(config, dataset, seed, device, on_epoch_end=on_epoch_end, on_step_end=on_step_end)

        final = curve[-1]                                       # the last epoch is always scored
        summary = {
            "variant": variant,
            "seed": seed,
            "device": device,
            "validation": {"split": "val", "n_samples": validation["n_samples"], "batch_size": validation["batch_size"]},
            "health": health_report(history),
            "final_val_mae": final["val_mae"],
            "forward_fill_val_mae": bar,
            "below_forward_fill": None if bar is None else final["val_mae"] < bar,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        tracker.summary.update(summary)
    return model, summary

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train one CSDI variant from the config and a seed (M3-1).")
    parser.add_argument("--seed", type=int, required=True, help="run seed, e.g. 42 (protocol.run_seeds)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto (the first available)")
    parser.add_argument("--epochs", type=int, help="override train.epochs; recorded in the saved config")
    parser.add_argument("--validate-every", type=int,
                        help="override validation.every_n_epochs; recorded in the saved config")
    parser.add_argument("--out", type=Path, help="output folder (default outputs/<variant>_seed<seed>)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete an earlier run's files in the output folder first")
    parser.add_argument("--wandb", choices=("online", "offline", "disabled"),
                        help="wandb mode (default: WANDB_MODE if set, else online)")
    args = parser.parse_args(argv)

    config = copy.deepcopy(yaml.safe_load(args.config.read_text()))
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.validate_every is not None:
        config["validation"]["every_n_epochs"] = args.validate_every
    device = pick_device(args.device)
    out_dir = args.out or OUTPUTS / f"{variant_name(config)}_seed{args.seed}"
    print(f"{variant_name(config)} seed {args.seed} on {device}, {config['train']['epochs']} epochs -> {out_dir}", flush=True)
    _, summary = run(config, load_dataset(args.dataset), args.seed, device, out_dir,
                     log=lambda line: print(line, flush=True), overwrite=args.overwrite,
                     wandb_mode=args.wandb)
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
