"""Launch one training run from the config and a seed (M3-1).

    uv run python -m diffusion.run --seed 42

Trains with ``diffusion.train.train``, scores the validation split every
``validation.every_n_epochs`` epochs on the holdout its regime trains for (``train.regime``, M6-2:
``EvalMasker``'s in the random regime, ``BlockEvalMasker``'s in the block-aware one), and writes everything to ``outputs/<run name>/``
(git-ignored) as it goes, so a run that is stopped keeps what it has done:

- ``config.yaml``: the resolved config the run used. ``--config`` on it reruns the run.
- ``history.jsonl``: one record per training step, ``{"epoch", "loss", "grad_norm", "lr"}``.
- ``curve.json``: one row per epoch: mean loss, largest gradient norm, seconds, the SHA-256 of
  what each paired stream gave the epoch (M6-2: its batches, its train masks, and its diffusion
  draws), and val MAE/RMSE on the epochs that were scored.
- ``model.pt``: the weights after the latest scored epoch. The last epoch is always scored, so a
  finished run leaves its final weights here, the ones that are scored (PROJECT_PLAN §3.1).
- ``summary.json``: the 500-step health report, the final val MAE, whether it is below
  forward fill's, the GPU it trained on, its pairing hashes (M6-2: the starting weights of each
  part, and each paired stream over the whole run), the SHA-256 of its final ``model.pt``
  (``model_sha256``, which ``diffusion.score`` checks), and the run's path on wandb
  (``entity/project/id``; null if untracked).
- ``checkpoint.pt`` (M6-2): the model, optimizer and learning-rate schedule, the epoch, the number
  of steps so far, the run's wandb path, and the device and GPU it was written on, rewritten whole
  after every validated epoch but the last (see ``run``).

The run refuses an output folder that is not empty, so two runs' files never mix; with
``--overwrite`` it first deletes the files a run writes (``RUN_FILES``), and nothing else. With
``--resume`` (M6-2) it continues the run in the folder from its checkpoint, on the same wandb run,
and ends with the same files and numbers as a run that was never stopped; a folder with no usable
checkpoint (none yet, one its ``history.jsonl`` and ``curve.json`` do not reach, or one written on
another device or GPU) is started again.

The same numbers stream to wandb as the run goes (M3-2), to the team and project in the
config's ``tracking`` block, under the name ``run_name()`` gives: every step's loss, gradient
norm and learning rate, every epoch's ``curve.json`` row, the config and output folder, and
``summary.json`` as the run's summary. ``--wandb`` picks the mode (online, offline, disabled);
without it wandb follows ``WANDB_MODE`` if that is set, else goes online, and so does ``run()``
called from Python without ``wandb_mode``. ``--tag`` adds tags that say what a run is for.

A paired run (``run(..., paired=True)`` or ``--paired``, the M6-2 harness's) starts only on the GPU
the timing test picked, ``protocol.paired_gpu`` (PROJECT_PLAN §3.3), and not at all before one is
picked.
"""

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from data.physionet import load_dataset
from diffusion.train import REGIMES, regime_name, train
from eval import BlockEvalMasker, EvalMasker, EvalSampler, masked_mae, masked_rmse, median_imputation

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
DATASET_PATH = ROOT / "data" / "processed" / "physionet_set_a.npz"
BASELINES_PATH = ROOT / "results" / "baselines.json"
OUTPUTS = ROOT / "outputs"
RUN_FILES = ("config.yaml", "history.jsonl", "curve.json", "model.pt", "summary.json", "checkpoint.pt")
CHECKPOINT_TEMP = "checkpoint.pt.tmp"       # written, then renamed: a session lost mid-save leaves the last one whole

# Where runs are logged (M3-2). Not part of a run: read from the repo's config, and left out
# of a run's saved config and its wandb config, so every saved config reruns the same way.
TRACKING = yaml.safe_load(CONFIG_PATH.read_text())["tracking"]
# The config blocks that are not part of a run, so are left out of its saved and wandb config: where
# runs are logged, and which runs diffusion.sweep launches (M6-2), which may grow (M8-4's ext1blind)
# without changing a run already trained.
NOT_A_RUN = ("tracking", "sweep")

# Health-report thresholds (ours, M3-1). A spike is a step whose gradient norm is this many
# times the median over the steps checked; an explosion is the last tenth of those steps
# sitting that far above the median, i.e. a spike training never comes back from.
SPIKE_FACTOR = 10.0
COLLAPSE_LOSS = 1.0e-3      # noise-prediction MSE this low means the targets are being leaked or skipped

def pick_device(name="auto"):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def gpu_name(device):
    """The GPU model ``device`` computes on, as ``nvidia-smi`` names it (``Tesla T4``); None off CUDA."""
    return torch.cuda.get_device_name(torch.device(device)) if str(device).startswith("cuda") else None

def require_paired_gpu(protocol, device):
    """Refuse a paired run on any GPU but ``protocol.paired_gpu`` (PROJECT_PLAN §3.3, M6-2).

    Shared seeds pair two runs only on one kind of GPU: on another, float rounding differs from the
    first optimizer step. Raises ``RuntimeError``, also while the timing test has not picked one.
    """
    wanted, found = protocol.get("paired_gpu"), gpu_name(device)
    if wanted is None:
        raise RuntimeError("protocol.paired_gpu is not set: M6-2's GPU timing test picks it, "
                           "and no paired run starts before it does")
    if found != wanted:
        raise RuntimeError(f"a paired run goes on a {wanted}, not on {found or device}: "
                           f"wait for a {wanted}, never take another GPU (README, Environment)")

# Extension encoders, as models.build_model attaches them; every other weight is shared by all
# four variants, so starts the same in each at a seed (README, Random streams).
ENCODERS = ("mask_encoder", "time_encoder")
# The streams a seed's variants share, each hashed as training draws from it (M6-2): the batches in
# their order, the train masks, and the diffusion timesteps and noise (PROJECT_PLAN §3.3).
PAIRED_STREAMS = ("data_order", "train_mask", "diffusion")

def weight_hashes(state_dict):
    """``{"shared": ..., "mask_encoder": ...}``: a SHA-256 of each part's weights, for the parts the model has."""
    hashes = {}
    for key in sorted(state_dict):
        part = key.split(".")[0] if key.split(".")[0] in ENCODERS else "shared"
        digest = hashes.setdefault(part, hashlib.sha256())
        digest.update(key.encode())
        digest.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return {part: digest.hexdigest() for part, digest in hashes.items()}

def chain_sha256(digests):
    """One SHA-256 over a run's per-epoch digests, in order: a whole stream as one value."""
    return hashlib.sha256("".join(digests).encode()).hexdigest()

# The variants (M6-1): each one's extension toggles in model:, every one written on or off, so
# setting a variant never depends on what the config held before. This table is the only place
# a variant is defined: with_variant() and --variant set it, VARIANT_FLAGS gives the toggle flags
# that switch on the same ones, and variant_name() reads it back. It lists every variant a run can
# be, not the sweep: the harness (M6-2) names the variants it trains, so a variant added here
# (M6-3's ext1blind, §4.1's control, trained only if M8-4 is triggered) is not swept by being added.
VARIANTS = {
    "base": {"use_mask_embedding": False, "use_time_encoding": False, "mask_embedding_blind": False},
    "ext1": {"use_mask_embedding": True, "use_time_encoding": False, "mask_embedding_blind": False},
    "ext2": {"use_mask_embedding": False, "use_time_encoding": True, "mask_embedding_blind": False},
    "ext1+2": {"use_mask_embedding": True, "use_time_encoding": True, "mask_embedding_blind": False},
    "ext1blind": {"use_mask_embedding": True, "use_time_encoding": False, "mask_embedding_blind": True},
}
# The toggle flags of each variant, for a config whose toggles are all off (the repo's). They only
# switch toggles on; --variant also switches them off.
VARIANT_FLAGS = {name: [f"--{toggle}" for toggle, on in toggles.items() if on]
                 for name, toggles in VARIANTS.items()}

def with_variant(config, name):
    """A copy of ``config`` set to variant ``name``: every toggle in ``VARIANTS`` written, on or off."""
    config = copy.deepcopy(config)
    config["model"].update(VARIANTS[name])
    return config

def variant_name(config):
    """``base``, ``ext1``, ``ext2``, ``ext1+2`` or ``ext1blind``, from the extension toggles (``VARIANTS``).

    A toggle a config does not have is off: a config saved before M6-3 has no ``mask_embedding_blind``.
    """
    toggles = {toggle: bool(config["model"].get(toggle, False)) for toggle in VARIANTS["base"]}
    for name, variant in VARIANTS.items():
        if variant == toggles:
            return name
    raise ValueError(f"the extension toggles {toggles} are none of the variants {list(VARIANTS)}")

def run_name(config, seed):
    """``<variant>_<regime>_seed<seed>``, e.g. ``base_random_seed42``: the run's name on wandb (M3-2)."""
    return f"{variant_name(config)}_{regime_name(config)}_seed{seed}"

def validation_masker(config):
    """The holdout a run's validation tracks, built from the run's own config: the one its regime
    trains for (PROJECT_PLAN §3.1), at the headline 10%."""
    protocol, data = config["protocol"], config["data"]
    grid = (data["n_timesteps"], data["n_features"])
    if REGIMES[regime_name(config)]["holdout"] == "block":
        return BlockEvalMasker(protocol["block_eval_masker_seed"], protocol["eval_holdout_fraction"],
                               protocol["block_window_hours"], grid)
    return EvalMasker(protocol["eval_masker_seed"], protocol["eval_holdout_fraction"], grid)

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

def history_records(path):
    """The records of a ``history.jsonl``, up to a line cut short by a session lost while writing it."""
    records = []
    for line in path.read_text().splitlines() if path.exists() else ():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return records

def resume_point(out_dir, config, device, log=print):
    """``(checkpoint, history, curve)`` to resume the unfinished run in ``out_dir`` from; None to start it again.

    ``history`` and ``curve`` are its ``history.jsonl`` records and ``curve.json`` rows up to the
    checkpoint's epoch; anything later is redone. None if it has no checkpoint yet; if the checkpoint
    was written on another device or GPU than ``device`` (review of #58: float rounding differs from
    one to another, so a run trained on two is neither one never stopped nor paired, yet its summary
    would name only the last); or if those files do not reach the checkpoint: Google Drive uploads
    each file in the background, so a session lost on it can leave a newer ``checkpoint.pt`` than
    them, and a run resumed from it would lack the steps and epochs in between (review of #57).
    Starting again is always right. Refuses a finished run (nothing to resume) and a run whose saved
    config is not ``config``: resuming with another config would end as neither run.
    """
    if (out_dir / "summary.json").exists():
        raise FileExistsError(f"{out_dir} holds a finished run: there is nothing to resume")
    if not (out_dir / "checkpoint.pt").exists():
        return None
    if yaml.safe_load((out_dir / "config.yaml").read_text()) != config:
        raise ValueError(f"the run in {out_dir} was started with another config: resume it with its own "
                         "(--config on its config.yaml)")
    checkpoint = torch.load(out_dir / "checkpoint.pt", map_location="cpu")
    done, here = checkpoint["epoch"], {"device": device, "gpu": gpu_name(device)}
    if {key: checkpoint.get(key) for key in here} != here:
        written_on = checkpoint.get("gpu") or checkpoint.get("device") or "a device it does not record"
        log(f"{out_dir.name}'s checkpoint was written on {written_on}, not on {here['gpu'] or device}: "
            "starting it again")
        return None
    history =[record for record in history_records(out_dir / "history.jsonl") if record["epoch"] <= done]
    try:
        curve = [row for row in json.loads((out_dir / "curve.json").read_text()) if row["epoch"] <= done]
    except (FileNotFoundError, json.JSONDecodeError):
        curve = []
    if len(history) != checkpoint.get("steps") or [row["epoch"] for row in curve] != list(range(done + 1)):
        log(f"{out_dir.name}'s history.jsonl and curve.json do not reach its checkpoint at epoch {done + 1}: "
            "starting it again")
        return None
    return checkpoint, history, curve

def run(config, dataset, seed, device, out_dir, log=print, overwrite=False, wandb_mode=None, tags=(), paired=False,
        resume=False):
    """Train one model and score it on the validation split as it goes; returns ``(model, summary)``.

    ``out_dir`` must be empty or new. With ``overwrite`` an earlier run's files are deleted first,
    so a run that stops early cannot leave them next to its own. With ``resume`` (M6-2) the
    unfinished run in ``out_dir`` goes on from its checkpoint, on the same wandb run, keeping its
    earlier steps and epochs (each stream's hash chain is rebuilt from them), and ends with the same
    files and numbers as a run that was never stopped; with no usable checkpoint (``resume_point``),
    the run starts again. The checkpoint is written after every validated epoch, when ``model.pt``
    is too, but not after the last: the summary follows the last, so a run stopped between the two
    redoes its last ``validation.every_n_epochs`` epochs, and the final ``model.pt`` whose hash the
    summary records is always one written in the same session (review of #57). ``wandb_mode`` is passed to
    ``wandb.init``: ``"online"``, ``"offline"``, ``"disabled"``, or ``None`` (the default) to let
    ``WANDB_MODE`` decide, as the command line does, so a harness looping over ``run()`` is
    tracked unless told otherwise. ``tags`` are added to the wandb run's variant, regime and seed.
    Nothing is written to ``out_dir`` until wandb has started, so a run that cannot start (no
    wandb login) can simply be run again. With ``paired`` (a sweep run, M6-2) it refuses to start,
    before writing or sending anything, on any GPU but ``protocol.paired_gpu``. A config whose
    toggles are none of the ``VARIANTS`` (``mask_embedding_blind`` without Ext 1, M6-3) or whose
    ``train.regime`` is none of the ``REGIMES`` is refused first, before ``overwrite`` deletes anything.
    """
    variant = variant_name(config)                  # raises ValueError for toggles that are no variant
    regime = regime_name(config)                    # ... and for a regime that is none of REGIMES
    if paired:
        require_paired_gpu(config["protocol"], device)
    config = {key: value for key, value in config.items() if key not in NOT_A_RUN}
    out_dir = Path(out_dir)
    point = resume_point(out_dir, config, device, log) if resume else None
    checkpoint, history, curve = (None, [], []) if point is None else point
    if checkpoint is None and out_dir.exists() and any(out_dir.iterdir()):
        if not (overwrite or resume):
            raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to replace the run in it")
        for name in (*RUN_FILES, CHECKPOINT_TEMP):  # only a run's own files: a mistaken --out must not wipe a folder
            (out_dir / name).unlink(missing_ok=True)

    protocol, validation = config["protocol"], config["validation"]
    masker = validation_masker(config)
    sampler = EvalSampler(protocol["eval_sampling_seed"], validation["n_samples"], validation["batch_size"])
    epochs, bar = config["train"]["epochs"], forward_fill_val_mae()
    clock = {"start": time.time()}

    def validated(epoch):                       # every every_n_epochs epochs, and the last
        return (epoch + 1) % validation["every_n_epochs"] == 0 or epoch + 1 == epochs

    # Pairing hashes (M6-2): the starting weights, then what each paired stream gives every epoch as
    # training draws it, so diffusion.results can check a seed's variants were given the same.
    init, epoch_streams = {}, {}
    same_wandb_run = {} if checkpoint is None or checkpoint["wandb_run"] is None else \
        {"id": checkpoint["wandb_run"].split("/")[-1], "resume": "must"}
    tracker = wandb.init(entity=TRACKING["wandb_entity"], project=TRACKING["wandb_project"],
                         mode=wandb_mode, name=run_name(config, seed), job_type="train",
                         tags=[variant, regime, f"seed{seed}", *tags],
                         config={**config, "seed": seed, "variant": variant, "regime": regime, "device": device,
                                 "gpu": gpu_name(device), "out_dir": str(out_dir.resolve())},
                         **same_wandb_run)

    def on_start(weights):                                      # as built, before any update
        init.update(weight_hashes(weights))

    def on_batch(epoch, values, mask, cond_mask, diffusion_state):
        if not epoch_streams:
            epoch_streams.update({stream: hashlib.sha256() for stream in PAIRED_STREAMS})
        epoch_streams["data_order"].update(values.numpy().tobytes())
        epoch_streams["data_order"].update(mask.numpy().tobytes())
        epoch_streams["train_mask"].update(cond_mask.numpy().tobytes())
        # The generator is re-seeded every epoch, so its state after each step says how far the
        # step's timesteps and noise took it: equal at every step means the same draws.
        epoch_streams["diffusion"].update(diffusion_state.numpy().tobytes())

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
               "seconds": round(time.time() - clock["start"], 1),
               **{f"{stream}_sha256": digest.hexdigest() for stream, digest in epoch_streams.items()}}
        epoch_streams.clear()
        line = f"epoch {epoch + 1}/{epochs}  loss {row['loss']:.4f}  max grad norm {row['grad_norm_max']:.2f}"
        if validated(epoch):
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

    def on_checkpoint(state):                       # after on_epoch_end, so the files hold the epoch too
        # Only after a validated epoch, with model.pt (a tenth of the Drive uploads of every epoch), and
        # never after the last (see the docstring). steps: how far history.jsonl must reach to resume.
        if not validated(state["epoch"]) or state["epoch"] + 1 == epochs:
            return
        torch.save({**state, "steps": len(history), "wandb_run": None if tracker.disabled else tracker.path,
                    "device": device, "gpu": gpu_name(device)},           # resumed only where it was written
                   out_dir / CHECKPOINT_TEMP)
        os.replace(out_dir / CHECKPOINT_TEMP, out_dir / "checkpoint.pt")

    with tracker:                                   # a run that raises is closed on wandb as failed
        out_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint is None:
            (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        else:
            log(f"resuming {out_dir.name} at epoch {checkpoint['epoch'] + 2}/{epochs}, from its checkpoint")
        (out_dir / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
        if curve:
            (out_dir / "curve.json").write_text(json.dumps(curve, indent=1) + "\n")
        model, _ = train(config, dataset, seed, device, on_epoch_end=on_epoch_end, on_step_end=on_step_end,
                         on_batch=on_batch, on_start=on_start, on_checkpoint=on_checkpoint, resume=checkpoint)

        final = curve[-1]                                       # the last epoch is always scored
        summary = {
            "variant": variant,
            "seed": seed,
            "device": device,
            "gpu": gpu_name(device),
            # one level deep: wandb's summary flattens it to pairing.<key>, which summary_block reads back
            "pairing": {**{f"init_{part}": digest for part, digest in init.items()},
                        **{stream: chain_sha256(row[f"{stream}_sha256"] for row in curve)
                           for stream in PAIRED_STREAMS}},
            "validation": {"split": "val", "holdout": REGIMES[regime]["holdout"],
                           "n_samples": validation["n_samples"], "batch_size": validation["batch_size"]},
            "health": health_report(history),
            "final_val_mae": final["val_mae"],
            # The final weights, as this session wrote them: diffusion.score refuses a model.pt that is not
            # them, e.g. an earlier epoch's that Google Drive kept when a session was lost (review of #57).
            "model_sha256": hashlib.sha256((out_dir / "model.pt").read_bytes()).hexdigest(),
            "forward_fill_val_mae": bar,
            "below_forward_fill": None if bar is None else final["val_mae"] < bar,
            # entity/project/id on wandb, so a later step (the gate's test scoring, M4-1) can find the run
            "wandb_run": None if tracker.disabled else tracker.path,
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
    parser.add_argument("--regime", choices=list(REGIMES),
                        help="set train.regime (M6-2): random, or block-aware (TrainMasker's mix strategy, "
                             "validated on the block holdout); recorded in the saved config")
    parser.add_argument("--variant", choices=list(VARIANTS),
                        help="set every extension toggle to this variant's, on or off (M6-1); recorded in "
                             "the saved config. Not with a toggle flag")
    parser.add_argument("--use_mask_embedding", action="store_true",
                        help="turn on model.use_mask_embedding (Extension 1, M4-2); recorded in the saved config")
    parser.add_argument("--use_time_encoding", action="store_true",
                        help="turn on model.use_time_encoding (Extension 2, M5-2); recorded in the saved config")
    parser.add_argument("--mask_embedding_blind", action="store_true",
                        help="turn on model.mask_embedding_blind (Ext 1-blind, M6-3), with --use_mask_embedding; "
                             "recorded in the saved config")
    parser.add_argument("--out", type=Path,
                        help="output folder (default outputs/<run name>, e.g. outputs/base_random_seed42)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete an earlier run's files in the output folder first")
    parser.add_argument("--resume", action="store_true",
                        help="continue the unfinished run in the output folder from its last epoch (M6-2), "
                             "with the same flags; start it again if it never finished one")
    parser.add_argument("--wandb", choices=("online", "offline", "disabled"),
                        help="wandb mode (default: WANDB_MODE if set, else online)")
    parser.add_argument("--tag", action="append", default=[],
                        help="a wandb tag saying what the run is for, e.g. gate, sweep, dry-run, debug (repeatable)")
    parser.add_argument("--paired", action="store_true",
                        help="a paired run (M6-2): refuse to start on any GPU but protocol.paired_gpu")
    args = parser.parse_args(argv)
    if args.variant is not None and (args.use_mask_embedding or args.use_time_encoding or args.mask_embedding_blind):
        parser.error("--variant sets every extension toggle; give it or the toggle flags, not both")
    if args.resume and args.overwrite:
        parser.error("--resume continues the run in the folder and --overwrite deletes it: give one")

    config = copy.deepcopy(yaml.safe_load(args.config.read_text()))
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.validate_every is not None:
        config["validation"]["every_n_epochs"] = args.validate_every
    if args.regime is not None:
        config["train"]["regime"] = args.regime
    if args.variant is not None:
        config = with_variant(config, args.variant)
    if args.use_mask_embedding:
        config["model"]["use_mask_embedding"] = True
    if args.use_time_encoding:
        config["model"]["use_time_encoding"] = True
    if args.mask_embedding_blind:
        config["model"]["mask_embedding_blind"] = True
    try:
        variant_name(config)
        regime_name(config)
    except ValueError as error:                 # e.g. --mask_embedding_blind without Ext 1
        parser.error(str(error))
    device = pick_device(args.device)
    out_dir = args.out or OUTPUTS / run_name(config, args.seed)             # the run's wandb name
    print(f"{variant_name(config)} {regime_name(config)} seed {args.seed} on {device}, {config['train']['epochs']} epochs -> {out_dir}", flush=True)
    _, summary = run(config, load_dataset(args.dataset), args.seed, device, out_dir,
                     log=lambda line: print(line, flush=True), overwrite=args.overwrite,
                     wandb_mode=args.wandb, tags=args.tag, paired=args.paired, resume=args.resume)
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
