"""The GPU timing test (M6-2): how long training and scoring take on this GPU, and the sweep's hours.

    uv run python -m diffusion.timing            # writes results/timing/<GPU>.json, e.g. tesla_t4.json
    uv run python -m diffusion.timing --tf32     # with TF32 on, for an L4 or A100: <GPU>_tf32.json

Run on each GPU everyone who runs a paired run can get (T4, L4 and A100 on Colab Pro), before the
dry run: PROJECT_PLAN §3.3 picks the one that finishes the Week-7 sweep soonest (README, Environment).

For base and Ext 1+2 it trains ``TIMED_EPOCHS`` (2) epochs at seed 42 through ``diffusion.run``, as
a sweep run trains (deterministic kernels, pairing hashes, validation), with wandb off, into
``outputs/timing/<variant>``. From each run's ``curve.json`` it takes the second epoch's seconds
(the first also builds the model) and that epoch's validation pass (the last epoch is always
validated). It then times the first ``TIMED_BATCHES`` (2) batches of a frozen test-split scoring
pass on the trained weights (``EvalSampler.batches`` with the protocol's draws and batch size, on
the random holdout); the second is the clean one, as the first also warms the GPU up. Nothing is
scored: no scores file is written and nothing goes to wandb.

The estimates cost each run at its variant's rate, for the cells the config's ``sweep`` block
names (``diffusion.sweep``'s). Ext 2's encoder (1,121 weights) adds about
nothing to base, and Ext 1's encoder is most of what Ext 1+2 adds, so base stands for ext2 and
Ext 1+2 for ext1 and ext1blind (a slight overestimate). The record, ``results/timing/<GPU>[_tf32].json``,
holds the GPU's exact name (what ``protocol.paired_gpu`` must say), the TF32 setting, the timings,
and the estimated hours of the Week-7 sweep and of M6-2's dry run, with the counts they assume.
"""

import argparse
import copy
import json
import math
import re
import time
from pathlib import Path

import torch
import yaml

from data.physionet import load_dataset
from diffusion.results import HEADLINE
from diffusion.run import CONFIG_PATH, DATASET_PATH, OUTPUTS, ROOT, gpu_name, pick_device, run, with_variant
from diffusion.score import holdout_masker
from diffusion.train import use_deterministic_kernels, use_tf32
from eval import EvalSampler

TIMING = ROOT / "results" / "timing"      # the records, one per GPU and TF32 setting
RUNS = OUTPUTS / "timing"                 # the timing runs' folders (git-ignored, like every run's)
TIMED = ("base", "ext1+2")                # the sweep's cheapest and dearest variants
TIMED_EPOCHS = 2                          # the second is the clean one
TIMED_BATCHES = 2                         # the second is the clean one
SEED = 42
# Whose timings each variant's runs are costed at (see the module docstring).
COST_OF = {"base": "base", "ext1": "ext1+2", "ext2": "base", "ext1+2": "ext1+2", "ext1blind": "ext1+2"}
DRY_RUN = {"epochs": 20, "seeds": 2}      # M6-2's dry run: every cell at 20 epochs and 2 seeds, scored as the sweep is

def n_validations(epochs, every):
    """How many validation passes a run of ``epochs`` makes: every ``every`` epochs, and the last."""
    return epochs // every + (epochs % every != 0)

def estimate_hours(timings, cells, epochs, every, n_seeds, n_batches, robustness_passes):
    """Hours to train and score every cell at ``n_seeds`` seeds, plus base's robustness passes.

    ``cells`` is ``{regime: [variant, ...]}``, as the config's ``sweep.regimes`` names the sweep's.
    ``timings[variant]`` has ``epoch_seconds``, ``validation_seconds`` and ``scoring_batch_seconds`` for
    base and ext1+2. Each run trains ``epochs`` epochs, validated every ``every``, and is scored under
    both headline holdouts; a scoring pass is ``n_batches`` batches.
    """
    seconds = 0.0
    for variants in cells.values():
        for variant in variants:
            t = timings[COST_OF[variant]]
            seconds += n_seeds * (epochs * t["epoch_seconds"]
                                  + n_validations(epochs, every) * t["validation_seconds"]
                                  + len(HEADLINE) * n_batches * t["scoring_batch_seconds"])
    seconds += robustness_passes * n_batches * timings["base"]["scoring_batch_seconds"]
    return seconds / 3600

def time_scoring(model, dataset, protocol, n_batches=TIMED_BATCHES):
    """Seconds each of the first ``n_batches`` batches of a frozen test-split scoring pass takes."""
    masker, _ = holdout_masker(protocol, dataset["values"].shape[1:], "random", protocol["eval_holdout_fraction"])
    sampler = EvalSampler(protocol["eval_sampling_seed"], protocol["eval_n_samples"], protocol["eval_batch_size"])
    idx = dataset["test_idx"]
    values, mask = dataset["values"][idx], dataset["mask"][idx]
    cond, _ = masker(mask, dataset["record_ids"][idx])
    seconds, batches = [], sampler.batches(model, values, mask, cond)
    try:
        clock = time.time()
        for _ in batches:                       # a batch's draws are on the CPU when it is yielded, so done
            seconds.append(round(time.time() - clock, 2))
            clock = time.time()
            if len(seconds) == n_batches:
                break
    finally:
        batches.close()                         # hands the model back in training mode
    return seconds

def time_gpu(config, dataset, device, tf32=False, log=print):
    """This GPU's timing record (see the module docstring)."""
    config = copy.deepcopy(config)
    epochs = config["train"]["epochs"]          # what the sweep trains; the timing runs train TIMED_EPOCHS
    config["train"]["epochs"] = TIMED_EPOCHS
    config["protocol"]["tf32"] = tf32
    use_deterministic_kernels()
    use_tf32(tf32)
    timings = {}
    for variant in TIMED:
        model, _ = run(with_variant(config, variant), dataset, SEED, device, RUNS / variant, log=log,
                       overwrite=True, wandb_mode="disabled")
        curve = json.loads((RUNS / variant / "curve.json").read_text())
        batches = time_scoring(model, dataset, config["protocol"])
        timings[variant] = {"epoch_seconds": curve[-1]["seconds"], "first_epoch_seconds": curve[0]["seconds"],
                            "validation_seconds": curve[-1]["val_seconds"],
                            "scoring_batch_seconds": batches[-1], "first_scoring_batch_seconds": batches[0]}
        log(f"{variant}: {timings[variant]}")

    protocol, every, cells = config["protocol"], config["validation"]["every_n_epochs"], config["sweep"]["regimes"]
    n_batches = math.ceil(len(dataset["test_idx"]) / protocol["eval_batch_size"])
    n_seeds, robustness = len(protocol["run_seeds"]), len(protocol["robustness_fractions"]) * len(protocol["run_seeds"])
    return {
        "gpu": gpu_name(device),
        "device": device,
        "tf32": tf32,
        "torch": torch.__version__,
        "timed": {"variants": list(TIMED), "seed": SEED, "epochs": TIMED_EPOCHS, "scoring_batches": TIMED_BATCHES,
                  "validation": {key: config["validation"][key] for key in ("n_samples", "batch_size")},
                  "sampler": {key: protocol[key] for key in ("eval_sampling_seed", "eval_n_samples", "eval_batch_size")}},
        "timings": timings,
        "estimate_hours": {
            "sweep": round(estimate_hours(timings, cells, epochs, every, n_seeds, n_batches, robustness), 1),
            "dry_run": round(estimate_hours(timings, cells, DRY_RUN["epochs"], every, DRY_RUN["seeds"], n_batches, 0), 1),
        },
        "assumes": {"sweep_cells": cells, "cost_of": COST_OF, "train_epochs": epochs, "validate_every": every,
                    "seeds": n_seeds, "holdouts": list(HEADLINE), "scoring_batches_per_pass": n_batches,
                    "robustness_passes": robustness, "dry_run": DRY_RUN},
    }

def record_name(record):
    """``tesla_t4``, ``nvidia_a100_sxm4_40gb_tf32``: the record's file name, from its GPU and TF32 setting."""
    name = re.sub(r"[^a-z0-9]+", "_", (record["gpu"] or record["device"]).lower()).strip("_")
    return f"{name}_tf32" if record["tf32"] else name

def main(argv=None):
    parser = argparse.ArgumentParser(description="Time training and scoring on this GPU, for M6-2's pick of the paired GPU.")
    parser.add_argument("--tf32", action="store_true", help="time with TF32 on (an L4 or A100; the T4 has none)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto (the first available)")
    parser.add_argument("--out", type=Path, help="record path (default results/timing/<GPU>[_tf32].json)")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    record = time_gpu(yaml.safe_load(args.config.read_text()), load_dataset(args.dataset), device, args.tf32,
                      log=lambda line: print(line, flush=True))
    out = args.out or TIMING / f"{record_name(record)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    hours = record["estimate_hours"]
    print(f"wrote {out}: Week-7 sweep about {hours['sweep']} h, dry run about {hours['dry_run']} h "
          f"on {record['gpu'] or device}{', TF32 on' if record['tf32'] else ''}")
    return record

if __name__ == "__main__":
    main()
