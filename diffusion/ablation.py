"""The ablation's driver (M8-1): three one-factor studies on the best variant, trained and scored as the sweep is.

    uv run python -m diffusion.ablation --study d_model --tag ablation --out <a folder on Google Drive>

Three bounded studies, not a grid (PROJECT_PLAN §5, ``Milestones/M08_Ablation_Runtime.md``). Each moves
one setting away from the sweep's, at seeds 42-44 in the random regime, every other setting the sweep's:

- ``d_model``: ``diffusion.channels`` at 64 / 128 / 256 (``model.d_model_sweep``) on Ext 1+2, the best
  variant (M7-2), with ``model.n_heads`` fixed at 8 (§3.6); scored on the random holdout;
- ``steps``: the diffusion steps T, ``diffusion.num_steps``, at 50 / 100 / 200 on Ext 1+2, each a full
  retrain; scored on the random holdout;
- ``encoder_layers``: Ext 1's encoder depth, ``model.mask_encoder_layers``, at 1 / 2 / 3 on the two
  variants with that encoder, Ext 1 and Ext 1+2; scored on the random and the block holdout.

The sweep's own value of each setting (64, 50, 2) is not trained again: the sweep's runs at seeds
42-44 are those cells. So each study trains its other two values, 24 runs in all. Each value trains
into ``<out>/<study>_<value>`` (e.g. ``<out>/d_model_128``) with ``diffusion.sweep.sweep``, so a run
goes on where a lost session stopped, trains and is scored on ``protocol.paired_gpu`` from committed
code (each folder's ``sweep.json`` records the tag and commit), and in the encoder-layers study Ext 1's
and Ext 1+2's pairing hashes are checked against each other at each seed. The same command, run again
in a new session, finishes what the last one started.

Ablation runs validate every ``VALIDATE_EVERY`` epochs, not the sweep's 10. Validation is tracking only:
every stream is reseeded at the start of each epoch (``diffusion.streams``), so it leaves training
unchanged, and at T = 200 the sweep's 20 passes would take about an hour a run. A run lost with its
session redoes at most that many epochs, since its checkpoint is written after each validated epoch.
"""

import argparse
import copy
from pathlib import Path

import yaml

from data.physionet import load_dataset
from diffusion.run import CONFIG_PATH, DATASET_PATH, OUTPUTS, pick_device
from diffusion.sweep import cells, sweep

SEEDS = (42, 43, 44)
VALIDATE_EVERY = 50
# Each study's one setting (config block, key), its values, the variants it trains and the holdouts it scores
STUDIES = {
    "d_model": {"setting": ("diffusion", "channels"), "values": (64, 128, 256),
                "variants": ("ext1+2",), "holdouts": ("random",)},
    "steps": {"setting": ("diffusion", "num_steps"), "values": (50, 100, 200),
              "variants": ("ext1+2",), "holdouts": ("random",)},
    "encoder_layers": {"setting": ("model", "mask_encoder_layers"), "values": (1, 2, 3),
                       "variants": ("ext1", "ext1+2"), "holdouts": ("random", "block")},
}

def setting(config, study):
    """``study``'s setting in ``config``."""
    block, key = STUDIES[study]["setting"]
    return config[block][key]

def study_config(config, study, value):
    """A copy of ``config`` with ``study``'s setting at ``value``, validated every ``VALIDATE_EVERY`` epochs."""
    config = copy.deepcopy(config)
    block, key = STUDIES[study]["setting"]
    config[block][key] = value
    config["validation"]["every_n_epochs"] = VALIDATE_EVERY
    return config

def trained_values(config, study):
    """The values ``study`` trains: all but ``config``'s own, whose runs the sweep already holds."""
    values, own = STUDIES[study]["values"], setting(config, study)
    if own not in values:
        raise ValueError(f"the config's {'.'.join(STUDIES[study]['setting'])} is {own}, not one of the {study} "
                         f"study's {list(values)}: the sweep's runs would be no cell of it")
    return [value for value in values if value != own]

def ablate(config, dataset, out, study, tag, device, seeds=SEEDS, paired=True, wandb_mode=None,
           config_path=CONFIG_PATH, log=print):
    """Train and score ``study``'s cells in ``out``, one folder per value; returns their run folders."""
    plan, run_dirs = STUDIES[study], []
    for value in trained_values(config, study):
        cell = study_config(config, study, value)
        folder = Path(out) / f"{study}_{value}"
        log(f"{'.'.join(plan['setting'])} = {value} -> {folder}")
        run_dirs += sweep(cell, dataset, folder, tag, device, cells(cell, ["random"], list(plan["variants"]), list(seeds)),
                          plan["holdouts"], paired=paired, wandb_mode=wandb_mode, config_path=config_path, log=log)
    return run_dirs

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and score one of M8-1's ablation studies, going on where a "
                                                 "session stopped.")
    parser.add_argument("--study", required=True, choices=list(STUDIES))
    parser.add_argument("--tag", required=True, help="the wandb tag every ablation run shares, e.g. ablation")
    parser.add_argument("--out", type=Path, help="the ablation's folder (default outputs/<tag>); on Colab, a folder "
                                                 "on Google Drive")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto (the first available)")
    parser.add_argument("--wandb", choices=("online", "disabled"),
                        help="wandb mode (default: WANDB_MODE if set, else online)")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    out, device, plan = args.out or OUTPUTS / args.tag, pick_device(args.device), STUDIES[args.study]
    print(f"{args.study}: {'.'.join(plan['setting'])} = {trained_values(config, args.study)} on "
          f"{', '.join(plan['variants'])} at seeds {' '.join(map(str, SEEDS))}, on {device} -> {out}", flush=True)
    ablate(config, load_dataset(args.dataset), out, args.study, args.tag, device, wandb_mode=args.wandb,
           config_path=args.config, log=lambda line: print(line, flush=True))

if __name__ == "__main__":
    main()
