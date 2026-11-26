"""The ablation's driver (M8-1): three one-factor studies on the best variant, trained and scored as the sweep is.

    uv run python -m diffusion.ablation --study d_model --tag ablation --out <a folder on Google Drive>

Three bounded studies, not a grid (PROJECT_PLAN §5, ``Milestones/M08_Ablation_Runtime.md``). Each moves
one setting away from the sweep's, at seeds 42-44, every other setting the sweep's:

- ``d_model``: ``diffusion.channels`` at 64 / 128 / 256 (``model.d_model_sweep``) on Ext 1+2, the best
  variant (M7-2), in the random regime, with ``model.n_heads`` fixed at 8 (§3.6); scored on the random
  holdout;
- ``steps``: the diffusion steps T, ``diffusion.num_steps``, at 50 / 100 / 200 on Ext 1+2 in the random
  regime, each a full retrain; scored on the random holdout. T moves alone: the schedule keeps upstream's
  beta range (quad, 1e-4 to 0.5), so a longer chain spends more of its steps near pure noise (README,
  The ablation);
- ``encoder_layers``: Ext 1's encoder depth, ``model.mask_encoder_layers``, at 1 / 2 / 3 on the two
  variants with that encoder: Ext 1+2 in the random regime, the only one it is trained in, and Ext 1
  block-aware, its primary regime (§3.1, §4.1: a random-trained model's block score only checks how it
  carries over to blocks); each scored on the random and the block holdout.

The sweep's own value of each setting (64, 50, 2) is not trained again: the sweep's runs at seeds
42-44 are those cells. So each study trains its other two values, 24 runs in all. Each value trains
into ``<out>/<study>_<value>`` (e.g. ``<out>/d_model_128``) with ``diffusion.sweep.sweep``, so a run
goes on where a lost session stopped and trains and is scored on ``protocol.paired_gpu`` from committed
code (each folder's ``sweep.json`` records the tag and commit). Each run is tagged on wandb with its
study and value (``d_model_128``) besides ``--tag``, since every value's run of a cell has the cell's
name. The same command, run again in a new session, finishes what the last one started.

A value's run is paired with the sweep's run of its cell at that seed, not with the study's other
runs, and its study unpairs what it changes: T changes no starting weight and no random number drawn,
dropout's included (its timesteps come from the same draws, spread over 0 to T - 1, with the same
noise); the encoder's depth changes that encoder's starting weights and, since a deeper encoder has more
dropout layers, every dropout mask; ``d_model`` changes every starting weight and every dropout mask,
leaving the batches, the train masks and the diffusion draws paired.

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
# Each study's one setting (config block, key), its values, the variants it trains in each regime (in the
# order it trains them, the random regime first as in the sweep) and the holdouts it scores. The encoder's
# depth trains Ext 1 block-aware, its primary regime (PROJECT_PLAN §3.1, §4.1), and Ext 1+2 in the random
# regime, the only one it is trained in (settled at the review of M8-1's driver).
STUDIES = {
    "d_model": {"setting": ("diffusion", "channels"), "values": (64, 128, 256),
                "runs": {"random": ("ext1+2",)}, "holdouts": ("random",)},
    "steps": {"setting": ("diffusion", "num_steps"), "values": (50, 100, 200),
              "runs": {"random": ("ext1+2",)}, "holdouts": ("random",)},
    "encoder_layers": {"setting": ("model", "mask_encoder_layers"), "values": (1, 2, 3),
                       "runs": {"random": ("ext1+2",), "block-aware": ("ext1",)}, "holdouts": ("random", "block")},
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

def study_cells(config, study, seeds=SEEDS):
    """``[(regime, variant, seed), ...]``: the cells each of ``study``'s values trains, regime by regime."""
    return [found for regime, variants in STUDIES[study]["runs"].items()
            for found in cells(config, [regime], list(variants), list(seeds))]

def value_name(study, value):
    """``<study>_<value>``, e.g. ``d_model_128``: the value's folder in ``out`` and its runs' wandb tag."""
    return f"{study}_{value}"

def ablate(config, dataset, out, study, tag, device, seeds=SEEDS, paired=True, wandb_mode=None,
           config_path=CONFIG_PATH, log=print):
    """Train and score ``study``'s cells in ``out``, one folder per value; returns their run folders."""
    plan, run_dirs = STUDIES[study], []
    for value in trained_values(config, study):
        cell, name = study_config(config, study, value), value_name(study, value)
        folder = Path(out) / name
        log(f"{'.'.join(plan['setting'])} = {value} -> {folder}")
        run_dirs += sweep(cell, dataset, folder, tag, device, study_cells(cell, study, seeds), plan["holdouts"],
                          paired=paired, wandb_mode=wandb_mode, config_path=config_path, tags=[name], log=log)
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
    runs = "; ".join(f"{', '.join(variants)} {regime}" for regime, variants in plan["runs"].items())
    print(f"{args.study}: {'.'.join(plan['setting'])} = {trained_values(config, args.study)} on {runs} "
          f"at seeds {' '.join(map(str, SEEDS))}, on {device} -> {out}", flush=True)
    ablate(config, load_dataset(args.dataset), out, args.study, args.tag, device, wandb_mode=args.wandb,
           config_path=args.config, log=lambda line: print(line, flush=True))

if __name__ == "__main__":
    main()
