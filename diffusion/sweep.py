"""The paired sweep's driver (M6-2): trains and scores every cell, and goes on where a session stopped.

    uv run python -m diffusion.sweep --tag dry-run --epochs 20 --seeds 42 43 --out <a folder on Google Drive>
    uv run python -m diffusion.sweep --tag sweep --out <a folder on Google Drive>

A cell is a regime, a variant and a seed. The config's ``sweep`` block names each regime's variants
(the random regime's four, the block-aware regime's base and Ext 1), and the seeds are
``protocol.run_seeds`` (42-46): 30 cells. They run regime by regime in that order, then seed by
seed, so a sweep cut short leaves whole seeds behind it, and the block-aware runs come after the 20
headline ones (PROJECT_PLAN §8). ``--regime``, ``--variants`` and ``--seeds`` run some of them;
``--variants`` takes any variant, e.g. ``--regime block-aware --variants ext1blind --seeds 42 43 44``
for M8-4.

Each cell, in ``<out>/<run name>`` (e.g. ``<out>/ext1_block-aware_seed42``):

1. is trained with ``diffusion.run.run(..., paired=True, resume=True)``, tagged with ``--tag``. A run
   that a lost session stopped goes on from its last epoch; a finished one (``summary.json``) is kept,
   once its saved config is checked to be the cell's;
2. has its pairing hashes checked against those of the regime's other variants at its seed already
   in ``<out>`` (``diffusion.results.unpaired``), so the sweep stops at the first cell that unpairs,
   not when the store is written after the last;
3. is scored on test under each of ``sweep.holdouts`` (``--holdouts``) with
   ``diffusion.score.score(..., paired=True)``, unless ``<out>`` already holds a scoring of its final
   weights with its median imputation beside it (``span_mae`` reads it), and each scoring is added to
   its wandb summary (``attach_to_wandb``), where the results store reads it.

So the same command, run again in a new session, finishes what the last one started. On Colab, put
``<out>`` on Google Drive: a session's own disk goes with it. ``<out>/sweep.json`` records the tag
and git commit the folder was started with, and another is refused: the store refuses a regime
trained from two commits. A paired sweep also refuses code with changes its commit does not hold
(``uncommitted``), since the commit is all the store checks. Every run trains and is scored on
``protocol.paired_gpu``; ``--unpaired`` lifts that and the commit check, for a check on another
machine (the store refuses its runs).

``--compare OTHER`` trains nothing: it checks that each cell finished in both ``<out>`` and
``OTHER`` left the same numbers in each (``reproduces``), for M6-2's rerun check: cells rerun into
another folder under another tag (``rerun``, since the store refuses two runs in one cell).
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from data.physionet import load_dataset
from diffusion.results import unpaired
from diffusion.run import (CONFIG_PATH, DATASET_PATH, NOT_A_RUN, OUTPUTS, ROOT, VARIANTS, pick_device,
                           require_paired_gpu, run, run_name, with_variant)
from diffusion.score import HOLDOUTS, attach_to_wandb, score, scores_name
from diffusion.train import REGIMES

def cells(config, regimes=None, variants=None, seeds=None):
    """``[(regime, variant, seed), ...]`` in the order the sweep runs them: regime, then seed, then variant.

    By default every regime in ``sweep.regimes`` with its variants, at every seed in
    ``protocol.run_seeds``; ``regimes``, ``variants`` and ``seeds`` narrow it (``variants`` may name a
    variant the regime does not list).
    """
    plan = config["sweep"]["regimes"]
    for regime in regimes or ():
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {list(REGIMES)}, not {regime!r}")
        if variants is None and regime not in plan:
            raise ValueError(f"sweep.regimes lists no variants for {regime!r}: name them")
    for variant in variants or ():
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}, not {variant!r}")
    seeds = seeds or config["protocol"]["run_seeds"]
    return [(regime, variant, seed) for regime in regimes or plan for seed in seeds
            for variant in variants or plan[regime]]

def cell_config(config, regime, variant, epochs=None):
    """The config a cell trains with: ``config`` set to the variant and regime, and to ``epochs`` if given."""
    config = with_variant(config, variant)
    config["train"]["regime"] = regime
    if epochs is not None:
        config["train"]["epochs"] = epochs
    return config

def current_commit():
    """The git commit the code runs from; None outside a git checkout."""
    try:
        found = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return found.stdout.strip()

def uncommitted(root=ROOT):
    """The tracked files in ``root``'s checkout that differ from its commit; empty outside a git checkout.

    Untracked files (data, outputs, a results store written mid-sweep) are not counted.
    """
    try:
        found = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line[3:] for line in found.stdout.splitlines()]

def claim(out, tag):
    """Record in ``<out>/sweep.json`` the tag and commit the folder's runs come from; refuse any other."""
    path, here = out / "sweep.json", {"tag": tag, "commit": current_commit()}
    if not path.exists():
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(here, indent=1) + "\n")
        return
    started = json.loads(path.read_text())
    if started["tag"] != tag:
        raise ValueError(f"{out} holds the runs tagged {started['tag']!r}, not {tag!r}: give its tag, or another folder")
    if started["commit"] != here["commit"]:
        raise ValueError(f"{out} was started from commit {started['commit']}, and the code here is at "
                         f"{here['commit']}: check out {started['commit']} to go on (the results store refuses "
                         "a regime trained from two commits)")

def final_scoring(run_dir, holdout, fraction):
    """``run_dir``'s test scoring of ``holdout`` if it scored the run's final weights and its median
    imputation is beside it (``span_mae`` reads it, and Google Drive may have kept one without the
    other when a session was lost); else None."""
    name = scores_name("test", holdout, fraction)
    path = run_dir / f"scores_{name}.json"
    if not (path.exists() and (run_dir / f"median_{name}.npy").exists()):
        return None
    scores = json.loads(path.read_text())
    return scores if scores["model_sha256"] == sha256(run_dir / "model.pt") else None

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_pairing(out, config, regime, seed):
    """Raise ``RuntimeError`` if the regime's finished runs at ``seed`` in ``out`` are not paired."""
    hashes = {}
    for variant in VARIANTS:
        summary = out / run_name(cell_config(config, regime, variant), seed) / "summary.json"
        if summary.exists():
            hashes[variant] = json.loads(summary.read_text())["pairing"]
    found = unpaired(regime, seed, hashes)
    if found:
        raise RuntimeError("the sweep's runs are not paired: " + "; ".join(found))

def sweep(config, dataset, out, tag, device, todo, holdouts, epochs=None, paired=True, wandb_mode=None,
          config_path=CONFIG_PATH, log=print):
    """Train and score every cell in ``todo`` (``cells()``) in ``out``; returns their run folders.

    ``holdouts`` are scored at ``protocol.eval_holdout_fraction``, under ``config_path``'s frozen
    protocol. See the module docstring for what is done, skipped, checked and refused.
    """
    if paired:                                                  # both before anything is written
        require_paired_gpu(config["protocol"], device)
        changed = uncommitted()
        if changed:
            raise RuntimeError(f"the code has changes its commit ({current_commit()}) does not hold: {changed}. "
                               "The results store checks only the commit: commit them or undo them")
    out = Path(out)
    claim(out, tag)
    fraction = config["protocol"]["eval_holdout_fraction"]
    run_dirs = []
    for i, (regime, variant, seed) in enumerate(todo, start=1):
        cell = cell_config(config, regime, variant, epochs)
        run_dir = out / run_name(cell, seed)
        run_dirs.append(run_dir)
        log(f"[{i}/{len(todo)}] {run_dir.name}")

        def indented(line):
            log(f"  {line}")

        if (run_dir / "summary.json").exists():
            if yaml.safe_load((run_dir / "config.yaml").read_text()) != {k: v for k, v in cell.items()
                                                                        if k not in NOT_A_RUN}:
                raise ValueError(f"{run_dir} holds a run trained with another config than the cell's "
                                 "(another --epochs?): give another folder")
            indented("trained")
        else:
            run(cell, dataset, seed, device, run_dir, log=indented, wandb_mode=wandb_mode, tags=[tag],
                paired=paired, resume=True)
        check_pairing(out, config, regime, seed)
        for holdout in holdouts:
            scores = final_scoring(run_dir, holdout, fraction)
            if scores is None:
                scores = score(run_dir, dataset, device, holdout, config_path=config_path, log=indented,
                               paired=paired)
            else:
                indented(f"scored on the {holdout} holdout")
            attach_to_wandb(scores, wandb_mode)
    return run_dirs

def reproduces(run_dir, other_dir):
    """What differs between two runs of one cell, in what they trained and scored; empty if nothing.

    Compared: every training step, every epoch's row but its seconds, the summary but its wandb run,
    the final weights, and each test scoring's numbers (MAE, RMSE, per-variable MAE, targets, weights).
    """
    a, b = Path(run_dir), Path(other_dir)
    differ = [name for name in ("history.jsonl", "model.pt") if sha256(a / name) != sha256(b / name)]

    def curve(folder):
        return [{k: v for k, v in row.items() if not k.endswith("seconds")}
                for row in json.loads((folder / "curve.json").read_text())]

    if curve(a) != curve(b):
        differ.append("curve.json")
    summaries = [{k: v for k, v in json.loads((folder / "summary.json").read_text()).items() if k != "wandb_run"}
                 for folder in (a, b)]
    differ += [f"summary.json {key}" for key in sorted(summaries[0].keys() | summaries[1].keys())
               if summaries[0].get(key) != summaries[1].get(key)]
    for name in sorted({path.name for folder in (a, b) for path in folder.glob("scores_*.json")}):
        if not ((a / name).exists() and (b / name).exists()):
            differ.append(f"{name} in one folder only")
            continue
        x, y = (json.loads((folder / name).read_text()) for folder in (a, b))
        differ += [f"{name} {key}" for key in ("mae", "rmse", "per_variable_mae", "n_targets", "model_sha256")
                   if x.get(key) != y.get(key)]
    return differ

def compare(out, other, config, todo, log=print):
    """``{run name: reproduces(...)}`` for every cell in ``todo`` finished in both ``out`` and ``other``."""
    found = {}
    for regime, variant, seed in todo:
        name = run_name(cell_config(config, regime, variant), seed)
        if (Path(out) / name / "summary.json").exists() and (Path(other) / name / "summary.json").exists():
            found[name] = reproduces(Path(out) / name, Path(other) / name)
            log(f"{name}: " + (f"differs in {', '.join(found[name])}" if found[name] else "the same numbers"))
    return found

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and score the paired sweep's cells, going on where a "
                                                 "session stopped (M6-2).")
    parser.add_argument("--tag", help="the wandb tag the results store collects the runs by, e.g. dry-run or "
                                      "sweep (needed to train)")
    parser.add_argument("--out", type=Path, help="the sweep's folder (default outputs/<tag>); on Colab, a folder "
                                                 "on Google Drive")
    parser.add_argument("--epochs", type=int, help="override train.epochs, e.g. 20 for the dry run")
    parser.add_argument("--seeds", type=int, nargs="+", help="run seeds (default protocol.run_seeds)")
    parser.add_argument("--regime", choices=list(REGIMES), help="only this regime's cells")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS),
                        help="only these variants; any variant, e.g. ext1blind for M8-4")
    parser.add_argument("--holdouts", nargs="+", choices=HOLDOUTS,
                        help="the test holdouts to score (default sweep.holdouts)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto (the first available)")
    parser.add_argument("--wandb", choices=("online", "disabled"),
                        help="wandb mode (default: WANDB_MODE if set, else online); the store reads only wandb")
    parser.add_argument("--unpaired", action="store_true",
                        help="train and score on any GPU, from code with uncommitted changes too: a check of the "
                             "driver, not the sweep (the store refuses its runs)")
    parser.add_argument("--compare", type=Path, metavar="OTHER",
                        help="train nothing: check each cell finished in --out and OTHER has the same numbers in both")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    try:
        todo = cells(config, None if args.regime is None else [args.regime], args.variants, args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if args.compare is not None:
        if args.out is None:
            parser.error("--compare checks --out against OTHER: give --out")
        found = compare(args.out, args.compare, config, todo, log=lambda line: print(line, flush=True))
        if not found or any(found.values()):
            print("no cell finished in both folders" if not found else "not every cell reproduces", flush=True)
            sys.exit(1)
        return found
    if args.tag is None:
        parser.error("--tag is needed to train: the results store collects the runs by it")
    out = args.out or OUTPUTS / args.tag
    device = pick_device(args.device)
    print(f"{len(todo)} cells on {device} -> {out}", flush=True)
    sweep(config, load_dataset(args.dataset), out, args.tag, device, todo, args.holdouts or config["sweep"]["holdouts"],
          args.epochs, paired=not args.unpaired, wandb_mode=args.wandb, config_path=args.config,
          log=lambda line: print(line, flush=True))
    print(f"done: collect and check them with  uv run python -m diffusion.results --tag {args.tag}", flush=True)

if __name__ == "__main__":
    main()
