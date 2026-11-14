"""The paired sweep's results store (M6-2): every run's test scores, kept only if the runs are paired.

    uv run python -m diffusion.results --tag dry-run         # writes results/dry-run.json

Reads every finished run with the tag from the config's ``tracking`` project on wandb, where
``diffusion.run`` streams each training run and ``diffusion.score`` adds each test scoring to the
run's summary (``random/0.1/test/mae``, ``.../rmse``, ``.../gpu``), so runs trained in different
Colab sessions are collected without their output folders. Writes ``results/<tag>.json``:

- ``runs[regime][variant][seed]``: the run, its wandb path, commit and training GPU, and under
  ``scores`` the MAE, RMSE and scoring GPU of each holdout scored (``random/0.1``, ``block/0.1``,
  and base's ``random/0.5`` and ``random/0.9`` once M7-1 scores them). M7-2's table reads these;
- ``configs[regime]``: the config every run of the regime trained with, extension toggles aside;
- ``gpu``: the one GPU every run trained and was scored on (``protocol.paired_gpu``);
- ``unscored``: the runs still missing a 10% holdout, whose score is null meanwhile, so the store
  can be written while the sweep runs.

The store is written only if the runs are paired (PROJECT_PLAN §3.3). Within a regime and seed,
every variant's pairing hashes (``diffusion.run``, ``summary.pairing``) must be equal: the batches
in their order, the train masks, and the shared parts' starting weights; and each encoder's
starting weights must be equal in every variant that has it (Ext 1's in ext1 and ext1+2, Ext 2's
in ext2 and ext1+2). Every run of a regime must have the same config but for the toggles (how a run
is validated aside: it does not touch training), and every run must have trained, and every
holdout been scored, on ``protocol.paired_gpu``, the same for all. Refused besides: two runs in one
cell, which wandb allows (a rerun gets the same name). Untag the one to leave out.
"""

import argparse
import json
import re
from pathlib import Path

from diffusion.gate import TRACKING_ONLY, flatten
from diffusion.health import summary_block
from diffusion.run import ENCODERS, ROOT, TRACKING, VARIANTS

RESULTS = ROOT / "results"
UNITS = "standardized (train-split z-scores)"
HEADLINE = ("random/0.1", "block/0.1")      # the two frozen holdouts every run is scored on (PROJECT_PLAN §3.1)
TOGGLES = tuple(VARIANTS["base"])           # the only config a seed's variants may differ in
LABELS = ("out_dir", "variant", "seed", "regime", "device", "gpu")     # which run, where: not what it trains
SCORE_KEY = re.compile(r"(random|block)/([0-9.]+)/test/(mae|rmse|gpu)")
STREAMS = ("data_order", "train_mask", "init_shared")

def fetch(tag):
    """Every finished run tagged ``tag`` in the tracking project, as the plain dicts ``record`` reads."""
    import wandb
    runs = wandb.Api(timeout=60).runs(f"{TRACKING['wandb_entity']}/{TRACKING['wandb_project']}",
                                      filters={"tags": {"$in": [tag]}, "state": "finished"})
    return [{"path": "/".join(run.path), "name": run.name, "config": dict(run.config),
             "summary": dict(run.summary._json_dict), "metadata": run.metadata or {}} for run in runs]

def scored_holdouts(summary):
    """``{"random/0.1": {"mae", "rmse", "gpu"}, ...}``: each test scoring ``diffusion.score`` added."""
    found = {}
    for key, value in summary.items():
        match = SCORE_KEY.fullmatch(key)
        if match:
            holdout, fraction, field = match.groups()
            found.setdefault(f"{holdout}/{fraction}", {})[field] = value
    return found

def shared_config(config):
    """What a run trained with, but for the toggles: equal across a regime's runs if they are paired."""
    config = {key: value for key, value in config.items() if key not in LABELS + TRACKING_ONLY}
    return {**config, "model": {key: value for key, value in config["model"].items() if key not in TOGGLES}}

def problems(runs):
    """Everything that unpairs ``runs`` or keeps one out of the store; empty if the store may be written."""
    found = []
    cells, regimes, seeds = {}, {}, {}
    for run in runs:
        config = run["config"]
        cells.setdefault((config["regime"], config["variant"], config["seed"]), []).append(run["path"])
        regimes.setdefault(config["regime"], []).append(run)
        seeds.setdefault((config["regime"], config["seed"]), []).append(run)
    found += [f"two runs of {variant} in the {regime} regime at seed {seed}: {paths}"
              for (regime, variant, seed), paths in cells.items() if len(paths) > 1]

    gpus = {run["config"]["protocol"].get("paired_gpu") for run in runs}
    if None in gpus:
        found.append("protocol.paired_gpu is not set in every run's config: M6-2's timing test picks it")
    elif len(gpus) > 1:
        found.append(f"runs set to different GPUs: {sorted(gpus)}")
    else:
        [gpu] = gpus
        for run in runs:
            if run["summary"].get("gpu") != gpu:
                found.append(f"{run['name']} trained on {run['summary'].get('gpu')}, not a {gpu}")
            found += [f"{run['name']}'s {holdout} was scored on {scores.get('gpu')}, not a {gpu}"
                      for holdout, scores in scored_holdouts(run["summary"]).items() if scores.get("gpu") != gpu]

    for regime, group in regimes.items():
        configs = [flatten(shared_config(run["config"])) for run in group]
        keys = sorted(set().union(*configs))
        differ = [key for key in keys if len({repr(config.get(key)) for config in configs}) > 1]
        if differ:
            found.append(f"the {regime} regime's runs differ in more than their toggles: {differ}")

    for (regime, seed), group in seeds.items():
        hashes = {run["config"]["variant"]: summary_block(run["summary"], "pairing") for run in group}
        for stream in STREAMS:
            values = {variant: pairing.get(stream) for variant, pairing in hashes.items()}
            if None in values.values() or len(set(values.values())) > 1:      # None: a run from before M6-2
                found.append(f"{stream} is missing or differs across the {regime} regime's variants "
                             f"at seed {seed}: {values}")
        for encoder in ENCODERS:
            key = f"init_{encoder}"
            have = {variant: pairing.get(key) for variant, pairing in hashes.items() if key in pairing}
            if len(set(have.values())) > 1:
                found.append(f"{encoder} starts from different weights across the {regime} regime's variants "
                             f"at seed {seed}: {have}")
    return found

def record(runs, tag):
    """The store of ``runs`` (``fetch``'s dicts); raises ``ValueError`` listing what unpairs them."""
    if not runs:
        raise ValueError(f"no finished run is tagged {tag!r}")
    found = problems(runs)
    if found:
        raise ValueError("; ".join(found))
    order = list(VARIANTS)
    runs = sorted(runs, key=lambda run: (run["config"]["regime"], order.index(run["config"]["variant"]),
                                         run["config"]["seed"]))
    store = {"tag": tag, "units": UNITS, "gpu": runs[0]["config"]["protocol"]["paired_gpu"],
             "configs": {}, "runs": {}, "unscored": []}
    for run in runs:
        config, scores = run["config"], scored_holdouts(run["summary"])
        store["configs"].setdefault(config["regime"], shared_config(config))
        store["runs"].setdefault(config["regime"], {}).setdefault(config["variant"], {})[str(config["seed"])] = {
            "run": run["name"],
            "wandb_run": run["path"],
            "commit": (run["metadata"].get("git") or {}).get("commit"),
            "gpu": run["summary"]["gpu"],
            "scores": {holdout: scores.get(holdout) for holdout in sorted(set(HEADLINE) | set(scores))},
        }
        store["unscored"] += [f"{run['name']}: {holdout}" for holdout in HEADLINE if holdout not in scores]
    return store

def main(argv=None, fetch=fetch):
    parser = argparse.ArgumentParser(description="Collect the paired sweep's test scores from wandb, checked (M6-2).")
    parser.add_argument("--tag", required=True, help="the runs' wandb tag, e.g. dry-run or sweep")
    parser.add_argument("--out", type=Path, help="store path (default results/<tag>.json)")
    args = parser.parse_args(argv)

    store = record(fetch(args.tag), args.tag)
    out = args.out or RESULTS / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(store, indent=1) + "\n")
    n_runs = sum(len(seeds) for variants in store["runs"].values() for seeds in variants.values())
    print(f"wrote {out}: {n_runs} paired runs on {store['gpu']}, {len(store['unscored'])} holdout(s) not scored yet")
    return store

if __name__ == "__main__":
    main()
