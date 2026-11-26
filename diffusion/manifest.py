"""The paired sweep's run manifest (M7-1): what each run is, what it cost, and how to train it again.

    uv run python -m diffusion.manifest --tag sweep          # reads outputs/sweep, writes results/sweep_manifest.json

The results store (``diffusion.results``) holds each run's outcomes, read from wandb. The manifest
holds what the store does not: each run's provenance and cost, read from the sweep's folder (``--out``,
default ``outputs/<tag>``; on Colab the sweep wrote it to Google Drive, so copy it here first). It
reads the folder, not wandb, because each extension's final gate is in ``model.pt`` alone: training
never logs it, and the sweep's weights are not uploaded. Writes ``results/<tag>_manifest.json``:

- ``tag``, ``commit``: from ``<out>/sweep.json``, the tag and git commit the folder was started with;
- ``runs[regime][variant][seed]``, keyed as the store is, so M7-2 joins the two files:
  - ``run``, ``wandb_run``, and ``config``, the run's ``config.yaml`` in full;
  - ``rerun``: the command that trains and scores the cell again, at the commit, into another folder
    under another tag (``<out>/sweep.json`` refuses a second tag), for ``diffusion.sweep --compare``,
    with a ``diffusion.score`` step for each scoring the driver does not do (base's 50% and 90%);
  - ``gpu``: the GPU it trained on (``summary.json``) and each scoring's (``scores_test_*.json``);
  - ``seconds``: training, the sum of ``curve.json``'s epoch ``seconds``; validation, the sum of its
    ``val_seconds``; and each scoring's ``seconds``. ``curve.json`` keeps one row per epoch, so a
    resumed run's lost epochs (trained, then lost with a session before their checkpoint) are not
    counted: this is what a run costs without interruption, not the GPU time billed;
  - ``weights``: the final weights' path (``model.pt``; ``checkpoint.pt`` is only for resuming, and
    holds an earlier epoch) and their SHA-256;
  - ``gates``: each extension's gate in ``model.pt`` (``diffusion.health.GATES``), which starts at 0;
    its size is how far training scaled the encoder's output up (PROJECT_PLAN §3.5). Base has none.

Every run ``sweep.cells`` plans must be in the folder, finished (``summary.json``), and consistent:
its config is its cell's, ``model.pt`` is the weights its summary and every scoring name, its curve has
each epoch once, it has both headline scorings (``results.HEADLINE``), it trained and was scored on
``protocol.paired_gpu``, and it has a gate for each extension it was trained with and no other.
Otherwise nothing is written, and the error lists every problem found.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch
import yaml

from diffusion.health import GATES
from diffusion.results import HEADLINE, RESULTS
from diffusion.run import CONFIG_PATH, OUTPUTS, ROOT, VARIANTS
from diffusion.sweep import cells

NOTES = {
    "seconds.train": "sum of curve.json's epoch seconds; a resumed run's lost epochs are not counted",
    "seconds.validation": "sum of curve.json's val_seconds (validation is for the curves only, PROJECT_PLAN §3.1)",
    "seconds.scoring": "each scoring's seconds, from its scores_test_<holdout>_<fraction>.json",
    "gates": "each extension's final gate, read from model.pt; it starts at 0 (PROJECT_PLAN §3.5)",
}

def shown(path, parts=1):
    """``path`` relative to the repo when inside it; outside it, only its last ``parts`` names, so no
    one's home folder goes into the manifest."""
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    return "<outside the repo>/" + "/".join(path.parts[-parts:])

def rerun_command(commit, regime, variant, seed, run, extra=()):
    """The command that trains and scores a cell again. The driver scores the headline holdouts; each
    scoring in ``extra`` (``holdout/fraction``, such as base's ``random/0.5``) is its own step, so the
    rerun's folder holds the same scores files and ``diffusion.sweep --compare`` can match them."""
    command = (f"git checkout {commit} && uv run python -m diffusion.sweep --tag rerun --out <another folder> "
               f"--regime {regime} --variants {variant} --seeds {seed}")
    for name in extra:
        holdout, fraction = name.split("/")
        command += (f" && uv run python -m diffusion.score <another folder>/{run} "
                    f"--holdout {holdout} --fraction {fraction} --paired")
    return command

def entry(run_dir, regime, variant, seed, protocol, commit, holdouts=HEADLINE):
    """One run's manifest entry, read from its folder; raises ``ValueError`` listing everything wrong with it.

    ``holdouts`` are the scorings it must have (``holdout/fraction``): the headline two, or fewer for an
    M8-1 ablation run, scored on the random holdout alone in two of its studies.
    """
    run_dir = Path(run_dir)
    needed = ("summary.json", "config.yaml", "curve.json", "model.pt")
    absent = [name for name in needed if not (run_dir / name).exists()]
    if absent:
        raise ValueError(f"{run_dir.name}: no {', '.join(absent)}"
                         + (" (the run has not finished)" if "summary.json" in absent else ""))
    summary = json.loads((run_dir / "summary.json").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    curve = json.loads((run_dir / "curve.json").read_text())
    weights_bytes = (run_dir / "model.pt").read_bytes()
    sha = hashlib.sha256(weights_bytes).hexdigest()
    weights = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=True)
    scorings = [json.loads(path.read_text()) for path in sorted(run_dir.glob("scores_test_*.json"))]
    problems = []

    toggles = {key: config["model"].get(key, False) for key in VARIANTS[variant]}
    if toggles != VARIANTS[variant]:
        problems.append(f"config's extension toggles {toggles} are not {variant}'s")
    if config["train"].get("regime") != regime:
        problems.append(f"config's regime is {config['train'].get('regime')!r}, not {regime!r}")
    if (summary.get("variant"), summary.get("seed")) != (variant, seed):
        problems.append(f"summary.json is {summary.get('variant')} seed {summary.get('seed')}, not {variant} seed {seed}")
    if summary.get("model_sha256") != sha:
        problems.append("model.pt is not the final weights summary.json records")
    epochs = [row["epoch"] for row in curve]
    if epochs != list(range(config["train"]["epochs"])):
        problems.append(f"curve.json's epochs are not 0-{config['train']['epochs'] - 1}, each once")

    gpu = {"train": summary.get("gpu"), "scoring": {}}
    seconds = {"train": round(sum(row["seconds"] for row in curve), 1),
               "validation": round(sum(row.get("val_seconds", 0.0) for row in curve), 1), "scoring": {}}
    for scoring in scorings:
        name = f"{scoring['holdout']}/{scoring['fraction']}"
        if scoring.get("split") != "test":
            problems.append(f"{name}: a scoring of the {scoring.get('split')} split")
        if scoring.get("model_sha256") != sha:
            problems.append(f"{name}: scored other weights than model.pt")
        gpu["scoring"][name] = scoring.get("gpu")
        seconds["scoring"][name] = scoring["seconds"]
    for name in holdouts:
        if name not in gpu["scoring"]:
            problems.append(f"no {name} scoring")
    for where, name in [("training", gpu["train"])] + [(f"{h} scoring", g) for h, g in gpu["scoring"].items()]:
        if name != protocol["paired_gpu"]:
            problems.append(f"{where} on {name}, not {protocol['paired_gpu']}")

    gates = {}
    for toggle, (key, field) in GATES.items():
        if config["model"].get(toggle, False):
            if key in weights:
                gates[field] = float(weights[key])
            else:
                problems.append(f"model.pt has no {key}, though the run trained with {toggle}")
        elif key in weights:
            problems.append(f"model.pt has {key}, though the run trained without {toggle}")

    if problems:
        raise ValueError(f"{run_dir.name}: " + "; ".join(problems))
    extra = [name for name in gpu["scoring"] if name not in HEADLINE]
    return {"run": run_dir.name, "wandb_run": summary.get("wandb_run"), "config": config,
            "rerun": rerun_command(commit, regime, variant, seed, run_dir.name, extra), "gpu": gpu, "seconds": seconds,
            "weights": {"path": shown(run_dir / "model.pt", parts=2), "sha256": sha}, "gates": gates}

def manifest(out, tag, config):
    """Every run ``sweep.cells(config)`` plans, from the sweep folder ``out``; raises ``ValueError`` listing
    every problem, in every run, if there is any."""
    out = Path(out)
    started = json.loads((out / "sweep.json").read_text())
    if started["tag"] != tag:
        raise ValueError(f"{out} was started with tag {started['tag']!r}, not {tag!r}")
    runs, problems = {}, []
    for regime, variant, seed in cells(config):
        run_dir = out / f"{variant}_{regime}_seed{seed}"
        if not run_dir.is_dir():
            problems.append(f"{run_dir.name}: no folder")
            continue
        try:
            runs.setdefault(regime, {}).setdefault(variant, {})[str(seed)] = entry(
                run_dir, regime, variant, seed, config["protocol"], started["commit"])
        except ValueError as error:
            problems.append(str(error))
    if problems:
        raise ValueError(f"{len(problems)} run(s) refused:\n" + "\n".join(problems))
    return {"tag": tag, "commit": started["commit"], "folder": shown(out), "gpu": config["protocol"]["paired_gpu"],
            "notes": NOTES, "runs": runs}

def gpu_hours(found):
    """Total GPU hours in a manifest: training, validation and every scoring of every run."""
    total = 0.0
    for variants in found["runs"].values():
        for seeds in variants.values():
            for run in seeds.values():
                total += run["seconds"]["train"] + run["seconds"]["validation"] + sum(run["seconds"]["scoring"].values())
    return total / 3600

def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the paired sweep's run manifest from its folder (M7-1).")
    parser.add_argument("--tag", required=True, help="the sweep's tag, e.g. sweep")
    parser.add_argument("--out", type=Path, help="the sweep's folder (default outputs/<tag>)")
    parser.add_argument("--manifest", type=Path, help="where to write it (default results/<tag>_manifest.json)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    found = manifest(args.out or OUTPUTS / args.tag, args.tag, config)
    path = args.manifest or RESULTS / f"{args.tag}_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, indent=1) + "\n")
    n_runs = sum(len(seeds) for variants in found["runs"].values() for seeds in variants.values())
    print(f"wrote {path}: {n_runs} runs at commit {found['commit'][:7]}, {gpu_hours(found):.1f} GPU hours on {found['gpu']}")
    return found

if __name__ == "__main__":
    main()
