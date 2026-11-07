"""The record of single-seed health runs (M5-3): every variant trains and validates end to end.

    uv run python -m diffusion.health bqoimtqw gwx2us22 --note base="the M4-1 gate run, reused"

Reads each training run from wandb, where ``diffusion.run`` streams everything it logs (M3-2),
so a teammate's run is recorded without its output folder, and writes
``results/health_seed<seed>.json``. A run is given by its wandb path (``entity/project/id``) or
its bare id in the config's ``tracking`` project. For each variant the record holds:

- where the run came from: its wandb path, tags, git commit, device, and machine;
- the 500-step health report, recomputed from the run's logged steps with
  ``diffusion.run.health_report`` and checked against the one the run wrote to its summary;
- the final val MAE @10%, forward fill's, and the val curve: tracking validation (``n_samples``
  draws on the val split), not the frozen protocol, which scores test with 100 (README, Scoring);
- training and validation seconds, the median epoch, and the slowest one;
- for a variant with Ext 1's encoder, the encoder's gate, read from the run's ``model.pt``: a
  ``model.pt`` uploaded to the wandb run, or a local one given with ``--weights``. It is null
  when neither is there.

**Health only, not an effect measurement** (M5-3): one seed, validation only. Whether a
variant helps waits for the paired 5-seed sweep (PROJECT_PLAN §3.3).

Timings compare only between runs on one machine: Ext 1's validation passes on an Apple M4 took
2.6 times as long as base's on an M4 Pro, mostly because of the machine (M5-3 review). The record
names each run's machine and says whether its timings compare.

Refused: an unfinished run, runs at different seeds or regimes, two runs of one variant, runs
validated with different settings, a run that logged fewer epochs than its config trains, and a
run whose summary does not match its logged steps (health report, final val MAE).
"""

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path

import torch

from diffusion.gate import flatten
from diffusion.run import ROOT, TRACKING, health_report

RESULTS = ROOT / "results"
STATUS = "health only, not an effect measurement"
LABELS = ("out_dir", "variant")     # name a run, not what it trains: left out of config_differences
VALIDATION_KEYS = ("validation.n_samples", "validation.batch_size", "protocol.eval_masker_seed",
                   "protocol.eval_sampling_seed", "protocol.eval_holdout_fraction")
# TODO(M6-2): the block-aware regime validates on BlockEvalMasker's holdout (PROJECT_PLAN §3.1).
HOLDOUT_OF_REGIME = {"random": "random"}
GATE_KEY = "mask_encoder.gate"      # MissingnessEncoder.gate, as models.build_model names it

def record_path(seed):
    return RESULTS / f"health_seed{seed}.json"

def wandb_path(run):
    """``entity/project/id`` for a wandb path or a bare run id in the config's tracking project."""
    return run if "/" in run else f"{TRACKING['wandb_entity']}/{TRACKING['wandb_project']}/{run}"

def fetch(path):
    """A finished run's logged numbers from wandb, as the plain dict ``run_entry`` reads."""
    import wandb
    run = wandb.Api(timeout=60).run(path)
    weights = None
    if any(f.name == "model.pt" for f in run.files()):
        with tempfile.TemporaryDirectory() as tmp:
            run.file("model.pt").download(root=tmp)
            weights = torch.load(Path(tmp) / "model.pt", map_location="cpu", weights_only=True)
    return {"path": "/".join(run.path), "name": run.name, "state": run.state, "tags": list(run.tags),
            "config": dict(run.config), "summary": dict(run.summary._json_dict),
            "metadata": run.metadata or {}, "history": list(run.scan_history(page_size=10000)),
            "weights": weights}

def machine(metadata):
    """The hardware a run ran on, from wandb's metadata: the Apple chip, else the NVIDIA GPU."""
    apple = metadata.get("apple")
    if apple:
        return f"{apple['name']} ({apple['gpuCores']}-core GPU, {apple['memoryGb']} GB)"
    return metadata.get("gpu")

def summary_block(summary, name):
    """``summary[name]``, whether wandb kept the block nested or flattened it to ``name.key``."""
    if isinstance(summary.get(name), dict):
        return summary[name]
    prefix = f"{name}."
    return {key[len(prefix):]: value for key, value in summary.items() if key.startswith(prefix)}

def same(a, b):
    """Equal, with floats compared to 1e-12: wandb's summary can round a float's last digit."""
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

def run_entry(run, note=None):
    """One run's part of the record; raises ``ValueError`` listing everything wrong with it."""
    config, summary = run["config"], run["summary"]
    rows = sorted(run["history"], key=lambda row: row["_step"])
    steps = [{"loss": row["train/loss"], "grad_norm": row["train/grad_norm"]}
             for row in rows if row.get("train/loss") is not None]
    epochs = [row for row in rows if row.get("epoch/seconds") is not None]
    scored = [row for row in epochs if row.get("val/mae") is not None]
    health = health_report(steps)

    problems = []
    if run["state"] != "finished":
        problems.append(f"state {run['state']!r}, not finished")
    if len(epochs) != config["train"]["epochs"]:
        problems.append(f"{len(epochs)} epochs logged, not the config's {config['train']['epochs']}")
    if not scored:
        problems.append("no validation pass logged")
    if not same(health, summary_block(summary, "health")):
        problems.append("the summary's health report is not what the logged steps give")
    if scored and not same(scored[-1]["val/mae"], summary.get("final_val_mae")):
        problems.append(f"summary final_val_mae {summary.get('final_val_mae')} is not the last val MAE "
                        f"logged, {scored[-1]['val/mae']}")
    gate = None
    if run.get("weights") is not None:
        if not config["model"]["use_mask_embedding"]:
            problems.append("weights given for a run without Ext 1's encoder")
        else:
            gate = round(float(run["weights"][GATE_KEY]), 4)
    if problems:
        raise ValueError(f"{run['path']} ({run['name']}): " + "; ".join(problems))

    seconds = [row["epoch/seconds"] for row in epochs]
    slowest = max(epochs, key=lambda row: row["epoch/seconds"])
    final = scored[-1]["val/mae"]
    bar = summary.get("forward_fill_val_mae")
    entry = {
        "run": run["name"],
        "wandb_run": run["path"],
        "tags": run["tags"],
        "commit": (run["metadata"].get("git") or {}).get("commit"),
        "device": config["device"],
        "machine": machine(run["metadata"]),
        **({"note": note} if note else {}),
        "health": health,
        "final_val_mae": final,
        "forward_fill_val_mae": bar,
        "below_forward_fill": None if bar is None else final < bar,
        "val_mae_by_epoch": {str(row["epoch"]): row["val/mae"] for row in scored},
        "train_seconds": round(sum(seconds), 1),
        "val_seconds": round(sum(row["val/seconds"] for row in scored), 1),
        "val_passes": len(scored),
        "epoch_seconds_median": round(statistics.median(seconds), 1),
        "slowest_epoch": {"epoch": slowest["epoch"], "seconds": slowest["epoch/seconds"]},
    }
    if config["model"]["use_mask_embedding"]:
        entry["mask_encoder_gate"] = gate
    return entry

def record(runs, notes=None, ticket="M5-3"):
    """The health record of ``runs`` (``fetch``'s dicts) at one seed; see the module docstring."""
    notes = notes or {}
    configs = {run["config"]["variant"]: run["config"] for run in runs}
    problems = []
    if len(configs) != len(runs):
        problems.append(f"two runs of one variant: {[run['config']['variant'] for run in runs]}")
    for key in ("seed", "regime"):
        if len({run["config"][key] for run in runs}) > 1:
            problems.append(f"runs at different {key}s: {[run['config'][key] for run in runs]}")
    flat = {variant: flatten({k: v for k, v in config.items() if k not in LABELS})
            for variant, config in configs.items()}
    for key in VALIDATION_KEYS:
        if len({repr(leaves.get(key)) for leaves in flat.values()}) > 1:
            problems.append(f"runs validated with different {key}: {[leaves.get(key) for leaves in flat.values()]}")
    regime = runs[0]["config"]["regime"]
    if regime not in HOLDOUT_OF_REGIME:
        problems.append(f"no validation holdout known for regime {regime!r}")
    unknown = set(notes) - set(configs)
    if unknown:
        problems.append(f"notes for variants not given: {sorted(unknown)}")
    if problems:
        raise ValueError("; ".join(problems))

    entries = {run["config"]["variant"]: run_entry(run, notes.get(run["config"]["variant"])) for run in runs}
    machines = {entry["machine"] for entry in entries.values()}
    first = runs[0]["config"]
    keys = sorted(set().union(*flat.values()))
    return {
        "status": STATUS,
        "ticket": ticket,
        "seed": first["seed"],
        "regime": regime,
        "validation": {"split": "val", "holdout": HOLDOUT_OF_REGIME[regime],
                       "fraction": first["protocol"]["eval_holdout_fraction"],
                       "n_samples": first["validation"]["n_samples"],
                       "batch_size": first["validation"]["batch_size"]},
        "timings": ("unknown: a run's machine is not in its wandb metadata" if None in machines
                    else f"comparable: every run is on {machines.pop()}" if len(machines) == 1
                    else "not comparable across runs: they ran on different machines (each run's machine)"),
        "runs": entries,
        "config_differences": {key: {variant: leaves.get(key) for variant, leaves in flat.items()}
                               for key in keys if len({repr(leaves.get(key)) for leaves in flat.values()}) > 1},
    }

def pairs(items, flag):
    """``["variant=value", ...]`` from the command line as ``{variant: value}``."""
    out = {}
    for item in items:
        variant, sep, value = item.partition("=")
        if not sep or not value:
            raise SystemExit(f"{flag} wants VARIANT=VALUE, got {item!r}")
        out[variant] = value
    return out

def main(argv=None, fetch=fetch):
    parser = argparse.ArgumentParser(description="Record single-seed health runs from wandb (M5-3).")
    parser.add_argument("runs", nargs="+", help="wandb run paths (entity/project/id) or bare ids in the tracking project")
    parser.add_argument("--weights", action="append", default=[], metavar="VARIANT=PATH",
                        help="a run's local model.pt, for Ext 1's gate when the wandb run has none (repeatable)")
    parser.add_argument("--note", action="append", default=[], metavar="VARIANT=TEXT",
                        help="a line saying where a run comes from, e.g. base='the M4-1 gate run, reused' (repeatable)")
    parser.add_argument("--ticket", default="M5-3")
    parser.add_argument("--out", type=Path, help="record path (default results/health_seed<seed>.json)")
    args = parser.parse_args(argv)

    runs = [fetch(wandb_path(run)) for run in args.runs]
    weights = pairs(args.weights, "--weights")
    by_variant = {run["config"]["variant"]: run for run in runs}
    for variant, path in weights.items():
        if variant not in by_variant:
            raise SystemExit(f"--weights for {variant!r}, which is not among the runs given")
        by_variant[variant]["weights"] = torch.load(path, map_location="cpu", weights_only=True)
    result = record(runs, pairs(args.note, "--note"), args.ticket)
    out = args.out or record_path(result["seed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {out}")
    return result

if __name__ == "__main__":
    main()
