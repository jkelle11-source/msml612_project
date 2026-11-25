"""The ablation's table (M8-1): each study's values against the sweep's, as paired Δ ± SE with a 95% CI and a verdict.

    uv run python -m diffusion.ablation_table --ablation <the ablation's folder> --sweep-dir <the sweep's folder>

Reads the run folders, not wandb: the read without the bad-value patients needs each run's saved
median. A study's value that is the sweep's (64, 50, 2) is read from the sweep's folder, whose runs
at seeds 42-44 are those cells (``diffusion.ablation``); the others from ``<ablation>/<study>_<value>``.
Writes ``results/ablation.json`` (every number, and each ablation run's manifest entry) and
``results/ablation.md`` (the tables). Nothing is written, and every problem is listed, unless:

- every run finished and its folder is consistent (``diffusion.manifest.entry``: its final weights, its
  curve, toggles and regime, the GPU it trained and was scored on), and each of its study's holdouts
  was scored the frozen way with its median beside it (``diffusion.sweep.final_scoring``), a median
  that gives exactly its scores' MAE;
- the sweep's runs in ``--sweep-dir`` are the ones ``results/sweep_manifest.json`` records (the same
  final weights, from its commit), so the third value is M7-2's runs and no other copy of them;
- the ablation's folders were started with one tag and one commit, and the rerun check
  (``<ablation>/rerun-check``: the sweep's ext1+2 at seed 42 trained again from that commit)
  reproduces the sweep's run exactly (``diffusion.sweep.reproduces``). Only then do the sweep's runs,
  from the sweep's own commit, stand for the studies' third value;
- each run's config is the sweep's run's at its variant and seed but for the study's setting (how a
  run validates aside: it does not touch training);
- each run's pairing hashes are the sweep's run's, but for those its study moves on purpose
  (``MOVES``): ``d_model`` resizes every weight, and the encoder's depth changes that encoder's weights
  alone. T changes none: its runs make the sweep's runs' very draws, the timesteps they give aside.

One factor at a time: each study keeps every other setting at the sweep's, so T is swept at the
sweep's ``d_model`` (64) alone, the plan's reduced design (§5). For each study, variant and holdout,
each value's test MAE at 10% (mean ± SE over seeds 42-44), and
each value but the sweep's against the sweep's: paired Δ ± SE, its 95% CI (t at 2 df, ±4.303 SE) and
its verdict by the ±1 SE bar, as M7-2's (``diffusion.headline``). Beside each Δ, the same Δ without
the test patients whose record holds a known bad value (``headline.bad_value_patients``): planned
before any M8 run (PROJECT_PLAN §3.5, Nov 25), a secondary result that changes no verdict.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from data.physionet import load_dataset
from diffusion.ablation import SEEDS, STUDIES, setting
from diffusion.gate import flatten
from diffusion.headline import bad_value_patients, mean_and_se, ms, paired, signed, verdict
from diffusion.manifest import entry, shown
from diffusion.results import RESULTS, UNITS
from diffusion.run import CONFIG_PATH, DATASET_PATH
from diffusion.score import holdout_masker, scores_name
from diffusion.sweep import final_scoring, reproduces
from eval import masked_mae

# The pairing hashes each study changes on purpose; every other one must be the sweep's run's. T changes
# none: the diffusion hash is of the stream's generator state after each step, and drawing a timestep
# from range(T) takes the generator as far whatever T is, so a T run makes the sweep run's draws.
MOVES = {"d_model": ("init_shared", "init_mask_encoder", "init_time_encoder"),
         "steps": (),
         "encoder_layers": ("init_mask_encoder",)}
LABELS = {"d_model": "d_model", "steps": "T", "encoder_layers": "encoder layers"}
RERUN = "ext1+2_random_seed42"          # the sweep's run the rerun check trains again (README, The ablation)
# P(|Δ| > SE) for a contrast with no true effect at 3 seeds: P(|T| > 1), T Student's t with 2 df
NULL_SEPARATION = 1 - 3 ** -0.5

def started(folder, problems=None):
    """The tag and commit a folder of the driver's was started with (``sweep.json``); with ``problems``,
    None for a folder without a readable one (Google Drive may not have synced it), listed there."""
    try:
        return json.loads((Path(folder) / "sweep.json").read_text())
    except (OSError, ValueError):
        if problems is None:
            raise
        problems.append(f"{Path(folder).name}: no readable sweep.json, the tag and commit it was started with")
        return None

def study_folders(ablation_dirs):
    """``{(study, value): folder}``: every ``<study>_<value>`` folder in ``ablation_dirs``."""
    found = {}
    for root in ablation_dirs:
        for folder in sorted(Path(root).iterdir()):
            study, _, value = folder.name.rpartition("_")
            if study in STUDIES and value.isdigit():
                if (study, int(value)) in found:
                    raise ValueError(f"{folder.name} is in two of {[shown(d) for d in ablation_dirs]}: keep one")
                found[(study, int(value))] = folder
    return found

def config_differences(config, sweeps):
    """The dotted keys ``config`` differs from the sweep's run's config ``sweeps`` in, how a run validates aside."""
    sweeps = {**sweeps, "model": {"mask_encoder_layers": 2, **sweeps["model"]}}     # saved before M8-1: depth 2
    ours, theirs = flatten(config), flatten(sweeps)
    return sorted(key for key in ours.keys() | theirs.keys()
                  if not key.startswith("validation.") and ours.get(key) != theirs.get(key))

def read(run_dir, variant, seed, holdouts, protocol, commit, data, problems):
    """One run's manifest entry, pairing hashes and MAE with and without the bad-value patients under each
    of ``holdouts``; None, with ``problems`` extended, if it may not be used."""
    fraction = protocol["eval_holdout_fraction"]
    try:
        found = entry(run_dir, "random", variant, seed, protocol, commit, [f"{h}/{fraction}" for h in holdouts])
    except ValueError as error:
        problems.append(str(error))
        return None
    found |= {"pairing": json.loads((run_dir / "summary.json").read_text())["pairing"], "mae": {}, "mae_without": {}}
    for holdout in holdouts:
        name = f"{holdout}/{fraction}"
        scores = final_scoring(run_dir, holdout, fraction, protocol, paired=True)
        if scores is None:
            problems.append(f"{run_dir.name}: no {name} scoring of its final weights done the frozen way, "
                            "with its median beside it")
            continue
        median, target = np.load(run_dir / f"median_{scores_name('test', holdout, fraction)}.npy"), data["targets"][holdout]
        if masked_mae(median, data["values"], target) != scores["mae"]:
            problems.append(f"{run_dir.name}: its {name} median does not give its scores' MAE")
            continue
        found["mae"][name] = scores["mae"]
        found["mae_without"][name] = masked_mae(median, data["values"], target * data["kept"][:, None, None])
    return found

def rerun_problems(rerun_dir, sweep_dir, commit):
    """What keeps the sweep's runs from standing for the studies' third value; empty if nothing."""
    if not (rerun_dir / RERUN / "summary.json").exists():
        return [f"no finished rerun check in {shown(rerun_dir)} ({RERUN}, trained from the ablation's commit)"]
    found = []
    claim = started(rerun_dir, found)
    if claim is not None and commit is not None and claim["commit"] != commit:      # None: the folders' problem
        found.append(f"the rerun check was trained from commit {claim['commit']}, not the ablation's {commit}")
    try:
        differ = reproduces(rerun_dir / RERUN, sweep_dir / RERUN)
    except (OSError, ValueError) as error:
        return found + [f"the rerun check could not be compared with the sweep's {RERUN}: {error}"]
    if differ:
        found.append(f"the rerun check does not reproduce the sweep's {RERUN}: it differs in {', '.join(differ)}. "
                     "Train the sweep's value in each study rather than reuse the sweep's runs")
    return found

def contrast(study, variant, holdout, value, own, mae, without):
    """One value against the sweep's: paired Δ ± SE, CI and verdict, and the same without the bad-value patients."""
    delta = paired({seed: mae[value][seed] - mae[own][seed] for seed in mae[own]})
    rest = paired({seed: without[value][seed] - without[own][seed] for seed in without[own]})
    names = {"value": f"{LABELS[study]} {value}", "own": f"{LABELS[study]} {own}"}
    excludes_0 = lambda d: d["ci95"][0] > 0 or d["ci95"][1] < 0
    return {"variant": variant, "holdout": holdout, "value": value, "against": own, "delta": delta,
            "se_separated": abs(delta["mean"]) > delta["se"], "ci_excludes_0": excludes_0(delta),
            "verdict": verdict(delta, {"value": 1, "own": -1}, "ablation", names),
            "delta_without": rest, "se_separated_without": abs(rest["mean"]) > rest["se"],
            "ci_excludes_0_without": excludes_0(rest)}

def table(ablation_dirs, sweep_dir, rerun_dir, dataset, config, manifest, studies=tuple(STUDIES)):
    """The ablation's record for ``studies``; raises ``ValueError`` listing every problem found.

    ``manifest`` is the sweep's run manifest (``results/sweep_manifest.json``), which the sweep's runs in
    ``sweep_dir`` must be.
    """
    protocol, problems = config["protocol"], []
    fraction, sweep_dir, rerun_dir = protocol["eval_holdout_fraction"], Path(sweep_dir), Path(rerun_dir)
    folders = {key: folder for key, folder in study_folders(ablation_dirs).items() if key[0] in studies}
    claims = {folder.name: started(folder, problems) for folder in folders.values()}
    claims = {name: claim for name, claim in claims.items() if claim is not None}
    commits = {claim["commit"] for claim in claims.values()}
    if len({claim["tag"] for claim in claims.values()}) > 1 or len(commits) > 1:
        problems.append(f"the ablation's folders were started with more than one tag or commit: {claims}")
    commit = next(iter(commits)) if len(commits) == 1 else None
    problems += rerun_problems(rerun_dir, sweep_dir, commit)
    sweeps_commit = (started(sweep_dir, problems) or {}).get("commit")
    if sweeps_commit != manifest["commit"]:
        problems.append(f"{shown(sweep_dir)} was started from commit {sweeps_commit}, and the sweep's manifest "
                        f"records {manifest['commit']}: give the sweep's own folder")

    idx = dataset["test_idx"]
    values, mask, record_ids = dataset["values"][idx], dataset["mask"][idx], dataset["record_ids"][idx]
    excluded = bad_value_patients(dataset)
    data = {"values": values, "kept": ~excluded,
            "targets": {holdout: holdout_masker(protocol, values.shape[1:], holdout, fraction)[0](mask, record_ids)[1]
                        for holdout in {h for study in studies for h in STUDIES[study]["holdouts"]}}}
    runs = {}
    for study in studies:
        plan, own = STUDIES[study], setting(config, study)
        for value in plan["values"]:
            if value == own:
                folder, at = sweep_dir, sweeps_commit
            elif (study, value) in folders:
                folder, at = folders[(study, value)], commit
            else:
                problems.append(f"no {study}_{value} folder in {[shown(d) for d in ablation_dirs]}")
                continue
            for variant in plan["variants"]:
                for seed in SEEDS:
                    run = read(folder / f"{variant}_random_seed{seed}", variant, seed, plan["holdouts"], protocol, at,
                               data, problems)
                    if run is not None:
                        runs[(study, value, variant, seed)] = run

    for (study, value, variant, seed), run in runs.items():
        own, key = setting(config, study), ".".join(STUDIES[study]["setting"])
        if value == own:
            listed = manifest["runs"]["random"].get(variant, {}).get(str(seed), {}).get("weights", {}).get("sha256")
            if run["weights"]["sha256"] != listed:
                problems.append(f"{shown(sweep_dir)}/{run['run']} is not the sweep's run its manifest records: "
                                "its model.pt is other weights")
            continue
        sweeps = runs.get((study, own, variant, seed))
        if sweeps is None:
            continue
        differ = config_differences(run["config"], sweeps["config"])
        if differ != [key] or flatten(run["config"])[key] != value:
            problems.append(f"{study}_{value}/{run['run']}: its config differs from the sweep's run in {differ}, "
                            f"not in {key} alone, at {value}")
        moved = [name for name in sorted(run["pairing"].keys() | sweeps["pairing"].keys())
                 if name not in MOVES[study] and run["pairing"].get(name) != sweeps["pairing"].get(name)]
        if moved:
            problems.append(f"{study}_{value}/{run['run']} is not paired with the sweep's run: its {moved} differ")
    problems = list(dict.fromkeys(problems))        # the sweep's runs are read once per study they stand in
    if problems:
        raise ValueError(f"{len(problems)} problem(s):\n" + "\n".join(problems))

    record = {"note": "M8-1's ablation: each study's values against the sweep's, at seeds 42-44 in the random regime",
              "units": UNITS, "gpu": protocol["paired_gpu"], "seeds": list(SEEDS),
              "commits": {"ablation": commit, "sweep": sweeps_commit},
              "rerun_check": {"run": RERUN, "folder": shown(rerun_dir), "reproduces": True},
              "excluded": {"rule": "test patients whose record holds a known bad value (data.physionet.known_bad_values)",
                           "record_ids": sorted(int(r) for r in record_ids[excluded]),
                           "planned": "before any M8 run (PROJECT_PLAN §3.5, Nov 25); changes no verdict"},
              "studies": {}, "runs": {}}
    for study in studies:
        plan, own = STUDIES[study], setting(config, study)
        cells, contrasts = {}, []
        for variant in plan["variants"]:
            for holdout in (f"{h}/{fraction}" for h in plan["holdouts"]):
                mae, without = ({value: {str(seed): runs[(study, value, variant, seed)][field][holdout] for seed in SEEDS}
                                 for value in plan["values"]} for field in ("mae", "mae_without"))
                cells.setdefault(variant, {})[holdout] = {
                    str(value): {"mae": mean_and_se(list(mae[value].values())), "per_seed": mae[value],
                                 "mae_without": mean_and_se(list(without[value].values()))}
                    for value in plan["values"]}
                contrasts += [contrast(study, variant, holdout, value, own, mae, without)
                              for value in plan["values"] if value != own]
        record["studies"][study] = {"setting": ".".join(plan["setting"]), "sweep_value": own,
                                    "values": list(plan["values"]), "variants": list(plan["variants"]),
                                    "cells": cells, "contrasts": contrasts}
        record["runs"][study] = {str(value): {variant: {str(seed): (
            {"run": f"{variant}_random_seed{seed}", "manifest": "results/sweep_manifest.json"} if value == own else
            {**{k: v for k, v in runs[(study, value, variant, seed)].items() if k not in ("pairing", "mae", "mae_without")},
             "rerun": f"git checkout {commit} && uv run python -m diffusion.ablation --study {study} --tag rerun "
                      "--out <another folder>   (trains the study's cells, this one among them)"})
            for seed in SEEDS} for variant in plan["variants"]} for value in plan["values"]}
    return record

def delta_text(stats):
    """``+0.0031 ± 0.0019``, signed as M7-2's table signs a Δ."""
    return f"{signed(stats['mean'])} ± {stats['se']:.4f}"

def ci(stats):
    return f"{signed(stats['ci95'][0])} to {signed(stats['ci95'][1])}"

def markdown(record):
    """``results/ablation.md``: one table per study."""
    lines = ["# The ablation (M8-1)", "",
             "Generated by `uv run python -m diffusion.ablation_table` from the ablation's and the sweep's run folders; "
             f"do not edit by hand. Test MAE at 10%, in {record['units']}, at seeds "
             f"{', '.join(map(str, record['seeds']))} in the random regime, every run trained and scored on one "
             f"{record['gpu']}. A study's value that is the sweep's is the sweep's own runs (commit "
             f"{record['commits']['sweep'][:7]}); the others were trained from commit {record['commits']['ablation'][:7]}, "
             f"and the rerun check ({record['rerun_check']['run']} trained again from it) reproduces the sweep's run exactly. "
             "One factor at a time: each study keeps every other setting at the sweep's, so T is swept at the "
             "sweep's d_model (64) alone, the plan's reduced design (§5).",
             "",
             "A Δ is a per-seed difference of MAEs, the value's minus the sweep's value's; negative is a gain. The verdict is "
             "the ±1 SE error bar, as M7-2's: \"SE-separated\" if Δ ± SE clears 0, else \"no significant effect\". The 95% "
             f"CI (t at 2 df) changes no verdict. At 3 seeds a contrast with no true effect clears ±1 SE "
             f"{NULL_SEPARATION:.0%} of the time, and nothing here corrects for the number of contrasts.",
             "",
             f"**Without the bad-value patients** is the same Δ leaving out the {len(record['excluded']['record_ids'])} "
             "test patients whose record holds a known bad value: planned before any M8 run (PROJECT_PLAN §3.5), so a "
             "secondary result here, and it changes no verdict."]
    for study, found in record["studies"].items():
        lines += ["", f"## {LABELS[study]} (`{found['setting']}`), the sweep's value {found['sweep_value']}", "",
                  "| Variant | Holdout | Value | MAE ± SE | Δ vs the sweep's ± SE | 95% CI | Without the bad-value patients | Verdict |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        by_cell = {(row["variant"], row["holdout"], row["value"]): row for row in found["contrasts"]}
        for variant, holdouts in found["cells"].items():
            for holdout, values in holdouts.items():
                for value, cell in values.items():
                    row = by_cell.get((variant, holdout, int(value)))
                    if row is None:
                        lines.append(f"| {variant} | {holdout} | {value} (the sweep's) | {ms(cell['mae'])} | — | — | — | — |")
                    else:
                        lines.append(f"| {variant} | {holdout} | {value} | {ms(cell['mae'])} | {delta_text(row['delta'])} | "
                                     f"{ci(row['delta'])} | {delta_text(row['delta_without'])} ({ci(row['delta_without'])}) | "
                                     f"{row['verdict']} |")
    return "\n".join(lines) + "\n"

def main(argv=None):
    parser = argparse.ArgumentParser(description="The ablation's table: each value against the sweep's (M8-1).")
    parser.add_argument("--ablation", type=Path, nargs="+", required=True,
                        help="the ablation's folder(s), holding <study>_<value> folders (two if two accounts ran it)")
    parser.add_argument("--sweep-dir", type=Path, required=True, help="the sweep's folder, read only")
    parser.add_argument("--rerun-check", type=Path, help="the rerun check's folder (default <first --ablation>/rerun-check)")
    parser.add_argument("--studies", nargs="+", choices=list(STUDIES), default=list(STUDIES))
    parser.add_argument("--sweep-manifest", type=Path, default=RESULTS / "sweep_manifest.json",
                        help="the sweep's run manifest, whose runs --sweep-dir must hold")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS / "ablation.json")
    parser.add_argument("--md", type=Path, default=RESULTS / "ablation.md")
    args = parser.parse_args(argv)

    record = table(args.ablation, args.sweep_dir, args.rerun_check or args.ablation[0] / "rerun-check",
                   load_dataset(args.dataset), yaml.safe_load(args.config.read_text()),
                   json.loads(args.sweep_manifest.read_text()), args.studies)
    for path in (args.out, args.md):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    args.md.write_text(markdown(record))
    rows = [row for found in record["studies"].values() for row in found["contrasts"]]
    print(f"wrote {args.out} and {args.md}: {len(rows)} contrasts, {sum(row['se_separated'] for row in rows)} SE-separated")
    return record

if __name__ == "__main__":
    main()
