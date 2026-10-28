"""The reproduction gate (PROJECT_PLAN §3.4, M4-1): base CSDI's test MAE @10% over at least 3 seeds.

    uv run python -m diffusion.gate outputs/base_random_seed42 outputs/base_random_seed43 outputs/base_random_seed44

Reads each run's random 10% test scores (``scores_test_random_0.1.json`` and its median, from
``diffusion.score``) and writes ``results/gate.json``: each seed's MAE and RMSE, their mean ± SE
(sample standard deviation over √n, as in BRITS's record), and the status:

- ``PASS``: at least ``protocol.gate_min_seeds`` seeds, averaging at most ``protocol.gate_mae_at_10pct``;
- ``FAIL``: as many seeds, averaging more;
- ``INCOMPLETE``: fewer seeds. A partial read, not a gate result.

Runs that would make the number mean something else are refused: a variant other than base, a
seed given twice, configs that differ between the runs (the tracking-only ``validation`` block
aside), scores not taken with the frozen holdout and sampler in ``configs/config.yaml``, scores of
weights other than the run's ``model.pt`` (``model_sha256``), or a median that does not give the
recorded MAE. Where the runs' config departs from the repo's, which is upstream's published setup
(PROJECT_PLAN §3.4), every departure is listed in the record, as M4-1 requires.

The record also carries M4-1's PaCO2/PaO2 check (``bad_value_check``). In BRITS (M3-3) a visible
bad pH dragged same-hour PaCO2 and PaO2 guesses off by up to 6,796 standardized units (README,
Learned baseline); the check shows whether CSDI's feature attention does the same.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from data.physionet import invert_scaler, known_bad_values, load_dataset
from diffusion.run import CONFIG_PATH, DATASET_PATH, ROOT
from diffusion.score import scores_name
from eval import EvalMasker, masked_mae

GATE_PATH = ROOT / "results" / "gate.json"
TRACKING_ONLY = ("tracking", "validation")      # where a run is logged and how it is watched, not what it trains

def mean_se(xs):
    """Mean and SE (sample standard deviation over √n) of ``xs``; SE is None for a single value."""
    xs = np.asarray(xs, dtype=np.float64)
    se = float(xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else None
    return {"mean": round(float(xs.mean()), 6), "se": None if se is None else round(se, 6)}

def flatten(config, prefix=""):
    """A config's leaves by dotted key: ``{"train.lr": 0.001, ...}``."""
    leaves = {}
    for key, value in config.items():
        if isinstance(value, dict):
            leaves.update(flatten(value, f"{prefix}{key}."))
        else:
            leaves[f"{prefix}{key}"] = value
    return leaves

def departures(config, reference):
    """Every leaf where ``config`` differs from ``reference``, ``{key: [config's, reference's]}``.

    The tracking-only blocks are left out: they cannot change what a run trains (README, Training).
    """
    a, b = (flatten({k: v for k, v in c.items() if k not in TRACKING_ONLY}) for c in (config, reference))
    return {key: [a.get(key), b.get(key)] for key in sorted(a.keys() | b.keys()) if a.get(key) != b.get(key)}

def load_run(run_dir, protocol, name):
    """A run's scores and saved config, refused if they are not a gate run's (see the module docstring)."""
    run_dir = Path(run_dir)
    scores = json.loads((run_dir / f"scores_{name}.json").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    frozen_sampler = {"eval_sampling_seed": protocol["eval_sampling_seed"],
                      "eval_n_samples": protocol["eval_n_samples"],
                      "eval_batch_size": protocol["eval_batch_size"]}
    problems = []
    if scores["variant"] != "base":
        problems.append(f"variant {scores['variant']!r}, not base")
    if scores["sampler"] != frozen_sampler:
        problems.append(f"scored with sampler {scores['sampler']}, not the frozen {frozen_sampler}")
    if scores["holdout_settings"] != {"eval_masker_seed": protocol["eval_masker_seed"]}:
        problems.append(f"scored on holdout {scores['holdout_settings']}, not eval_masker_seed {protocol['eval_masker_seed']}")
    if hashlib.sha256((run_dir / "model.pt").read_bytes()).hexdigest() != scores["model_sha256"]:
        problems.append("its scores are of other weights than its model.pt (retrained since?); score it again")
    if problems:
        raise ValueError(f"{run_dir}: " + "; ".join(problems))
    return scores, config

def bad_value_check(dataset, idx, median, cond, target, features=("PaCO2", "PaO2")):
    """Does a visible bad value drag other variables' guesses with it? (M4-1; README, Learned baseline)

    ``median`` is a run's ``(N, T, D)`` guess for the patients ``idx``, scored on ``target`` with
    ``cond`` visible. A patient *shows* a bad value when one of ``data.physionet.known_bad_values``
    is visible to the model. Errors are in standardized units unless named raw. Returns:

    - ``mae`` and ``mae_without_patients_showing_a_bad_value``: the pooled MAE, and the same with
      every such patient left out, as the README reports for BRITS;
    - for each of ``features``: its MAE (standardized and raw), target count, and its largest miss
      (record, hour, raw truth and guess, and whether that patient shows a bad value);
    - ``next_to_a_visible_bad_ph``: every target of ``features`` at an hour where the patient's pH
      is visible and bad, the case where BRITS's feature regression blew up.
    """
    values, record_ids = dataset["values"][idx], dataset["record_ids"][idx]
    names, std, mean = list(dataset["features"]), dataset["std"], dataset["mean"]
    bad = known_bad_values(dataset)[idx] & (cond == 1)
    shows_bad = bad.any(axis=(1, 2))
    miss = np.abs(median.astype(np.float64) - values)
    raw_truth, raw_guess = invert_scaler(values, mean, std), invert_scaler(median, mean, std)

    def target_row(n, t, j):
        return {"record_id": int(record_ids[n]), "hour": int(t), "feature": names[j],
                "truth_raw": round(float(raw_truth[n, t, j]), 4), "guess_raw": round(float(raw_guess[n, t, j]), 4),
                "miss": round(float(miss[n, t, j]), 4)}

    clean_target = target * ~shows_bad[:, None, None]
    result = {
        "n_patients_showing_a_bad_value": int(shows_bad.sum()),
        "mae": round(masked_mae(median, values, target), 6),
        "mae_without_patients_showing_a_bad_value":         # None: every patient shows one
            round(masked_mae(median, values, clean_target), 6) if clean_target.any() else None,
    }
    for name in features:
        j = names.index(name)
        hidden = target[..., j] == 1
        if not hidden.any():
            result[name] = {"n_targets": 0}
            continue
        n, t = np.unravel_index(np.argmax(np.where(hidden, miss[..., j], -1.0)), hidden.shape)
        result[name] = {
            "n_targets": int(hidden.sum()),
            "mae": round(float(miss[..., j][hidden].mean()), 6),
            "mae_raw": round(float(miss[..., j][hidden].mean() * std[j]), 4),
            "largest_miss": target_row(n, t, j) | {"patient_shows_a_bad_value": bool(shows_bad[n])},
        }
    ph = names.index("pH")
    result["next_to_a_visible_bad_ph"] = [
        target_row(n, t, names.index(name)) | {"visible_ph_raw": round(float(raw_truth[n, t, ph]), 4)}
        for n, t in zip(*np.nonzero(bad[..., ph]))
        for name in features if target[n, t, names.index(name)] == 1
    ]
    return result

def gate(run_dirs, dataset, config_path=CONFIG_PATH):
    """The gate record for ``run_dirs``: what ``results/gate.json`` holds."""
    reference = yaml.safe_load(Path(config_path).read_text())
    protocol, grid = reference["protocol"], (reference["data"]["n_timesteps"], reference["data"]["n_features"])
    fraction = protocol["eval_holdout_fraction"]
    name = scores_name("test", "random", fraction)
    runs = [load_run(run_dir, protocol, name) for run_dir in run_dirs]

    seeds = [scores["seed"] for scores, _ in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"a seed is given twice: {seeds}")
    for run_dir, (_, config) in zip(run_dirs[1:], runs[1:]):
        if departures(config, runs[0][1]):
            raise ValueError(f"{run_dir}'s config differs from {run_dirs[0]}'s: {departures(config, runs[0][1])}")

    idx = dataset["test_idx"]
    cond, target = EvalMasker(protocol["eval_masker_seed"], fraction, grid)(dataset["mask"][idx], dataset["record_ids"][idx])
    per_seed = {}
    for run_dir, (scores, _) in zip(run_dirs, runs):
        median = np.load(Path(run_dir) / f"median_{name}.npy")
        if masked_mae(median, dataset["values"][idx], target) != scores["mae"]:
            raise ValueError(f"{run_dir}: median_{name}.npy does not give the MAE in its scores_{name}.json")
        summary = json.loads((Path(run_dir) / "summary.json").read_text())
        curve = json.loads((Path(run_dir) / "curve.json").read_text())
        per_seed[str(scores["seed"])] = {
            "run": scores["run"],
            "wandb_run": scores["wandb_run"],
            "device": scores["device"],
            "mae": round(scores["mae"], 6),
            "rmse": round(scores["rmse"], 6),
            "n_targets": scores["n_targets"],
            "tracked_val_mae": summary["final_val_mae"],        # 5 draws on val, for comparison only
            "healthy_first_500_steps": summary["health"]["healthy"],
            "train_seconds": round(sum(row["seconds"] for row in curve), 1),
            "val_seconds": round(sum(row.get("val_seconds", 0) for row in curve), 1),
            "score_seconds": scores["seconds"],
            "bad_values": bad_value_check(dataset, idx, median, cond, target),
        }

    maes = [scores["mae"] for scores, _ in runs]
    passed = float(np.mean(maes)) <= protocol["gate_mae_at_10pct"]          # unrounded, so 0.2400004 fails
    return {
        "status": "INCOMPLETE" if len(runs) < protocol["gate_min_seeds"] else "PASS" if passed else "FAIL",
        "gate": {"mae_at_most": protocol["gate_mae_at_10pct"], "min_seeds": protocol["gate_min_seeds"],
                 "split": "test", "holdout": "random", "fraction": fraction},
        "units": "standardized (train-split z-scores)",
        "split_seed": reference["data"]["split_seed"],
        "eval_masker_seed": protocol["eval_masker_seed"],
        "sampler": runs[0][0]["sampler"],
        "n_patients": len(idx),
        "n_targets": int(target.sum()),
        "seeds": sorted(seeds),
        "mean_se": {"mae": mean_se(maes), "rmse": mean_se([scores["rmse"] for scores, _ in runs])},
        "per_seed": dict(sorted(per_seed.items())),
        "config_departures": departures(runs[0][1], reference),
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description="The reproduction gate over scored base runs (M4-1).")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="the base runs' output folders")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="where the frozen protocol is read from")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--out", type=Path, default=GATE_PATH)
    args = parser.parse_args(argv)

    record = gate(args.run_dirs, load_dataset(args.dataset), config_path=args.config)
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    for seed, row in record["per_seed"].items():
        print(f"seed {seed}: MAE {row['mae']:.4f}  RMSE {row['rmse']:.4f}")
    mae = record["mean_se"]["mae"]
    print(f"mean MAE {mae['mean']:.4f} ± {mae['se'] or 0:.4f} over {len(record['seeds'])} seeds "
          f"(gate: <= {record['gate']['mae_at_most']}, >= {record['gate']['min_seeds']} seeds): "
          f"{record['status']}  -> {args.out}")

if __name__ == "__main__":
    main()
