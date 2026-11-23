"""Reconstructed trajectories for the qualitative read (M7-3): the data behind the plots.

    uv run python -m diffusion.trajectories                  # writes results/trajectories/trajectories.npz and .json
    python -m diffusion.trajectories_plot                    # any Python with numpy and matplotlib: the figure

Five test patients by three variables (HR, serum sodium, GCS), in raw clinical units, on the frozen
10% random holdout (``EvalMasker``): what the model was shown, the held-out values it was scored on,
the best variant's imputations, and the two baselines (mean imputation, forward fill). The sweep's
weights are not in Git or wandb: copy the variant's run folders from Drive into ``--sweep`` (default
``outputs/sweep``) first. Every seed's ``model.pt`` is checked against its SHA-256 in
``results/sweep_manifest.json``, and refused if it differs.

- **Patients**: chosen by a fixed rule, not by eye (``pick_patients``): among the test patients with
  at least one held-out value and at least two visible values in each of the three variables, five
  evenly spaced in the test split's order.
- **Draws**: ``--draws`` (100) from one seed's weights (``--seed``, 42), with ``EvalSampler``'s frozen
  noise seed. The band and the few draws plotted come from them. They are the trajectories the model
  believes, not the protocol's scoring: that pass drew 100 per patient over the whole test split, and
  its median for each seed (``median_test_random_0.1.npy``) is kept beside them, every seed's, to show
  whether the picture depends on the seed.
- **Trajectories**: the model's draws keep the visible values and fill only what was hidden. The
  sampler's output at a visible value is never meant to be read (nor scored), so it is replaced.
- **Test-split MAE**, per variable in raw units, over every test patient: each seed's protocol median,
  and the two baselines. The five patients are an illustration; these are the numbers behind it.

The draws and the curves are written to ``trajectories.npz`` (only numpy arrays, so the plotting step
needs nothing of this repo's environment) and what was chosen and scored to ``trajectories.json``.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from data.physionet import load_dataset
from diffusion.run import CONFIG_PATH, DATASET_PATH, OUTPUTS, ROOT, pick_device
from eval import EvalMasker, EvalSampler, median_imputation, per_variable_mae
from eval.baselines import forward_fill, mean_imputation
from models.build import build_model

VARIABLES = ("HR", "Na", "GCS")             # heart rate, serum sodium, Glasgow Coma Score
N_PATIENTS = 5
MIN_VISIBLE = 2                             # visible values per variable, so forward fill has something to carry
OUT = ROOT / "results" / "trajectories"
MANIFEST_PATH = ROOT / "results" / "sweep_manifest.json"

def checked_runs(manifest, regime, variant, sweep):
    """``{seed: run folder}`` for every seed of the cell, each ``model.pt`` checked against the manifest."""
    runs, problems = {}, []
    for seed, run in manifest["runs"][regime][variant].items():
        run_dir = Path(sweep) / Path(run["weights"]["path"]).parent.name
        weights = run_dir / "model.pt"
        if not weights.exists():
            problems.append(f"{weights} is missing: copy {run_dir.name} from the sweep's Drive folder")
        elif hashlib.sha256(weights.read_bytes()).hexdigest() != run["weights"]["sha256"]:
            problems.append(f"{weights} is not the weights the manifest records for {run['run']}")
        else:
            runs[int(seed)] = run_dir
    if problems:
        raise ValueError("; ".join(problems))
    return runs

def pick_patients(cond, target, columns, n=N_PATIENTS, min_visible=MIN_VISIBLE):
    """Positions of ``n`` patients, evenly spaced among those eligible, in the order given.

    Eligible: at least one held-out value (``target``) and at least ``min_visible`` visible values
    (``cond``) in each of ``columns``, so every panel has something imputed to compare and something
    for forward fill to carry. ``cond`` and ``target`` are ``(N, T, D)``.
    """
    held_out, visible = target[:, :, columns].sum(axis=1), cond[:, :, columns].sum(axis=1)
    eligible = np.flatnonzero((held_out >= 1).all(axis=1) & (visible >= min_visible).all(axis=1))
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} patients have every variable held out and visible")
    return eligible[np.linspace(0, len(eligible) - 1, n).round().astype(int)]

def raw(z, mean, std):
    """Train-split z-scores back to clinical units: ``z * std + mean``, per variable (last axis)."""
    return np.asarray(z, dtype=np.float64) * std + mean

def trajectories(dataset, runs, seed, draws, device, config_path=CONFIG_PATH, log=print):
    """The arrays ``trajectories.npz`` holds and the record ``trajectories.json`` holds, for ``runs``."""
    protocol = yaml.safe_load(Path(config_path).read_text())["protocol"]
    features = [str(f) for f in dataset["features"]]
    columns = [features.index(v) for v in VARIABLES]
    mean, std = dataset["mean"][columns].astype(np.float64), dataset["std"][columns].astype(np.float64)

    idx = dataset["test_idx"]
    values, mask = dataset["values"][idx], dataset["mask"][idx]
    cond, target = EvalMasker.from_config(config_path)(mask, dataset["record_ids"][idx])
    rows = pick_patients(cond, target, columns)

    run_config = yaml.safe_load((runs[seed] / "config.yaml").read_text())
    model = build_model(run_config, device)
    model.load_state_dict(torch.load(io.BytesIO((runs[seed] / "model.pt").read_bytes()), map_location=device))
    log(f"{draws} draws for {len(rows)} patients from {runs[seed].name} on {device}")
    sampler = EvalSampler(protocol["eval_sampling_seed"], draws, protocol["eval_batch_size"])
    samples = sampler(model, values[rows], mask[rows], cond[rows])[..., columns]        # (K, 5, T, 3)

    medians = {s: np.load(run_dir / "median_test_random_0.1.npy") for s, run_dir in sorted(runs.items())}
    guesses = {"forward_fill": forward_fill(values, cond), "mean_imputation": mean_imputation(values)}

    def at(x):                                                      # the five patients, three variables
        return x[rows][..., columns]

    # The sampler generates only what was hidden from it: at the visible values its output is
    # whatever the denoiser left there (off by thousands of units), never scored, so a trajectory
    # keeps the visible values and takes the model's only in between, as CSDI's imputation does.
    visible, truth = at(cond).astype(bool), raw(at(values), mean, std)

    def imputed(z):
        return np.where(visible, truth, raw(z, mean, std))

    arrays = {
        "hours": np.arange(values.shape[1]),
        "variables": np.array(VARIABLES),
        "record_ids": dataset["record_ids"][idx][rows],
        "truth": truth,
        "visible": visible,
        "held_out": at(target).astype(bool),
        "observed": at(mask).astype(bool),
        "draws": imputed(samples).astype(np.float32),
        "median": imputed(median_imputation(samples)),
        "seed_medians": np.stack([imputed(at(m)) for m in medians.values()]),
        "seeds": np.array(list(medians)),
        "forward_fill": raw(at(guesses["forward_fill"]), mean, std),
        "mean_imputation": raw(at(guesses["mean_imputation"]), mean, std),
    }

    def panel_mae(guess):                                           # (5, 3): each panel's held-out values
        err = np.abs(guess - arrays["truth"]) * arrays["held_out"]
        return (err.sum(axis=1) / arrays["held_out"].sum(axis=1)).round(3).tolist()

    low, high = np.quantile(arrays["draws"], [0.05, 0.95], axis=0)
    inside = (arrays["truth"] >= low) & (arrays["truth"] <= high) & arrays["held_out"]
    record = {
        "variant": json.loads((runs[seed] / "summary.json").read_text())["variant"], "seed": seed, "draws": draws,
        "holdout": "random/0.1", "split": "test", "units": "raw clinical units (train-split z-scores undone)",
        "variables": list(VARIABLES),
        "patients": {
            "rule": f"at least one held-out value and at least {MIN_VISIBLE} visible values in each variable; "
                    f"{N_PATIENTS} evenly spaced among the eligible, in the test split's order",
            "eligible": int(((target[:, :, columns].sum(1) >= 1).all(1)
                             & (cond[:, :, columns].sum(1) >= MIN_VISIBLE).all(1)).sum()),
            "record_ids": arrays["record_ids"].tolist(),
            "held_out": arrays["held_out"].sum(axis=1).tolist(),
            "visible": arrays["visible"].sum(axis=1).tolist(),
        },
        "panel_mae": {"median_of_draws": panel_mae(arrays["median"]),
                      f"protocol_median_seed{seed}": panel_mae(arrays["seed_medians"][list(medians).index(seed)]),
                      "forward_fill": panel_mae(arrays["forward_fill"]),
                      "mean_imputation": panel_mae(arrays["mean_imputation"])},
        "held_out_inside_90pct_band": {v: [int(inside[:, :, j].sum()), int(arrays["held_out"][:, :, j].sum())]
                                       for j, v in enumerate(VARIABLES)},
        "test_mae": {
            **{f"seed{s}": dict(zip(VARIABLES, per_variable_mae(m, values, target, dataset["std"])[columns].round(3).tolist()))
               for s, m in medians.items()},
            **{name: dict(zip(VARIABLES, per_variable_mae(g, values, target, dataset["std"])[columns].round(3).tolist()))
               for name, g in guesses.items()},
        },
        "test_held_out": dict(zip(VARIABLES, target[:, :, columns].sum(axis=(0, 1)).astype(int).tolist())),
    }
    return arrays, record

def main(argv=None):
    parser = argparse.ArgumentParser(description="The data behind M7-3's trajectory plots.")
    parser.add_argument("--variant", default="ext1+2", help="the best variant (M7-2); its runs must be in --sweep")
    parser.add_argument("--regime", default="random")
    parser.add_argument("--seed", type=int, default=42, help="the seed whose weights are sampled")
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--sweep", type=Path, default=OUTPUTS / "sweep", help="the sweep's run folders, copied from Drive")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    runs = checked_runs(json.loads(args.manifest.read_text()), args.regime, args.variant, args.sweep)
    print(f"{len(runs)} runs of {args.variant} match the manifest's SHA-256", flush=True)
    arrays, record = trajectories(load_dataset(args.dataset), runs, args.seed, args.draws, pick_device(args.device),
                                  log=lambda line: print(line, flush=True))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "trajectories.npz", **arrays)
    (args.out / "trajectories.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("patients", "panel_mae", "test_mae")}, indent=1))

if __name__ == "__main__":
    main()
