"""Reconstructed trajectories for the qualitative read (M7-3): the data behind the plots.

    uv run python -m diffusion.trajectories                  # writes results/trajectories/trajectories.npz and .json
    uv run python -m diffusion.trajectories --draw-check     # writes results/trajectories/draw_check.json
    uv run --group plots python -m diffusion.trajectories_plot   # the figure

Five test patients by three variables (HR, serum sodium, GCS), in raw clinical units, on the frozen
10% random holdout (``EvalMasker``): what the model was shown, the held-out values it was scored on,
the imputations of the variant PROJECT_PLAN §3.5 picks (Ext 1+2: the lowest mean test MAE, not
separated from base), and the two baselines (mean imputation, forward fill). The sweep's
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
  the two baselines, and linear interpolation between the visible values. Interpolation is not one
  of the protocol's baselines and is not plotted: it is the simplest guess that reads both sides of
  a gap, so it shows how much of the model's margin over forward fill that alone would explain. The
  five patients are an illustration; these are the numbers behind it.
- **Draw check** (``--draw-check``): the same kind of draws for every eligible patient, not only the
  five, scored two ways: how many held-out values the 5-95% band covers, and how far the draws move
  over 1 to 12 hours against how far the charted values really move.

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
LAGS = (1, 2, 4, 8, 12)                     # hours, for the draw check
VARIANT_NOTES = {"ext1+2": "lowest mean test MAE, not separated from base"}     # PROJECT_PLAN §3.5
PATIENT_RULE = (f"at least one held-out value and at least {MIN_VISIBLE} visible values in each variable; "
                f"{N_PATIENTS} evenly spaced among the eligible, in the test split's order")
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

def eligible_rows(cond, target, columns, min_visible=MIN_VISIBLE):
    """Positions of the patients with a held-out value and ``min_visible`` visible values in each of ``columns``."""
    held_out, visible = target[:, :, columns].sum(axis=1), cond[:, :, columns].sum(axis=1)
    return np.flatnonzero((held_out >= 1).all(axis=1) & (visible >= min_visible).all(axis=1))

def pick_patients(cond, target, columns, n=N_PATIENTS, min_visible=MIN_VISIBLE):
    """Positions of ``n`` patients, evenly spaced among those eligible, in the order given.

    Eligible: at least one held-out value (``target``) and at least ``min_visible`` visible values
    (``cond``) in each of ``columns``, so every panel has something imputed to compare and something
    for forward fill to carry. ``cond`` and ``target`` are ``(N, T, D)``.
    """
    eligible = eligible_rows(cond, target, columns, min_visible)
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} patients have every variable held out and visible")
    return eligible[np.linspace(0, len(eligible) - 1, n).round().astype(int)]

def raw(z, mean, std):
    """Train-split z-scores back to clinical units: ``z * std + mean``, per variable (last axis)."""
    return np.asarray(z, dtype=np.float64) * std + mean

def linear_interpolation(values, cond):
    """Straight lines between each variable's visible values, flat before the first and after the last.

    Only boxes where ``cond == 1`` are read, so held-out values cannot leak in. A variable the patient
    never shows gets the training mean, 0, as in forward fill. ``values`` and ``cond`` are ``(N, T, D)``.
    """
    values, cond = np.asarray(values, dtype=np.float64), np.asarray(cond) == 1
    hours = np.arange(values.shape[1])
    filled = np.zeros(values.shape)
    for n, k in zip(*np.nonzero(cond.any(axis=1))):
        seen = cond[n, :, k]
        filled[n, :, k] = np.interp(hours, hours[seen], values[n, seen, k])
    return filled

def band_coverage(draws, truth, held_out, variables):
    """Held-out values inside the 5-95% band of ``draws`` ``(K, N, T, V)``, per variable and over all."""
    low, high = np.quantile(draws, [0.05, 0.95], axis=0)
    inside = (truth >= low) & (truth <= high) & held_out
    counts = {v: (int(inside[..., j].sum()), int(held_out[..., j].sum())) for j, v in enumerate(variables)}
    counts["all"] = (int(inside.sum()), int(held_out.sum()))
    return {v: {"inside": a, "held_out": b, "fraction": round(a / b, 3)} for v, (a, b) in counts.items()}

def lag_changes(draws, truth, visible, observed, lags=LAGS):
    """Mean |change| over each lag for one variable; ``draws`` are ``(K, N, T)``, the rest ``(N, T)``.

    The draws and their median are read where both hours were hidden from the model, the real values
    where both hours were charted. A lag with no such pair gets ``None``.
    """
    def mean(x):
        return round(float(x.mean()), 3) if x.size else None

    median = median_imputation(draws)
    rows = {}
    for lag in lags:
        hidden = ~visible[:, lag:] & ~visible[:, :-lag]
        charted = observed[:, lag:] & observed[:, :-lag]
        rows[str(lag)] = {"draws": mean(np.abs(draws[:, :, lag:] - draws[:, :, :-lag])[:, hidden]),
                          "median": mean(np.abs(median[:, lag:] - median[:, :-lag])[hidden]),
                          "real": mean(np.abs(truth[:, lag:] - truth[:, :-lag])[charted]),
                          "hidden_pairs": int(hidden.sum()), "charted_pairs": int(charted.sum())}
    return rows

def residual_correlation(draws, visible):
    """Correlation of (draw - median) between neighbouring hidden hours; ``None`` if there are none.

    Near 0, a draw's departure from the median at one hour says nothing about the next hour's.
    """
    hidden = ~visible[:, 1:] & ~visible[:, :-1]
    if not hidden.any():
        return None
    residual = draws - median_imputation(draws)
    return round(float(np.corrcoef(residual[:, :, :-1][:, hidden].ravel(), residual[:, :, 1:][:, hidden].ravel())[0, 1]), 3)

def load_weights(run_dir, device):
    """The model of ``run_dir``, built from its saved config, with its final weights."""
    model = build_model(yaml.safe_load((run_dir / "config.yaml").read_text()), device)
    model.load_state_dict(torch.load(io.BytesIO((run_dir / "model.pt").read_bytes()), map_location=device))
    return model

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

    model = load_weights(runs[seed], device)
    log(f"{draws} draws for {len(rows)} patients from {runs[seed].name} on {device}")
    sampler = EvalSampler(protocol["eval_sampling_seed"], draws, protocol["eval_batch_size"])
    samples = sampler(model, values[rows], mask[rows], cond[rows])[..., columns]        # (K, 5, T, 3)

    medians = {s: np.load(run_dir / "median_test_random_0.1.npy") for s, run_dir in sorted(runs.items())}
    guesses = {"forward_fill": forward_fill(values, cond), "mean_imputation": mean_imputation(values),
               "linear_interpolation": linear_interpolation(values, cond)}

    def at(x):                                                      # the five patients, three variables
        return x[rows][..., columns]

    # The sampler generates only what was hidden from it: at the visible values its output is
    # whatever the denoiser left there (off by hundreds to thousands of units), never scored, so a
    # trajectory keeps the visible values and takes the model's only in between, as CSDI's imputation does.
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

    coverage = band_coverage(arrays["draws"], arrays["truth"], arrays["held_out"], VARIABLES)
    variant = json.loads((runs[seed] / "summary.json").read_text())["variant"]
    record = {
        "variant": variant, "variant_note": VARIANT_NOTES.get(variant), "seed": seed, "draws": draws,
        "holdout": "random/0.1", "split": "test", "units": "raw clinical units (train-split z-scores undone)",
        "variables": list(VARIABLES),
        "patients": {
            "rule": PATIENT_RULE,
            "eligible": len(eligible_rows(cond, target, columns)),
            "record_ids": arrays["record_ids"].tolist(),
            "held_out": arrays["held_out"].sum(axis=1).tolist(),
            "visible": arrays["visible"].sum(axis=1).tolist(),
        },
        "panel_mae": {"median_of_draws": panel_mae(arrays["median"]),
                      f"protocol_median_seed{seed}": panel_mae(arrays["seed_medians"][list(medians).index(seed)]),
                      "forward_fill": panel_mae(arrays["forward_fill"]),
                      "mean_imputation": panel_mae(arrays["mean_imputation"])},
        "held_out_inside_90pct_band": {v: [coverage[v]["inside"], coverage[v]["held_out"]] for v in VARIABLES},
        "test_mae": {
            **{f"seed{s}": dict(zip(VARIABLES, per_variable_mae(m, values, target, dataset["std"])[columns].round(3).tolist()))
               for s, m in medians.items()},
            **{name: dict(zip(VARIABLES, per_variable_mae(g, values, target, dataset["std"])[columns].round(3).tolist()))
               for name, g in guesses.items()},
        },
        "test_held_out": dict(zip(VARIABLES, target[:, :, columns].sum(axis=(0, 1)).astype(int).tolist())),
    }
    return arrays, record

def draw_check(dataset, runs, seed, draws, device, config_path=CONFIG_PATH, log=print):
    """Band coverage and movement of the draws over every eligible test patient: ``draw_check.json``."""
    protocol = yaml.safe_load(Path(config_path).read_text())["protocol"]
    features = [str(f) for f in dataset["features"]]
    columns = [features.index(v) for v in VARIABLES]
    mean, std = dataset["mean"][columns].astype(np.float64), dataset["std"][columns].astype(np.float64)

    idx = dataset["test_idx"]
    values, mask = dataset["values"][idx], dataset["mask"][idx]
    cond, target = EvalMasker.from_config(config_path)(mask, dataset["record_ids"][idx])
    rows = eligible_rows(cond, target, columns)

    model = load_weights(runs[seed], device)
    log(f"{draws} draws for {len(rows)} patients from {runs[seed].name} on {device}")
    sampler = EvalSampler(protocol["eval_sampling_seed"], draws, protocol["eval_batch_size"])
    parts, done = [], 0
    for _, batch in sampler.batches(model, values[rows], mask[rows], cond[rows]):
        parts.append(batch[..., columns])
        done += batch.shape[1]
        log(f"{done} of {len(rows)} patients")
    samples = raw(np.concatenate(parts, axis=1), mean, std)                             # (K, N, T, 3)
    truth = raw(values[rows][..., columns], mean, std)
    visible, held_out, observed = (x[rows][..., columns].astype(bool) for x in (cond, target, mask))
    return {
        "variant": json.loads((runs[seed] / "summary.json").read_text())["variant"], "seed": seed, "draws": draws,
        "holdout": "random/0.1", "split": "test", "units": "raw clinical units (train-split z-scores undone)",
        "patients": {"rule": PATIENT_RULE, "n": len(rows)},
        "band_5_95": band_coverage(samples, truth, held_out, VARIABLES),
        "mean_abs_change": {v: lag_changes(samples[..., j], truth[..., j], visible[..., j], observed[..., j])
                            for j, v in enumerate(VARIABLES)},
        "draw_residual_lag1_correlation": {v: residual_correlation(samples[..., j], visible[..., j])
                                           for j, v in enumerate(VARIABLES)},
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description="The data behind M7-3's trajectory plots.")
    parser.add_argument("--variant", default="ext1+2",
                        help="the variant to read; PROJECT_PLAN §3.5 picks ext1+2. Its runs must be in --sweep")
    parser.add_argument("--regime", default="random")
    parser.add_argument("--seed", type=int, default=42, help="the seed whose weights are sampled")
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--draw-check", action="store_true",
                        help="write draw_check.json (every eligible patient) instead of the five patients' data")
    parser.add_argument("--sweep", type=Path, default=OUTPUTS / "sweep", help="the sweep's run folders, copied from Drive")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    runs = checked_runs(json.loads(args.manifest.read_text()), args.regime, args.variant, args.sweep)
    print(f"{len(runs)} runs of {args.variant} match the manifest's SHA-256", flush=True)
    dataset, device = load_dataset(args.dataset), pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.draw_check:
        record = draw_check(dataset, runs, args.seed, args.draws, device, log=lambda line: print(line, flush=True))
        (args.out / "draw_check.json").write_text(json.dumps(record, indent=1) + "\n")
        print(json.dumps(record["band_5_95"], indent=1))
        return
    arrays, record = trajectories(dataset, runs, args.seed, args.draws, device,
                                  log=lambda line: print(line, flush=True))
    np.savez_compressed(args.out / "trajectories.npz", **arrays)
    (args.out / "trajectories.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("patients", "panel_mae", "test_mae")}, indent=1))

if __name__ == "__main__":
    main()
