"""The headline table (M7-2): each variant's mean ± SE, and each paired Δ ± SE with its 95% CI and verdict.

    uv run python -m diffusion.headline                   # writes results/headline.json and results/headline.md
    uv run python -m diffusion.headline --skip-post-hoc   # without outputs/sweep: no post-hoc read

Reads the sweep's results store (``results/sweep.json``, ``diffusion.results``), its run manifest
(``results/sweep_manifest.json``, ``diffusion.manifest``) for the extensions' gates, and the baselines'
records (``results/baselines.json``, ``results/learned_baseline.json``). Cited numbers are in ``CITED``.
Everything is test MAE at the frozen holdouts, in standardized units (PROJECT_PLAN §3.2).

The rules (PROJECT_PLAN §3.3, §3.5):

- **Paired.** A contrast is a weighted sum of variants' scores taken seed by seed, inside one regime
  (``Ext 2 − base`` is ``{ext2: +1, base: −1}``), so its SE is the sample SD of the per-seed values over
  √n, not a combination of two means' SEs. Every variant in it must have run at the same seeds; no
  contrast mixes the random and block-aware regimes.
- **Verdict: the ±1 SE error bar** (the plan's "SE-separated"; settled at M7-2). A contrast whose
  Δ ± SE crosses 0 is "no significant effect"; one whose bar clears 0 is SE-separated, as a gain
  (lower MAE) or a harm. With 4 degrees of freedom, a contrast with no true effect clears its bar
  37% of the time (``NULL_SEPARATION``), and nothing here corrects for the number of contrasts.
- **95% CI** beside each, Δ ± t·SE with t the 97.5% point of Student's t at n − 1 degrees of freedom
  (2.776 at 5 seeds, 4.303 at 3). Each verdict says whether the CI excludes 0 and, unless it does,
  the largest gain it allows. The CI changes no verdict.
- **Gates** beside each extension's Δ vs. base, from the manifest's final ``model.pt`` gates: mean
  |gate| ± SE over seeds, with each seed's signed value. The sign means nothing, and no threshold
  reads a gate as used or ignored.
- **M8-4's trigger** is Ext 1's block contrast (block-aware regime, block holdout): M8-4 runs only
  if it is SE-separated (§4.1).

The **post-hoc read** (``post_hoc``; chosen on Nov 25 after the results were seen, so it changes no
verdict) repeats every contrast, pooled and per variable, without the 12 test patients whose record
holds a known bad value (``data.physionet.known_bad_values``; README, Dataset), as ``results/gate.json``
does for the gate; ``results/headline.md`` shows each beside its Δ. From M8 on the same cut
(``bad_value_patients``) is planned before the results (PROJECT_PLAN §3.5). It needs each run's saved median (``median_test_<holdout>_0.1.npy``) in the sweep's folder
(``--sweep-dir``, default ``outputs/sweep``, copied from Drive) and the dataset, and refuses a median
whose MAE is not exactly the store's, which would be another scoring's.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from data.physionet import known_bad_values, load_dataset
from diffusion.results import HEADLINE, RESULTS, UNITS, sweep_plan
from diffusion.run import CONFIG_PATH, DATASET_PATH, OUTPUTS, VARIANTS
from diffusion.score import holdout_masker
from eval import masked_mae, per_variable_mae

# The 97.5% point of Student's t by degrees of freedom: a 95% CI is Δ ± t·SE (PROJECT_PLAN §3.5)
T975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582, 6: 2.446912, 7: 2.364624,
        8: 2.306004, 9: 2.262157}
# P(|Δ| > SE) for a contrast with no true effect at 5 seeds: P(|T| > 1), T Student's t with 4 df
NULL_SEPARATION = 0.374

# Cited, not run by us: random holdout, test MAE (mean, SE) by fraction. Tashiro et al. 2021, Table 3,
# report "the mean and the standard error for five trials", each testing a different fifth of the
# patients (their Appendix E.2.1). The independent reimplementation's numbers are PROJECT_PLAN §6's.
CITED = {
    "brits_cited": {"label": "BRITS", "source": "Tashiro 2021, Table 3",
                    "mae": {"0.1": (0.284, 0.001), "0.5": (0.368, 0.002), "0.9": (0.517, 0.002)}},
    "v_rin": {"label": "V-RIN", "source": "Tashiro 2021, Table 3",
              "mae": {"0.1": (0.271, 0.001), "0.5": (0.365, 0.002), "0.9": (0.606, 0.006)}},
    "csdi_paper": {"label": "CSDI (paper)", "source": "Tashiro 2021, Table 3",
                   "mae": {"0.1": (0.217, 0.001), "0.5": (0.301, 0.002), "0.9": (0.481, 0.003)}},
    "csdi_cfmi": {"label": "CSDI (independent reimpl.)", "source": "CFMI 2025",
                  "mae": {"0.1": (0.228, None), "0.5": (0.319, None), "0.9": (0.530, None)}},
}

def difference(a, b):
    """The contrast ``a − b``: ``{a: +1, b: −1}``."""
    return {a: 1, b: -1}

INTERACTION = {"ext1+2": 1, "ext1": -1, "ext2": -1, "base": 1}     # (Ext 1+2 − Ext 2) − (Ext 1 − base)
NAMES = {"base": "base", "ext1": "Ext 1", "ext2": "Ext 2", "ext1+2": "Ext 1+2"}

# Every contrast reported: (group, regime, holdout, weights, variable). ``variable`` None is the pooled
# MAE; a name is that variable's MAE (Ext 2's primary regime, PROJECT_PLAN §4.2).
CONTRASTS = (
    ("main", "random", "random/0.1", difference("ext1", "base"), None),
    ("main", "random", "random/0.1", difference("ext2", "base"), None),
    ("main", "random", "random/0.1", difference("ext1+2", "base"), None),
    ("block", "block-aware", "block/0.1", difference("ext1", "base"), None),
    ("block", "random", "block/0.1", difference("ext1", "base"), None),
    ("stratified", "random", "block/0.1", difference("ext2", "base"), None),
    ("stratified", "random", "block/0.1", difference("ext1+2", "base"), None),
    ("stratified", "block-aware", "random/0.1", difference("ext1", "base"), None),
    ("ext2_variables", "random", "random/0.1", difference("ext2", "base"), "HR"),
    ("ext2_variables", "random", "random/0.1", difference("ext2", "base"), "SysABP"),
    *(("interaction", "random", holdout, weights, None)
      for holdout in HEADLINE
      for weights in (difference("ext1+2", "ext1"), difference("ext1+2", "ext2"), INTERACTION)),
)
GROUPS = {
    "main": "Main effects (confirmatory) and the combined model: random regime, random holdout",
    "block": "Ext 1's block contrast (confirmatory; M8-4's trigger), with the random regime's carry-over check",
    "stratified": "The other contrasts vs. base under the other holdout",
    "ext2_variables": "Ext 2's primary regime (§4.2): heart rate and systolic arterial pressure, random holdout",
    "interaction": "Interaction (exploratory): Ext 1+2 vs. each alone, and the interaction term",
}
ROLES = {"main": "confirmatory", "block": "confirmatory", "stratified": "reported beside the confirmatory ones",
         "ext2_variables": "Ext 2's primary regime", "interaction": "exploratory"}

def t975(df):
    """The 97.5% point of Student's t at ``df`` degrees of freedom."""
    if df not in T975:
        raise ValueError(f"no t value for {df} degrees of freedom (a contrast needs 2 to 10 seeds)")
    return T975[df]

def mean_and_se(xs):
    """Mean and SE (sample SD over √n) of ``xs``, unrounded; SE is None for one value."""
    xs = np.asarray(xs, dtype=np.float64)
    return {"mean": float(xs.mean()), "se": float(xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else None,
            "n_seeds": len(xs)}

def paired(per_seed):
    """Mean, SE and 95% CI of a contrast's per-seed values, ``{seed: value}``."""
    if len(per_seed) < 2:
        raise ValueError(f"a paired contrast needs at least 2 seeds, not {sorted(per_seed)}")
    per_seed = dict(sorted(per_seed.items(), key=lambda item: int(item[0])))   # a fixed order: the same last digit
    stats = mean_and_se(list(per_seed.values()))
    half = t975(len(per_seed) - 1) * stats["se"]
    return {**stats, "ci95": [stats["mean"] - half, stats["mean"] + half], "t": t975(len(per_seed) - 1),
            "per_seed": per_seed}

def cell_scores(store, regime, variant, holdout, variable=None):
    """``{seed: MAE}`` of one regime's variant under ``holdout``: pooled, or ``variable``'s alone."""
    runs = store["runs"].get(regime, {}).get(variant)
    if not runs:
        raise ValueError(f"the store has no {variant} runs in the {regime} regime")
    found = {}
    for seed, run in runs.items():
        scoring = run["scores"].get(holdout)
        if scoring is None:
            raise ValueError(f"{run['run']} has no {holdout} scoring")
        found[seed] = scoring["mae"] if variable is None else scoring["per_variable_mae"][variable]
    return found

def contrast_values(store, regime, holdout, weights, variable=None):
    """A contrast's per-seed values, ``{seed: Σ weight · MAE}``, from one regime's runs only.

    Refused unless every variant in it ran at the same seeds: a seed one of them lacks is unpaired.
    """
    scores = {variant: cell_scores(store, regime, variant, holdout, variable) for variant in weights}
    seeds = {variant: frozenset(found) for variant, found in scores.items()}
    if len(set(seeds.values())) > 1:
        raise ValueError(f"the {regime} regime's {sorted(weights)} did not run at the same seeds: "
                         f"{ {variant: sorted(found, key=int) for variant, found in seeds.items()} }")
    return {seed: sum(weight * scores[variant][seed] for variant, weight in weights.items())
            for seed in sorted(next(iter(seeds.values())), key=int)}

def contrast_name(weights, variable=None):
    """``Ext 2 − base``, ``(Ext 1+2 − Ext 2) − (Ext 1 − base)``, or ``Ext 2 − base, HR``."""
    if weights == INTERACTION:
        name = "(Ext 1+2 − Ext 2) − (Ext 1 − base)"
    else:
        [a], [b] = ([v for v, w in weights.items() if w == sign] for sign in (1, -1))
        name = f"{NAMES[a]} − {NAMES[b]}"
    return name if variable is None else f"{name}, {variable}"

def signed(x, digits=4):
    """``+0.0031`` or ``−0.0011``, with a real minus sign; ``0.0000`` for anything that rounds to 0."""
    text = f"{x:+.{digits}f}"
    return f"{0:.{digits}f}" if float(text) == 0 else text.replace("-", "−")

def verdict(delta, weights, group):
    """The written verdict: SE-separated (±1 SE) or "no significant effect", and what the 95% CI says.

    Unless the CI excludes 0, it also says the largest gain the CI allows (a gain lowers MAE: −low).
    """
    mean, se, (low, high) = delta["mean"], delta["se"], delta["ci95"]
    prefix = "Exploratory. " if group == "interaction" else ""
    gain = f"it allows a gain of at most {max(0.0, -low):.4f}"
    if abs(mean) <= se:
        return f"{prefix}No significant effect: Δ ± SE crosses 0. The 95% CI includes 0; {gain}."
    direction = "better" if mean < 0 else "worse"
    if weights == INTERACTION:
        found = f"together the extensions do {direction} than their separate effects add up to"
    else:
        [a], [b] = ([NAMES[v] for v, w in weights.items() if w == sign] for sign in (1, -1))
        found = f"{a} is {direction} than {b}"
    found = f"{prefix}SE-separated: {found}, " + (f"by {abs(mean) / se:.1f} SE." if se > 0 else "by the same at every seed (SE 0).")
    if low > 0 or high < 0:
        return f"{found} The 95% CI excludes 0."
    return f"{found} The 95% CI includes 0, so this is not established at 95%; {gain}."

def gate_summary(manifest, regime, variant):
    """``{gate: {"abs": mean ± SE of |gate|, "signed": {seed: gate}}}`` for one regime's variant."""
    runs = manifest["runs"][regime][variant]
    fields = sorted(set().union(*(run["gates"] for run in runs.values())))
    return {field: {"abs": mean_and_se([abs(run["gates"][field]) for run in runs.values()]),
                    "signed": {seed: run["gates"][field] for seed, run in sorted(runs.items(), key=lambda i: int(i[0]))}}
            for field in fields}

def gates_beside(manifest, regime, weights):
    """The gates of the extension in an ``X − base`` contrast; None for any other contrast or for base."""
    if weights == INTERACTION or weights.get("base") != -1:
        return None
    [variant] = [v for v, w in weights.items() if w == 1]
    return gate_summary(manifest, regime, variant) or None

def contrasts(store, manifest=None):
    """Every contrast in ``CONTRASTS``, with its paired Δ, verdict and (given ``manifest``) gates."""
    found = []
    for group, regime, holdout, weights, variable in CONTRASTS:
        delta = paired(contrast_values(store, regime, holdout, weights, variable))
        row = {"group": group, "role": ROLES[group], "name": contrast_name(weights, variable), "regime": regime,
               "holdout": holdout, "variable": variable, "weights": weights, "delta": delta,
               "se_separated": abs(delta["mean"]) > delta["se"],
               "ci_excludes_0": delta["ci95"][0] > 0 or delta["ci95"][1] < 0,
               "verdict": verdict(delta, weights, group)}
        if manifest is not None:
            row["gates"] = gates_beside(manifest, regime, weights)
        found.append(row)
    return found

def variant_rows(store, regimes):
    """``{regime: {variant: {holdout: {"mae", "rmse", "per_seed"}}}}``: mean ± SE over seeds, every scoring."""
    rows = {}
    for regime in regimes:
        for variant in (v for v in VARIANTS if v in store["runs"].get(regime, {})):
            runs = store["runs"][regime][variant]
            for holdout in sorted(set().union(*(run["scores"] for run in runs.values()))):
                have = {seed: run["scores"][holdout] for seed, run in sorted(runs.items(), key=lambda i: int(i[0]))
                        if run["scores"].get(holdout)}
                rows.setdefault(regime, {}).setdefault(variant, {})[holdout] = {
                    "mae": mean_and_se([s["mae"] for s in have.values()]),
                    "rmse": mean_and_se([s["rmse"] for s in have.values()]),
                    "per_seed": {seed: {"mae": s["mae"], "rmse": s["rmse"]} for seed, s in have.items()}}
    return rows

def baseline_rows(baselines, learned, protocol):
    """Mean imputation and forward fill (deterministic: one value) and our BRITS (mean ± SE), test split,
    by holdout; refused if a record was scored on other holdouts than the frozen ones."""
    frozen = {"random": {"eval_masker_seed": protocol["eval_masker_seed"]},
              "block": {"block_eval_masker_seed": protocol["block_eval_masker_seed"],
                        "block_window_hours": protocol["block_window_hours"]}}
    rows = {}
    for holdout, settings in frozen.items():
        for name, record in (("results/baselines.json", baselines), ("results/learned_baseline.json", learned["mean_se"])):
            recorded = {key: record[holdout].get(key) for key in settings}
            if recorded != settings:
                raise ValueError(f"{name}'s {holdout} holdout is {recorded}, not the frozen {settings}")
        for fraction, splits in baselines[holdout]["fractions"].items():
            test, brits = splits["test"], learned["mean_se"][holdout]["fractions"][fraction]["test"]
            if brits["n_targets"] != test["n_targets"]:
                raise ValueError(f"BRITS and the baselines scored different {holdout}/{fraction} targets")
            rows[f"{holdout}/{fraction}"] = {
                "n_patients": test["n_patients"], "n_targets": test["n_targets"],
                "mean_imputation": test["mean_imputation"], "forward_fill": test["forward_fill"],
                "brits": {**brits["brits"], "n_seeds": len(learned["seeds"])}}
    return rows

def joined(store, manifest):
    """Refuse a manifest that is not of the store's runs: every cell must name the same run on wandb."""
    problems = []
    for regime, variants in store["runs"].items():
        for variant, runs in variants.items():
            for seed, run in runs.items():
                other = manifest["runs"].get(regime, {}).get(variant, {}).get(seed)
                if other is None or (other["run"], other["wandb_run"]) != (run["run"], run["wandb_run"]):
                    problems.append(f"{run['run']}: the manifest has {None if other is None else other['wandb_run']}, "
                                    f"the store {run['wandb_run']}")
    if problems:
        raise ValueError("the manifest is not of the store's runs: " + "; ".join(problems))

def headline(store, manifest, baselines, learned, protocol, regimes):
    """The headline record (``results/headline.json``) but for its post-hoc read."""
    if store["unscored"] or store["missing"]:
        raise ValueError(f"the store is not complete: unscored {store['unscored']}, missing {store['missing']}")
    if store["units"] != UNITS:
        raise ValueError(f"the store is in {store['units']!r}, not {UNITS!r}")
    joined(store, manifest)
    found = contrasts(store, manifest)
    [trigger] = [row for row in found if (row["group"], row["regime"]) == ("block", "block-aware")]
    triggered = trigger["se_separated"]
    return {
        "tag": store["tag"], "units": UNITS, "gpu": store["gpu"], "split": "test",
        "rule": {"verdict": "SE-separated when the ±1 SE error bar clears 0, else \"no significant effect\" "
                            "(PROJECT_PLAN §3.3, §3.5)",
                 "ci95": "Δ ± t·SE, t the 97.5% point of Student's t at n − 1 degrees of freedom; changes no verdict",
                 "null_separation": NULL_SEPARATION, "corrected_for_multiple_contrasts": False},
        "m8_4": {"triggered": triggered, "line": (
            f"Ext 1's block contrast (block-aware regime, block holdout) is {signed(trigger['delta']['mean'])} "
            f"± {trigger['delta']['se']:.4f}, 95% CI {signed(trigger['delta']['ci95'][0])} to "
            f"{signed(trigger['delta']['ci95'][1])}: "
            + ("SE-separated, so M8-4's attribution checks run." if triggered else
               "no significant effect, so M8-4's attribution checks do not run (§4.1: a null needs no control)."))},
        "variants": variant_rows(store, regimes),
        "baselines": baseline_rows(baselines, learned, protocol),
        "cited": CITED,
        "contrasts": found,
        "gates": {regime: {variant: gate_summary(manifest, regime, variant)
                           for variant in manifest["runs"].get(regime, {}) if variant != "base"}
                  for regime in regimes},
    }

def bad_value_patients(dataset):
    """Which test patients' records hold a known bad value, ``(N_test,)`` bool, whatever the holdout."""
    return known_bad_values(dataset)[dataset["test_idx"]].any(axis=(1, 2))

def post_hoc(store, sweep_dir, dataset, protocol):
    """Every contrast again without the test patients whose record holds a known bad value.

    Post-hoc (chosen after the results were seen) and changes no verdict. Each run's saved median must
    give exactly the store's MAE, so it is the median that was scored.
    """
    idx = dataset["test_idx"]
    values, mask, record_ids = dataset["values"][idx], dataset["mask"][idx], dataset["record_ids"][idx]
    bad, excluded = known_bad_values(dataset)[idx], bad_value_patients(dataset)
    clean = json.loads(json.dumps(store))          # a copy whose MAEs become the ones without those patients
    holdouts, problems = {}, []
    for holdout in HEADLINE:
        name, fraction = holdout.split("/")
        masker, _ = holdout_masker(protocol, values.shape[1:], name, float(fraction))
        cond, target = masker(mask, record_ids)
        kept = target * ~excluded[:, None, None]
        holdouts[holdout] = {"n_targets": int(target.sum()), "n_targets_excluded": int(target.sum() - kept.sum()),
                             "n_patients_showing_a_bad_value": int((bad & (cond == 1)).any(axis=(1, 2)).sum()),
                             "n_bad_values_scored": int((bad & (target == 1)).sum())}
        for regime, variants in store["runs"].items():
            for variant, runs in variants.items():
                for seed, run in runs.items():
                    median = np.load(Path(sweep_dir) / run["run"] / f"median_test_{name}_{fraction}.npy")
                    if masked_mae(median, values, target) != run["scores"][holdout]["mae"]:
                        problems.append(f"{run['run']}'s median_test_{name}_{fraction}.npy does not give the store's MAE")
                        continue
                    scores = clean["runs"][regime][variant][seed]["scores"][holdout]
                    scores["mae"] = masked_mae(median, values, kept)
                    scores["per_variable_mae"] = {str(feature): None if np.isnan(mae) else float(mae) for feature, mae
                                                  in zip(dataset["features"], per_variable_mae(median, values, kept))}
    if problems:
        raise ValueError("; ".join(problems))
    rows = []
    for group, regime, holdout, weights, variable in CONTRASTS:
        without = paired(contrast_values(clean, regime, holdout, weights, variable))
        rows.append({"group": group, "name": contrast_name(weights, variable), "regime": regime, "holdout": holdout,
                     "variable": variable, "delta": paired(contrast_values(store, regime, holdout, weights, variable)),
                     "delta_without": without, "se_separated_without": abs(without["mean"]) > without["se"],
                     "ci_excludes_0_without": without["ci95"][0] > 0 or without["ci95"][1] < 0})
    return {"note": "post-hoc, chosen on Nov 25 after the results were seen; changes no verdict",
            "excluded": {"rule": "test patients whose record holds a known bad value (data.physionet.known_bad_values)",
                         "record_ids": sorted(int(r) for r in record_ids[excluded]), "holdouts": holdouts},
            "mae_without": {regime: {variant: {seed: {holdout: run["scores"][holdout]["mae"] for holdout in HEADLINE}
                                               for seed, run in runs.items()}
                                     for variant, runs in variants.items()}
                            for regime, variants in clean["runs"].items()},
            "contrasts": rows}

def ms(stats, digits=4):
    """``0.2197 ± 0.0012``, or the mean alone when there is no SE."""
    return f"{stats['mean']:.{digits}f}" + ("" if stats.get("se") is None else f" ± {stats['se']:.{digits}f}")

def gate_text(gates):
    """``mask_encoder_gate |g| 0.20 ± 0.03 (−0.18, −0.18, −0.20, +0.16, +0.30)`` per gate, for a table cell."""
    if not gates:
        return "—"
    return "; ".join(f"{field.replace('_gate', '')} |g| {ms(g['abs'], 3)} "
                     f"({', '.join(signed(v, 2) for v in g['signed'].values())})"
                     for field, g in gates.items())

def markdown(record):
    """``results/headline.md``: the tables and every verdict, generated from the record."""
    variants, base = record["variants"], record["baselines"]
    random_n = base["random/0.1"]
    lines = [
        "# Headline table (M7-2)", "",
        f"Generated by `uv run python -m diffusion.headline` from `results/sweep.json` (tag `{record['tag']}`), "
        "`results/sweep_manifest.json`, `results/baselines.json` and `results/learned_baseline.json`; do not edit by "
        "hand. Test MAE at 10%, in standardized units (train-split z-scores), over the test split's "
        f"{random_n['n_patients']} patients "
        f"({random_n['n_targets']:,} held-out values under each holdout). Every run trained and was scored on one "
        f"{record['gpu']}.", "",
        "## Mean ± SE", "",
        "| Model | Random 10% | Block 10% | Seeds | Source |", "| --- | --- | --- | --- | --- |",
    ]
    for key, label in (("mean_imputation", "Mean imputation"), ("forward_fill", "Forward fill")):
        lines.append(f"| {label} | {base['random/0.1'][key]['mae']:.4f} | {base['block/0.1'][key]['mae']:.4f} | — | "
                     "our run (deterministic) |")
    lines.append(f"| BRITS (ours) | {ms(base['random/0.1']['brits']['mae'])} | {ms(base['block/0.1']['brits']['mae'])} | "
                 f"{base['random/0.1']['brits']['n_seeds']} | our run (M3-3) |")
    for cited in record["cited"].values():
        mean, se = cited["mae"]["0.1"]
        lines.append(f"| {cited['label']} | {mean:.3f}{'' if se is None else f' ± {se:.3f}'} | — | — | {cited['source']} |")
    for regime, title in (("random", "random regime"), ("block-aware", "block-aware regime")):
        for variant, holdouts in variants.get(regime, {}).items():
            label = "CSDI base (ours)" if variant == "base" else f"+ {NAMES[variant]}"
            lines.append(f"| **{label}**, {title} | {ms(holdouts['random/0.1']['mae'])} | {ms(holdouts['block/0.1']['mae'])} "
                         f"| {holdouts['random/0.1']['mae']['n_seeds']} | our run, paired seeds |")
    lines += [
        "",
        "- **What our ± covers.** Training randomness only: the test split is fixed, so every run, ours and the "
        "baselines', scores the same 800 patients. The cited standard errors (CSDI, Table 3) also cover which "
        "patients were tested, because each of CSDI's five trials tested a different fifth of them (its "
        "Appendix E.2.1). Mean imputation and forward fill have no seed, so no ±.",
        "- **Regimes.** The block-aware rows are trained with `TrainMasker`'s `mix` strategy and are paired only "
        "with each other; no Δ below crosses regimes (PROJECT_PLAN §3.1).",
        "- **Block holdout.** One of its targets is a recorded pH of 733, which every model misses by about 520 "
        "standardized units: about +0.02 on every block MAE, nearly the same for every model (README, Baselines).",
        "- **Cited rows** are on the random holdout only; the block holdout has no published counterpart. RMSE is "
        "in `results/headline.json`; the known bad values dominate it (README, Dataset).",
        "",
        "## Paired Δ vs. base ± SE, 95% CI, verdict", "",
        f"A Δ is a per-seed difference of MAEs inside one regime; negative is a gain. The verdict is the ±1 SE "
        f"error bar (PROJECT_PLAN §3.3): \"SE-separated\" if Δ ± SE clears 0, else \"no significant effect\". The "
        f"95% CI (t at n − 1 df) changes no verdict. With 4 df a contrast with no true effect clears ±1 SE "
        f"{record['rule']['null_separation']:.0%} of the time, nothing here corrects for the "
        f"{len(record['contrasts'])} contrasts, and they share runs, so they are not independent. Gates (§3.5): "
        "mean |gate| ± SE over seeds, then each seed's signed value (42–46); a gate starts at 0, and its sign "
        "means nothing.", "",
        f"**M8-4:** {record['m8_4']['line']}",
    ]
    hoc = record.get("post_hoc")
    without = without_by_contrast(record)
    if hoc is not None:
        lines += ["", f"**Without the {len(hoc['excluded']['record_ids'])} (post-hoc)** is the same Δ ± SE, with its "
                  f"95% CI, leaving out the {len(hoc['excluded']['record_ids'])} test patients whose record holds a known "
                  "bad value (last section). It was chosen after the results were seen, so it **changes no verdict**, "
                  "and a CI there that excludes 0 is not a claim."]
    extra = f" Without the {len(hoc['excluded']['record_ids'])} (post-hoc) |" if hoc is not None else ""
    for group, title in GROUPS.items():
        lines += ["", f"### {title}", "", f"| Contrast | Regime | Holdout | Δ ± SE | 95% CI |{extra} Verdict | Gates |",
                  "| --- | --- | --- | --- | --- |" + (" --- |" if hoc is not None else "") + " --- | --- |"]
        for row in (r for r in record["contrasts"] if r["group"] == group):
            d = row["delta"]
            cell = ""
            if hoc is not None:
                w = without[contrast_key(row)]
                cell = f" {signed(w['mean'])} ± {w['se']:.4f} ({signed(w['ci95'][0])} to {signed(w['ci95'][1])}) |"
            lines.append(f"| {row['name']} | {row['regime']} | {row['holdout']} | {signed(d['mean'])} ± {d['se']:.4f} | "
                         f"{signed(d['ci95'][0])} to {signed(d['ci95'][1])} |{cell} {row['verdict']} | "
                         f"{gate_text(row.get('gates'))} |")
    lines += ["", "## Gates", "", "| Regime | Variant | Gate | Mean \\|gate\\| ± SE | Signed, seeds 42–46 |",
              "| --- | --- | --- | --- | --- |"]
    for regime, by_variant in record["gates"].items():
        for variant, gates in by_variant.items():
            for field, g in gates.items():
                lines.append(f"| {regime} | {NAMES[variant]} | `{field}` | {ms(g['abs'], 3)} | "
                             + ", ".join(signed(v, 3) for v in g["signed"].values()) + " |")
    lines += ["", "Each gate is how far training scaled its encoder's output up from 0; no threshold reads one as "
              "used or ignored. Whether an extension helped is read from its Δ and CI, never from its gate."]
    lines += ["", "## Post-hoc: without the patients whose record holds a known bad value", ""]
    if hoc is None:
        lines.append("Not computed (`--skip-post-hoc`): it needs the sweep's saved medians in `outputs/sweep`.")
    else:
        counts = hoc["excluded"]["holdouts"]
        lines += [
            f"Chosen on Nov 25, after the results were seen, so it **changes no verdict**. It leaves out the "
            f"{len(hoc['excluded']['record_ids'])} test patients whose record holds a known bad value (README, "
            f"Dataset): {counts['random/0.1']['n_targets_excluded']} of {counts['random/0.1']['n_targets']:,} "
            f"targets under each holdout. Under the random holdout "
            f"{counts['random/0.1']['n_patients_showing_a_bad_value']} of them show one to the model and none is scored; "
            f"under the block holdout {counts['block/0.1']['n_patients_showing_a_bad_value']} show one and "
            f"{counts['block/0.1']['n_bad_values_scored']} are scored. The gate record makes the same cut "
            f"(`results/gate.json`). Its Δs are the \"Without the {len(hoc['excluded']['record_ids'])} (post-hoc)\" "
            "column of every contrast table above. From M8 on, the same cut is planned before the results, and "
            "reported beside every paired Δ (PROJECT_PLAN §3.5)."]
    return "\n".join(lines) + "\n"

def contrast_key(row):
    """Which contrast a row is, in the record's ``contrasts`` or in its post-hoc read."""
    return row["group"], row["regime"], row["holdout"], row["name"]

def without_by_contrast(record):
    """``{contrast: Δ without the excluded patients}``; refused unless the post-hoc read covers every contrast."""
    hoc = record.get("post_hoc")
    if hoc is None:
        return {}
    found = {contrast_key(row): row["delta_without"] for row in hoc["contrasts"]}
    absent = [row["name"] for row in record["contrasts"] if contrast_key(row) not in found]
    if absent or len(found) != len(record["contrasts"]):
        raise ValueError(f"the post-hoc read does not cover the same contrasts: missing {absent}")
    return found

def main(argv=None):
    parser = argparse.ArgumentParser(description="The headline table: mean ± SE and paired Δ ± SE (M7-2).")
    parser.add_argument("--store", type=Path, default=RESULTS / "sweep.json")
    parser.add_argument("--manifest", type=Path, default=RESULTS / "sweep_manifest.json")
    parser.add_argument("--baselines", type=Path, default=RESULTS / "baselines.json")
    parser.add_argument("--learned", type=Path, default=RESULTS / "learned_baseline.json")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--sweep-dir", type=Path, default=OUTPUTS / "sweep", help="the sweep's folder, for the post-hoc read")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--skip-post-hoc", action="store_true", help="write the record without the post-hoc read")
    parser.add_argument("--out", type=Path, default=RESULTS / "headline.json")
    parser.add_argument("--md", type=Path, default=RESULTS / "headline.md")
    args = parser.parse_args(argv)

    read = lambda path: json.loads(Path(path).read_text())
    protocol = yaml.safe_load(args.config.read_text())["protocol"]
    store = read(args.store)
    record = headline(store, read(args.manifest), read(args.baselines), read(args.learned), protocol,
                      list(sweep_plan(args.config)))
    record["post_hoc"] = None if args.skip_post_hoc else post_hoc(store, args.sweep_dir, load_dataset(args.dataset), protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    args.md.write_text(markdown(record))
    separated = [row["name"] + f" ({row['regime']}, {row['holdout']})" for row in record["contrasts"] if row["se_separated"]]
    print(f"wrote {args.out} and {args.md}: {len(record['contrasts'])} contrasts, {len(separated)} SE-separated")
    print(record["m8_4"]["line"])
    return record

if __name__ == "__main__":
    main()
