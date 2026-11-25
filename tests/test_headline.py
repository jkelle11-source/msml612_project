"""The headline table (M7-2): paired Δ, SE and CI worked out by hand, verdicts, gates, refusals, the
post-hoc read on a tiny dataset recomputed another way, and the committed record against its files."""
import copy
import json
import math

import numpy as np
import pytest
import yaml
from conftest import CONFIG, ROOT
from data.physionet import FEATURES
from diffusion.headline import (CONTRASTS, INTERACTION, NULL_SEPARATION, T975, baseline_rows, contrast_values,
                                contrasts, difference, gate_summary, headline, main, markdown, paired, post_hoc,
                                signed, verdict)
from diffusion.results import UNITS, sweep_plan
from diffusion.score import holdout_masker
from eval import masked_mae

PROTOCOL = CONFIG["protocol"]
REGIMES = list(sweep_plan())
SEEDS = ("42", "43", "44", "45", "46")
CELLS = {"random": ("base", "ext1", "ext2", "ext1+2"), "block-aware": ("base", "ext1")}
GATE_FIELDS = {"base": (), "ext1": ("mask_encoder_gate",), "ext2": ("time_encoder_gate",),
               "ext1+2": ("mask_encoder_gate", "time_encoder_gate")}
RESULTS = ROOT / "results"
REAL_SWEEP = ROOT / "outputs" / "sweep"

def make_store(mae=None, seeds=SEEDS):
    """A complete store whose every MAE is ``mae(regime, variant, holdout, seed)``; HR's is 0.1 lower and
    SysABP's 0.2 lower, so per-variable contrasts move with the pooled ones."""
    mae = mae or (lambda regime, variant, holdout, seed: 0.2)
    runs = {}
    for regime, variants in CELLS.items():
        for variant in variants:
            for seed in seeds:
                scores = {}
                for holdout in ("random/0.1", "block/0.1"):
                    m = mae(regime, variant, holdout, seed)
                    scores[holdout] = {"mae": m, "rmse": 2 * m, "gpu": "Some GPU",
                                       "per_variable_mae": {"HR": m - 0.1, "SysABP": m - 0.2}}
                runs.setdefault(regime, {}).setdefault(variant, {})[seed] = {
                    "run": f"{variant}_{regime}_seed{seed}", "wandb_run": f"e/p/{variant}-{regime}-{seed}",
                    "commit": "abc", "gpu": "Some GPU", "scores": scores}
    return {"tag": "sweep", "units": UNITS, "gpu": "Some GPU", "configs": {}, "runs": runs,
            "unscored": [], "missing": []}

def make_manifest(store, gate=lambda regime, variant, field, seed: 0.1):
    return {"runs": {regime: {variant: {seed: {"run": run["run"], "wandb_run": run["wandb_run"],
                                               "gates": {f: gate(regime, variant, f, seed) for f in GATE_FIELDS[variant]}}
                                        for seed, run in runs.items()}
                              for variant, runs in variants.items()}
                     for regime, variants in store["runs"].items()}}

def real(name):
    return json.loads((RESULTS / name).read_text())

# ---------------------------------------------------------------- paired Δ, SE, CI

def test_a_paired_delta_worked_out_by_hand():
    """Base's MAEs spread over seeds; Ext 1 is base plus 0.1, 0.3, 0.2, 0.4, 0.0 at seeds 42-46.

    Mean 0.2; deviations −0.1, 0.1, 0, 0.2, −0.2, so the squares sum to 0.10, the variance (n − 1) is
    0.025, SD 0.158114, SE 0.158114 / √5 = 0.0707107; CI 0.2 ± 2.776445 · 0.0707107 = 0.2 ± 0.196324.
    Two independent means would give an SE of about 0.7 here: base alone spreads from 1 to 5.
    """
    shift = dict(zip(SEEDS, (0.1, 0.3, 0.2, 0.4, 0.0)))
    store = make_store(lambda regime, variant, holdout, seed: float(SEEDS.index(seed) + 1)
                       + (shift[seed] if variant == "ext1" else 0.0))
    delta = paired(contrast_values(store, "random", "random/0.1", difference("ext1", "base")))
    assert delta["mean"] == pytest.approx(0.2)
    assert delta["se"] == pytest.approx(0.0707107, abs=1e-7)
    assert delta["ci95"] == pytest.approx([0.0036756, 0.3963244], abs=1e-7)
    assert delta["t"] == 2.776445 and delta["n_seeds"] == 5
    assert delta["per_seed"] == pytest.approx(shift)

def test_seeds_are_paired_by_name_not_by_order():
    store = make_store(lambda regime, variant, holdout, seed: int(seed) / 100 + (0.01 if variant == "ext1" else 0))
    shuffled = copy.deepcopy(store)
    runs = shuffled["runs"]["random"]["ext1"]
    shuffled["runs"]["random"]["ext1"] = {seed: runs[seed] for seed in reversed(SEEDS)}
    for found in (store, shuffled):
        delta = paired(contrast_values(found, "random", "random/0.1", difference("ext1", "base")))
        assert delta["mean"] == pytest.approx(0.01) and delta["se"] == pytest.approx(0.0, abs=1e-12)

def test_seeds_are_paired_even_where_the_variants_rank_their_seeds_differently():
    """Base 0.20-0.24 rising over seeds 42-46; Ext 1 is base plus 0.05, 0.03, 0.01, −0.01, −0.03, so it falls
    from 0.25 to 0.21. Paired: mean 0.01, deviations ±0.04, ±0.02, 0, squares sum 0.004, variance 0.001,
    SE √0.001 / √5 = 0.0141421. Pairing the smallest with the smallest instead would give 0.01 ± 0."""
    shift = dict(zip(SEEDS, (0.05, 0.03, 0.01, -0.01, -0.03)))
    store = make_store(lambda regime, variant, holdout, seed: 0.20 + SEEDS.index(seed) / 100
                       + (shift[seed] if variant == "ext1" else 0.0))
    delta = paired(contrast_values(store, "random", "random/0.1", difference("ext1", "base")))
    assert delta["mean"] == pytest.approx(0.01) and delta["se"] == pytest.approx(0.0141421, abs=1e-7)

def test_the_interaction_term_worked_out_by_hand():
    """MAEs base 0.30, Ext 1 0.28, Ext 2 0.29, Ext 1+2 0.25 at every seed but 46, where Ext 1+2 is 0.26:
    the term is 0.25 − 0.28 − 0.29 + 0.30 = −0.02 (−0.01 at seed 46)."""
    level = {"base": 0.30, "ext1": 0.28, "ext2": 0.29, "ext1+2": 0.25}
    store = make_store(lambda regime, variant, holdout, seed: level[variant] + (0.01 if (variant, seed) == ("ext1+2", "46") else 0))
    per_seed = contrast_values(store, "random", "random/0.1", INTERACTION)
    assert per_seed == pytest.approx({"42": -0.02, "43": -0.02, "44": -0.02, "45": -0.02, "46": -0.01})

def test_a_variant_missing_a_seed_is_refused():
    store = make_store()
    del store["runs"]["random"]["ext1"]["46"]
    with pytest.raises(ValueError, match="same seeds"):
        contrast_values(store, "random", "random/0.1", difference("ext1", "base"))

def test_no_contrast_crosses_regimes():
    """Raising every block-aware base MAE by 0.05 moves only the block-aware contrasts, each by −0.05."""
    before = contrasts(make_store())
    after = contrasts(make_store(lambda regime, variant, holdout, seed:
                                 0.25 if (regime, variant) == ("block-aware", "base") else 0.2))
    for old, new in zip(before, after):
        expected = -0.05 if new["regime"] == "block-aware" else 0.0
        assert new["delta"]["mean"] == pytest.approx(old["delta"]["mean"] + expected)

def test_t_values_match_an_independent_integration():
    """Each 97.5% point by bisection on Student's t CDF, integrated numerically from its density."""
    def cdf(x, df):
        u = np.linspace(0.0, x, 200001)
        density = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2)) * (1 + u ** 2 / df) ** (-(df + 1) / 2)
        return 0.5 + np.trapezoid(density, u)
    for df, t in T975.items():
        low, high = 0.0, 20.0
        for _ in range(60):
            mid = (low + high) / 2
            low, high = (mid, high) if cdf(mid, df) < 0.975 else (low, mid)
        assert t == pytest.approx(low, abs=2e-5), df
    assert NULL_SEPARATION == pytest.approx(2 * (1 - cdf(1.0, 4)), abs=5e-4)

def test_one_seed_is_refused():
    with pytest.raises(ValueError, match="at least 2 seeds"):
        paired({"42": 0.1})

# ---------------------------------------------------------------- verdicts

@pytest.mark.parametrize("delta, weights, group, expected", [
    ({"mean": 0.001, "se": 0.002, "ci95": [-0.004, 0.006]}, difference("ext1", "base"), "main",
     "No significant effect: Δ ± SE crosses 0. The 95% CI includes 0; it allows a gain of at most 0.0040."),
    ({"mean": 0.002, "se": 0.002, "ci95": [-0.003, 0.007]}, difference("ext1", "base"), "main",     # the bar touches 0
     "No significant effect: Δ ± SE crosses 0. The 95% CI includes 0; it allows a gain of at most 0.0030."),
    ({"mean": 0.003, "se": 0.002, "ci95": [-0.002, 0.008]}, difference("ext2", "base"), "main",
     "SE-separated: Ext 2 is worse than base, by 1.5 SE. The 95% CI includes 0, so this is not established "
     "at 95%; it allows a gain of at most 0.0020."),
    ({"mean": -0.004, "se": 0.001, "ci95": [-0.007, -0.001]}, difference("ext1+2", "ext1"), "interaction",
     "Exploratory. SE-separated: Ext 1+2 is better than Ext 1, by 4.0 SE. The 95% CI excludes 0."),
    ({"mean": -0.003, "se": 0.001, "ci95": [-0.006, 0.0002]}, INTERACTION, "interaction",
     "Exploratory. SE-separated: together the extensions do better than their separate effects add up to, by "
     "3.0 SE. The 95% CI includes 0, so this is not established at 95%; it allows a gain of at most 0.0060."),
])
def test_verdicts(delta, weights, group, expected):
    assert verdict(delta, weights, group) == expected

def test_every_contrast_is_reported_with_its_verdict():
    store = make_store(lambda regime, variant, holdout, seed: 0.2 + int(seed) * 1e-4 * (variant != "base"))
    found = contrasts(store, make_manifest(store))
    assert [(row["regime"], row["holdout"], row["weights"], row["variable"]) for row in found] == \
           [(regime, holdout, weights, variable) for _, regime, holdout, weights, variable in CONTRASTS]
    for row in found:
        assert row["se_separated"] == (abs(row["delta"]["mean"]) > row["delta"]["se"])
        assert row["verdict"].startswith("Exploratory. " if row["group"] == "interaction" else ("SE-separated"
                                         if row["se_separated"] else "No significant effect"))

def test_the_error_bar_decides_and_the_ci_does_not():
    """Ext 1 − base 0.03, −0.01, 0.02, 0, 0.01: mean 0.01, deviations ±0.02, ±0.01, 0, squares sum 0.001,
    variance 0.00025, SE 0.0070711, so Δ is 1.4 SE from 0 (SE-separated), while the CI, 0.01 ± 0.019632,
    includes 0."""
    shift = dict(zip(SEEDS, (0.03, -0.01, 0.02, 0.0, 0.01)))
    store = make_store(lambda regime, variant, holdout, seed: 0.2 + (shift[seed] if variant == "ext1" else 0.0))
    [row] = [r for r in contrasts(store) if (r["group"], r["name"]) == ("main", "Ext 1 − base")]
    assert row["delta"]["se"] == pytest.approx(0.0070711, abs=1e-7)
    assert row["se_separated"] and not row["ci_excludes_0"]
    assert row["verdict"] == ("SE-separated: Ext 1 is worse than base, by 1.4 SE. The 95% CI includes 0, so this "
                              "is not established at 95%; it allows a gain of at most 0.0096.")

# ---------------------------------------------------------------- gates

def test_gate_sizes_not_signs():
    """Gates −0.2, 0.2, −0.1, 0.1, 0.3: |gate| 0.2, 0.2, 0.1, 0.1, 0.3, mean 0.18; deviations 0.02, 0.02,
    −0.08, −0.08, 0.12, squares sum 0.028, variance 0.007, SE √0.007 / √5 = 0.0374166. The signed mean
    would be 0.06."""
    signed = dict(zip(SEEDS, (-0.2, 0.2, -0.1, 0.1, 0.3)))
    store = make_store()
    found = gate_summary(make_manifest(store, lambda regime, variant, field, seed: signed[seed]), "random", "ext1")
    assert found["mask_encoder_gate"]["abs"]["mean"] == pytest.approx(0.18)
    assert found["mask_encoder_gate"]["abs"]["se"] == pytest.approx(0.0374166, abs=1e-7)
    assert found["mask_encoder_gate"]["signed"] == signed

def test_gates_go_beside_each_extensions_delta_vs_base_only():
    store = make_store()
    for row in contrasts(store, make_manifest(store)):
        if row["weights"] == INTERACTION or row["weights"].get("base") != -1:
            expected = ()
        else:
            [variant] = [v for v, w in row["weights"].items() if w == 1]
            expected = GATE_FIELDS[variant]
        assert tuple(row["gates"] or ()) == expected, row["name"]

# ---------------------------------------------------------------- the record and its refusals

def tiny_baselines():
    test = {"n_patients": 800, "n_targets": 100, "mean_imputation": {"mae": 0.7, "rmse": 1.0},
            "forward_fill": {"mae": 0.45, "rmse": 0.8}}
    brits = {"n_patients": 800, "n_targets": 100, "brits": {"mae": {"mean": 0.27, "se": 0.001}, "rmse": {"mean": 0.6, "se": 0.01}}}
    random = {"eval_masker_seed": PROTOCOL["eval_masker_seed"]}
    block = {"block_eval_masker_seed": PROTOCOL["block_eval_masker_seed"], "block_window_hours": PROTOCOL["block_window_hours"]}
    baselines = {"random": {**random, "fractions": {"0.1": {"test": test}}}, "block": {**block, "fractions": {"0.1": {"test": test}}}}
    learned = {"seeds": [42, 43, 44, 45, 46], "mean_se": {"random": {**random, "fractions": {"0.1": {"test": brits}}},
                                                          "block": {**block, "fractions": {"0.1": {"test": brits}}}}}
    return baselines, learned

@pytest.mark.parametrize("separated", [True, False])
def test_m8_4_is_triggered_by_the_block_aware_block_contrast_alone(separated):
    """Ext 1's block-aware block Δ is 0.01 ± 0 (separated) or 0.01 at seeds 42-43 and −0.01 at 44-46."""
    def mae(regime, variant, holdout, seed):
        if (regime, variant, holdout) != ("block-aware", "ext1", "block/0.1"):
            return 0.2
        return 0.21 if separated or seed in ("42", "43") else 0.19
    store = make_store(mae)
    record = headline(store, make_manifest(store), *tiny_baselines(), PROTOCOL, REGIMES)
    assert record["m8_4"]["triggered"] is separated
    assert record["m8_4"]["line"].endswith("so M8-4's attribution checks run." if separated else
                                           "so M8-4's attribution checks do not run (§4.1: a null needs no control).")

@pytest.mark.parametrize("field", ["unscored", "missing"])
def test_an_incomplete_store_is_refused(field):
    store = make_store()
    store[field] = ["ext1_random_seed46: block/0.1"]
    with pytest.raises(ValueError, match="not complete"):
        headline(store, make_manifest(store), *tiny_baselines(), PROTOCOL, REGIMES)

def test_a_manifest_of_other_runs_is_refused():
    store = make_store()
    manifest = make_manifest(store)
    manifest["runs"]["random"]["ext2"]["44"]["wandb_run"] = "e/p/someone-else"
    with pytest.raises(ValueError, match="ext2_random_seed44"):
        headline(store, manifest, *tiny_baselines(), PROTOCOL, REGIMES)

@pytest.mark.parametrize("which", [0, 1])
def test_baselines_scored_on_another_holdout_are_refused(which):
    records = tiny_baselines()
    target = records[0] if which == 0 else records[1]["mean_se"]
    target["block"]["block_eval_masker_seed"] += 1
    with pytest.raises(ValueError, match="not the frozen"):
        baseline_rows(*records, PROTOCOL)

# ---------------------------------------------------------------- the post-hoc read

def tiny_dataset():
    """Six patients charted at random, test split 1-4. Raw values sit in range (pH 7.4 ± 0.1); patient 2
    (a test patient) has a pH of 735 at hour 5 and patient 0 (not a test patient) a Temp of 0."""
    rng = np.random.default_rng(0)
    features = list(FEATURES)
    mean = np.zeros(len(features))
    std = np.ones(len(features))
    for name, (m, s) in {"pH": (7.4, 0.1), "Temp": (37.0, 1.0), "K": (4.0, 1.0), "PaO2": (100.0, 1.0)}.items():
        mean[features.index(name)], std[features.index(name)] = m, s
    mask = (rng.random((6, 48, 35)) < 0.5).astype(np.float32)
    values = (rng.uniform(-1, 1, (6, 48, 35)) * mask).astype(np.float32)
    ph, temp = features.index("pH"), features.index("Temp")
    mask[2, 5, ph] = mask[0, 7, temp] = 1
    values[2, 5, ph], values[0, 7, temp] = (735 - 7.4) / 0.1, (0 - 37.0) / 1.0
    return {"values": values, "mask": mask, "mean": mean, "std": std, "features": np.array(features),
            "record_ids": np.arange(1000, 1006), "test_idx": np.array([1, 2, 3, 4])}

def tiny_sweep(folder, dataset, seeds=("42", "43")):
    """Medians for every cell (truth plus noise that differs by variant and seed), and the store of their MAEs."""
    idx = dataset["test_idx"]
    values, mask, ids = dataset["values"][idx], dataset["mask"][idx], dataset["record_ids"][idx]
    targets = {h: holdout_masker(PROTOCOL, (48, 35), h.split("/")[0], 0.1)[0](mask, ids)[1] for h in ("random/0.1", "block/0.1")}
    medians = {}
    def mae(regime, variant, holdout, seed):
        rng = np.random.default_rng([int(seed), list(CELLS["random"]).index(variant), list(CELLS).index(regime),
                                     ("random/0.1", "block/0.1").index(holdout)])
        median = (values + rng.normal(0, 0.3, values.shape)).astype(np.float32)
        medians[(regime, variant, holdout, seed)] = median
        return masked_mae(median, values, targets[holdout])
    store = make_store(mae, seeds)
    for (regime, variant, holdout, seed), median in medians.items():
        run_dir = folder / f"{variant}_{regime}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / f"median_test_{holdout.replace('/', '_')}.npy", median)
    return store, medians

def test_the_post_hoc_read_recomputed_by_leaving_the_patient_out(tmp_path):
    """Only test patient 2 is left out (patient 0's bad value is outside the test split). Each holdout
    draws a generator per patient from its record id, so scoring patients 1, 3 and 4 alone gives the
    targets they have in the full split: their MAE there is the post-hoc MAE."""
    dataset = tiny_dataset()
    store, medians = tiny_sweep(tmp_path, dataset)
    found = post_hoc(store, tmp_path, dataset, PROTOCOL)
    assert found["excluded"]["record_ids"] == [1002]
    keep = np.array([0, 2, 3])                     # test patients 1, 3, 4, by position in the test split
    idx = dataset["test_idx"][keep]
    clean = copy.deepcopy(store)
    features = list(dataset["features"])
    for holdout in ("random/0.1", "block/0.1"):
        _, target = holdout_masker(PROTOCOL, (48, 35), holdout.split("/")[0], 0.1)[0](dataset["mask"][idx], dataset["record_ids"][idx])
        for (regime, variant, h, seed), median in medians.items():
            if h == holdout:
                expected = masked_mae(median[keep], dataset["values"][idx], target)
                assert found["mae_without"][regime][variant][seed][holdout] == pytest.approx(expected, rel=1e-12)
                scores = clean["runs"][regime][variant][seed]["scores"][holdout]
                scores["mae"] = expected
                for name in ("HR", "SysABP"):         # one variable's targets alone, from the kept patients
                    j = features.index(name)
                    scores["per_variable_mae"][name] = masked_mae(median[keep][..., j], dataset["values"][idx][..., j],
                                                                  target[..., j])
    assert [(row["regime"], row["holdout"], row["variable"]) for row in found["contrasts"]] == \
           [(regime, holdout, variable) for _, regime, holdout, _, variable in CONTRASTS]
    for row, (_, regime, holdout, weights, variable) in zip(found["contrasts"], CONTRASTS):
        assert row["delta"] == paired(contrast_values(store, regime, holdout, weights, variable))
        expected = paired(contrast_values(clean, regime, holdout, weights, variable))
        assert row["delta_without"]["mean"] == pytest.approx(expected["mean"], abs=1e-12)
        assert row["delta_without"]["se"] == pytest.approx(expected["se"], abs=1e-12)
        assert row["se_separated_without"] == (abs(expected["mean"]) > expected["se"])

def test_each_contrast_row_shows_its_own_delta_without_the_patients(tmp_path):
    """The post-hoc Δ sits on its own contrast's row, beside the Δ over all patients; the verdict is the same."""
    dataset = tiny_dataset()
    store, _ = tiny_sweep(tmp_path, dataset)
    record = headline(store, make_manifest(store), *tiny_baselines(), PROTOCOL, REGIMES)
    record["post_hoc"] = post_hoc(store, tmp_path, dataset, PROTOCOL)
    md = markdown(record)
    rows = [line for line in md.splitlines() if " | random/0.1 | " in line or " | block/0.1 | " in line]
    assert len(rows) == len(CONTRASTS)
    for line, row, hoc in zip(rows, record["contrasts"], record["post_hoc"]["contrasts"]):
        d, w = row["delta"], hoc["delta_without"]
        assert line.startswith(f"| {row['name']} | {row['regime']} | {row['holdout']} | ")
        assert (f"| {signed(d['mean'])} ± {d['se']:.4f} | {signed(d['ci95'][0])} to {signed(d['ci95'][1])} | "
                f"{signed(w['mean'])} ± {w['se']:.4f} ({signed(w['ci95'][0])} to {signed(w['ci95'][1])}) | "
                f"{row['verdict']} |") in line
    record["post_hoc"]["contrasts"].pop(3)
    with pytest.raises(ValueError, match="does not cover the same contrasts"):
        markdown(record)

def test_a_median_that_is_not_the_scored_one_is_refused(tmp_path):
    dataset = tiny_dataset()
    store, _ = tiny_sweep(tmp_path, dataset)
    path = tmp_path / "ext2_random_seed43" / "median_test_block_0.1.npy"
    np.save(path, np.load(path) + 1e-3)
    with pytest.raises(ValueError, match="ext2_random_seed43's median_test_block_0.1.npy"):
        post_hoc(store, tmp_path, dataset, PROTOCOL)

# ---------------------------------------------------------------- the committed record

def fresh_record():
    record = headline(real("sweep.json"), real("sweep_manifest.json"), real("baselines.json"),
                      real("learned_baseline.json"), PROTOCOL, REGIMES)
    return json.loads(json.dumps(record, ensure_ascii=False))

def test_the_committed_record_matches_its_files():
    committed = real("headline.json")
    assert {key: value for key, value in committed.items() if key != "post_hoc"} == fresh_record()
    assert (RESULTS / "headline.md").read_text() == markdown(committed)

def test_the_committed_record_labels_every_null_and_carries_every_ci():
    committed, md = real("headline.json"), (RESULTS / "headline.md").read_text()
    assert len(committed["contrasts"]) == len(CONTRASTS)
    for row in committed["contrasts"]:
        if not row["se_separated"]:
            assert "No significant effect" in row["verdict"] and "a gain of at most" in row["verdict"]
        assert row["verdict"] in md and len(row["delta"]["ci95"]) == 2
    assert committed["m8_4"]["line"] in md

@pytest.mark.skipif(not (REAL_SWEEP / "base_random_seed42" / "median_test_random_0.1.npy").exists(),
                    reason="the sweep's folder is not here (outputs/sweep)")
def test_the_committed_post_hoc_read_matches_the_sweeps_medians():
    from data.physionet import load_dataset
    from diffusion.run import DATASET_PATH
    found = post_hoc(real("sweep.json"), REAL_SWEEP, load_dataset(DATASET_PATH), PROTOCOL)
    assert real("headline.json")["post_hoc"] == json.loads(json.dumps(found, ensure_ascii=False))

def test_main_without_the_post_hoc_read(tmp_path, capsys):
    out, md = tmp_path / "headline.json", tmp_path / "headline.md"
    main(["--skip-post-hoc", "--out", str(out), "--md", str(md)])
    written = json.loads(out.read_text())
    assert written["post_hoc"] is None and "Not computed (`--skip-post-hoc`)" in md.read_text()
    assert "| Without the" not in md.read_text()          # no post-hoc column without the post-hoc read
    assert md.read_text() == markdown(written)
    assert "M8-4's attribution checks" in capsys.readouterr().out
