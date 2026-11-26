"""M8-1's ablation table: each value against the sweep's, from real tiny runs, and every refusal."""
import hashlib
import json
import re
import shutil

import numpy as np
import pytest
import yaml
from data.physionet import save_dataset
from diffusion import ablation_table
from diffusion.ablation import SEEDS, STUDIES, ablate
from diffusion.ablation_table import MOVES, main, markdown, table
from diffusion.headline import signed
from diffusion.score import holdout_masker
from diffusion.sweep import cells, sweep
from test_sweep import quick_config

T4 = "Tesla T4"

def quiet(line):
    pass

def first_left_out(dataset):
    """Test patient 0 stands in for one with a known bad value: every fake patient's raw values are out of range."""
    return np.arange(len(dataset["test_idx"])) == 0

# The regime each variant's encoder-depth runs train in (diffusion.ablation.STUDIES): Ext 1 block-aware, its primary one
REGIME = {"ext1+2": "random", "ext1": "block-aware"}

def name(variant, seed):
    return f"{variant}_{REGIME[variant]}_seed{seed}"

@pytest.fixture(scope="module")
def trained(tmp_path_factory, small_dataset):
    """Tiny runs on a "T4": the sweep's Ext 1+2 (random) and Ext 1 (block-aware) at encoder depth 2, the
    encoder-layers study at depths 1 and 3, and the rerun check, at seeds 42-44."""
    root = tmp_path_factory.mktemp("trained")
    config = quick_config()
    config["protocol"]["paired_gpu"] = T4
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    go = {"paired": False, "wandb_mode": "disabled", "config_path": root / "config.yaml", "log": quiet}
    with pytest.MonkeyPatch.context() as patch:
        for module in ("diffusion.run", "diffusion.score"):
            patch.setattr(f"{module}.gpu_name", lambda device: T4)
        sweep(config, small_dataset, root / "sweep", "sweep", "cpu",
              cells(config, ["random"], ["ext1+2"], list(SEEDS)) + cells(config, ["block-aware"], ["ext1"], list(SEEDS)),
              ["random", "block"], **go)
        ablate(config, small_dataset, root / "ablation", "encoder_layers", "ablation", "cpu", **go)
        sweep(config, small_dataset, root / "ablation" / "rerun-check", "rerun-check", "cpu", [("random", "ext1+2", 42)],
              ["random", "block"], **go)
    return root, config

@pytest.fixture
def tree(trained, tmp_path, monkeypatch):
    """A copy of the trained folders to break, with test patient 0 as the one bad-value patient."""
    shutil.copytree(trained[0], tmp_path / "copy")
    monkeypatch.setattr(ablation_table, "bad_value_patients", first_left_out)
    return tmp_path / "copy", trained[1]

def manifest_of(sweep_dir):
    """The sweep's run manifest as far as the table reads it: the folder's commit and each run's final weights."""
    runs = {}
    for run_dir in sorted(path for path in sweep_dir.glob("*_seed*") if path.is_dir()):
        cell, _, seed = run_dir.name.rpartition("_seed")
        variant, _, regime = cell.partition("_")
        runs.setdefault(regime, {}).setdefault(variant, {})[seed] = {
            "weights": {"sha256": hashlib.sha256((run_dir / "model.pt").read_bytes()).hexdigest()}}
    return {"commit": json.loads((sweep_dir / "sweep.json").read_text())["commit"], "runs": runs}

def tabled(root, config, dataset, manifest=None, studies=("encoder_layers",)):
    return table([root / "ablation"], root / "sweep", root / "ablation" / "rerun-check", dataset, config,
                 manifest or manifest_of(root / "sweep"), list(studies))

def scored(run_dir, holdout):
    return json.loads((run_dir / f"scores_test_{holdout.replace('/', '_')}.json").read_text())["mae"]

# --- the numbers ------------------------------------------------------------------------

def test_each_value_is_set_against_the_sweeps_seed_by_seed(tree, small_dataset):
    root, config = tree
    found = tabled(root, config, small_dataset)["studies"]["encoder_layers"]
    assert (found["setting"], found["sweep_value"]) == ("model.mask_encoder_layers", 2)
    assert [(row["regime"], row["variant"], row["holdout"], row["value"]) for row in found["contrasts"]] == [
        (REGIME[variant], variant, holdout, depth) for variant in ("ext1+2", "ext1") for holdout in ("random/0.1", "block/0.1")
        for depth in (1, 3)]
    assert found["runs"] == {"random": ["ext1+2"], "block-aware": ["ext1"]}
    for row in found["contrasts"]:
        expected = {str(seed): scored(root / "ablation" / f"encoder_layers_{row['value']}" / name(row["variant"], seed),
                                      row["holdout"])
                    - scored(root / "sweep" / name(row["variant"], seed), row["holdout"]) for seed in SEEDS}
        assert row["delta"]["per_seed"] == expected
        assert row["delta"]["mean"] == pytest.approx(np.mean(list(expected.values())))
        assert row["delta"]["se"] == pytest.approx(np.std(list(expected.values()), ddof=1) / np.sqrt(len(SEEDS)))
        assert row["se_separated"] == (abs(row["delta"]["mean"]) > row["delta"]["se"])
        assert re.match(r"(No significant effect|SE-separated: encoder layers \d is (better|worse) than encoder layers 2)",
                        row["verdict"])
        cell = found["cells"][row["regime"]][row["variant"]][row["holdout"]][str(row["value"])]
        assert cell["mae"]["mean"] == pytest.approx(np.mean(list(cell["per_seed"].values())))

def test_the_read_without_the_bad_value_patients_leaves_them_out(tree, small_dataset):
    # recomputed with numpy from the saved medians, patient 0's targets dropped
    root, config = tree
    record = tabled(root, config, small_dataset)
    idx = small_dataset["test_idx"]
    values, mask, record_ids = (small_dataset[key][idx] for key in ("values", "mask", "record_ids"))
    assert record["excluded"]["record_ids"] == [int(record_ids[0])]
    for row in record["studies"]["encoder_layers"]["contrasts"]:
        holdout, fraction = row["holdout"].split("/")
        target = holdout_masker(config["protocol"], (48, 35), holdout, float(fraction))[0](mask, record_ids)[1].astype(bool)
        target[0] = False

        def mae(run_dir):
            median = np.load(run_dir / f"median_test_{holdout}_{fraction}.npy")
            return float(np.abs(median.astype(np.float64) - values)[target].mean())

        expected = {str(seed): mae(root / "ablation" / f"encoder_layers_{row['value']}" / name(row["variant"], seed))
                    - mae(root / "sweep" / name(row["variant"], seed)) for seed in SEEDS}
        assert row["delta_without"]["per_seed"] == pytest.approx(expected, rel=1e-5)
        assert row["delta_without"]["per_seed"] != row["delta"]["per_seed"]

def test_the_sweeps_runs_may_come_from_a_commit_of_their_own(tree, small_dataset):
    # as the real sweep's are: an earlier commit, whose saved configs have no encoder-depth key (depth 2)
    root, config = tree
    edit_json(root / "sweep" / "sweep.json", lambda claim: claim.update(commit="1" * 40))
    for path in (root / "sweep").glob("*/config.yaml"):
        edit_yaml(path, lambda saved: saved["model"].pop("mask_encoder_layers"))
    record = tabled(root, config, small_dataset)
    assert record["commits"]["sweep"] == "1" * 40 and record["rerun_check"]["reproduces"]

def test_a_study_scored_on_the_random_holdout_alone_is_set_against_the_sweeps_older_configs(tree, small_dataset,
                                                                                            monkeypatch):
    # The T study at the tiny model's T and a little more; the sweep's configs saved before the depth key
    root, config = tree
    monkeypatch.setitem(STUDIES["steps"], "values", (10, 12))
    for module in ("diffusion.run", "diffusion.score"):
        monkeypatch.setattr(f"{module}.gpu_name", lambda device: T4)
    ablate(config, small_dataset, root / "ablation", "steps", "ablation", "cpu", paired=False, wandb_mode="disabled",
           config_path=root / "config.yaml", log=quiet)
    for path in (root / "sweep").glob("*/config.yaml"):
        edit_yaml(path, lambda saved: saved["model"].pop("mask_encoder_layers"))
    record = tabled(root, config, small_dataset, studies=["steps"])
    [row] = record["studies"]["steps"]["contrasts"]
    assert (row["regime"], row["variant"], row["holdout"], row["value"], row["against"]) == (
        "random", "ext1+2", "random/0.1", 12, 10)
    # How many of each chain's steps sit near pure noise (none at these tiny Ts), and the T table says so
    noise = record["studies"]["steps"]["near_pure_noise"]
    assert noise["steps"] == {"10": quad_near_pure_noise(config, 10), "12": quad_near_pure_noise(config, 12)}
    assert (noise["schedule"], noise["beta"], noise["signal_below"]) == ("quad", [1.0e-4, 0.5], 0.05)
    assert ("quad β range, 0.0001 to 0.5, so a longer chain spends more of its steps near pure noise, where the data's "
            "signal √ᾱ_t is under 0.05: 0 of 10, 0 of 12 steps.") in markdown(record)

def quad_near_pure_noise(config, steps):
    """Steps whose sqrt(alpha_bar_t) is under 0.05, counted from the quad schedule's formula, not the model."""
    diffusion = config["diffusion"]
    beta = np.linspace(diffusion["beta_start"] ** 0.5, diffusion["beta_end"] ** 0.5, steps) ** 2
    return int((np.sqrt(np.cumprod(1 - beta)) < 0.05).sum())

def test_the_t_tables_steps_near_pure_noise_are_the_reused_models():
    # At the study's values, with upstream's beta range: 8 of 50, 32 of 100 and 91 of 200 (README, The ablation)
    config = quick_config()
    assert [ablation_table.near_pure_noise(config, steps) for steps in (50, 100, 200)] == [
        quad_near_pure_noise(config, steps) for steps in (50, 100, 200)] == [8, 32, 91]

def test_each_study_changes_only_the_pairing_hashes_it_names(trained, small_dataset, tmp_path, monkeypatch):
    root, config = trained

    def moved(study_dir, run="ext1+2_random_seed42"):
        sweeps = json.loads((root / "sweep" / run / "summary.json").read_text())["pairing"]
        pairing = json.loads((study_dir / run / "summary.json").read_text())["pairing"]
        return {key for key in pairing.keys() | sweeps.keys() if pairing.get(key) != sweeps.get(key)}

    assert moved(root / "ablation" / "encoder_layers_3") == set(MOVES["encoder_layers"])
    assert moved(root / "ablation" / "encoder_layers_1", "ext1_block-aware_seed42") == set(MOVES["encoder_layers"])
    monkeypatch.setitem(STUDIES["d_model"], "values", (16, 32))        # the tiny model's width, and twice it
    monkeypatch.setitem(STUDIES["steps"], "values", (10, 12))          # its T, and a little more
    for module in ("diffusion.run", "diffusion.score"):
        monkeypatch.setattr(f"{module}.gpu_name", lambda device: T4)
    for study in ("d_model", "steps"):
        ablate(config, small_dataset, tmp_path, study, "ablation", "cpu", seeds=[42], paired=False, wandb_mode="disabled",
               config_path=root / "config.yaml", log=quiet)
    assert moved(tmp_path / "d_model_32") == set(MOVES["d_model"])
    assert moved(tmp_path / "steps_12") == set(MOVES["steps"])

# --- what keeps a run out ---------------------------------------------------------------

def edit_json(path, change):
    found = json.loads(path.read_text())
    change(found)
    path.write_text(json.dumps(found))

def edit_yaml(path, change):
    found = yaml.safe_load(path.read_text())
    change(found)
    path.write_text(yaml.safe_dump(found))

def run_of(depth, variant, seed):
    return f"ablation/encoder_layers_{depth}/{name(variant, seed)}"

RERUN = "ablation/rerun-check"
BROKEN = {
    "a run unpaired": (
        lambda root: edit_json(root / run_of(1, "ext1", 43) / "summary.json",
                               lambda summary: summary["pairing"].update(data_order="other")),
        "encoder_layers_1/ext1_block-aware_seed43 is not paired with the sweep's run: its ['data_order'] differ"),
    "another learning rate": (
        lambda root: edit_yaml(root / run_of(3, "ext1+2", 44) / "config.yaml", lambda config: config["train"].update(lr=1.0e-4)),
        "encoder_layers_3/ext1+2_random_seed44: its config differs from the sweep's run in "
        "['model.mask_encoder_layers', 'train.lr']"),
    "a depth run trained in the other regime": (
        lambda root: edit_yaml(root / run_of(3, "ext1", 42) / "config.yaml",
                               lambda config: config["train"].update(regime="random")),
        "ext1_block-aware_seed42: config's regime is 'random', not 'block-aware'"),
    "trained on another GPU": (
        lambda root: edit_json(root / run_of(3, "ext1", 42) / "summary.json", lambda summary: summary.update(gpu="NVIDIA L4")),
        "ext1_block-aware_seed42: training on NVIDIA L4, not Tesla T4"),
    "not finished": (lambda root: (root / run_of(1, "ext1+2", 42) / "summary.json").unlink(),
                     "ext1+2_random_seed42: no summary.json (the run has not finished)"),
    "a value not trained": (lambda root: shutil.rmtree(root / "ablation" / "encoder_layers_3"), "no encoder_layers_3 folder"),
    "a scoring done another way": (
        lambda root: edit_json(root / run_of(1, "ext1", 44) / "scores_test_block_0.1.json",
                               lambda scores: scores["sampler"].update(eval_n_samples=7)),
        "ext1_block-aware_seed44: no block/0.1 scoring of its final weights done the frozen way"),
    "a median that was not scored": (
        lambda root: np.save(root / run_of(3, "ext1", 43) / "median_test_random_0.1.npy",
                             np.zeros_like(np.load(root / run_of(3, "ext1", 43) / "median_test_random_0.1.npy"))),
        "ext1_block-aware_seed43: its random/0.1 median does not give its scores' MAE"),
    "a sweep folder holding other weights": (
        lambda root: (shutil.rmtree(root / "sweep" / "ext1+2_random_seed43"),
                      shutil.copytree(root / run_of(3, "ext1+2", 43), root / "sweep" / "ext1+2_random_seed43")),
        "ext1+2_random_seed43 is not the sweep's run its manifest records: its model.pt is other weights"),
    "a block-aware sweep run holding other weights": (
        lambda root: (shutil.rmtree(root / "sweep" / "ext1_block-aware_seed44"),
                      shutil.copytree(root / run_of(1, "ext1", 44), root / "sweep" / "ext1_block-aware_seed44")),
        "ext1_block-aware_seed44 is not the sweep's run its manifest records: its model.pt is other weights"),
    "a sweep folder from another commit than its manifest's": (
        lambda root: edit_json(root / "sweep" / "sweep.json", lambda claim: claim.update(commit="4" * 40)),
        "was started from commit 4444444444444444444444444444444444444444, and the sweep's manifest records"),
    "a study folder without its sweep.json": (lambda root: (root / "ablation" / "encoder_layers_1" / "sweep.json").unlink(),
                                              "encoder_layers_1: no readable sweep.json"),
    "a study folder from another commit": (
        lambda root: edit_json(root / "ablation" / "encoder_layers_3" / "sweep.json", lambda claim: claim.update(commit="2" * 40)),
        "the ablation's folders were started with more than one tag or commit"),
    "no rerun check": (lambda root: shutil.rmtree(root / RERUN), "no finished rerun check"),
    "a rerun check from another commit": (
        lambda root: edit_json(root / RERUN / "sweep.json", lambda claim: claim.update(commit="3" * 40)),
        "the rerun check was trained from commit 333"),
    "a rerun check that does not reproduce": (
        lambda root: (root / RERUN / "ext1+2_random_seed42" / "history.jsonl").open("a").write("\n"),
        "the rerun check does not reproduce the sweep's ext1+2_random_seed42: it differs in history.jsonl"),
}

@pytest.mark.parametrize("brake, refusal", BROKEN.values(), ids=BROKEN.keys())
def test_what_is_not_paired_or_not_frozen_is_refused(tree, small_dataset, brake, refusal):
    root, config = tree
    manifest = manifest_of(root / "sweep")              # the sweep as M7-1 recorded it, before anything breaks
    brake(root)
    with pytest.raises(ValueError, match=re.escape(refusal)) as refused:
        tabled(root, config, small_dataset, manifest)
    assert "not the ablation's None" not in str(refused.value)          # the rerun check is not blamed for it

# --- the command line -------------------------------------------------------------------

def test_the_command_line_writes_the_record_and_its_tables(tree, small_dataset, tmp_path):
    root, config = tree
    save_dataset(small_dataset, tmp_path / "data.npz")
    (tmp_path / "sweep_manifest.json").write_text(json.dumps(manifest_of(root / "sweep")))
    record = main(["--ablation", str(root / "ablation"), "--sweep-dir", str(root / "sweep"), "--studies", "encoder_layers",
                   "--sweep-manifest", str(tmp_path / "sweep_manifest.json"), "--config", str(root / "config.yaml"),
                   "--dataset", str(tmp_path / "data.npz"), "--out", str(tmp_path / "ablation.json"),
                   "--md", str(tmp_path / "new folder" / "ablation.md")])
    assert json.loads((tmp_path / "ablation.json").read_text()) == json.loads(json.dumps(record))
    tables = (tmp_path / "new folder" / "ablation.md").read_text()
    assert tables.count("| ext1+2 | random | block/0.1 |") == 3 and tables.count("| ext1 | block-aware | random/0.1 |") == 3
    assert tables.count("2 (the sweep's)") == 4 and "encoder depth is studied block-aware" in tables
    runs = record["runs"]["encoder_layers"]
    assert runs["2"]["block-aware"]["ext1"]["42"] == {"run": "ext1_block-aware_seed42", "manifest": "results/sweep_manifest.json"}
    assert runs["3"]["block-aware"]["ext1"]["42"]["config"]["model"]["mask_encoder_layers"] == 3
    assert runs["3"]["block-aware"]["ext1"]["42"]["config"]["train"]["regime"] == "block-aware"
    assert runs["3"]["random"]["ext1+2"]["42"]["rerun"].startswith(f"git checkout {record['commits']['ablation']} && ")

def test_the_read_without_the_bad_value_patients_is_labelled_where_it_separates(tree, small_dataset):
    # Planned before any M8 run (PROJECT_PLAN §3.5), so where it clears ±1 SE the table says so
    root, config = tree
    record = tabled(root, config, small_dataset)
    rows = record["studies"]["encoder_layers"]["contrasts"]
    for separated, label in ((True, "SE-separated"), (False, "no significant effect")):
        for row in rows:
            row["se_separated_without"] = separated
        tables = markdown(record)
        for row in rows:
            without = row["delta_without"]
            assert (f"{signed(without['mean'])} ± {without['se']:.4f} ({signed(without['ci95'][0])} to "
                    f"{signed(without['ci95'][1])}), {label} | {row['verdict']} |") in tables
