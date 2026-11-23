import copy
import hashlib
import json

import numpy as np
import pytest
import yaml
from conftest import CONFIG
from data.physionet import save_dataset
from diffusion.run import run, with_variant
from diffusion.trajectories import VARIABLES, checked_runs, main, pick_patients, raw
from eval import EvalMasker

def quiet(line):
    pass

def manifest_for(run_dir, sha=None):
    """A manifest naming ``run_dir`` as ext1+2's seed-42 run, as results/sweep_manifest.json does."""
    sha = sha or hashlib.sha256((run_dir / "model.pt").read_bytes()).hexdigest()
    return {"runs": {"random": {"ext1+2": {"42": {"run": run_dir.name, "weights": {
        "path": f"outputs/sweep/{run_dir.name}/model.pt", "sha256": sha}}}}}}

@pytest.fixture(scope="module")
def charted(small_dataset):
    """The fake patients with HR, sodium and GCS charted every hour, so every test patient has some
    of each held out (with 8 values a variable, the fake patients have none)."""
    dataset = copy.deepcopy(small_dataset)
    columns = [list(dataset["features"]).index(v) for v in VARIABLES]
    rng = np.random.default_rng(1)
    dataset["mask"][:, :, columns] = 1
    dataset["values"][:, :, columns] = rng.normal(size=dataset["values"][:, :, columns].shape).astype(np.float32)
    return dataset

@pytest.fixture(scope="module")
def sweep(charted, tmp_path_factory):
    """A tiny finished ext1+2 run in a sweep folder, with a protocol median beside it, and a manifest."""
    root = tmp_path_factory.mktemp("sweep")
    config = with_variant(CONFIG, "ext1+2")
    config["diffusion"].update(channels=16, layers=1, num_steps=5)
    config["train"].update(epochs=1, batch_size=8)
    config["validation"].update(every_n_epochs=1, n_samples=1, batch_size=8)
    run_dir = root / "sweep" / "ext1+2_random_seed42"
    run(config, charted, 42, "cpu", run_dir, log=quiet, wandb_mode="disabled")
    median = np.random.default_rng(2).normal(size=(len(charted["test_idx"]), 48, 35)).astype(np.float32)
    np.save(run_dir / "median_test_random_0.1.npy", median)
    (root / "manifest.json").write_text(json.dumps(manifest_for(run_dir)))
    save_dataset(charted, root / "data.npz")
    return root

# --- the weights are the sweep's -----------------------------------------------------------

def test_weights_matching_the_manifest_are_taken(sweep):
    manifest = json.loads((sweep / "manifest.json").read_text())
    assert checked_runs(manifest, "random", "ext1+2", sweep / "sweep") == {42: sweep / "sweep" / "ext1+2_random_seed42"}

def test_weights_that_differ_or_are_missing_are_refused(sweep, tmp_path):
    run_dir = sweep / "sweep" / "ext1+2_random_seed42"
    with pytest.raises(ValueError, match="is not the weights the manifest records"):
        checked_runs(manifest_for(run_dir, sha="0" * 64), "random", "ext1+2", sweep / "sweep")
    with pytest.raises(ValueError, match="is missing: copy ext1\\+2_random_seed42"):
        checked_runs(manifest_for(run_dir), "random", "ext1+2", tmp_path)

# --- the patients ---------------------------------------------------------------------------

def test_patients_are_evenly_spaced_among_the_eligible():
    cond, target = np.zeros((20, 48, 3)), np.zeros((20, 48, 3))
    eligible = [1, 3, 4, 8, 9, 12, 15, 19, 2]
    for i in eligible:
        cond[i, :2] = 1
        target[i, 5] = 1
    cond[2, :2, 1] = 0                                              # 2: sodium not visible twice
    assert pick_patients(cond, target, [0, 1, 2]).tolist() == [1, 4, 9, 12, 19]
    target[0, 5, :2] = cond[0, :2] = 1                              # 0: GCS never held out
    assert pick_patients(cond, target, [0, 1, 2]).tolist() == [1, 4, 9, 12, 19]
    with pytest.raises(ValueError, match="only 8 patients"):
        pick_patients(cond, target, [0, 1, 2], n=9)

def test_raw_undoes_the_z_scoring():
    mean, std = np.array([80.0, 139.0, 11.0]), np.array([15.0, 4.0, 4.0])
    assert raw(np.array([[0.0, 1.0, -2.0]]), mean, std).tolist() == [[80.0, 143.0, 3.0]]

# --- end to end ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def written(sweep):
    out = sweep / "trajectories"
    main(["--sweep", str(sweep / "sweep"), "--manifest", str(sweep / "manifest.json"), "--dataset",
          str(sweep / "data.npz"), "--device", "cpu", "--draws", "4", "--out", str(out)])
    return dict(np.load(out / "trajectories.npz")), json.loads((out / "trajectories.json").read_text())

def test_five_patients_three_variables_in_clinical_units(written, charted):
    arrays, record = written
    assert arrays["draws"].shape == (4, 5, 48, 3) and arrays["truth"].shape == (5, 48, 3)
    assert arrays["variables"].tolist() == list(VARIABLES) == record["variables"]
    assert record["variant"] == "ext1+2" and record["seed"] == 42 and record["draws"] == 4
    idx, columns = charted["test_idx"], [list(charted["features"]).index(v) for v in VARIABLES]
    rows = [list(charted["record_ids"][idx]).index(r) for r in arrays["record_ids"]]
    z = charted["values"][idx][rows][..., columns]
    assert np.allclose(arrays["truth"], z * charted["std"][columns] + charted["mean"][columns])

def test_visible_and_held_out_are_the_frozen_holdouts(written, charted):
    arrays, _ = written
    idx, columns = charted["test_idx"], [list(charted["features"]).index(v) for v in VARIABLES]
    cond, target = EvalMasker.from_config()(charted["mask"][idx], charted["record_ids"][idx])
    rows = [list(charted["record_ids"][idx]).index(r) for r in arrays["record_ids"]]
    assert np.array_equal(arrays["visible"], cond[rows][..., columns] == 1)
    assert np.array_equal(arrays["held_out"], target[rows][..., columns] == 1)
    assert arrays["held_out"].any(axis=1).all()                     # every panel has something to compare

def test_trajectories_keep_the_visible_values(written):
    # The sampler's output at a visible value is not an imputation; every curve keeps the value.
    arrays, _ = written
    visible = arrays["visible"]
    for name in ("median", "forward_fill"):
        assert np.allclose(arrays[name][visible], arrays["truth"][visible])
    assert all(np.allclose(draw[visible], arrays["truth"][visible], atol=1e-3) for draw in arrays["draws"])
    assert all(np.allclose(m[visible], arrays["truth"][visible]) for m in arrays["seed_medians"])
    assert not np.allclose(arrays["median"][arrays["held_out"]], arrays["truth"][arrays["held_out"]])

def test_the_record_scores_each_method_on_the_held_out_values(written):
    _, record = written
    assert set(record["panel_mae"]) == {"median_of_draws", "protocol_median_seed42", "forward_fill", "mean_imputation"}
    assert all(np.array(v).shape == (5, 3) for v in record["panel_mae"].values())
    assert set(record["test_mae"]) == {"seed42", "forward_fill", "mean_imputation"}
    assert all(set(v) == set(VARIABLES) for v in record["test_mae"].values())

def test_the_figure_is_drawn(written, tmp_path):
    pytest.importorskip("matplotlib")                               # not in this repo's environment
    from diffusion.trajectories_plot import figure
    arrays, record = written
    fig = figure(arrays, record)
    assert fig.axes[0].get_title() == "Heart rate (bpm)" and len(fig.axes) == 15
