import copy
import hashlib
import json

import numpy as np
import pytest
from conftest import CONFIG
from data.physionet import save_dataset
from diffusion.run import run, with_variant
from diffusion.trajectories import (
    VARIABLES,
    band_coverage,
    checked_runs,
    lag_changes,
    linear_interpolation,
    load_weights,
    main,
    pick_patients,
    raw,
    residual_correlation,
)
from eval import EvalMasker, EvalSampler

SEEDS = (42, 43)

def quiet(line):
    pass

def manifest_for(*run_dirs, sha=None):
    """A manifest naming ``run_dirs`` as ext1+2's runs, as results/sweep_manifest.json does."""
    return {"runs": {"random": {"ext1+2": {run_dir.name[-2:]: {"run": run_dir.name, "weights": {
        "path": f"outputs/sweep/{run_dir.name}/model.pt",
        "sha256": sha or hashlib.sha256((run_dir / "model.pt").read_bytes()).hexdigest()}}
        for run_dir in run_dirs}}}}

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
    """Two tiny finished ext1+2 runs in a sweep folder, each with a protocol median beside it, and a manifest."""
    root = tmp_path_factory.mktemp("sweep")
    config = with_variant(CONFIG, "ext1+2")
    config["diffusion"].update(channels=16, layers=1, num_steps=5)
    config["train"].update(epochs=1, batch_size=8)
    config["validation"].update(every_n_epochs=1, n_samples=1, batch_size=8)
    run_dirs = [root / "sweep" / f"ext1+2_random_seed{seed}" for seed in SEEDS]
    for seed, run_dir in zip(SEEDS, run_dirs):
        run(config, charted, seed, "cpu", run_dir, log=quiet, wandb_mode="disabled")
        median = np.random.default_rng(seed).normal(size=(len(charted["test_idx"]), 48, 35)).astype(np.float32)
        np.save(run_dir / "median_test_random_0.1.npy", median)
    (root / "manifest.json").write_text(json.dumps(manifest_for(*run_dirs)))
    save_dataset(charted, root / "data.npz")
    return root

def holdout(charted):
    """The test split's values, mask and frozen conditioning and target masks."""
    idx = charted["test_idx"]
    values, mask = charted["values"][idx], charted["mask"][idx]
    cond, target = EvalMasker.from_config()(mask, charted["record_ids"][idx])
    return values, mask, cond, target

def columns_of(charted):
    return [list(charted["features"]).index(v) for v in VARIABLES]

def sampled(sweep, charted, seed, record_ids, draws=4):
    """The draws those patients get from ``seed``'s weights with the frozen noise, in clinical units."""
    values, mask, cond, _ = holdout(charted)
    ids = list(charted["record_ids"][charted["test_idx"]])
    rows = [ids.index(r) for r in record_ids]
    protocol = CONFIG["protocol"]
    sampler = EvalSampler(protocol["eval_sampling_seed"], draws, protocol["eval_batch_size"])
    z = sampler(load_weights(sweep / "sweep" / f"ext1+2_random_seed{seed}", "cpu"), values[rows], mask[rows], cond[rows])
    columns = columns_of(charted)
    return raw(z[..., columns], charted["mean"][columns].astype(np.float64), charted["std"][columns].astype(np.float64))

# --- the weights are the sweep's -----------------------------------------------------------

def test_weights_matching_the_manifest_are_taken(sweep):
    manifest = json.loads((sweep / "manifest.json").read_text())
    assert checked_runs(manifest, "random", "ext1+2", sweep / "sweep") == {
        seed: sweep / "sweep" / f"ext1+2_random_seed{seed}" for seed in SEEDS}

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

# --- the helpers, worked by hand --------------------------------------------------------------

def test_linear_interpolation_by_hand():
    values = np.array([[[0.0, 5.0], [99.0, 5.0], [4.0, 5.0], [99.0, 5.0], [99.0, 5.0]],
                       [[99.0, 1.0], [99.0, 2.0], [3.0, 3.0], [99.0, 4.0], [99.0, 5.0]]])
    cond = np.zeros(values.shape)
    cond[0, [0, 2], 0] = 1                                          # 0 to 4 over two hours, then flat
    cond[1, 2, 0] = 1                                               # one visible value: flat both ways
    cond[1, [1, 4], 1] = 1                                          # 2 at hour 1 to 5 at hour 4
    filled = linear_interpolation(values, cond)
    assert filled[0, :, 0].tolist() == [0, 2, 4, 4, 4]
    assert filled[0, :, 1].tolist() == [0, 0, 0, 0, 0]              # never visible: the training mean
    assert filled[1, :, 0].tolist() == [3, 3, 3, 3, 3]
    assert filled[1, :, 1].tolist() == [2, 2, 3, 4, 5]

def test_band_coverage_by_hand():
    draws = np.arange(101.0)[:, None, None, None] * np.ones((101, 1, 5, 1))     # the 5-95% band is [5, 95]
    truth = np.array([[[4.0], [5.0], [95.0], [96.0], [50.0]]])
    held_out = np.array([[[True], [True], [True], [True], [False]]])            # 50 is inside but not scored
    assert band_coverage(draws, truth, held_out, ("HR",)) == {
        "HR": {"inside": 2, "held_out": 4, "fraction": 0.5},
        "all": {"inside": 2, "held_out": 4, "fraction": 0.5}}

def test_lag_changes_by_hand():
    draws = np.array([[[9.0, 1.0, 3.0, 6.0, 9.0]], [[9.0, 1.0, 1.0, 1.0, 9.0]]])    # lower median: 9 1 1 1 9
    truth = np.array([[10.0, 12.0, 0.0, 15.0, 11.0]])
    visible = np.array([[True, False, False, False, True]])
    observed = np.array([[True, True, False, True, True]])
    found = lag_changes(draws, truth, visible, observed, lags=(1, 2, 3))
    # lag 1: hidden pairs (1,2), (2,3) move 2, 3 and 0, 0; charted pairs (0,1), (3,4) move 2 and 4
    assert found["1"] == {"draws": 1.25, "median": 0.0, "real": 3.0, "hidden_pairs": 2, "charted_pairs": 2}
    # lag 2: hidden pair (1,3) moves 5 and 0; charted pair (1,3) moves 3
    assert found["2"] == {"draws": 2.5, "median": 0.0, "real": 3.0, "hidden_pairs": 1, "charted_pairs": 1}
    # lag 3: no hidden pair; charted pairs (0,3), (1,4) move 5 and 1
    assert found["3"] == {"draws": None, "median": None, "real": 3.0, "hidden_pairs": 0, "charted_pairs": 2}

def test_residual_correlation_by_hand():
    hidden = np.zeros((1, 3), dtype=bool)
    same_way = np.array([[[-1.0, -2.0, -4.0]], [[0.0, 0.0, 0.0]], [[1.0, 2.0, 4.0]]])     # next = 2 x this
    opposite = np.array([[[-1.0, 1.0, -1.0]], [[0.0, 0.0, 0.0]], [[1.0, -1.0, 1.0]]])     # next = -this
    assert residual_correlation(same_way, hidden) == 1.0
    assert residual_correlation(opposite, hidden) == -1.0
    # A median that moves from hour to hour is taken out first: only the departures from it count.
    assert residual_correlation(same_way + np.array([10.0, 20.0, 30.0]), hidden) == 1.0
    assert residual_correlation(opposite + np.array([10.0, 40.0, 5.0]), hidden) == -1.0
    assert residual_correlation(same_way, np.ones((1, 3), dtype=bool)) is None

# --- end to end ------------------------------------------------------------------------------

def written_for(sweep, name, *extra):
    out = sweep / name
    main(["--sweep", str(sweep / "sweep"), "--manifest", str(sweep / "manifest.json"), "--dataset",
          str(sweep / "data.npz"), "--device", "cpu", "--draws", "4", "--out", str(out), *extra])
    return out

@pytest.fixture(scope="module")
def written(sweep):
    out = written_for(sweep, "trajectories")
    return dict(np.load(out / "trajectories.npz")), json.loads((out / "trajectories.json").read_text())

@pytest.fixture(scope="module")
def written_seed43(sweep):
    out = written_for(sweep, "trajectories43", "--seed", "43")
    return dict(np.load(out / "trajectories.npz")), json.loads((out / "trajectories.json").read_text())

def test_five_patients_three_variables_in_clinical_units(written, charted):
    arrays, record = written
    assert arrays["draws"].shape == (4, 5, 48, 3) and arrays["truth"].shape == (5, 48, 3)
    assert arrays["variables"].tolist() == list(VARIABLES) == record["variables"]
    assert record["variant"] == "ext1+2" and record["seed"] == 42 and record["draws"] == 4
    assert record["variant_note"] == "lowest mean test MAE, not separated from base"
    idx, columns = charted["test_idx"], columns_of(charted)
    rows = [list(charted["record_ids"][idx]).index(r) for r in arrays["record_ids"]]
    z = charted["values"][idx][rows][..., columns]
    assert np.allclose(arrays["truth"], z * charted["std"][columns] + charted["mean"][columns])

def test_visible_and_held_out_are_the_frozen_holdouts(written, charted):
    arrays, _ = written
    idx, columns = charted["test_idx"], columns_of(charted)
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

@pytest.mark.parametrize("seed", SEEDS)
def test_the_draws_are_that_seeds_weights_with_the_frozen_noise(seed, written, written_seed43, sweep, charted):
    arrays, _ = written if seed == 42 else written_seed43
    expected = sampled(sweep, charted, seed, arrays["record_ids"].tolist())
    other = sampled(sweep, charted, 85 - seed, arrays["record_ids"].tolist())
    hidden = ~arrays["visible"]
    assert np.allclose(arrays["draws"][:, hidden], expected[:, hidden], rtol=1e-5, atol=1e-3)
    assert not np.allclose(arrays["draws"][:, hidden], other[:, hidden], rtol=1e-5, atol=1e-3)

def test_the_median_is_the_lower_middle_draw(written):
    arrays, _ = written                                             # 4 draws: the 2nd smallest, as upstream scores
    assert np.allclose(arrays["median"], np.sort(arrays["draws"], axis=0)[1], rtol=1e-5, atol=1e-3)

def test_the_baseline_curves_read_only_the_visible_values(written, charted):
    arrays, _ = written
    mean = charted["mean"][columns_of(charted)]
    for i, j in np.ndindex(5, 3):
        seen = np.flatnonzero(arrays["visible"][i, :, j])
        last = np.searchsorted(seen, arrays["hours"], side="right") - 1         # before the first: the first
        assert np.allclose(arrays["forward_fill"][i, :, j], arrays["truth"][i, seen[np.maximum(last, 0)], j])
        assert np.allclose(arrays["mean_imputation"][i, :, j], mean[j])

def test_the_panel_scores_are_each_panels_held_out_errors(written):
    arrays, record = written
    assert set(record["panel_mae"]) == {"median_of_draws", "protocol_median_seed42", "forward_fill", "mean_imputation"}
    curves = {"median_of_draws": arrays["median"], "forward_fill": arrays["forward_fill"],
              "mean_imputation": arrays["mean_imputation"], "protocol_median_seed42": arrays["seed_medians"][0]}
    for name, curve in curves.items():
        for i, j in np.ndindex(5, 3):
            held = arrays["held_out"][i, :, j]
            expected = np.abs(curve[i, held, j] - arrays["truth"][i, held, j]).mean()
            assert record["panel_mae"][name][i][j] == pytest.approx(expected, abs=6e-4)

def test_the_band_counts_are_the_held_out_values_inside_it(written):
    arrays, record = written
    low, high = np.quantile(arrays["draws"], [0.05, 0.95], axis=0)
    for j, variable in enumerate(VARIABLES):
        held = arrays["held_out"][..., j]
        truth = arrays["truth"][..., j]
        inside = ((truth >= low[..., j]) & (truth <= high[..., j]) & held).sum()
        assert record["held_out_inside_90pct_band"][variable] == [int(inside), int(held.sum())]

def test_the_test_scores_are_held_out_errors_in_clinical_units(written, sweep, charted):
    _, record = written
    assert set(record["test_mae"]) == {"seed42", "seed43", "forward_fill", "mean_imputation", "linear_interpolation"}
    values, _, cond, target = holdout(charted)
    columns, std = columns_of(charted), charted["std"]

    def score(guess, c):
        hidden = target[:, :, c] == 1
        return np.abs(np.asarray(guess, dtype=np.float64) - values[:, :, c].astype(np.float64))[hidden].mean() * std[c]

    def carried(c, how):
        out = np.zeros(values.shape[:2])
        for n in range(len(values)):
            seen = np.flatnonzero(cond[n, :, c] == 1)
            if how == "carry":
                last = np.searchsorted(seen, np.arange(48), side="right") - 1
                out[n] = values[n, seen[np.maximum(last, 0)], c]
            else:
                out[n] = np.interp(np.arange(48), seen, values[n, seen, c].astype(np.float64))
        return out

    for variable, c in zip(VARIABLES, columns):
        for seed in SEEDS:
            median = np.load(sweep / "sweep" / f"ext1+2_random_seed{seed}" / "median_test_random_0.1.npy")
            assert record["test_mae"][f"seed{seed}"][variable] == pytest.approx(score(median[:, :, c], c), abs=6e-4)
        assert record["test_mae"]["mean_imputation"][variable] == pytest.approx(score(np.zeros(values.shape[:2]), c), abs=6e-4)
        assert record["test_mae"]["forward_fill"][variable] == pytest.approx(score(carried(c, "carry"), c), abs=6e-4)
        assert record["test_mae"]["linear_interpolation"][variable] == pytest.approx(score(carried(c, "line"), c), abs=6e-4)
        assert record["test_held_out"][variable] == int((target[:, :, c] == 1).sum())

def test_the_figure_draws_each_method_and_marks_the_held_out_values(written):
    pytest.importorskip("matplotlib")                               # the optional ``plots`` dependency group
    from matplotlib import pyplot as plt

    from diffusion.trajectories_plot import FORWARD_FILL, MEAN, figure
    arrays, record = written
    fig = figure(arrays, record)
    assert fig.axes[0].get_title() == "Heart rate (bpm)" and len(fig.axes) == 15
    assert record["variant_note"] in fig.get_suptitle()
    for (i, j), ax in zip(np.ndindex(5, 3), fig.axes):
        curves = {line.get_color(): line for line in ax.lines if line.get_marker() in ("None", None, "")}
        assert np.array_equal(curves[FORWARD_FILL].get_ydata(), arrays["forward_fill"][i, :, j])
        assert np.array_equal(curves[MEAN].get_ydata(), arrays["mean_imputation"][i, :, j])
        markers = [line for line in ax.lines if line.get_marker() == "o"]
        rings = [line for line in markers if line.get_markerfacecolor() == "white"]
        dots = [line for line in markers if line.get_markerfacecolor() != "white"]
        assert len(rings) == len(dots) == 1
        assert np.array_equal(rings[0].get_xdata(), arrays["hours"][arrays["held_out"][i, :, j]])
        assert np.array_equal(dots[0].get_xdata(), arrays["hours"][arrays["visible"][i, :, j]])
        assert not rings[0].get_clip_on() and not dots[0].get_clip_on()         # whole markers at hours 0 and 47
    plt.close(fig)

# --- the draw check -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def draw_checked(sweep):
    out = written_for(sweep, "draw_check", "--draw-check")
    return json.loads((out / "draw_check.json").read_text()), out

def test_the_draw_check_scores_every_eligible_patient(draw_checked, sweep, charted):
    record, out = draw_checked
    assert [p.name for p in out.iterdir()] == ["draw_check.json"]   # the five patients' data stay as they are
    values, mask, cond, target = holdout(charted)
    columns = columns_of(charted)
    held, seen = target[:, :, columns].sum(axis=1), cond[:, :, columns].sum(axis=1)
    rows = np.flatnonzero((held >= 1).all(axis=1) & (seen >= 2).all(axis=1))
    assert record["patients"]["n"] == len(rows) >= 5
    ids = charted["record_ids"][charted["test_idx"]][rows].tolist()
    draws = sampled(sweep, charted, 42, ids)
    truth = raw(values[rows][..., columns], charted["mean"][columns].astype(np.float64),
                charted["std"][columns].astype(np.float64))
    held_out = target[rows][..., columns] == 1
    assert record["band_5_95"] == band_coverage(draws, truth, held_out, VARIABLES)
    for j, variable in enumerate(VARIABLES):
        assert record["mean_abs_change"][variable] == lag_changes(
            draws[..., j], truth[..., j], cond[rows][..., columns[j]] == 1, mask[rows][..., columns[j]] == 1)
        assert record["draw_residual_lag1_correlation"][variable] == residual_correlation(
            draws[..., j], cond[rows][..., columns[j]] == 1)
