import copy
import hashlib
import json

import numpy as np
import pytest
import yaml
from conftest import CONFIG
from data.physionet import FEATURES, save_dataset
from diffusion.gate import bad_value_check, departures, gate, main, mean_se
from eval import EvalMasker, masked_mae, masked_rmse

SAMPLER = {"eval_sampling_seed": 2021, "eval_n_samples": 100, "eval_batch_size": 16}

def saved_config():
    """The repo's config as a run saves it: without ``tracking``, validated every 25 epochs."""
    config = {k: copy.deepcopy(v) for k, v in CONFIG.items() if k != "tracking"}
    config["validation"]["every_n_epochs"] = 25
    return config

def scored_run(tmp_path, dataset, seed, miss, config=None, **score_overrides):
    """A finished, scored base run whose median misses every test target by ``miss``, built by hand."""
    run_dir = tmp_path / f"base_random_seed{seed}"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config or saved_config(), sort_keys=False))
    weights = f"weights of seed {seed}".encode()
    (run_dir / "model.pt").write_bytes(weights)
    (run_dir / "summary.json").write_text(json.dumps({"final_val_mae": 0.25, "health": {"healthy": True}}))
    (run_dir / "curve.json").write_text(json.dumps([{"seconds": 30.0}, {"seconds": 31.5, "val_seconds": 300.0}]))

    idx = dataset["test_idx"]
    values = dataset["values"][idx]
    _, target = EvalMasker.from_config()(dataset["mask"][idx], dataset["record_ids"][idx])
    median = (values + miss * target).astype(np.float32)
    np.save(run_dir / "median_test_random_0.1.npy", median)
    scores = {"run": run_dir.name, "variant": "base", "seed": seed, "mae": masked_mae(median, values, target),
              "rmse": masked_rmse(median, values, target), "n_targets": int(target.sum()),
              "holdout_settings": {"eval_masker_seed": 2012}, "sampler": dict(SAMPLER), "device": "mps",
              "seconds": 12000.0, "model_sha256": hashlib.sha256(weights).hexdigest(),
              "wandb_run": f"DATA612/msml612-csdi/run{seed}"} | score_overrides
    (run_dir / "scores_test_random_0.1.json").write_text(json.dumps(scores))
    return run_dir

def with_threshold(tmp_path, threshold):
    config = copy.deepcopy(CONFIG)
    config["protocol"]["gate_mae_at_10pct"] = threshold
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path

# --- mean ± SE --------------------------------------------------------------------------

def test_mean_and_se_by_hand():
    # mean 0.24; deviations -0.01, 0, 0.01, so the sample sd is sqrt(0.0002 / 2) = 0.01 and SE 0.01 / sqrt(3)
    assert mean_se([0.23, 0.24, 0.25]) == {"mean": 0.24, "se": 0.005774}
    assert mean_se([0.3]) == {"mean": 0.3, "se": None}

# --- the status -------------------------------------------------------------------------

def test_three_seeds_at_most_the_threshold_pass(small_dataset, tmp_path):
    runs = [scored_run(tmp_path, small_dataset, seed, miss) for seed, miss in ((42, 0.20), (43, 0.22), (44, 0.24))]
    record = gate(runs, small_dataset)
    assert record["status"] == "PASS" and record["seeds"] == [42, 43, 44]
    maes = [record["per_seed"][s]["mae"] for s in ("42", "43", "44")]
    assert maes == pytest.approx([0.20, 0.22, 0.24], abs=1e-5)
    # mean 0.22; deviations -0.02, 0, 0.02, so the sample sd is 0.02 and the SE 0.02 / sqrt(3)
    assert record["mean_se"]["mae"]["mean"] == pytest.approx(0.22, abs=1e-5)
    assert record["mean_se"]["mae"]["se"] == pytest.approx(0.02 / np.sqrt(3), abs=1e-5)

def test_the_threshold_itself_passes_and_anything_above_fails(small_dataset, tmp_path):
    runs = [scored_run(tmp_path, small_dataset, seed, miss) for seed, miss in ((42, 0.20), (43, 0.22), (44, 0.24))]
    mean = float(np.mean([json.loads((r / "scores_test_random_0.1.json").read_text())["mae"] for r in runs]))
    assert gate(runs, small_dataset, config_path=with_threshold(tmp_path, mean))["status"] == "PASS"
    assert gate(runs, small_dataset, config_path=with_threshold(tmp_path, mean - 1e-9))["status"] == "FAIL"

def test_fewer_seeds_than_the_gate_needs_are_incomplete_not_a_pass(small_dataset, tmp_path):
    runs = [scored_run(tmp_path, small_dataset, seed, 0.1) for seed in (42, 43)]
    assert gate(runs, small_dataset)["status"] == "INCOMPLETE"

def test_the_record_carries_what_the_report_needs(small_dataset, tmp_path):
    runs = [scored_run(tmp_path, small_dataset, seed, 0.25) for seed in (42, 43, 44)]
    record = gate(runs, small_dataset)
    assert record["gate"] == {"mae_at_most": 0.24, "min_seeds": 3, "split": "test", "holdout": "random", "fraction": 0.1}
    assert record["sampler"] == SAMPLER and record["eval_masker_seed"] == 2012 and record["split_seed"] == 612
    row = record["per_seed"]["43"]
    assert row["wandb_run"] == "DATA612/msml612-csdi/run43" and row["tracked_val_mae"] == 0.25
    assert (row["train_seconds"], row["val_seconds"], row["score_seconds"]) == (61.5, 300.0, 12000.0)
    assert record["config_departures"] == {}            # validate-every 25 is tracking only

# --- refused runs -----------------------------------------------------------------------

@pytest.mark.parametrize("overrides", [
    {"variant": "ext1"},
    {"sampler": SAMPLER | {"eval_n_samples": 5}},       # the tracked validation's draws
    {"sampler": SAMPLER | {"eval_batch_size": 64}},
    {"holdout_settings": {"eval_masker_seed": 1}},
    {"model_sha256": "0" * 64},                         # retrained since it was scored
    {"mae": 0.1},                                       # the median no longer gives the recorded MAE
])
def test_a_run_that_is_not_a_frozen_base_scoring_is_refused(small_dataset, tmp_path, overrides):
    runs = [scored_run(tmp_path, small_dataset, 42, 0.2, **overrides),
            *(scored_run(tmp_path, small_dataset, seed, 0.2) for seed in (43, 44))]
    with pytest.raises(ValueError):
        gate(runs, small_dataset)

def test_a_seed_twice_is_refused(small_dataset, tmp_path):
    runs = [scored_run(tmp_path, small_dataset, 42, 0.2)]
    (tmp_path / "copy").mkdir()
    runs.append(scored_run(tmp_path / "copy", small_dataset, 42, 0.2))
    runs.append(scored_run(tmp_path, small_dataset, 43, 0.2))
    with pytest.raises(ValueError, match="twice"):
        gate(runs, small_dataset)

def test_runs_trained_differently_are_refused(small_dataset, tmp_path):
    other = saved_config()
    other["train"]["lr"] = 5.0e-4
    runs = [scored_run(tmp_path, small_dataset, 42, 0.2), scored_run(tmp_path, small_dataset, 43, 0.2, config=other),
            scored_run(tmp_path, small_dataset, 44, 0.2)]
    with pytest.raises(ValueError, match="differs"):
        gate(runs, small_dataset)

def test_a_departure_from_the_repos_config_is_listed(small_dataset, tmp_path):
    tuned = saved_config()
    tuned["train"]["epochs"] = 300
    runs = [scored_run(tmp_path, small_dataset, seed, 0.2, config=tuned) for seed in (42, 43, 44)]
    assert gate(runs, small_dataset)["config_departures"] == {"train.epochs": [300, 200]}

def test_departures_skip_the_tracking_only_blocks_and_see_added_keys():
    config, reference = saved_config(), copy.deepcopy(CONFIG)
    assert departures(config, reference) == {}
    config["model"]["ema"] = 0.999
    config["diffusion"]["num_steps"] = 100
    assert departures(config, reference) == {"diffusion.num_steps": [100, 50], "model.ema": [0.999, None]}

# --- the PaCO2/PaO2 check ---------------------------------------------------------------

def two_patients():
    """Two patients in raw units (mean 0, std 1, so raw = standardized), worked by hand.

    Patient 0 (record 7) shows a pH of 735 at hour 3, where its PaCO2 (40) and PaO2 (80) are
    held out, and its HR at hour 5 is held out. Patient 1 (record 8) is clean but for a pH of 1
    that is itself held out, which it does not *show*; its PaCO2 at hour 3 is held out.
    """
    T, D = 48, len(FEATURES)
    j = {name: FEATURES.index(name) for name in ("pH", "PaCO2", "PaO2", "HR")}
    values, mask, target = (np.zeros((2, T, D), np.float32) for _ in range(3))
    for n, t, name, value, held_out in [(0, 3, "pH", 735.0, False), (0, 3, "PaCO2", 40.0, True),
                                        (0, 3, "PaO2", 80.0, True), (0, 5, "HR", 90.0, True),
                                        (1, 2, "pH", 1.0, True), (1, 3, "PaCO2", 35.0, True),
                                        (1, 3, "pH", 7.4, False)]:
        values[n, t, j[name]], mask[n, t, j[name]], target[n, t, j[name]] = value, 1, held_out
    dataset = {"values": values, "mask": mask, "mean": np.zeros(D, np.float32), "std": np.ones(D, np.float32),
               "features": np.array(FEATURES), "record_ids": np.array([7, 8])}
    median = values.copy()
    for n, t, name, miss in [(0, 3, "PaCO2", 60.0), (0, 3, "PaO2", -2.0), (0, 5, "HR", 1.0),
                             (1, 2, "pH", 6.0), (1, 3, "PaCO2", 0.5)]:
        median[n, t, j[name]] += miss
    return dataset, median, mask - target, target

def test_the_bad_value_check_by_hand():
    dataset, median, cond, target = two_patients()
    check = bad_value_check(dataset, np.arange(2), median, cond, target)
    assert check["n_patients_showing_a_bad_value"] == 1                 # patient 1's bad pH is hidden, not shown
    assert check["mae"] == pytest.approx((60 + 2 + 1 + 6 + 0.5) / 5)
    assert check["mae_without_patients_showing_a_bad_value"] == pytest.approx((6 + 0.5) / 2)
    assert check["PaCO2"] == {"n_targets": 2, "mae": 30.25, "mae_raw": 30.25,
                              "largest_miss": {"record_id": 7, "hour": 3, "feature": "PaCO2", "truth_raw": 40.0,
                                               "guess_raw": 100.0, "miss": 60.0, "patient_shows_a_bad_value": True}}
    assert check["PaO2"]["largest_miss"]["guess_raw"] == 78.0 and check["PaO2"]["mae"] == 2.0
    assert check["next_to_a_visible_bad_ph"] == [
        {"record_id": 7, "hour": 3, "feature": "PaCO2", "truth_raw": 40.0, "guess_raw": 100.0, "miss": 60.0,
         "visible_ph_raw": 735.0},
        {"record_id": 7, "hour": 3, "feature": "PaO2", "truth_raw": 80.0, "guess_raw": 78.0, "miss": 2.0,
         "visible_ph_raw": 735.0},
    ]

def test_raw_units_undo_the_scaling():
    dataset, median, cond, target = two_patients()
    dataset["std"] = dataset["std"] * 10                # one standardized unit is now 10 raw units
    dataset["mean"] = dataset["mean"] + 5
    check = bad_value_check(dataset, np.arange(2), median, cond, target)
    assert check["PaCO2"]["mae"] == 30.25 and check["PaCO2"]["mae_raw"] == 302.5
    assert check["PaCO2"]["largest_miss"]["truth_raw"] == 405.0

def test_a_feature_with_nothing_held_out_is_reported_empty():
    dataset, median, cond, target = two_patients()
    target[..., FEATURES.index("PaO2")] = 0
    assert bad_value_check(dataset, np.arange(2), median, cond, target)["PaO2"] == {"n_targets": 0}

# --- the command line -------------------------------------------------------------------

def test_the_command_line_writes_the_record(small_dataset, tmp_path, capsys):
    runs = [scored_run(tmp_path, small_dataset, seed, 0.2) for seed in (42, 43, 44)]
    save_dataset(small_dataset, tmp_path / "data.npz")
    main([*map(str, runs), "--dataset", str(tmp_path / "data.npz"), "--out", str(tmp_path / "gate.json")])
    record = json.loads((tmp_path / "gate.json").read_text())
    assert record["status"] == "PASS" and set(record["per_seed"]) == {"42", "43", "44"}
    assert "PASS" in capsys.readouterr().out
