import copy
import json

import numpy as np
import pytest
import yaml
from conftest import CONFIG
from data.physionet import save_dataset
from diffusion.run import health_report, main, run, validate, variant_name
from diffusion.train import train
from eval import EvalMasker

def tiny_config(epochs=2):
    """A model small enough to train and validate in a few seconds on the 40 fake patients."""
    config = copy.deepcopy(CONFIG)
    config["diffusion"].update(channels=16, layers=1, num_steps=10)
    config["train"].update(epochs=epochs, batch_size=8)
    config["validation"].update(every_n_epochs=1, n_samples=2, batch_size=8)
    return config

def history(loss, grad_norm):
    return [{"epoch": 0, "loss": l, "grad_norm": g, "lr": 1.0e-3} for l, g in zip(loss, grad_norm)]

# --- the 500-step health report ---------------------------------------------------------

def test_a_steadily_learning_run_is_healthy():
    report = health_report(history(np.linspace(1.0, 0.3, 500), np.full(500, 2.0)))
    assert report["healthy"] and report["learning"] and report["spikes"] == []
    assert report["n_steps"] == 500

def test_only_the_first_n_steps_are_checked():
    steps = history(np.linspace(1.0, 0.3, 600), [2.0] * 500 + [float("nan")] * 100)
    assert health_report(steps)["finite"] and not health_report(steps, n_steps=600)["finite"]

def test_an_isolated_spike_is_listed_but_not_failed():
    norm = np.full(500, 2.0)
    norm[100] = 80.0                                        # one bad value made a target
    report = health_report(history(np.linspace(1.0, 0.3, 500), norm))
    assert report["healthy"]
    assert report["spikes"] == [{"step": 100, "loss": pytest.approx(np.linspace(1.0, 0.3, 500)[100]), "grad_norm": 80.0}]
    assert report["grad_norm_max_step"] == 100

def test_a_gradient_that_runs_away_and_stays_up_has_exploded():
    norm = np.concatenate([np.full(400, 2.0), np.full(100, 500.0)])
    report = health_report(history(np.linspace(1.0, 0.3, 500), norm))
    assert report["exploded"] and not report["healthy"]

@pytest.mark.parametrize("loss, failed", [
    (np.full(500, 0.8), "learning"),                        # flat: not learning
    (np.geomspace(1.0, 1.0e-6, 500), "collapsed"),          # toward zero: targets leaked or skipped
])
def test_flat_or_collapsed_loss_is_unhealthy(loss, failed):
    report = health_report(history(loss, np.full(500, 2.0)))
    assert not report["healthy"]
    assert report[failed] is (failed == "collapsed")

def test_a_nan_is_unhealthy():
    loss = np.linspace(1.0, 0.3, 500)
    loss[10] = np.nan
    assert not health_report(history(loss, np.full(500, 2.0)))["finite"]

# --- validation scores EvalMasker's held-out targets, as the baselines are --------------

def test_validation_scores_exactly_the_eval_masker_targets(small_dataset):
    masker = EvalMasker(CONFIG["protocol"]["eval_masker_seed"], 0.1)
    idx = small_dataset["val_idx"]
    values = small_dataset["values"][idx]
    expected_cond, target = masker(small_dataset["mask"][idx], small_dataset["record_ids"][idx])
    seen = {}

    def sampler(model, values_in, mask_in, cond_in):
        seen["cond"] = cond_in
        guess = values_in + 2.0 * target                    # wrong by exactly 2 on the targets only
        return np.stack([guess, guess, guess - 1.0])        # median of the 3 draws is `guess`

    scores = validate(None, small_dataset, sampler, masker)
    assert np.array_equal(seen["cond"], expected_cond)
    assert ((seen["cond"] == 1) & (target == 1)).sum() == 0   # the model never sees a scored value
    assert scores == {"mae": pytest.approx(2.0), "rmse": pytest.approx(2.0)}

# --- one run: outputs, reproduction from its own saved config ---------------------------

def test_a_run_writes_its_outputs_as_it_goes(small_dataset, tmp_path):
    lines = []
    _, summary = run(tiny_config(), small_dataset, 42, "cpu", tmp_path, log=lines.append)
    steps = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    curve = json.loads((tmp_path / "curve.json").read_text())
    assert len(steps) == 2 * 4 and [s["epoch"] for s in steps] == [0] * 4 + [1] * 4     # 28 patients / 8
    assert [row["epoch"] for row in curve] == [0, 1] and all("val_mae" in row for row in curve)
    assert (tmp_path / "model.pt").exists() and (tmp_path / "best.pt").exists()
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == tiny_config()
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert summary["health"]["n_steps"] == 8 and summary["variant"] == "base"
    assert summary["best_val_mae"] == min(row["val_mae"] for row in curve)
    assert len(lines) == 2 and "val MAE" in lines[0]

def test_validation_runs_every_n_epochs_and_on_the_last(small_dataset, tmp_path):
    config = tiny_config(epochs=3)
    config["validation"]["every_n_epochs"] = 2
    run(config, small_dataset, 42, "cpu", tmp_path, log=lambda line: None)
    curve = json.loads((tmp_path / "curve.json").read_text())
    assert ["val_mae" in row for row in curve] == [False, True, True]

def test_the_command_line_reruns_a_run_exactly_from_its_saved_config(small_dataset, tmp_path):
    save_dataset(small_dataset, tmp_path / "data.npz")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(tiny_config(epochs=5)))
    common = ["--seed", "42", "--dataset", str(tmp_path / "data.npz"), "--device", "cpu"]
    main([*common, "--config", str(tmp_path / "config.yaml"), "--epochs", "2", "--out", str(tmp_path / "first")])
    main([*common, "--config", str(tmp_path / "first" / "config.yaml"), "--out", str(tmp_path / "rerun")])
    saved = yaml.safe_load((tmp_path / "first" / "config.yaml").read_text())
    assert saved["train"]["epochs"] == 2                                    # --epochs is recorded
    for name in ("history.jsonl", "summary.json"):
        assert (tmp_path / "first" / name).read_text() == (tmp_path / "rerun" / name).read_text()
    strip = lambda rows: [{k: v for k, v in row.items() if not k.endswith("seconds")} for row in rows]
    curves = [json.loads((tmp_path / d / "curve.json").read_text()) for d in ("first", "rerun")]
    assert strip(curves[0]) == strip(curves[1])

def test_the_step_hook_sees_every_record_as_it_is_made(small_dataset):
    seen = []
    _, recorded = train(tiny_config(), small_dataset, 42, "cpu", on_step_end=seen.append)
    assert seen == recorded

def test_variant_names_follow_the_toggles():
    config = copy.deepcopy(CONFIG)
    names = []
    for ext1, ext2 in ((False, False), (True, False), (False, True), (True, True)):
        config["model"].update(use_mask_embedding=ext1, use_time_encoding=ext2)
        names.append(variant_name(config))
    assert names == ["base", "ext1", "ext2", "ext1+2"]
