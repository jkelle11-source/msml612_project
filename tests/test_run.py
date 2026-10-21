import copy
import json

import numpy as np
import pytest
import yaml
from conftest import CONFIG
from data.physionet import save_dataset
from diffusion.run import RUN_FILES, health_report, main, run, validate, variant_name
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

def launch(tmp_path, dataset, *args):
    """``main`` on ``dataset`` and a 3-epoch tiny config, both saved under ``tmp_path``."""
    save_dataset(dataset, tmp_path / "data.npz")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(tiny_config(epochs=3)))
    main(["--seed", "42", "--dataset", str(tmp_path / "data.npz"), "--device", "cpu",
          "--config", str(tmp_path / "config.yaml"), *args])

def quiet(line):
    pass

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

def test_a_run_shorter_than_the_steps_checked_is_not_healthy():
    report = health_report(history(np.linspace(1.0, 0.3, 350), np.full(350, 2.0)))    # e.g. --epochs 2
    assert report["n_steps"] == 350 and not report["complete"] and not report["healthy"]
    assert report["finite"] and report["learning"]          # it fails on length alone

@pytest.mark.parametrize("length, n_steps", [(500, 500), (600, 500), (100, 100)])
def test_a_history_with_at_least_n_steps_is_complete(length, n_steps):
    report = health_report(history(np.linspace(1.0, 0.3, length), np.full(length, 2.0)), n_steps=n_steps)
    assert report["complete"] and report["healthy"]

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
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(RUN_FILES)      # no best.pt: final weights are scored
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == tiny_config()
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert summary["health"]["n_steps"] == 8 and summary["variant"] == "base"
    assert summary["final_val_mae"] == curve[-1]["val_mae"]
    assert summary["below_forward_fill"] == (summary["final_val_mae"] < summary["forward_fill_val_mae"])
    assert len(lines) == 2 and "val MAE" in lines[0]

def test_validation_runs_every_n_epochs_and_on_the_last(small_dataset, tmp_path):
    config = tiny_config(epochs=3)
    config["validation"]["every_n_epochs"] = 2
    run(config, small_dataset, 42, "cpu", tmp_path, log=quiet)
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

def test_validate_every_overrides_the_config_and_is_recorded(small_dataset, tmp_path):
    launch(tmp_path, small_dataset, "--validate-every", "2", "--out", str(tmp_path / "run"))
    saved = yaml.safe_load((tmp_path / "run" / "config.yaml").read_text())
    curve = json.loads((tmp_path / "run" / "curve.json").read_text())
    assert saved["validation"]["every_n_epochs"] == 2                       # tiny_config says 1
    assert ["val_mae" in row for row in curve] == [False, True, True]       # epoch 2, and the last

# --- an output folder holds one run -----------------------------------------------------

def test_a_run_refuses_a_folder_that_is_not_empty(small_dataset, tmp_path):
    (tmp_path / "summary.json").write_text("an earlier run's")
    with pytest.raises(FileExistsError, match="--overwrite"):
        run(tiny_config(), small_dataset, 42, "cpu", tmp_path, log=quiet)
    assert [p.name for p in tmp_path.iterdir()] == ["summary.json"]           # refused before writing
    assert (tmp_path / "summary.json").read_text() == "an earlier run's"

def test_overwrite_deletes_the_earlier_runs_files_and_nothing_else(small_dataset, tmp_path, monkeypatch):
    run(tiny_config(), small_dataset, 42, "cpu", tmp_path, log=quiet)
    (tmp_path / "notes.txt").write_text("not a run file")

    def session_lost(*args, **kwargs):                      # dies before its first scored epoch
        raise RuntimeError("session lost")

    monkeypatch.setattr("diffusion.run.train", session_lost)
    with pytest.raises(RuntimeError, match="session lost"):
        run(tiny_config(), small_dataset, 42, "cpu", tmp_path, log=quiet, overwrite=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.yaml", "history.jsonl", "notes.txt"]
    assert (tmp_path / "history.jsonl").read_text() == ""   # the new run's, not the old one's

def test_the_command_line_reuses_a_folder_only_with_overwrite(small_dataset, tmp_path):
    out = ["--out", str(tmp_path / "run")]
    launch(tmp_path, small_dataset, *out)
    with pytest.raises(FileExistsError):
        launch(tmp_path, small_dataset, *out)
    launch(tmp_path, small_dataset, *out, "--overwrite")
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == sorted(RUN_FILES)

def test_the_step_hook_sees_every_record_as_it_is_made(small_dataset):
    seen = []
    _, recorded = train(tiny_config(), small_dataset, 42, "cpu", on_step_end=seen.append)
    assert seen == recorded

def test_a_step_hook_that_edits_its_record_leaves_the_history_alone(small_dataset):
    def careless_hook(record):                              # e.g. a logger that adds its own keys
        record["loss"] = -1.0
        record["step"] = 0

    _, recorded = train(tiny_config(), small_dataset, 42, "cpu", on_step_end=careless_hook)
    assert all(set(r) == {"epoch", "loss", "grad_norm", "lr"} and r["loss"] >= 0 for r in recorded)

def test_variant_names_follow_the_toggles():
    config = copy.deepcopy(CONFIG)
    names = []
    for ext1, ext2 in ((False, False), (True, False), (False, True), (True, True)):
        config["model"].update(use_mask_embedding=ext1, use_time_encoding=ext2)
        names.append(variant_name(config))
    assert names == ["base", "ext1", "ext2", "ext1+2"]
