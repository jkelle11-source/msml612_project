import copy
import json

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
import diffusion.ext2_health as ext2_health
from conftest import CONFIG
from data.physionet import save_dataset
from diffusion.ext2_health import epochs_for, main, status, variable_mae
from eval import EvalMasker

def tiny_config():
    """A model small enough to train and validate in a few seconds on the 40 fake patients."""
    config = copy.deepcopy(CONFIG)
    config["diffusion"].update(channels=16, layers=1, num_steps=10)
    config["train"].update(batch_size=8)
    config["validation"].update(every_n_epochs=1, n_samples=2, batch_size=8)
    return config

# --- epochs_for: the fewest epochs that reach the health report's 500 steps ---------------

@pytest.mark.parametrize("n_train, batch_size, n_steps, epochs", [
    (2800, 16, 500, 3),         # ours: 175 steps an epoch, 525 in 3
    (2800, 16, 525, 3),
    (2800, 16, 526, 4),
    (2800, 16, 175, 1),
    (2801, 16, 528, 3),         # the last, short batch is a step too: 176 an epoch
    (28, 8, 8, 2),              # the 40 fake patients' 28 training patients: 4 steps an epoch
])
def test_epochs_for(n_train, batch_size, n_steps, epochs):
    assert epochs_for({"train": {"batch_size": batch_size}}, {"train_idx": np.arange(n_train)}, n_steps) == epochs

# --- status: PASS only for a healthy run whose encoder is in use --------------------------

SCORES = {"HR": {"mae": 0.2}, "SysABP": {"mae": 0.3}, "all": {"mae": 0.25}}

def test_status_passes_a_healthy_run_whose_encoder_is_in_use():
    assert status({"healthy": True}, -0.01, 0.5, SCORES) == "PASS"         # the gate's sign is free

@pytest.mark.parametrize("health, gate, grad, scores", [
    ({"healthy": False}, -0.01, 0.5, SCORES),                               # unhealthy, or under 500 steps
    ({"healthy": True}, 0.0, 0.5, SCORES),                                  # the gate never opened
    ({"healthy": True}, -0.01, 0.0, SCORES),                                # no gradient reached the encoder
    ({"healthy": True}, -0.01, 0.5, {**SCORES, "HR": {"mae": float("nan")}}),
])
def test_status_fails_otherwise(health, gate, grad, scores):
    assert status(health, gate, grad, scores) == "FAIL"

# --- variable_mae: the named variables alone, on the run's own validation holdout ---------

class ZeroImputer(nn.Module):
    """Stands in for the model: every draw is 0, so the median guess is 0 and a variable's MAE is
    the mean |value| over its held-out targets."""
    def evaluate(self, batch, n_samples, generator=None):
        B, T, D = batch["observed_data"].shape
        return (torch.zeros(B, n_samples, D, T),)

def test_variable_mae_scores_each_named_variable_on_the_validation_holdout(small_dataset):
    scores = variable_mae(ZeroImputer(), small_dataset, tiny_config())
    idx = small_dataset["val_idx"]
    values, mask = small_dataset["values"][idx], small_dataset["mask"][idx]
    _, target = EvalMasker.from_config()(mask, small_dataset["record_ids"][idx])
    features = list(small_dataset["features"])
    assert set(scores) == {"HR", "SysABP", "all"}
    for name in ("HR", "SysABP"):
        j = features.index(name)
        hidden = target[..., j] == 1
        expected = np.abs(values[..., j][hidden]).mean()
        assert scores[name]["n_targets"] == int(hidden.sum()) > 0
        assert scores[name]["mae"] == pytest.approx(expected, abs=1e-6)
        assert scores[name]["mae_raw"] == pytest.approx(expected * small_dataset["std"][j], abs=1e-3)
    assert scores["all"]["n_targets"] == int(target.sum())
    assert scores["all"]["mae"] == pytest.approx(np.abs(values[target == 1]).mean(), abs=1e-6)

# --- the whole check, through main ----------------------------------------------------------

def launch(tmp, dataset, *args):
    config = tiny_config()
    config["model"]["use_mask_embedding"] = True        # the check turns Ext 1 off: Ext 2 alone, then base
    save_dataset(dataset, tmp / "data.npz")
    (tmp / "config.yaml").write_text(yaml.safe_dump(config))
    main(["--device", "cpu", "--config", str(tmp / "config.yaml"), "--dataset", str(tmp / "data.npz"),
          "--runs-dir", str(tmp / "runs"), "--steps", "8", "--out", str(tmp / "record.json"), *args])
    return json.loads((tmp / "record.json").read_text())

@pytest.fixture(scope="module")
def checked(small_dataset, tmp_path_factory):
    """The check on the 40 fake patients with a tiny model, 8 steps (2 epochs); every call to
    diffusion.run.run is recorded with the config it was given."""
    tmp = tmp_path_factory.mktemp("ext2_health")
    calls, real_run = [], ext2_health.run
    def recording_run(config, *args, **kwargs):
        calls.append((copy.deepcopy(config), kwargs))
        return real_run(config, *args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ext2_health, "run", recording_run)
        record = launch(tmp, small_dataset)
    return record, calls, tmp

def test_ext2_then_base_go_through_run_untracked(checked):
    # Untracked: run() would name the Ext 2 run ext2_random_seed42, the full health run's name.
    _, calls, _ = checked
    assert [(c["model"]["use_time_encoding"], c["model"]["use_mask_embedding"]) for c, _ in calls] == [(True, False), (False, False)]
    assert all(kwargs["wandb_mode"] == "disabled" for _, kwargs in calls)
    assert all(c["train"]["epochs"] == 2 for c, _ in calls)

def test_the_record_says_what_was_run(checked):
    record, _, _ = checked
    assert (record["variant"], record["base_reference"]["variant"]) == ("ext2", "base")
    assert (record["seed"], record["epochs"], record["steps"]) == (42, 2, 8)
    assert record["lr_drops_at_epochs"] == [1, 1]           # max(1, int(f · 2)) for f = 0.75, 0.9
    assert "not an effect" in record["note"] and "not an effect" in record["base_reference"]["note"]
    assert record["torch"] == torch.__version__ and record["machine"]

def test_a_run_under_500_steps_is_not_passed(checked):
    record, _, _ = checked
    assert record["health"]["n_steps"] == 8 and record["health"]["complete"] is False
    assert record["status"] == "FAIL"

def test_gradients_reach_the_encoder_and_its_gate_opens(checked):
    # 8 steps: step 1 trains upstream's zero-initialised last layer, step 2 the gate, then all.
    record, _, _ = checked
    enc = record["time_encoder"]
    assert enc["gate_last_step"] != 0 and enc["grad_norm_last_step"] > 0
    assert enc["d_k"] == 16
    assert enc["n_parameters"] == 2 + 2 * 15 + (16 * 16 + 16) + 1        # w0, w, proj at channels 16, gate
    assert record["n_parameters"] - record["base_reference"]["n_parameters"] == enc["n_parameters"]

def test_variable_scores_reproduce_each_runs_own_validation(checked):
    # The same masker and frozen sampler as run()'s validation, so "all" is its final val MAE.
    record, _, _ = checked
    for r in (record, record["base_reference"]):
        assert r["variable_mae"]["all"]["mae"] == pytest.approx(r["final_val_mae"], abs=1e-6)
        assert all(np.isfinite(r["variable_mae"][name]["mae"]) and r["variable_mae"][name]["n_targets"] > 0
                   for name in ("HR", "SysABP"))

def test_the_run_folders_hold_each_variants_weights(checked):
    # model.pt is saved at the last epoch's validation, after the last step: the weights recorded.
    record, _, tmp = checked
    ext2 = torch.load(tmp / "runs" / "ext2" / "model.pt")
    base = torch.load(tmp / "runs" / "base" / "model.pt")
    assert record["time_encoder"]["gate_last_step"] == ext2["time_encoder.gate"].item()
    assert not any(k.startswith(("time_encoder.", "mask_encoder.")) for k in base)
    assert not any(k.startswith("mask_encoder.") for k in ext2)

def test_a_rerun_replaces_its_run_folders_and_gives_the_same_record(checked, small_dataset):
    record, _, tmp = checked
    again = launch(tmp, small_dataset)                      # the run folders are not empty now
    drop = lambda r: {k: v for k, v in r.items() if k not in ("seconds", "base_reference")}
    assert drop(again) == drop(record)
    assert again["base_reference"]["final_val_mae"] == record["base_reference"]["final_val_mae"]
