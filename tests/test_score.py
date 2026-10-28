import copy
import hashlib
import json
import shutil

import numpy as np
import pytest
import yaml
from conftest import CONFIG
from data.physionet import save_dataset
from diffusion.run import run
from diffusion.score import attach_to_wandb, main, score
from eval import BlockEvalMasker, EvalMasker, EvalSampler, masked_mae, median_imputation
from models.build import build_model
import torch

# Each draw misses a held-out value by one of these; the lower middle of the four, the median
# upstream scores, is +0.5 (np.median would give +1.25). Every other box is missed by 7, so
# scoring a single box that is not a target would move the MAE off 0.5.
TARGET_MISSES = np.array([0.5, 2.0, -1.0, 3.0], dtype=np.float32)
ELSEWHERE = 7.0

def tiny_config():
    """A model small enough to train in a second on the 40 fake patients (as tests/test_run.py)."""
    config = copy.deepcopy(CONFIG)
    config["diffusion"].update(channels=16, layers=1, num_steps=10)
    config["train"].update(epochs=1, batch_size=8)
    config["validation"].update(every_n_epochs=1, n_samples=2, batch_size=8)
    return config

def quiet(line):
    pass

@pytest.fixture(scope="module")
def finished_run(small_dataset, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("runs") / "base_random_seed42"
    run(tiny_config(), small_dataset, 42, "cpu", out_dir, log=quiet, wandb_mode="disabled")
    return out_dir

@pytest.fixture
def run_dir(finished_run, tmp_path):
    """A fresh copy of the finished run, so a test can change it."""
    return shutil.copytree(finished_run, tmp_path / finished_run.name)

def quick_protocol(tmp_path, n_samples=2):
    """The repo's config with fewer draws, for the tests that run the real sampler."""
    config = copy.deepcopy(CONFIG)
    config["protocol"]["eval_n_samples"] = n_samples
    path = tmp_path / "quick_config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path

@pytest.fixture
def fake_sampler(monkeypatch):
    """``EvalSampler.batches`` replaced by draws that miss by known amounts; records what it was given."""
    seen = []

    def batches(self, model, values, mask, cond_mask):
        seen.append({"seed": self.seed, "n_samples": self.n_samples, "batch_size": self.batch_size,
                     "values": values, "mask": mask, "cond_mask": cond_mask})
        for start in range(0, len(values), self.batch_size):
            rows = slice(start, start + self.batch_size)
            target = mask[rows] - cond_mask[rows]
            miss = np.where(target[None] == 1, TARGET_MISSES[:, None, None, None], ELSEWHERE)
            yield rows, (values[rows][None] + miss).astype(np.float32)

    monkeypatch.setattr(EvalSampler, "batches", batches)
    return seen

# --- what is scored ---------------------------------------------------------------------

def test_the_median_is_scored_on_exactly_the_held_out_values(run_dir, small_dataset, fake_sampler):
    scores = score(run_dir, small_dataset, "cpu", log=quiet)
    assert scores["mae"] == pytest.approx(0.5, abs=1e-5) and scores["rmse"] == pytest.approx(0.5, abs=1e-5)
    idx = small_dataset["test_idx"]
    _, target = EvalMasker.from_config()(small_dataset["mask"][idx], small_dataset["record_ids"][idx])
    assert scores["n_targets"] == int(target.sum()) > 0 and scores["n_patients"] == len(idx)

def test_it_scores_the_test_split_unless_told_otherwise(run_dir, small_dataset, fake_sampler):
    score(run_dir, small_dataset, "cpu", log=quiet)
    score(run_dir, small_dataset, "cpu", split="val", log=quiet)
    for seen, split in zip(fake_sampler, ("test", "val")):
        idx = small_dataset[f"{split}_idx"]
        assert np.array_equal(seen["values"], small_dataset["values"][idx])          # in the split's order

def test_the_model_sees_only_the_frozen_holdouts_conditioning_mask(run_dir, small_dataset, fake_sampler):
    score(run_dir, small_dataset, "cpu", log=quiet)
    score(run_dir, small_dataset, "cpu", holdout="block", log=quiet)
    idx = small_dataset["test_idx"]
    mask, ids = small_dataset["mask"][idx], small_dataset["record_ids"][idx]
    for seen, masker in zip(fake_sampler, (EvalMasker.from_config(), BlockEvalMasker.from_config())):
        assert np.array_equal(seen["cond_mask"], masker(mask, ids)[0])
    assert not np.array_equal(fake_sampler[0]["cond_mask"], fake_sampler[1]["cond_mask"])

def test_the_robustness_rates_hold_out_their_own_fraction(run_dir, small_dataset, fake_sampler):
    scores = score(run_dir, small_dataset, "cpu", fraction=0.5, log=quiet)
    idx = small_dataset["test_idx"]
    cond, _ = EvalMasker(CONFIG["protocol"]["eval_masker_seed"], 0.5)(small_dataset["mask"][idx],
                                                                       small_dataset["record_ids"][idx])
    assert np.array_equal(fake_sampler[0]["cond_mask"], cond) and scores["fraction"] == 0.5

@pytest.mark.parametrize("holdout, fraction", [("random", 0.3), ("block", 0.5), ("blocks", 0.1)])
def test_holdouts_outside_the_frozen_protocol_are_refused(run_dir, small_dataset, holdout, fraction):
    with pytest.raises(ValueError):
        score(run_dir, small_dataset, "cpu", holdout=holdout, fraction=fraction, log=quiet)

def test_the_sampler_is_the_frozen_protocols(run_dir, small_dataset, fake_sampler):
    scores = score(run_dir, small_dataset, "cpu", log=quiet)
    protocol = CONFIG["protocol"]
    assert (fake_sampler[0]["seed"], fake_sampler[0]["n_samples"], fake_sampler[0]["batch_size"]) == (2021, 100, 16)
    assert scores["sampler"] == {"eval_sampling_seed": protocol["eval_sampling_seed"],
                                 "eval_n_samples": protocol["eval_n_samples"],
                                 "eval_batch_size": protocol["eval_batch_size"]}

# --- the weights scored -----------------------------------------------------------------

def test_the_runs_own_weights_are_scored(run_dir, small_dataset, tmp_path):
    config = quick_protocol(tmp_path)
    scores = score(run_dir, small_dataset, "cpu", config_path=config, log=quiet)

    model = build_model(yaml.safe_load((run_dir / "config.yaml").read_text()), "cpu")      # by hand
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    idx = small_dataset["test_idx"]
    values, mask = small_dataset["values"][idx], small_dataset["mask"][idx]
    cond, target = EvalMasker.from_config()(mask, small_dataset["record_ids"][idx])
    median = median_imputation(EvalSampler(2021, 2, 16)(model, values, mask, cond))
    assert np.array_equal(np.load(run_dir / "median_test_random_0.1.npy"), median)
    assert scores["mae"] == masked_mae(median, values, target)

    torch.manual_seed(0)
    with torch.no_grad():                                       # other weights score differently
        for p in model.parameters():
            p.add_(torch.randn(p.shape) * 0.1)
    torch.save(model.state_dict(), run_dir / "model.pt")
    assert score(run_dir, small_dataset, "cpu", config_path=config, log=quiet)["mae"] != scores["mae"]

def test_the_scores_record_the_hash_of_the_weights_scored(run_dir, small_dataset, fake_sampler):
    scores = score(run_dir, small_dataset, "cpu", log=quiet)
    assert scores["model_sha256"] == hashlib.sha256((run_dir / "model.pt").read_bytes()).hexdigest()

def test_an_unfinished_run_is_refused(run_dir, small_dataset, fake_sampler):
    (run_dir / "summary.json").unlink()             # model.pt alone may be an earlier epoch's weights
    with pytest.raises(FileNotFoundError, match="not finished"):
        score(run_dir, small_dataset, "cpu", log=quiet)

def test_held_out_values_cannot_reach_the_guess(run_dir, small_dataset, tmp_path):
    config = quick_protocol(tmp_path)
    score(run_dir, small_dataset, "cpu", config_path=config, log=quiet)
    guess = np.load(run_dir / "median_test_random_0.1.npy")

    idx = small_dataset["test_idx"]
    _, target = EvalMasker.from_config()(small_dataset["mask"][idx], small_dataset["record_ids"][idx])
    tampered = dict(small_dataset, values=small_dataset["values"].copy())
    tampered["values"][idx] += np.where(target == 1, 1e3, 0).astype(np.float32)
    score(run_dir, tampered, "cpu", config_path=config, log=quiet)
    assert np.array_equal(np.load(run_dir / "median_test_random_0.1.npy"), guess)

def test_scoring_twice_gives_identical_files(run_dir, small_dataset, tmp_path):
    config = quick_protocol(tmp_path)
    first = score(run_dir, small_dataset, "cpu", config_path=config, log=quiet)
    median = (run_dir / "median_test_random_0.1.npy").read_bytes()
    second = score(run_dir, small_dataset, "cpu", config_path=config, log=quiet)
    assert {k: v for k, v in first.items() if k != "seconds"} == {k: v for k, v in second.items() if k != "seconds"}
    assert (run_dir / "median_test_random_0.1.npy").read_bytes() == median

# --- what is written --------------------------------------------------------------------

def test_both_files_are_written_named_for_what_was_scored(run_dir, small_dataset, fake_sampler):
    scores = score(run_dir, small_dataset, "cpu", holdout="block", log=quiet)
    assert json.loads((run_dir / "scores_test_block_0.1.json").read_text()) == scores
    median = np.load(run_dir / "median_test_block_0.1.npy")
    assert median.shape == (len(small_dataset["test_idx"]), 48, 35) and median.dtype == np.float32
    assert scores["holdout_settings"] == {"block_eval_masker_seed": 8080, "block_window_hours": 3}

def test_the_scores_name_the_run_and_its_seed(run_dir, small_dataset, fake_sampler):
    scores = score(run_dir, small_dataset, "cpu", log=quiet)
    assert (scores["run"], scores["variant"], scores["seed"]) == ("base_random_seed42", "base", 42)
    assert set(scores["per_variable_mae"]) == set(small_dataset["features"])
    assert scores["holdout_settings"] == {"eval_masker_seed": 2012}

def test_the_running_mae_is_reported_after_every_batch(run_dir, small_dataset, fake_sampler):
    lines = []
    score(run_dir, small_dataset, "cpu", log=lines.append)
    assert len(lines) == -(-len(small_dataset["test_idx"]) // 16)
    assert "running MAE 0.5000" in lines[-1]

def test_the_command_line_scores_a_run(run_dir, small_dataset, fake_sampler, tmp_path, capsys):
    save_dataset(small_dataset, tmp_path / "data.npz")
    main([str(run_dir), "--dataset", str(tmp_path / "data.npz"), "--device", "cpu", "--holdout", "block"])
    assert (run_dir / "scores_test_block_0.1.json").exists()
    assert '"mae"' in capsys.readouterr().out

# --- wandb ------------------------------------------------------------------------------

class FakeApi:
    updates = []

    def run(self, path):
        api = self

        class Summary:
            def update(self, values):
                api.updates.append((path, values))

        return type("Run", (), {"summary": Summary()})()

@pytest.fixture
def fake_api(monkeypatch):
    FakeApi.updates = []
    monkeypatch.setattr("diffusion.score.wandb.Api", FakeApi)
    return FakeApi.updates

SCORES = {"holdout": "random", "fraction": 0.1, "split": "test", "mae": 0.23, "rmse": 0.5,
          "wandb_run": "DATA612/msml612-csdi/abc123"}

def test_scores_join_the_runs_wandb_summary_under_the_baselines_keys(fake_api):
    assert attach_to_wandb(SCORES, "online")
    assert fake_api == [("DATA612/msml612-csdi/abc123", {"random/0.1/test/mae": 0.23, "random/0.1/test/rmse": 0.5})]

def test_nothing_is_sent_unless_online_or_for_an_untracked_run(fake_api, monkeypatch):
    assert not attach_to_wandb(SCORES, "disabled")
    assert not attach_to_wandb(SCORES, "offline")               # the public API is always online
    assert not attach_to_wandb(SCORES)                          # conftest's WANDB_MODE=disabled
    assert not attach_to_wandb(SCORES | {"wandb_run": None}, "online")
    monkeypatch.delenv("WANDB_MODE")
    assert attach_to_wandb(SCORES)                              # unset: online, as diffusion.run
    assert len(fake_api) == 1
