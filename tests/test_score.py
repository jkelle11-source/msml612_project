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
from diffusion.score import attach_to_wandb, main, score, span_mae
from diffusion.train import use_tf32
from eval import BlockEvalMasker, EvalMasker, EvalSampler, OffGridBlockEvalMasker, masked_mae, median_imputation
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

@pytest.mark.parametrize("holdout, fraction", [("random", 0.3), ("block", 0.5), ("offgrid", 0.5), ("blocks", 0.1)])
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

def test_the_offgrid_holdout_is_scored_under_a_name_of_its_own(run_dir, small_dataset, fake_sampler):
    # §4.1's grid check (M6-3): base's and Ext 1's runs scored, without retraining, off the grid.
    scores = score(run_dir, small_dataset, "cpu", holdout="offgrid", log=quiet)
    idx = small_dataset["test_idx"]
    cond, target = OffGridBlockEvalMasker.from_config()(small_dataset["mask"][idx], small_dataset["record_ids"][idx])
    assert np.array_equal(fake_sampler[0]["cond_mask"], cond) and scores["n_targets"] == int(target.sum())
    assert json.loads((run_dir / "scores_test_offgrid_0.1.json").read_text()) == scores
    assert (run_dir / "median_test_offgrid_0.1.npy").exists() and not (run_dir / "scores_test_block_0.1.json").exists()
    assert scores["holdout_settings"] == {"offgrid_block_eval_masker_seed": 9090, "block_window_hours": 3,
                                          "offgrid_block_offset_hours": 1}

# --- span_mae: a holdout's MAE over some hours, from a saved scoring (§4.1's grid check, M6-3) ------

def miss_by_the_hour(run_dir, small_dataset, name):
    """Overwrite a scoring's saved median so every box misses by its hour (0, 1, ..., 47): the MAE over
    any set of targets is then their mean hour, which a test can work out without span_mae."""
    values = small_dataset["values"][small_dataset["test_idx"]]
    np.save(run_dir / f"median_{name}.npy", values + np.arange(values.shape[1], dtype=np.float32)[None, :, None])

def test_span_mae_is_the_holdouts_mae_over_those_hours_alone(run_dir, small_dataset, fake_sampler):
    scores = score(run_dir, small_dataset, "cpu", holdout="block", log=quiet)
    assert span_mae(run_dir, small_dataset, "block", range(48)) == {      # every hour: the score itself
        "mae": pytest.approx(scores["mae"]), "n_targets": scores["n_targets"]}
    miss_by_the_hour(run_dir, small_dataset, "test_block_0.1")
    idx = small_dataset["test_idx"]
    _, target = BlockEvalMasker.from_config()(small_dataset["mask"][idx], small_dataset["record_ids"][idx])
    hour = np.nonzero(target)[1]                                    # each block target's hour
    inside = hour[(hour >= 1) & (hour <= 45)]
    assert 0 < len(inside) < len(hour)                              # the fake patients lose hours 0, 46 or 47 too
    span = span_mae(run_dir, small_dataset, "block", OffGridBlockEvalMasker.from_config().hours)
    assert span == {"mae": pytest.approx(inside.mean()), "n_targets": len(inside)}

def test_the_offgrid_holdout_over_its_own_hours_is_all_of_it(run_dir, small_dataset, fake_sampler):
    score(run_dir, small_dataset, "cpu", holdout="offgrid", log=quiet)
    miss_by_the_hour(run_dir, small_dataset, "test_offgrid_0.1")
    hours = OffGridBlockEvalMasker.from_config().hours
    assert span_mae(run_dir, small_dataset, "offgrid", hours) == span_mae(run_dir, small_dataset, "offgrid", range(48))

def test_span_mae_refuses_a_scoring_with_another_holdout_or_sampler(run_dir, small_dataset, fake_sampler, tmp_path):
    quick = quick_protocol(tmp_path)                                # 2 draws: not the frozen 100
    score(run_dir, small_dataset, "cpu", holdout="block", config_path=quick, log=quiet)
    with pytest.raises(ValueError, match="frozen holdout and sampler"):
        span_mae(run_dir, small_dataset, "block", range(1, 46))
    assert span_mae(run_dir, small_dataset, "block", range(1, 46), config_path=quick)["n_targets"] > 0

def test_span_mae_refuses_weights_that_are_not_the_ones_scored(run_dir, small_dataset, fake_sampler):
    score(run_dir, small_dataset, "cpu", holdout="block", log=quiet)
    weights = torch.load(run_dir / "model.pt")
    key = next(k for k, v in weights.items() if v.is_floating_point())
    weights[key] = weights[key] + 1
    torch.save(weights, run_dir / "model.pt")
    with pytest.raises(ValueError, match="not the weights"):
        span_mae(run_dir, small_dataset, "block", range(1, 46))

def test_span_mae_needs_a_scoring_of_that_holdout_and_a_target_in_the_span(run_dir, small_dataset, fake_sampler):
    score(run_dir, small_dataset, "cpu", holdout="block", log=quiet)
    with pytest.raises(FileNotFoundError):
        span_mae(run_dir, small_dataset, "offgrid", range(1, 46))   # never scored off the grid
    with pytest.raises(ValueError, match="no hidden boxes"):
        span_mae(run_dir, small_dataset, "block", range(0))

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

SCORES = {"holdout": "random", "fraction": 0.1, "split": "test", "mae": 0.23, "rmse": 0.5, "gpu": "Tesla T4", "tf32": False,
          "sampler": {"eval_sampling_seed": 2021, "eval_n_samples": 100, "eval_batch_size": 16},
          "holdout_settings": {"eval_masker_seed": 7}, "per_variable_mae": {"HR": 0.1, "SysABP": None},
          "wandb_run": "DATA612/msml612-csdi/abc123"}

def test_scores_join_the_runs_wandb_summary_under_the_baselines_keys(fake_api):
    assert attach_to_wandb(SCORES, "online")
    assert fake_api == [("DATA612/msml612-csdi/abc123", {
        "random/0.1/test/mae": 0.23, "random/0.1/test/rmse": 0.5,
        # M6-2: what diffusion.results collects and checks, flat as wandb keeps a block anyway
        "random/0.1/test/gpu": "Tesla T4", "random/0.1/test/tf32": False,
        "random/0.1/test/sampler.eval_sampling_seed": 2021, "random/0.1/test/sampler.eval_n_samples": 100,
        "random/0.1/test/sampler.eval_batch_size": 16, "random/0.1/test/holdout_settings.eval_masker_seed": 7,
        "random/0.1/test/per_variable_mae.HR": 0.1, "random/0.1/test/per_variable_mae.SysABP": None})]

def test_scores_from_before_the_gpu_was_recorded_send_none_for_it(fake_api):
    assert attach_to_wandb({k: v for k, v in SCORES.items() if k != "gpu"}, "online")
    assert fake_api[0][1]["random/0.1/test/gpu"] is None

def test_a_real_scoring_sends_its_settings_and_every_variables_mae(run_dir, small_dataset, fake_sampler, fake_api):
    scores = score(run_dir, small_dataset, "cpu", holdout="block", log=quiet)
    attach_to_wandb({**scores, "wandb_run": "DATA612/msml612-csdi/abc123"}, "online")
    sent = fake_api[0][1]
    assert sent["block/0.1/test/sampler.eval_n_samples"] == CONFIG["protocol"]["eval_n_samples"]
    assert sent["block/0.1/test/holdout_settings.block_window_hours"] == CONFIG["protocol"]["block_window_hours"]
    assert {key for key in sent if "/per_variable_mae." in key} == {
        f"block/0.1/test/per_variable_mae.{name}" for name in small_dataset["features"]}

def test_nothing_is_sent_unless_online_or_for_an_untracked_run(fake_api, monkeypatch):
    assert not attach_to_wandb(SCORES, "disabled")
    assert not attach_to_wandb(SCORES, "offline")               # the public API is always online
    assert not attach_to_wandb(SCORES)                          # conftest's WANDB_MODE=disabled
    assert not attach_to_wandb(SCORES | {"wandb_run": None}, "online")
    monkeypatch.delenv("WANDB_MODE")
    assert attach_to_wandb(SCORES)                              # unset: online, as diffusion.run
    assert len(fake_api) == 1

# --- the paired GPU (M6-2) --------------------------------------------------------------

def picked(tmp_path, gpu):
    """The repo's config with ``protocol.paired_gpu`` set to ``gpu``."""
    config = copy.deepcopy(CONFIG)
    config["protocol"]["paired_gpu"] = gpu
    path = tmp_path / "picked_config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path

def test_the_scores_record_the_gpu_they_were_scored_on(run_dir, small_dataset, fake_sampler, monkeypatch):
    assert score(run_dir, small_dataset, "cpu", log=quiet)["gpu"] is None
    monkeypatch.setattr("diffusion.score.gpu_name", lambda device: "Tesla T4")      # as on a Colab T4
    assert score(run_dir, small_dataset, "cpu", log=quiet)["gpu"] == "Tesla T4"
    assert json.loads((run_dir / "scores_test_random_0.1.json").read_text())["gpu"] == "Tesla T4"

@pytest.mark.parametrize("gpu, refusal", [(None, "timing test"), ("Tesla T4", "not on cpu")])
def test_a_paired_scoring_refuses_any_gpu_but_the_picked_one(gpu, refusal, run_dir, small_dataset, fake_sampler, tmp_path):
    with pytest.raises(RuntimeError, match=refusal):
        score(run_dir, small_dataset, "cpu", config_path=picked(tmp_path, gpu), log=quiet, paired=True)
    assert not list(run_dir.glob("scores_*")) and fake_sampler == []       # refused before scoring

@pytest.mark.parametrize("tf32, precision", [(True, "tf32"), (False, "ieee")])
def test_a_scoring_runs_and_records_the_protocols_tf32(tf32, precision, run_dir, small_dataset, fake_sampler, tmp_path):
    config = copy.deepcopy(CONFIG)
    config["protocol"]["tf32"] = tf32
    (tmp_path / "tf32_config.yaml").write_text(yaml.safe_dump(config))
    try:
        scores = score(run_dir, small_dataset, "cpu", config_path=tmp_path / "tf32_config.yaml", log=quiet)
        assert torch.backends.cuda.matmul.fp32_precision == torch.backends.cudnn.conv.fp32_precision == precision
    finally:
        use_tf32(False)
    assert scores["tf32"] is tf32
    assert json.loads((run_dir / "scores_test_random_0.1.json").read_text())["tf32"] is tf32

def test_the_command_line_scores_a_paired_run_only_on_the_picked_gpu(run_dir, small_dataset, fake_sampler, tmp_path):
    save_dataset(small_dataset, tmp_path / "data.npz")
    with pytest.raises(RuntimeError, match="not on cpu"):
        main([str(run_dir), "--dataset", str(tmp_path / "data.npz"), "--device", "cpu",
              "--config", str(picked(tmp_path, "Tesla T4")), "--wandb", "disabled", "--paired"])
    assert not list(run_dir.glob("scores_*")) and fake_sampler == []       # refused before scoring

def test_a_paired_scoring_on_the_picked_gpu_runs(run_dir, small_dataset, fake_sampler, tmp_path, monkeypatch):
    monkeypatch.setattr("diffusion.run.gpu_name", lambda device: "Tesla T4")
    monkeypatch.setattr("diffusion.score.gpu_name", lambda device: "Tesla T4")
    scores = score(run_dir, small_dataset, "cpu", config_path=picked(tmp_path, "Tesla T4"), log=quiet, paired=True)
    assert scores["gpu"] == "Tesla T4" and scores["mae"] == pytest.approx(0.5)
