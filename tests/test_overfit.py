import copy

import numpy as np
import pytest
import torch
from conftest import CONFIG
from diffusion.overfit import overfit
from eval import BlockEvalMasker, EvalSampler

def tiny_config():
    config = copy.deepcopy(CONFIG)
    config["diffusion"].update(channels=16, layers=1, num_steps=10)
    return config

def quiet(*args):
    pass

@pytest.fixture(scope="module")
def record(small_dataset):
    return overfit(tiny_config(), small_dataset, epochs=2, steps_per_epoch=3,
                   sampler=EvalSampler(2021, 2, 16), log=quiet)

def test_the_record_scores_the_block_targets_of_the_first_train_patients(record, small_dataset):
    idx = small_dataset["train_idx"][:10]
    _, target = BlockEvalMasker.from_config()(small_dataset["mask"][idx], small_dataset["record_ids"][idx])
    assert record["patients"] == small_dataset["record_ids"][idx].tolist()
    assert record["n_targets"] == int(target.sum()) > 0
    assert record["zero_guess_mae"] == pytest.approx(np.abs(small_dataset["values"][idx][target == 1]).mean(), abs=1e-6)

def test_the_record_says_what_was_run(record):
    assert (record["variant"], record["epochs"], record["steps_per_epoch"], record["seed"]) == ("ext1", 2, 3, 42)
    assert record["status"] == ("PASS" if record["mae"] < record["mae_below"] else "FAIL")
    assert np.isfinite([record["mae"], record["rmse"], record["loss_first_epoch"], record["loss_last_epoch"]]).all()
    assert record["torch"] == torch.__version__ and record["machine"]       # bit-for-bit only on one machine

def test_gradients_reach_the_encoder_and_its_gate_opens(record):
    assert record["encoder_grad_norm_last_step"] > 0
    assert record["encoder_gate_last_step"] != 0                            # starts at 0

def test_the_base_reference_runs_the_same_protocol_without_the_encoder(record, small_dataset):
    base = overfit(tiny_config(), small_dataset, epochs=2, steps_per_epoch=3,
                   sampler=EvalSampler(2021, 2, 16), ext1=False, log=quiet)
    assert base["variant"] == "base" and base["n_parameters"] < record["n_parameters"]
    assert base["encoder_grad_norm_last_step"] is None and base["encoder_gate_last_step"] is None
    same = ("patients", "n_targets", "zero_guess_mae", "epochs", "steps_per_epoch", "seed", "sampler")
    assert {k: base[k] for k in same} == {k: record[k] for k in same}
    assert np.isfinite([base["mae"], base["rmse"]]).all()

def test_the_holdout_follows_the_config_it_is_given(small_dataset):
    # --config used to reach the model but not the holdout (M4-2 review).
    config = tiny_config()
    config["protocol"]["block_eval_masker_seed"] += 1
    other = overfit(config, small_dataset, epochs=1, steps_per_epoch=1, sampler=EvalSampler(2021, 1, 16), log=quiet)
    idx = small_dataset["train_idx"][:10]
    values, mask, ids = small_dataset["values"][idx], small_dataset["mask"][idx], small_dataset["record_ids"][idx]
    protocol, grid = CONFIG["protocol"], (CONFIG["data"]["n_timesteps"], CONFIG["data"]["n_features"])
    expected = {}
    for seed in (protocol["block_eval_masker_seed"], config["protocol"]["block_eval_masker_seed"]):
        masker = BlockEvalMasker(seed, protocol["eval_holdout_fraction"], protocol["block_window_hours"], grid)
        _, target = masker(mask, ids)
        expected[seed] = np.abs(values[target == 1]).mean()
    assert other["zero_guess_mae"] == pytest.approx(expected[config["protocol"]["block_eval_masker_seed"]], abs=1e-6)
    assert other["zero_guess_mae"] != pytest.approx(expected[CONFIG["protocol"]["block_eval_masker_seed"]], abs=1e-6)

def test_the_same_seed_gives_the_same_record(record, small_dataset):
    again = overfit(tiny_config(), small_dataset, epochs=2, steps_per_epoch=3,
                    sampler=EvalSampler(2021, 2, 16), log=quiet)
    assert {k: v for k, v in again.items() if k != "seconds"} == {k: v for k, v in record.items() if k != "seconds"}
