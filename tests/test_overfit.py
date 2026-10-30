import copy

import numpy as np
import pytest
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
    assert (record["epochs"], record["steps_per_epoch"], record["seed"]) == (2, 3, 42)
    assert record["status"] == ("PASS" if record["mae"] < record["mae_below"] else "FAIL")
    assert np.isfinite([record["mae"], record["rmse"], record["loss_first_epoch"], record["loss_last_epoch"]]).all()

def test_gradients_reach_the_encoder(record):
    assert record["encoder_grad_norm_last_step"] > 0

def test_the_same_seed_gives_the_same_record(record, small_dataset):
    again = overfit(tiny_config(), small_dataset, epochs=2, steps_per_epoch=3,
                    sampler=EvalSampler(2021, 2, 16), log=quiet)
    assert {k: v for k, v in again.items() if k != "seconds"} == {k: v for k, v in record.items() if k != "seconds"}
