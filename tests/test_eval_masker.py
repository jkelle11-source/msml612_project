import hashlib

import numpy as np
import pytest
import torch
from conftest import CONFIG, RAW_DIR
from eval.masking import EvalMasker

needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")

def random_mask(n=50, seed=0, observed_rate=0.2):
    rng = np.random.default_rng(seed)
    return (rng.random((n, 48, 35)) < observed_rate).astype(np.float32)

@pytest.fixture(scope="module")
def masker() -> EvalMasker:
    return EvalMasker.from_config()

# --- config ---------------------------------------------------------------------------

def test_from_config_reads_the_frozen_protocol(masker: EvalMasker):
    assert masker.seed == CONFIG["protocol"]["eval_masker_seed"] == 2012
    assert masker.fraction == CONFIG["protocol"]["eval_holdout_fraction"] == 0.10
    assert masker.grid == (CONFIG["data"]["n_timesteps"], CONFIG["data"]["n_features"]) == (48, 35)

def test_eval_seed_is_not_a_run_seed():
    assert CONFIG["protocol"]["eval_masker_seed"] not in CONFIG["protocol"]["run_seeds"]

def test_maskers_and_metrics_import_from_the_package():
    import eval                                 # the README and config name them as eval.EvalMasker etc.
    assert eval.EvalMasker is EvalMasker
    assert {"EvalMasker", "TrainMasker", "masked_mae", "masked_rmse", "per_variable_mae"} <= set(eval.__all__)
    assert all(callable(getattr(eval, name)) for name in eval.__all__)

# --- hand-checkable fixture -------------------------------------------------------------

def test_tiny_fixture_hides_floor_ten_percent():
    mask = np.zeros((3, 48, 35), dtype=np.float32)
    mask[0, :25, 0] = 1        # 25 observed -> 2 hidden
    mask[1, :9, 3] = 1         #  9 observed -> 0 hidden (floor)
    # patient 2 has nothing observed -> 0 hidden, no crash
    cond, target = EvalMasker(2012, 0.1)(mask, np.array([1, 2, 3]))
    assert target.reshape(3, -1).sum(axis=1).tolist() == [2, 0, 0]
    assert (target[0, 25:] == 0).all() and (target[0, :, 1:] == 0).all()
    assert (cond[1] == mask[1]).all() and (cond[2] == 0).all()

# --- shape and correctness ------------------------------------------------------------

def test_outputs_keep_shape_and_dtype(masker: EvalMasker):
    mask = random_mask()
    cond, target = masker(mask, np.arange(50))
    for out in (cond, target):
        assert out.shape == mask.shape
        assert out.dtype == mask.dtype
        assert set(np.unique(out)) <= {0.0, 1.0}

def test_targets_are_observed_and_disjoint_from_conditioning(masker: EvalMasker):
    mask = random_mask()
    cond, target = masker(mask, np.arange(50))
    assert (target <= mask).all()             # only hide values we actually have
    assert (cond * target == 0).all()         # a target is never also conditioning
    assert (cond + target == mask).all()      # nothing observed is lost

def test_each_patient_gets_exactly_floor_ten_percent(masker: EvalMasker):
    mask = random_mask()
    _, target = masker(mask, np.arange(50))
    observed = mask.reshape(50, -1).sum(axis=1)
    assert (target.reshape(50, -1).sum(axis=1) == (observed * 0.1).astype(int)).all()

def test_input_mask_is_not_modified(masker: EvalMasker):
    mask = random_mask()
    before = mask.copy()
    masker(mask, np.arange(50))
    assert (mask == before).all()

def test_mismatched_record_ids_raise(masker: EvalMasker):
    with pytest.raises(ValueError):
        masker(random_mask(n=5), np.arange(4))

def test_memory_layout_does_not_change_the_holdout(masker: EvalMasker):
    mask, ids = random_mask(), np.arange(50)
    cond, target = masker(mask, ids)
    assert target.sum() > 0
    for same_values in (np.asfortranarray(mask), np.repeat(mask, 2, axis=2)[:, :, ::2]):
        assert not same_values.flags["C_CONTIGUOUS"]
        other_cond, other_target = masker(same_values, ids)
        assert (other_target == target).all() and (other_cond == cond).all()

def test_wrong_grid_layout_raises(masker: EvalMasker):
    mask = random_mask(n=5)
    for wrong in (mask.transpose(0, 2, 1), mask[:, :47], mask[0], mask.reshape(5, -1)):
        with pytest.raises(ValueError, match="mask must be"):       # (N, D, T) would hide different values
            masker(wrong, np.arange(len(wrong)))

# --- the frozen-protocol guarantees ----------------------------------------------------

def test_byte_identical_across_run_seeds():
    mask, ids = random_mask(), np.arange(50)
    outputs = []
    for run_seed in CONFIG["protocol"]["run_seeds"]:
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        outputs.append(EvalMasker.from_config()(mask, ids))
    for cond, target in outputs[1:]:
        assert cond.tobytes() == outputs[0][0].tobytes()
        assert target.tobytes() == outputs[0][1].tobytes()

def test_targets_do_not_depend_on_patient_order_or_subset(masker: EvalMasker):
    mask, ids = random_mask(), np.arange(1000, 1050)
    _, full = masker(mask, ids)
    order = np.random.default_rng(1).permutation(50)
    _, shuffled = masker(mask[order], ids[order])
    assert (shuffled == full[order]).all()
    _, subset = masker(mask[10:20], ids[10:20])
    assert (subset == full[10:20]).all()

def test_different_patients_get_different_patterns(masker: EvalMasker):
    mask = np.repeat(random_mask(n=1), 2, axis=0)   # two patients, identical observations
    _, target = masker(mask, np.array([1, 2]))
    assert not (target[0] == target[1]).all()

def test_the_seed_actually_controls_the_holdout():
    mask, ids = random_mask(), np.arange(50)
    _, a = EvalMasker(2012, 0.1)(mask, ids)
    _, b = EvalMasker(2013, 0.1)(mask, ids)
    assert not (a == b).all()

def test_holdout_is_pinned_to_a_known_fingerprint(masker: EvalMasker):
    # Golden value: if this changes, the eval holdout changed and every past result is invalid.
    _, target = masker(random_mask(), np.arange(50))
    assert hashlib.sha256(target.tobytes()).hexdigest() == GOLDEN_SHA256

GOLDEN_SHA256 = "589e52e1f8ccd3d677f50a46f0a6823ebfe2e430c7a884e5895ea40985f5cfbb"

# --- dry run on the real pipeline output -----------------------------------------------

def test_dry_run_on_built_dataset(masker: EvalMasker, small_dataset: dict[str, np.ndarray]):
    for split in ("val_idx", "test_idx"):
        idx = small_dataset[split]
        mask, values = small_dataset["mask"][idx], small_dataset["values"][idx]
        cond, target = masker(mask, small_dataset["record_ids"][idx])
        assert (cond + target == mask).all()
        assert target.sum() == sum(int(m.sum() * 0.1) for m in mask)
        model_input = values * cond                       # what a model would be given
        assert (model_input[target == 1] == 0).all()      # hidden values never reach it
        assert np.isfinite(values[target == 1]).all()     # but are there to score against

@needs_raw_data
def test_set_a_holds_out_about_ten_percent(masker: EvalMasker, set_a: dict[str, np.ndarray]):
    cond, target = masker(set_a["mask"], set_a["record_ids"])
    assert (cond + target == set_a["mask"]).all()
    assert target.sum() / set_a["mask"].sum() == pytest.approx(0.1, abs=0.005)
    assert (target.reshape(len(target), -1).sum(axis=1) > 0).mean() > 0.95
