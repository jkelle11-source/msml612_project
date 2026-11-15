import hashlib

import numpy as np
import pytest
import torch
from conftest import CONFIG, RAW_DIR
from eval.masking import BlockEvalMasker, EvalMasker, OffGridBlockEvalMasker, block_targets

needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")

GRID = (CONFIG["data"]["n_timesteps"], CONFIG["data"]["n_features"])
WINDOW = CONFIG["protocol"]["block_window_hours"]
OFFSET = CONFIG["protocol"]["offgrid_block_offset_hours"]
SLOTS = range(OFFSET, GRID[0] - WINDOW + 1, WINDOW)     # the off-grid windows' first hours: 1, 4, ..., 43

def random_mask(n=50, seed=0, observed_rate=0.2):
    """(N, T, D) masks; the defaults are the ones the block holdout's golden fingerprint uses."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, *GRID)) < observed_rate).astype(np.float32)

def per_window(x):
    """Sum of ``x`` ((N, T, D)) in each off-grid window: (N, 15), hours 1-3, 4-6, ..., 43-45."""
    return np.stack([x[:, h:h + WINDOW].sum(axis=(1, 2)) for h in SLOTS], axis=1)

def per_grid_slot(x):
    """Sum of ``x`` ((N, T, D)) in each of the grid's 16 slots, hours 0-2, ..., 45-47."""
    return x.reshape(len(x), GRID[0] // WINDOW, -1).sum(axis=-1)

def counts(target):
    return target.reshape(len(target), -1).sum(axis=1)

@pytest.fixture(scope="module")
def masker() -> OffGridBlockEvalMasker:
    return OffGridBlockEvalMasker.from_config()

# --- config ---------------------------------------------------------------------------

def test_offgrid_protocol_constants():
    protocol = CONFIG["protocol"]
    assert protocol["offgrid_block_eval_masker_seed"] == 9090 and protocol["offgrid_block_offset_hours"] == 1
    other_seeds = {protocol["eval_masker_seed"], protocol["block_eval_masker_seed"], protocol["eval_sampling_seed"],
                   CONFIG["data"]["split_seed"], *protocol["run_seeds"]}
    assert protocol["offgrid_block_eval_masker_seed"] not in other_seeds
    assert 0 < OFFSET < WINDOW and len(SLOTS) == 15                 # off the grid, one window fewer

def test_from_config_reads_the_frozen_protocol(masker):
    protocol = CONFIG["protocol"]
    assert (masker.seed, masker.fraction, masker.window, masker.offset, masker.grid) == (
        protocol["offgrid_block_eval_masker_seed"], protocol["eval_holdout_fraction"],
        protocol["block_window_hours"], protocol["offgrid_block_offset_hours"], GRID)

def test_offgrid_masker_imports_from_the_package():
    import eval
    assert eval.OffGridBlockEvalMasker is OffGridBlockEvalMasker and "OffGridBlockEvalMasker" in eval.__all__

# --- block_targets with an offset ------------------------------------------------------

def test_offset_zero_is_the_grid_draw_for_draw():
    # BlockEvalMasker's holdout is block_targets at offset 0; its golden fingerprint is pinned in
    # tests/test_block_eval_masker.py, and the class itself still hides on the grid.
    assert BlockEvalMasker.offset == 0 and BlockEvalMasker.from_config().offset == 0
    for seed in range(20):
        mask = random_mask(1, seed)[0]
        count = int(mask.sum() * 0.1)
        assert np.array_equal(block_targets(mask, count, np.random.default_rng(seed), WINDOW),
                              block_targets(mask, count, np.random.default_rng(seed), WINDOW, 0))

def test_toy_patient_by_hand():
    toy = np.array([[1, 1], [1, 0], [0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.float32)   # hours 0-5
    # 6 hours, window 3, offset 1: one slot, hours 1-3 (3 observed). Hour 0 and hours 4-5 are never hidden.
    target = block_targets(toy, 3, np.random.default_rng(0), 3, 1)
    assert target.tolist() == [[0, 0], [1, 0], [0, 1], [1, 0], [0, 0], [0, 0]]
    with pytest.raises(ValueError, match="greater than 3 observed values in hours 1-3"):
        block_targets(toy, 4, np.random.default_rng(0), 3, 1)    # 7 observed, but only 3 in the slot

@pytest.mark.parametrize("offset", [-1, WINDOW, WINDOW + 1])
def test_offset_outside_the_window_is_refused(offset):
    with pytest.raises(ValueError, match="offset"):
        block_targets(random_mask(1)[0], 1, np.random.default_rng(0), WINDOW, offset)

# --- the off-grid holdout --------------------------------------------------------------

def test_each_patient_loses_block_eval_maskers_count(masker):
    mask = random_mask()
    cond, target = masker(mask, np.arange(50))
    _, block_target = BlockEvalMasker.from_config()(mask, np.arange(50))
    assert (counts(target) == counts(block_target)).all()
    assert (counts(target) == (counts(mask) * 0.1).astype(int)).all()
    assert (cond + target == mask).all() and (target <= mask).all()
    assert not np.array_equal(target, block_target)                 # the same count, elsewhere

def test_no_hidden_window_starts_on_a_multiple_of_three_hours(masker):
    mask = random_mask()
    _, target = masker(mask, np.arange(50))
    assert not target[:, [0, GRID[0] - 2, GRID[0] - 1]].any()       # hours 0, 46, 47: in no window
    # Every window is hidden whole or not at all, apart from one trimmed one, and the windows
    # start at hours 1, 4, ..., 43, so no hidden window starts on a multiple of 3.
    observed, hidden = per_window(mask), per_window(target)
    assert (((hidden > 0) & (hidden < observed)).sum(axis=1) <= 1).all()
    assert all(h % WINDOW != 0 for h in SLOTS)
    assert ((hidden == observed) & (observed > 0)).any(axis=1).all()    # each patient loses a whole one

def test_the_windows_straddle_the_grids_slots(masker):
    # On the grid's tiling the same targets are not whole slots: a hidden window spans two of them.
    mask = random_mask()
    _, target = masker(mask, np.arange(50))
    observed, hidden = per_grid_slot(mask), per_grid_slot(target)
    assert (((hidden > 0) & (hidden < observed)).sum(axis=1) >= 2).all()

def test_targets_are_identical_across_run_seeds_and_runs(masker):
    mask, ids = random_mask(), np.arange(50)
    outputs = []
    for run_seed in (42, 43, 44, 45, 46):                            # global RNGs moved: no effect
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        outputs.append(OffGridBlockEvalMasker.from_config()(mask, ids)[1])
    assert all(np.array_equal(outputs[0], out) for out in outputs[1:])

def test_targets_do_not_depend_on_patient_order_or_subset(masker):
    mask, ids = random_mask(), np.arange(50)
    _, whole = masker(mask, ids)
    order = np.random.default_rng(1).permutation(50)
    _, shuffled = masker(mask[order], ids[order])
    assert np.array_equal(shuffled, whole[order])
    _, some = masker(mask[10:20], ids[10:20])
    assert np.array_equal(some, whole[10:20])

def test_the_seed_and_the_offset_control_the_holdout():
    mask, ids = random_mask(), np.arange(50)
    _, a = OffGridBlockEvalMasker(9090, 0.1, 3, 1)(mask, ids)
    assert not np.array_equal(a, OffGridBlockEvalMasker(9091, 0.1, 3, 1)(mask, ids)[1])
    assert not np.array_equal(a, OffGridBlockEvalMasker(9090, 0.1, 3, 2)(mask, ids)[1])

def test_wrong_grid_layout_and_mismatched_ids_raise(masker):
    mask = random_mask(5)
    with pytest.raises(ValueError):
        masker(mask.transpose(0, 2, 1), np.arange(5))
    with pytest.raises(ValueError):
        masker(mask, np.arange(4))

def test_holdout_is_pinned_to_a_known_fingerprint(masker):
    # Golden value: if this changes, the off-grid holdout changed and every off-grid result is invalid.
    _, target = masker(random_mask(), np.arange(50))
    assert hashlib.sha256(target.tobytes()).hexdigest() == GOLDEN_SHA256

GOLDEN_SHA256 = "3e01bc7a40d00c9031097a06e6100612f00cd5400733f80f4f0a8b959a21ae69"

# --- on the real pipeline output ----------------------------------------------------------

def test_dry_run_on_built_dataset(masker, small_dataset):
    for split in ("val_idx", "test_idx"):
        idx = small_dataset[split]
        mask = small_dataset["mask"][idx]
        cond, target = masker(mask, small_dataset["record_ids"][idx])
        _, block_target = BlockEvalMasker.from_config()(mask, small_dataset["record_ids"][idx])
        assert (cond + target == mask).all() and (counts(target) == counts(block_target)).all()

@needs_raw_data
def test_set_a_loses_block_eval_maskers_count_off_the_grid(masker, set_a):
    mask, ids = set_a["mask"], set_a["record_ids"]
    cond, target = masker(mask, ids)
    _, block_target = BlockEvalMasker.from_config()(mask, ids)
    _, random_target = EvalMasker.from_config()(mask, ids)
    assert (counts(target) == counts(block_target)).all() and target.sum() == random_target.sum()
    assert (cond + target == mask).all()
    assert not target[:, [0, 46, 47]].any()
    observed, hidden = per_window(mask), per_window(target)
    assert (((hidden > 0) & (hidden < observed)).sum(axis=1) <= 1).all()
    # Nearly as on the grid (94%, tests/test_block_eval_masker.py), most patients lose a whole window.
    assert ((hidden == observed) & (observed > 0)).any(axis=1).mean() == pytest.approx(0.9345, abs=0.005)
