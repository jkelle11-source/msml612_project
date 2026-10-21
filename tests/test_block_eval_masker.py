import hashlib

import numpy as np
import pytest
import torch
from conftest import CONFIG, RAW_DIR
from eval.masking import BlockEvalMasker, EvalMasker, block_targets

needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")

GRID = (CONFIG["data"]["n_timesteps"], CONFIG["data"]["n_features"])
WINDOW = CONFIG["protocol"]["block_window_hours"]

def random_patient(seed=0, observed_rate=0.2, grid=GRID):
    """One patient's (T, D) mask; Set A's median patient has about 20% of the grid observed."""
    rng = np.random.default_rng(seed)
    return (rng.random(grid) < observed_rate).astype(np.float32)

def random_mask(n=50, seed=0, observed_rate=0.2):
    """(N, T, D) masks; the defaults are the fixture ``EvalMasker``'s golden fingerprint uses."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, *GRID)) < observed_rate).astype(np.float32)

def per_slot(x, window=WINDOW):
    """Sum of ``x`` ((T, D) or (N, T, D)) in each slot: hours s*window to (s+1)*window, every feature."""
    *patients, T, D = x.shape
    return x.reshape(*patients, T // window, window * D).sum(axis=-1)

@pytest.fixture(scope="module")
def masker() -> BlockEvalMasker:
    return BlockEvalMasker.from_config()

# --- config ---------------------------------------------------------------------------

def test_block_protocol_constants():
    protocol = CONFIG["protocol"]
    assert protocol["block_window_hours"] == 3
    assert GRID[0] % protocol["block_window_hours"] == 0         # 16 fixed slots
    other_seeds = {protocol["eval_masker_seed"], protocol["eval_sampling_seed"],
                   CONFIG["data"]["split_seed"], *protocol["run_seeds"]}
    assert protocol["block_eval_masker_seed"] == 8080
    assert protocol["block_eval_masker_seed"] not in other_seeds

def test_block_masker_imports_from_the_package():
    import eval                                 # the README names it as eval.BlockEvalMasker
    assert eval.BlockEvalMasker is BlockEvalMasker
    assert eval.block_targets is block_targets
    assert {"BlockEvalMasker", "block_targets"} <= set(eval.__all__)

# --- block_targets: hand-checkable fixture ---------------------------------------------

TOY = np.array([[1, 0], [1, 1], [0, 0],      # slot 0 (hours 0-2): 3 observed
                [1, 1], [0, 1], [1, 0]],     # slot 1 (hours 3-5): 4 observed
               dtype=np.float32)

@pytest.mark.parametrize("seed, order, hidden_per_slot", [
    (0, [0, 1], [3, 2]),      # slot 0 fits whole (3 <= 5); slot 1 trimmed to the 2 left
    (3, [1, 0], [1, 4]),      # slot 1 fits whole (4 <= 5); slot 0 trimmed to the 1 left
])
def test_toy_patient_by_hand(seed, order, hidden_per_slot):
    assert np.random.default_rng(seed).permutation(2).tolist() == order     # the shuffle this seed gives
    target = block_targets(TOY, 5, np.random.default_rng(seed), 3)
    assert per_slot(target, 3).tolist() == hidden_per_slot
    assert (target <= TOY).all()

def test_count_zero_hides_nothing_and_count_all_hides_everything():
    mask = random_patient()
    assert not block_targets(mask, 0, np.random.default_rng(0), WINDOW).any()
    assert (block_targets(mask, int(mask.sum()), np.random.default_rng(0), WINDOW) == mask).all()
    nothing = np.zeros(GRID, dtype=np.float32)
    assert not block_targets(nothing, 0, np.random.default_rng(0), WINDOW).any()

def test_asking_for_more_than_is_observed_raises():
    assert block_targets(TOY, 7, np.random.default_rng(0), 3).sum() == 7       # all 7 observed
    with pytest.raises(ValueError, match="observed"):
        block_targets(TOY, 8, np.random.default_rng(0), 3)

# --- block_targets: shape and correctness ----------------------------------------------

@pytest.mark.parametrize("window", [1, WINDOW, 4, GRID[0]])
@pytest.mark.parametrize("observed_rate", [0.02, 0.2, 0.8])
def test_hides_exactly_count_observed_values(window, observed_rate):
    for seed in range(10):
        mask = random_patient(seed, observed_rate)
        n_observed = int(mask.sum())
        for count in (0, 1, n_observed // 10, n_observed // 2, n_observed):
            target = block_targets(mask, count, np.random.default_rng(seed), window)
            assert target.shape == mask.shape and target.dtype == mask.dtype
            assert set(np.unique(target)) <= {0.0, 1.0}
            assert (target <= mask).all()                # only values we actually have
            assert target.sum() == count

def test_every_slot_is_hidden_whole_or_untouched_apart_from_one():
    for seed in range(50):
        mask = random_patient(seed)
        target = block_targets(mask, int(mask.sum() * 0.1), np.random.default_rng(seed), WINDOW)
        observed, hidden = per_slot(mask, WINDOW), per_slot(target, WINDOW)
        assert ((hidden == 0) | (hidden == observed)).sum() >= len(observed) - 1

@pytest.mark.parametrize("seed", range(30))
def test_walks_the_shuffled_slots_and_stops_at_the_first_that_does_not_fit(seed):
    # Sparse patients with any count: after the trimmed slot, later slots that would still fit
    # must hide nothing (the walk stops; it does not skip ahead to a smaller slot).
    mask = random_patient(seed, observed_rate=0.1)
    count = int(np.random.default_rng(seed + 1000).integers(0, mask.sum() + 1))
    target = block_targets(mask, count, np.random.default_rng(seed), WINDOW)
    observed, hidden = per_slot(mask, WINDOW), per_slot(target, WINDOW)
    left = count
    for s in np.random.default_rng(seed).permutation(len(observed)):        # the same shuffle
        expected = observed[s] if observed[s] <= left else left
        assert hidden[s] == expected, f"slot {s}"
        left -= expected

def test_draws_one_permutation_then_at_most_one_choice():
    # The draw order is part of the frozen holdout. A slot that exactly fills the count is
    # hidden whole, with no draw; only a trimmed slot draws, once, over its observed values.
    mask = random_patient()
    n_slots = GRID[0] // WINDOW
    order = np.random.default_rng(0).permutation(n_slots)
    observed = per_slot(mask, WINDOW).astype(int)
    exact_fills = (0, observed[order[0]], observed[order[:3]].sum(), observed.sum())
    trimmed = (observed[order[:3]].sum() + 1, observed[order[0]] - 1)
    for count in (*exact_fills, *trimmed):
        used, twin = np.random.default_rng(0), np.random.default_rng(0)
        block_targets(mask, int(count), used, WINDOW)
        twin.permutation(n_slots)
        if count in trimmed:
            done = np.cumsum(observed[order]) <= count
            misfit = order[done.sum()]
            twin.choice(observed[misfit], int(count - observed[order[:done.sum()]].sum()), replace=False)
        assert used.random() == twin.random(), f"count {count}"

def test_returns_a_fresh_array_and_leaves_the_mask_alone():
    mask = random_patient()
    before = mask.copy()
    target = block_targets(mask, int(mask.sum()) // 10, np.random.default_rng(0), WINDOW)
    assert (mask == before).all()
    assert not np.shares_memory(target, mask)

def test_memory_layout_does_not_change_the_targets():
    mask = random_patient()
    count = int(mask.sum()) // 10
    target = block_targets(mask, count, np.random.default_rng(0), WINDOW)
    assert target.sum() > 0
    for same_values in (np.asfortranarray(mask), np.repeat(mask, 2, axis=1)[:, ::2]):
        assert not same_values.flags["C_CONTIGUOUS"]
        assert (block_targets(same_values, count, np.random.default_rng(0), WINDOW) == target).all()

@pytest.mark.parametrize("window", [5, 7, GRID[0] + 1])
def test_window_that_does_not_divide_the_hours_raises(window):
    with pytest.raises(ValueError, match="must divide"):
        block_targets(random_patient(), 10, np.random.default_rng(0), window)

# --- block_targets: randomness ---------------------------------------------------------

def test_randomness_comes_only_from_the_generator():
    mask = random_patient()
    count = int(mask.sum()) // 10
    outputs = []
    for run_seed in CONFIG["protocol"]["run_seeds"]:
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        outputs.append(block_targets(mask, count, np.random.default_rng(8080), WINDOW))
    assert all((out == outputs[0]).all() for out in outputs)

def test_the_generator_controls_the_targets():
    mask = random_patient()
    count = int(mask.sum()) // 10
    a = block_targets(mask, count, np.random.default_rng(1), WINDOW)
    b = block_targets(mask, count, np.random.default_rng(2), WINDOW)
    assert not (a == b).all()

# --- BlockEvalMasker: config ------------------------------------------------------------

def test_from_config_reads_the_frozen_protocol(masker: BlockEvalMasker):
    assert masker.seed == CONFIG["protocol"]["block_eval_masker_seed"] == 8080
    assert masker.fraction == CONFIG["protocol"]["eval_holdout_fraction"] == 0.10    # shared with EvalMasker
    assert masker.window == CONFIG["protocol"]["block_window_hours"] == 3
    assert masker.grid == GRID == (48, 35)

# --- BlockEvalMasker: hand-checkable fixtures ---------------------------------------------

def test_tiny_fixture_hides_floor_ten_percent():
    mask = np.zeros((3, *GRID), dtype=np.float32)
    mask[0, :25, 0] = 1        # 25 observed -> 2 hidden
    mask[1, :9, 3] = 1         #  9 observed -> 0 hidden (floor)
    # patient 2 has nothing observed -> 0 hidden, no crash
    cond, target = BlockEvalMasker(8080, 0.1, 3)(mask, np.array([1, 2, 3]))
    assert target.reshape(3, -1).sum(axis=1).tolist() == [2, 0, 0]
    assert (target[0, 25:] == 0).all() and (target[0, :, 1:] == 0).all()
    assert (cond[1] == mask[1]).all() and (cond[2] == 0).all()

def test_one_value_an_hour_loses_one_whole_slot_and_one_value_of_the_next():
    mask = np.zeros((1, *GRID), dtype=np.float32)
    mask[0, :, 0] = 1          # 48 observed, 3 in every slot -> 4 hidden: a whole slot, then 1 trimmed
    _, target = BlockEvalMasker(8080, 0.1, 3)(mask, np.array([7]))
    assert sorted(per_slot(target[0]).tolist()) == [0] * 14 + [1, 3]

# --- BlockEvalMasker: shape and correctness -----------------------------------------------

def test_outputs_keep_shape_and_dtype(masker: BlockEvalMasker):
    mask = random_mask()
    cond, target = masker(mask, np.arange(50))
    for out in (cond, target):
        assert out.shape == mask.shape
        assert out.dtype == mask.dtype
        assert set(np.unique(out)) <= {0.0, 1.0}

def test_targets_are_observed_and_disjoint_from_conditioning(masker: BlockEvalMasker):
    mask = random_mask()
    cond, target = masker(mask, np.arange(50))
    assert (target <= mask).all()             # only hide values we actually have
    assert (cond * target == 0).all()         # a target is never also conditioning
    assert (cond + target == mask).all()      # nothing observed is lost

def test_each_patient_loses_exactly_eval_maskers_count(masker: BlockEvalMasker):
    mask, ids = random_mask(), np.arange(50)
    _, target = masker(mask, ids)
    _, random_target = EvalMasker.from_config()(mask, ids)
    counts = target.reshape(50, -1).sum(axis=1)
    assert (counts == random_target.reshape(50, -1).sum(axis=1)).all()
    assert (counts == (mask.reshape(50, -1).sum(axis=1) * 0.1).astype(int)).all()
    assert not (target == random_target).all()        # same counts, different places

def test_targets_fill_whole_slots_apart_from_one_trimmed_slot(masker: BlockEvalMasker):
    mask = random_mask()
    _, target = masker(mask, np.arange(50))
    observed, hidden = per_slot(mask), per_slot(target)
    assert (((hidden > 0) & (hidden < observed)).sum(axis=1) <= 1).all()
    assert ((hidden == observed) & (observed > 0)).any(axis=1).all()    # every patient here loses a whole slot

def test_each_patient_is_block_targets_with_their_own_generator(masker: BlockEvalMasker):
    # Pins the seeding: one generator per patient, seeded from (seed, record_id).
    mask, ids = random_mask(), np.arange(1000, 1050)
    _, target = masker(mask, ids)
    for i, record_id in enumerate(ids):
        count = int(np.count_nonzero(mask[i]) * 0.1)
        rng = np.random.default_rng([8080, record_id])
        assert (target[i] == block_targets(mask[i], count, rng, WINDOW)).all()

def test_input_mask_is_not_modified(masker: BlockEvalMasker):
    mask = random_mask()
    before = mask.copy()
    masker(mask, np.arange(50))
    assert (mask == before).all()

def test_mismatched_record_ids_raise(masker: BlockEvalMasker):
    with pytest.raises(ValueError):
        masker(random_mask(n=5), np.arange(4))

def test_memory_layout_does_not_change_the_holdout(masker: BlockEvalMasker):
    mask, ids = random_mask(), np.arange(50)
    cond, target = masker(mask, ids)
    assert target.sum() > 0
    for same_values in (np.asfortranarray(mask), np.repeat(mask, 2, axis=2)[:, :, ::2]):
        assert not same_values.flags["C_CONTIGUOUS"]
        other_cond, other_target = masker(same_values, ids)
        assert (other_target == target).all() and (other_cond == cond).all()

def test_wrong_grid_layout_raises(masker: BlockEvalMasker):
    mask = random_mask(n=5)
    for wrong in (mask.transpose(0, 2, 1), mask[:, :47], mask[0], mask.reshape(5, -1)):
        with pytest.raises(ValueError, match="mask must be"):       # (N, D, T) would cut slots across features
            masker(wrong, np.arange(len(wrong)))

# --- BlockEvalMasker: the frozen-protocol guarantees --------------------------------------

def test_byte_identical_across_run_seeds():
    mask, ids = random_mask(), np.arange(50)
    outputs = []
    for run_seed in CONFIG["protocol"]["run_seeds"]:
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        outputs.append(BlockEvalMasker.from_config()(mask, ids))
    for cond, target in outputs[1:]:
        assert cond.tobytes() == outputs[0][0].tobytes()
        assert target.tobytes() == outputs[0][1].tobytes()

def test_targets_do_not_depend_on_patient_order_or_subset(masker: BlockEvalMasker):
    mask, ids = random_mask(), np.arange(1000, 1050)
    _, full = masker(mask, ids)
    order = np.random.default_rng(1).permutation(50)
    _, shuffled = masker(mask[order], ids[order])
    assert (shuffled == full[order]).all()
    _, subset = masker(mask[10:20], ids[10:20])
    assert (subset == full[10:20]).all()

def test_different_patients_get_different_patterns(masker: BlockEvalMasker):
    mask = np.repeat(random_mask(n=1), 2, axis=0)   # two patients, identical observations
    _, target = masker(mask, np.array([1, 2]))
    assert not (target[0] == target[1]).all()

def test_the_seed_actually_controls_the_holdout():
    mask, ids = random_mask(), np.arange(50)
    _, a = BlockEvalMasker(8080, 0.1, 3)(mask, ids)
    _, b = BlockEvalMasker(8081, 0.1, 3)(mask, ids)
    assert not (a == b).all()

def test_holdout_is_pinned_to_a_known_fingerprint(masker: BlockEvalMasker):
    # Golden value: if this changes, the block holdout changed and every past block result is invalid.
    _, target = masker(random_mask(), np.arange(50))
    assert hashlib.sha256(target.tobytes()).hexdigest() == GOLDEN_SHA256

GOLDEN_SHA256 = "b75f346f4dac4fcd4e71c4c4f24d72f3c0990d8543ca1e5fd553700622ee20d6"

# --- BlockEvalMasker: dry run on the real pipeline output ---------------------------------

def test_dry_run_on_built_dataset(masker: BlockEvalMasker, small_dataset: dict[str, np.ndarray]):
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
def test_set_a_holds_out_eval_maskers_ten_percent(masker: BlockEvalMasker, set_a: dict[str, np.ndarray]):
    cond, target = masker(set_a["mask"], set_a["record_ids"])
    _, random_target = EvalMasker.from_config()(set_a["mask"], set_a["record_ids"])
    assert (cond + target == set_a["mask"]).all()
    assert target.sum() == random_target.sum()
    assert target.sum() / set_a["mask"].sum() == pytest.approx(0.1, abs=0.005)

@needs_raw_data
def test_set_a_most_patients_lose_a_whole_window(masker: BlockEvalMasker, set_a: dict[str, np.ndarray]):
    # M2-4's case for 3 hours: most patients lose at least one complete window before the trim.
    _, target = masker(set_a["mask"], set_a["record_ids"])
    observed, hidden = per_slot(set_a["mask"]), per_slot(target)
    assert ((hidden == observed) & (observed > 0)).any(axis=1).mean() == pytest.approx(0.94, abs=0.005)
    assert (((hidden > 0) & (hidden < observed)).sum(axis=1) <= 1).all()
