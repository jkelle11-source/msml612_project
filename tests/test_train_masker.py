import hashlib

import numpy as np
import pytest
import torch
from conftest import CONFIG, RAW_DIR
from eval.masking import EvalMasker, TrainMasker

needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")
ACCELERATOR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
WINDOW = CONFIG["protocol"]["block_window_hours"]
STRATEGIES = {"random": {}, "mix": {"strategy": "mix", "window": WINDOW}}

def random_mask(n=32, seed=0, observed_rate=0.2):
    rng = np.random.default_rng(seed)
    return torch.from_numpy((rng.random((n, 48, 35)) < observed_rate).astype(np.float32))

def drawn_ratios(seed, n):
    """The per-patient ratios a fresh TrainMasker(seed) draws on its first call."""
    return torch.rand(n, 1, generator=torch.Generator().manual_seed(seed)).squeeze(1)

def hidden_per_patient(target):
    return target.reshape(len(target), -1).sum(1)

def batches(dataset, idx, batch_size=8):
    for start in range(0, len(idx), batch_size):
        rows = idx[start:start + batch_size]
        yield torch.from_numpy(dataset["values"][rows]), torch.from_numpy(dataset["mask"][rows])

# --- 1 + 2: fresh every step, but reproducible from the seed ---------------------------

def test_a_different_mask_every_call():
    masker, mask = TrainMasker(42), random_mask()
    targets = [masker(mask)[1] for _ in range(5)]
    for a, b in zip(targets, targets[1:]):
        assert not torch.equal(a, b)

def test_same_seed_gives_the_same_sequence():
    mask = random_mask()
    first, second = TrainMasker(42), TrainMasker(42)
    for _ in range(10):
        cond_a, target_a = first(mask)
        cond_b, target_b = second(mask)
        assert torch.equal(cond_a, cond_b) and torch.equal(target_a, target_b)

def test_different_seeds_give_different_sequences():
    mask = random_mask()
    assert not torch.equal(TrainMasker(42)(mask)[1], TrainMasker(43)(mask)[1])

# --- 3: paired seeds stay paired -------------------------------------------------------

def test_other_random_draws_do_not_shift_the_masks():
    mask = random_mask()
    torch.manual_seed(42)
    plain = TrainMasker(42)
    expected = [plain(mask)[1] for _ in range(5)]

    torch.manual_seed(42)
    busy = TrainMasker(42)
    for step in range(5):
        torch.nn.Linear(35, 128)            # like Ext 1's extra layers drawing init weights
        torch.rand(1000)                    # like dropout / diffusion noise
        np.random.rand(1000)
        assert torch.equal(busy(mask)[1], expected[step])

# --- 4: valid split --------------------------------------------------------------------

def test_targets_are_observed_and_disjoint_from_conditioning():
    mask = random_mask()
    masker = TrainMasker(42)
    for _ in range(20):
        cond, target = masker(mask)
        assert (target <= mask).all()
        assert (cond * target == 0).all()
        assert torch.equal(cond + target, mask)

def test_outputs_keep_shape_dtype_and_values():
    mask = random_mask()
    cond, target = TrainMasker(42)(mask)
    for out in (cond, target):
        assert out.shape == mask.shape
        assert out.dtype == mask.dtype
        assert set(out.unique().tolist()) <= {0.0, 1.0}

def test_input_mask_is_not_modified():
    mask = random_mask()
    before = mask.clone()
    TrainMasker(42)(mask)
    assert torch.equal(mask, before)

@pytest.mark.skipif(ACCELERATOR is None, reason="no GPU available")
def test_gpu_masks_stay_on_gpu_and_match_cpu():
    mask = random_mask()
    cond, target = TrainMasker(42)(mask.to(ACCELERATOR))
    assert cond.device.type == target.device.type == ACCELERATOR
    _, cpu_target = TrainMasker(42)(mask)
    assert torch.equal(target.cpu(), cpu_target)

# --- 5: right count --------------------------------------------------------------------

def test_hidden_count_matches_the_drawn_ratio():
    mask = random_mask()
    _, target = TrainMasker(42)(mask)
    observed = hidden_per_patient(mask)
    assert torch.equal(hidden_per_patient(target), (observed * drawn_ratios(42, 32)).round())

def test_hidden_share_covers_zero_to_one_and_averages_half():
    mask = random_mask(n=1)
    masker = TrainMasker(0)
    shares = torch.stack([hidden_per_patient(masker(mask)[1]) for _ in range(500)]) / mask.sum()
    assert shares.mean().item() == pytest.approx(0.5, abs=0.05)
    assert shares.min().item() < 0.05 and shares.max().item() > 0.95

def test_every_observed_value_is_equally_likely_to_be_hidden():
    # The count tests above can't see a masker that always hides the same values (say, the earliest).
    mask = random_mask(n=1)
    masker, hits, calls = TrainMasker(0), torch.zeros_like(mask), 2000
    for _ in range(calls):
        hits += masker(mask)[1]
    rate = (hits / calls)[mask == 1]
    assert 0.4 < rate.min() and rate.max() < 0.6          # each one is hidden about half the time

# --- 6: no eval leakage ----------------------------------------------------------------

def test_eval_targets_never_reach_training_split(small_dataset):
    idx = small_dataset["val_idx"]
    eval_cond, eval_target = EvalMasker.from_config()(small_dataset["mask"][idx], small_dataset["record_ids"][idx])
    eval_target = torch.from_numpy(eval_target)
    masker = TrainMasker(42)
    for _ in range(20):
        cond, target = masker(torch.from_numpy(eval_cond))
        assert (cond * eval_target == 0).all()      # not visible to the model
        assert (target * eval_target == 0).all()    # not trained on either

# --- 7: small example you can check by hand --------------------------------------------

def test_tiny_fixture_by_hand():
    mask = torch.zeros(2, 48, 35)
    mask[0, :10, 0] = 1                     # 10 observed; patient 1 has nothing
    r = drawn_ratios(7, 2)
    cond, target = TrainMasker(7)(mask)
    assert hidden_per_patient(target).tolist() == [round(10 * r[0].item()), 0]
    assert (target[0, 10:] == 0).all() and (target[0, :, 1:] == 0).all()
    assert (cond[1] == 0).all() and (target[1] == 0).all()

# --- 8 + 9: dry run like the training loop will use it ---------------------------------

def run_training_epochs(dataset, epochs=2, **strategy):
    masker = TrainMasker(42, **strategy)
    seen = {}
    for epoch in range(epochs):
        for step, (values, mask) in enumerate(batches(dataset, dataset["train_idx"])):
            cond, target = masker(mask)
            model_input = values * cond
            assert torch.equal(cond + target, mask)
            assert (model_input[target == 1] == 0).all()                 # targets hidden from the model
            assert torch.isfinite(values[target == 1]).all()              # but there to compute the loss
            if epoch == 1:
                assert not torch.equal(target, seen[step])                # same batch, new mask next epoch
            seen[step] = target
    return seen

@pytest.mark.parametrize("strategy", STRATEGIES.values(), ids=list(STRATEGIES))
def test_dry_run_on_built_dataset(small_dataset, strategy):
    first = run_training_epochs(small_dataset, **strategy)
    again = run_training_epochs(small_dataset, **strategy)
    for step in first:
        assert torch.equal(first[step], again[step])                      # whole run is reproducible

@needs_raw_data
@pytest.mark.parametrize("strategy", STRATEGIES.values(), ids=list(STRATEGIES))
def test_dry_run_on_set_a(set_a, strategy):
    run_training_epochs(set_a, epochs=1, **strategy)

# --- 10: the mix strategy (M6-2, the block-aware regime) -------------------------------

# 5 calls on random_mask(), SHA-256 over the targets' bytes. RANDOM's is from the code before mix
# existed. MIX's is from a replay of the docstring's reads on a generator of our own (block_targets
# for the tails patients), not from TrainMasker. The reads are part of the protocol: changing their
# order, the coin or the seed's range changes every block-aware run's masks.
RANDOM_FINGERPRINT = "cf6654e6d16cb4b39b38eb7f79bfd9343092363115fc39590d0dd17befa3b898"
MIX_FINGERPRINT = "54b04302d7ccb9925a58b7a851dc623053066d3cfdbb1ff26425563c01bec1e4"

def mix_masker(seed):
    return TrainMasker(seed, "mix", WINDOW)

def fingerprint(masker, mask, calls=5):
    digest = hashlib.sha256()
    for _ in range(calls):
        digest.update(masker(mask)[1].numpy().tobytes())
    return digest.hexdigest()

def mix_tails(seed, mask, calls):
    """Which patients each of a fresh mix_masker(seed)'s first ``calls`` calls sends to block
    targets, replayed in the documented order: ratios, scores, coins, one seed per tails patient."""
    generator = torch.Generator().manual_seed(seed)
    tails = []
    for _ in range(calls):
        torch.rand(len(mask), 1, generator=generator)
        torch.rand(len(mask), mask[0].numel(), generator=generator)
        tails.append(torch.rand(len(mask), generator=generator) < 0.5)
        for _ in range(int(tails[-1].sum())):
            torch.randint(2**62, (1,), generator=generator)
    return tails

def per_slot(x):
    """Sum of ``x`` (B, T, D) in each slot, hours s*WINDOW to (s+1)*WINDOW across all features."""
    B, T, D = x.shape
    return x.reshape(B, T // WINDOW, WINDOW * D).sum(-1)

def test_random_strategy_is_unchanged_by_mix():
    mask = random_mask()
    for masker in (TrainMasker(42), TrainMasker(42, "random"), TrainMasker(42, "random", WINDOW)):
        assert fingerprint(masker, mask) == RANDOM_FINGERPRINT

def test_mix_stream_is_frozen():
    assert fingerprint(mix_masker(42), random_mask()) == MIX_FINGERPRINT

def test_mix_tiny_example_by_hand():
    # Two identical patients on a 6-hour x 2-feature grid; a 3-hour window makes 2 slots of 6 cells.
    # Each has 9 observed values: 4 in slot 0 (hours 0-2), 5 in slot 1 (hours 3-5).
    patient = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1], [1, 1], [0, 1]], dtype=torch.float32)
    mask = torch.stack([patient, patient])
    # TrainMasker(0)'s reads: ratios 0.4963, 0.7682, so round(r * 9) = 4 and 7 to hide; then coins
    # 0.9527, 0.0362, so patient 0 is heads (random's split) and patient 1 tails (block targets).
    assert drawn_ratios(0, 2).tolist() == pytest.approx([0.4963, 0.7682], abs=1e-4)
    cond, target = TrainMasker(0, "mix", window=3)(mask)

    def hidden(t):
        return sorted(map(tuple, t.nonzero().tolist()))                  # (hour, feature)

    assert hidden(target[0]) == [(2, 1), (3, 0), (3, 1), (4, 1)]
    assert hidden(target[0]) == hidden(TrainMasker(0)(mask)[1][0])      # random's split
    # slot 0 comes first and its 4 fit, so it's hidden whole; slot 1's 5 are more than the 3 left,
    # so it's trimmed to 3 and the walk stops
    assert hidden(target[1]) == [(0, 0), (0, 1), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1)]
    assert torch.equal(cond + target, mask)

def test_mix_heads_keep_the_random_split_and_tails_keep_its_count():
    mask = random_mask()
    _, mixed = mix_masker(42)(mask)
    _, plain = TrainMasker(42)(mask)                    # a first call reads the same ratios and scores
    tails = mix_tails(42, mask, calls=1)[0]
    assert 0 < tails.sum() < len(mask)
    assert torch.equal(mixed[~tails], plain[~tails])
    assert not torch.equal(mixed[tails], plain[tails])
    assert torch.equal(hidden_per_patient(mixed), hidden_per_patient(plain))

def test_mix_tails_hide_whole_slots():
    mask, masker = random_mask(), mix_masker(42)
    for tails in mix_tails(42, mask, calls=10):
        _, target = masker(mask)
        hidden, observed = per_slot(target[tails]), per_slot(mask[tails])
        partial = (hidden > 0) & (hidden < observed)
        assert (partial.sum(1) <= 1).all()              # each slot untouched or hidden whole, bar one
        assert (hidden.sum(1) > 0).any()

def test_mix_coin_is_fair():
    # Read from the masks, not a replay: on a first call a heads patient gets random's split exactly,
    # and a tails patient a different one (bar the rare patient who hides none or all of theirs).
    mask = random_mask()
    changed = [(mix_masker(seed)(mask)[1] != TrainMasker(seed)(mask)[1]).flatten(1).any(1)
               for seed in range(50)]
    assert 0.45 < torch.cat(changed).float().mean().item() < 0.55          # of 1,600 patients

def test_mix_other_random_draws_do_not_shift_the_masks():
    mask = random_mask()
    torch.manual_seed(42)
    np.random.seed(42)
    plain = mix_masker(42)
    expected = [plain(mask)[1] for _ in range(5)]

    torch.manual_seed(42)
    np.random.seed(42)
    busy = mix_masker(42)
    for step in range(5):
        torch.nn.Linear(35, 128)            # like Ext 1's extra layers drawing init weights
        torch.rand(1000)                    # like dropout / diffusion noise
        np.random.rand(1000)                # block_targets draws from NumPy, but never the global stream
        assert torch.equal(busy(mask)[1], expected[step])

def test_mix_split_is_valid():
    mask = random_mask()
    mask[0] = 0                                         # a patient with nothing observed hides nothing
    before = mask.clone()
    assert any(tails[0] for tails in mix_tails(42, mask, calls=20))       # ... even on tails
    masker = mix_masker(42)
    for _ in range(20):
        cond, target = masker(mask)
        assert cond.shape == target.shape == mask.shape and cond.dtype == target.dtype == mask.dtype
        assert set(target.unique().tolist()) <= {0.0, 1.0}
        assert (target <= mask).all() and (cond * target == 0).all()
        assert torch.equal(cond + target, mask)
        assert (target[0] == 0).all()
    assert torch.equal(mask, before)

@pytest.mark.skipif(ACCELERATOR is None, reason="no GPU available")
def test_mix_gpu_masks_stay_on_gpu_and_match_cpu():
    mask = random_mask()
    cond, target = mix_masker(42)(mask.to(ACCELERATOR))
    assert cond.device.type == target.device.type == ACCELERATOR
    assert torch.equal(target.cpu(), mix_masker(42)(mask)[1])

def test_bad_strategy_arguments_raise():
    with pytest.raises(ValueError, match="strategy must be one of"):
        TrainMasker(42, "block")
    with pytest.raises(ValueError, match="needs the block window"):
        TrainMasker(42, "mix")
