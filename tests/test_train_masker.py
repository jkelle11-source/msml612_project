import numpy as np
import pytest
import torch
from conftest import RAW_DIR
from eval.masking import EvalMasker, TrainMasker

needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")
ACCELERATOR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None

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

def run_training_epochs(dataset, epochs=2):
    masker = TrainMasker(42)
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

def test_dry_run_on_built_dataset(small_dataset):
    first = run_training_epochs(small_dataset)
    again = run_training_epochs(small_dataset)
    for step in first:
        assert torch.equal(first[step], again[step])                      # whole run is reproducible

@needs_raw_data
def test_dry_run_on_set_a(set_a):
    run_training_epochs(set_a, epochs=1)
