import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from conftest import RAW_DIR, ROOT
from data.physionet import invert_scaler
from eval import BlockEvalMasker, EvalMasker, masked_errors, masked_mae
from eval.baselines import RESULTS_PATH, baseline_results, forward_fill, mean_imputation

needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")

def random_values(n=6, seed=0):
    rng = np.random.default_rng(seed)
    mask = (rng.random((n, 48, 35)) < 0.2).astype(np.float32)
    return (rng.normal(size=(n, 48, 35)) * mask).astype(np.float32), mask

def raw_mean_per_feature(dataset, idx):
    """Each feature's mean in raw units over the observed values of patients ``idx``."""
    raw = invert_scaler(dataset["values"][idx], dataset["mean"], dataset["std"])
    return np.nanmean(np.where(dataset["mask"][idx] == 1, raw, np.nan), axis=(0, 1))

def score_mean_imputation(dataset, idx):
    mask, values = dataset["mask"][idx], dataset["values"][idx]
    _, target = EvalMasker.from_config()(mask, dataset["record_ids"][idx])
    return masked_mae(mean_imputation(values), values, target), values, target

# --- D1: mean imputation ----------------------------------------------------------------

def test_mean_imputation_is_a_zero_grid_of_the_same_shape():
    values, _ = random_values()
    guess = mean_imputation(values)
    assert guess.shape == (6, 48, 35)
    assert guess.dtype == np.float32
    assert (guess == 0).all()

def test_mean_imputation_score_by_hand():
    values, target = np.zeros((1, 48, 35)), np.zeros((1, 48, 35))
    for hour, truth in zip([0, 5, 47], [1.0, -2.0, 3.0]):
        values[0, hour, 2] = truth
        target[0, hour, 2] = 1
    assert masked_mae(mean_imputation(values), values, target) == 2.0     # (1 + 2 + 3) / 3

def test_zero_is_the_training_mean_not_the_test_mean(small_dataset):
    d = small_dataset
    guess_raw = invert_scaler(mean_imputation(d["values"][d["test_idx"]]), d["mean"], d["std"])
    train_mean = raw_mean_per_feature(d, d["train_idx"])
    test_mean = raw_mean_per_feature(d, d["test_idx"])
    assert guess_raw[0, 0] == pytest.approx(train_mean, rel=1e-5)          # every feature, in raw units
    assert np.allclose(guess_raw, guess_raw[0, 0])                         # same guess at every box
    assert not np.allclose(guess_raw[0, 0], test_mean, rtol=1e-3)          # not peeking at the test patients

def test_mean_imputation_ignores_the_patient():
    values, _ = random_values()
    expected = mean_imputation(values)
    for changed in (values * 100, values + 5, np.full_like(values, np.nan)):
        assert np.array_equal(mean_imputation(changed), expected)

def test_mean_imputation_unusual_inputs_give_the_same_result():
    values, _ = random_values()
    expected = mean_imputation(values)
    wide = np.repeat(values, 2, axis=0)
    for variant in (
        np.asfortranarray(values),            # column-by-column memory layout
        wide[::2],                            # sliced, not contiguous
        values.astype(np.float64),
        torch.from_numpy(values),
    ):
        guess = mean_imputation(variant)
        assert guess.dtype == np.float32
        assert np.array_equal(guess, expected)

def test_mean_imputation_returns_a_fresh_array_and_leaves_the_input_alone():
    values, _ = random_values()
    before = values.copy()
    guess = mean_imputation(values)
    assert not np.shares_memory(guess, values)
    guess += 1                                # changing the guess can't reach the data
    assert np.array_equal(values, before)

@pytest.mark.parametrize("shape", [(6, 35, 48), (48, 35), (6, 48, 34), (6, 47, 35), (6, 48, 35, 1)])
def test_mean_imputation_rejects_other_shapes(shape):
    with pytest.raises(ValueError):
        mean_imputation(np.zeros(shape, dtype=np.float32))

def test_mean_imputation_dry_run_on_built_dataset(small_dataset):
    mae, values, target = score_mean_imputation(small_dataset, small_dataset["test_idx"])
    assert mae == pytest.approx(np.abs(values[target == 1]).mean())
    assert 0 < mae < 5

@needs_raw_data
def test_mean_imputation_on_set_a_test_split(set_a):
    mae, _, _ = score_mean_imputation(set_a, set_a["test_idx"])
    assert mae == pytest.approx(0.7053, abs=5e-5)                          # pinned: no randomness involved

# --- D2: forward fill -------------------------------------------------------------------

def score_forward_fill(dataset, idx):
    mask, values = dataset["mask"][idx], dataset["values"][idx]
    cond, target = EvalMasker.from_config()(mask, dataset["record_ids"][idx])
    return masked_mae(forward_fill(values, cond), values, target), values, target

def forward_fill_by_loop(values, cond):
    """The three rules spelled out one box at a time, to check the vectorised version against."""
    guess = np.zeros(values.shape, dtype=np.float32)
    for n in range(values.shape[0]):
        for d in range(values.shape[2]):
            seen = [t for t in range(values.shape[1]) if cond[n, t, d] == 1]
            for t in range(values.shape[1]):
                earlier = [s for s in seen if s <= t]
                if earlier:
                    guess[n, t, d] = values[n, earlier[-1], d]      # 1. carry the last visible value
                elif seen:
                    guess[n, t, d] = values[n, seen[0], d]          # 2. leading gap: first visible value
    return guess                                                    # 3. never visible: training mean, 0

def test_forward_fill_by_hand():
    values, cond = np.zeros((1, 48, 35), dtype=np.float32), np.zeros((1, 48, 35), dtype=np.float32)
    for hour, value in [(2, 7.0), (5, 5.0)]:
        values[0, hour, 0], cond[0, hour, 0] = value, 1
    values[0, 3, 0] = 99.0                                          # observed but hidden: must not be used
    guess = forward_fill(values, cond)
    assert guess[0, :7, 0].tolist() == [7.0, 7.0, 7.0, 7.0, 7.0, 5.0, 5.0]
    assert (guess[0, 7:, 0] == 5.0).all()                           # carried to the last hour
    assert (guess[0, :, 1:] == 0).all()                             # never visible: the training mean

def test_forward_fill_matches_a_plain_loop():
    values, mask = random_values()
    cond, _ = EvalMasker.from_config()(mask, np.arange(len(mask)))
    assert np.array_equal(forward_fill(values, cond), forward_fill_by_loop(values, cond))

def test_forward_fill_never_reads_hidden_values():
    values, mask = random_values()
    cond, _ = EvalMasker.from_config()(mask, np.arange(len(mask)))
    expected = forward_fill(values, cond)
    hidden = cond == 0                                              # targets and unobserved boxes
    for changed in (values * 100, values + 5, np.full_like(values, np.nan)):
        assert np.array_equal(forward_fill(np.where(hidden, changed, values), cond), expected)

def test_forward_fill_stays_within_its_patient_and_feature():
    values, cond = np.zeros((3, 48, 35), dtype=np.float32), np.zeros((3, 48, 35), dtype=np.float32)
    values[1, 10, 4], cond[1, 10, 4] = 3.0, 1
    guess = forward_fill(values, cond)
    assert (guess[1, :, 4] == 3.0).all()
    guess[1, :, 4] = 0
    assert (guess == 0).all()                                       # nothing leaked anywhere else

def test_forward_fill_unusual_inputs_give_the_same_result():
    values, mask = random_values()
    cond, _ = EvalMasker.from_config()(mask, np.arange(len(mask)))
    expected = forward_fill(values, cond)
    wide_values, wide_cond = np.repeat(values, 2, axis=0), np.repeat(cond, 2, axis=0)
    for v, c in (
        (np.asfortranarray(values), np.asfortranarray(cond)),       # column-by-column memory layout
        (wide_values[::2], wide_cond[::2]),                         # sliced, not contiguous
        (values.astype(np.float64), cond.astype(bool)),
        (torch.from_numpy(values), torch.from_numpy(cond)),
    ):
        guess = forward_fill(v, c)
        assert guess.dtype == np.float32
        assert np.array_equal(guess, expected)

def test_forward_fill_returns_a_fresh_array_and_leaves_the_inputs_alone():
    values, mask = random_values()
    cond, _ = EvalMasker.from_config()(mask, np.arange(len(mask)))
    before_values, before_cond = values.copy(), cond.copy()
    guess = forward_fill(values, cond)
    assert not np.shares_memory(guess, values)
    guess += 1                                                      # changing the guess can't reach the data
    assert np.array_equal(values, before_values) and np.array_equal(cond, before_cond)

@pytest.mark.parametrize("shape", [(6, 35, 48), (48, 35), (6, 48, 34), (6, 47, 35), (6, 48, 35, 1)])
def test_forward_fill_rejects_other_shapes(shape):
    with pytest.raises(ValueError):
        forward_fill(np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32))

def test_forward_fill_rejects_a_mask_for_other_patients():
    with pytest.raises(ValueError):
        forward_fill(np.zeros((6, 48, 35)), np.zeros((5, 48, 35)))

def test_forward_fill_dry_run_on_built_dataset(small_dataset):
    mae, _, _ = score_forward_fill(small_dataset, small_dataset["test_idx"])
    assert 0 < mae < 5

@needs_raw_data
def test_forward_fill_on_set_a_test_split(set_a):
    mae, _, _ = score_forward_fill(set_a, set_a["test_idx"])
    assert mae == pytest.approx(0.4487, abs=5e-5)                          # pinned: no randomness involved
    assert mae < score_mean_imputation(set_a, set_a["test_idx"])[0]        # it beats the floor

# --- D3: what the baselines are scored on -----------------------------------------------

@needs_raw_data
@pytest.mark.parametrize("baseline", [lambda values, cond: mean_imputation(values), forward_fill],
                         ids=["mean", "forward_fill"])
def test_baselines_are_scored_on_exactly_the_held_out_targets(set_a, baseline):
    idx = set_a["test_idx"]
    mask, values = set_a["mask"][idx], set_a["values"][idx]
    cond, target = EvalMasker.from_config()(mask, set_a["record_ids"][idx])
    assert np.array_equal(target, mask - cond)                             # observed minus conditioning
    assert target.sum() == sum(int(n * 0.10) for n in mask.sum(axis=(1, 2)))

    guess = baseline(values, cond)
    hidden = target == 1
    by_hand = guess[hidden].astype(np.float64) - values[hidden].astype(np.float64)
    assert np.array_equal(masked_errors(guess, values, target), by_hand)   # one error per target, in order
    off_target_ignored = np.where(hidden, guess, np.nan)                   # wreck every other box
    assert masked_mae(off_target_ignored, values, target) == masked_mae(guess, values, target)

# --- D4: the results log ----------------------------------------------------------------

def test_baseline_results_layout(small_dataset):
    results = baseline_results(small_dataset)
    assert results["random"]["eval_masker_seed"] == 2012
    assert set(results["random"]["fractions"]) == {"0.1", "0.5", "0.9"}   # headline + robustness rates
    for rate in results["random"]["fractions"].values():
        for split in ("val", "test"):
            row = rate[split]
            assert row["n_patients"] == len(small_dataset[f"{split}_idx"])
            assert set(row["mean_imputation"]) == set(row["forward_fill"]) == {"mae", "rmse"}
    json.dumps(results)                                                    # plain JSON types only

def test_the_10_percent_entry_is_the_headline_score(small_dataset):
    d = small_dataset
    row = baseline_results(d)["random"]["fractions"]["0.1"]["test"]
    assert row["mean_imputation"]["mae"] == round(score_mean_imputation(d, d["test_idx"])[0], 6)
    assert row["forward_fill"]["mae"] == round(score_forward_fill(d, d["test_idx"])[0], 6)

def test_each_rate_hides_its_own_fraction(small_dataset):
    d = small_dataset
    for key, rate in baseline_results(d)["random"]["fractions"].items():
        for split in ("val", "test"):
            n_observed = d["mask"][d[f"{split}_idx"]].sum(axis=(1, 2))
            assert rate[split]["n_targets"] == sum(int(int(n) * float(key)) for n in n_observed)

@needs_raw_data
@pytest.mark.parametrize("fraction, mean_mae, ffill_mae", [(0.5, 0.7132, 0.4881), (0.9, 0.7108, 0.5908)])
def test_robustness_rates_on_set_a_test_split(set_a, fraction, mean_mae, ffill_mae):
    idx = set_a["test_idx"]
    mask, values = set_a["mask"][idx], set_a["values"][idx]
    cond, target = EvalMasker(2012, fraction)(mask, set_a["record_ids"][idx])
    assert masked_mae(mean_imputation(values), values, target) == pytest.approx(mean_mae, abs=5e-5)
    assert masked_mae(forward_fill(values, cond), values, target) == pytest.approx(ffill_mae, abs=5e-5)

def test_importing_eval_leaves_the_baselines_script_alone():
    # `python -m eval.baselines` runs the module as __main__. If importing the package had
    # already imported it, Python would run it twice and warn (runpy's RuntimeWarning).
    code = "import sys, eval; assert 'eval.baselines' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)

@needs_raw_data
@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="results/baselines.json not written yet")
def test_recorded_baselines_match_a_fresh_run(set_a):
    # Fails if the code, the data pipeline or the frozen protocol changed without
    # re-running `uv run python -m eval.baselines` to rewrite results/baselines.json.
    assert json.loads(RESULTS_PATH.read_text()) == baseline_results(set_a)

# --- M2-4: the block holdout --------------------------------------------------------------

def test_block_entry_sits_next_to_random(small_dataset):
    block = baseline_results(small_dataset)["block"]
    assert block["block_eval_masker_seed"] == 8080
    assert block["block_window_hours"] == 3
    assert set(block["fractions"]) == {"0.1"}                              # the headline rate only
    for split in ("val", "test"):
        row = block["fractions"]["0.1"][split]
        assert row["n_patients"] == len(small_dataset[f"{split}_idx"])
        assert set(row["mean_imputation"]) == set(row["forward_fill"]) == {"mae", "rmse"}

def test_block_entry_hides_the_random_headline_count(small_dataset):
    results = baseline_results(small_dataset)
    for split in ("val", "test"):
        block_row = results["block"]["fractions"]["0.1"][split]
        assert block_row["n_targets"] == results["random"]["fractions"]["0.1"][split]["n_targets"]

def test_block_entry_is_scored_on_block_eval_maskers_targets(small_dataset):
    d = small_dataset
    row = baseline_results(d)["block"]["fractions"]["0.1"]["test"]
    idx = d["test_idx"]
    values, mask = d["values"][idx], d["mask"][idx]
    cond, target = BlockEvalMasker.from_config()(mask, d["record_ids"][idx])
    assert row["mean_imputation"]["mae"] == round(masked_mae(mean_imputation(values), values, target), 6)
    assert row["forward_fill"]["mae"] == round(masked_mae(forward_fill(values, cond), values, target), 6)

@needs_raw_data
def test_block_holdout_on_set_a_test_split(set_a):
    idx = set_a["test_idx"]
    mask, values = set_a["mask"][idx], set_a["values"][idx]
    cond, target = BlockEvalMasker.from_config()(mask, set_a["record_ids"][idx])
    assert masked_mae(mean_imputation(values), values, target) == pytest.approx(0.7338, abs=5e-5)
    assert masked_mae(forward_fill(values, cond), values, target) == pytest.approx(0.4786, abs=5e-5)

KNOWN_BAD = {"pH": lambda x: (x < 6.5) | (x > 8.0), "Temp": lambda x: x < 25, "K": lambda x: x > 20}

@needs_raw_data
def test_block_test_targets_include_three_known_bad_values(set_a):
    # The README's † footnote under Baselines; the bad values themselves are listed under Dataset.
    features = list(set_a["features"])
    raw = invert_scaler(set_a["values"], set_a["mean"], set_a["std"])
    scored = {}
    for split in ("val", "test"):
        idx = set_a[f"{split}_idx"]
        _, target = BlockEvalMasker.from_config()(set_a["mask"][idx], set_a["record_ids"][idx])
        scored[split] = []
        for name, is_bad in KNOWN_BAD.items():
            k = features.index(name)
            held_out = raw[idx][..., k][target[..., k] == 1]
            scored[split] += [(name, round(float(x), 1)) for x in held_out[is_bad(held_out)]]
    assert sorted(scored["val"]) == []
    assert sorted(scored["test"]) == [("Temp", 0.0), ("pH", 1.0), ("pH", 733.0)]

# --- M3-3: any guesser through the same scoring, and the baselines on wandb ---------------

def test_guessers_default_to_mean_and_forward_fill(small_dataset):
    from eval.baselines import BASELINES
    assert baseline_results(small_dataset) == baseline_results(small_dataset, BASELINES)
    assert list(BASELINES) == ["mean_imputation", "forward_fill"]

def test_a_guesser_is_shown_only_the_conditioning_mask_and_gets_the_same_layout(small_dataset):
    seen = []

    def spy(values, cond_mask):                             # forward fill, recording what it is given
        seen.append(cond_mask)
        return forward_fill(values, cond_mask)

    ours, theirs = baseline_results(small_dataset, {"spy": spy}), baseline_results(small_dataset)
    for holdout in ("random", "block"):
        for fraction, rows in theirs[holdout]["fractions"].items():
            for split, row in rows.items():
                assert ours[holdout]["fractions"][fraction][split] == {
                    "n_patients": row["n_patients"], "n_targets": row["n_targets"], "spy": row["forward_fill"]}
    assert len(seen) == 8                                   # 3 random rates + 1 block, val and test
    for cond in seen:                                       # a strict subset of the split's observed values
        split = "val" if len(cond) == len(small_dataset["val_idx"]) else "test"
        full = small_dataset["mask"][small_dataset[f"{split}_idx"]]
        assert ((cond == 1) <= (full == 1)).all() and cond.sum() < full.sum()

def test_baselines_are_logged_to_wandb(small_dataset, monkeypatch):
    import wandb
    from eval.baselines import log_to_wandb, wandb_summary

    runs = []

    class FakeRun:
        def __init__(self, **kwargs):
            self.init, self.summary = kwargs, {}
            runs.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(wandb, "init", FakeRun)
    results = baseline_results(small_dataset)
    log_to_wandb(results, "offline")
    assert [r.init["name"] for r in runs] == ["mean_imputation", "forward_fill"]
    assert {r.init["job_type"] for r in runs} == {"baseline"} and {r.init["mode"] for r in runs} == {"offline"}
    for r in runs:
        name = r.init["name"]
        assert r.summary == wandb_summary(results, name)
        assert r.summary["random/0.1/test/mae"] == results["random"]["fractions"]["0.1"]["test"][name]["mae"]
        assert r.summary["val/mae"] == results["random"]["fractions"]["0.1"]["val"][name]["mae"]
        assert r.summary["block/0.1/val/rmse"] == results["block"]["fractions"]["0.1"]["val"][name]["rmse"]
        assert len(r.summary) == 4 * 2 * 2 + 2              # 4 holdouts x 2 splits x 2 metrics, + val/*
