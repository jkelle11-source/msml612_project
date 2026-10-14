import math

import numpy as np
import pytest
import torch
from conftest import RAW_DIR
from data.physionet import invert_scaler
from eval.masking import EvalMasker
from eval.metrics import masked_errors, masked_mae, masked_rmse, median_imputation, per_variable_mae

needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")

def hand_fixture():
    """2 patients x 2 hours x 2 features; 4 hidden boxes with errors 1, -2, 3, 0."""
    truth = np.zeros((2, 2, 2))
    pred = np.zeros((2, 2, 2))
    target = np.zeros((2, 2, 2))
    for box, err in zip([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], [1, -2, 3, 0]):
        pred[box] = err
        target[box] = 1
    return pred, truth, target

def random_case(seed=0, shape=(20, 48, 35)):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=shape)
    pred = truth + rng.normal(scale=0.5, size=shape)
    target = (rng.random(shape) < 0.1).astype(np.float32)
    return pred, truth, target

def score_dataset(dataset, idx, rng):
    """Dry run like M2-3: EvalMasker -> fake K-draw model output -> median -> metrics."""
    mask, values = dataset["mask"][idx], dataset["values"][idx]
    cond, target = EvalMasker.from_config()(mask, dataset["record_ids"][idx])
    samples = values[None] + rng.normal(scale=0.3, size=(5, *values.shape))    # (K, B, 48, D)
    guess = median_imputation(samples)
    perfect = np.repeat(values[None], 5, axis=0)
    return dict(
        target=target, values=values,
        model_mae=masked_mae(guess, values, target),
        model_rmse=masked_rmse(guess, values, target),
        perfect_mae=masked_mae(median_imputation(perfect), values, target),
        mean_impute_mae=masked_mae(np.zeros_like(values), values, target),
        per_var_raw=per_variable_mae(guess, values, target, std=dataset["std"]),
    )

# --- 1: the ticket's hand-computed example ---------------------------------------------

def test_hand_computed_mae_and_rmse():
    pred, truth, target = hand_fixture()
    assert masked_mae(pred, truth, target) == 1.5                       # (1 + 2 + 3 + 0) / 4
    assert masked_rmse(pred, truth, target) == pytest.approx(math.sqrt(3.5))   # sqrt((1+4+9+0)/4)
    assert sorted(masked_errors(pred, truth, target).tolist()) == [-2, 0, 1, 3]

# --- 2 + 3: what gets scored and how it's averaged -------------------------------------

def test_only_hidden_boxes_count():
    pred, truth, target = hand_fixture()
    pred[target == 0] = 1e9
    pred[0, 1, 0] = np.nan                  # a NaN outside the targets must not leak in
    truth[target == 0] = -1e9
    assert masked_mae(pred, truth, target) == 1.5
    assert masked_rmse(pred, truth, target) == pytest.approx(math.sqrt(3.5))
    assert np.isfinite(per_variable_mae(pred, truth, target)).all()

def test_pooled_over_boxes_not_averaged_per_patient():
    truth, pred, target = np.zeros((2, 4, 1)), np.zeros((2, 4, 1)), np.zeros((2, 4, 1))
    target[0, 0] = 1
    pred[0, 0] = 4                          # patient A: 1 hidden box, error 4
    target[1, :3] = 1                       # patient B: 3 hidden boxes, error 0
    assert masked_mae(pred, truth, target) == 1.0       # per-patient-first would give 2.0

# --- 4 + 5: sanity -----------------------------------------------------------------------

def test_perfect_guess_scores_zero():
    _, truth, target = random_case()
    assert masked_mae(truth, truth, target) == 0
    assert masked_rmse(truth, truth, target) == 0

def test_rmse_at_least_mae_and_equal_for_constant_errors():
    pred, truth, target = random_case()
    assert masked_rmse(pred, truth, target) > masked_mae(pred, truth, target)
    assert masked_rmse(truth + 0.7, truth, target) == pytest.approx(masked_mae(truth + 0.7, truth, target))

# --- 6 + 7: fail loudly ------------------------------------------------------------------

def test_bad_input_raises():
    pred, truth, target = hand_fixture()
    with pytest.raises(ValueError):
        masked_mae(pred[:1], truth, target)             # would otherwise broadcast silently
    with pytest.raises(ValueError):
        masked_rmse(pred, truth, target[..., :1])
    with pytest.raises(ValueError):
        masked_mae(pred, truth, np.zeros_like(target))  # nothing to score
    with pytest.raises(ValueError):
        per_variable_mae(pred, truth, np.zeros_like(target))

def test_nan_guess_at_a_hidden_box_gives_nan():
    pred, truth, target = hand_fixture()
    pred[0, 0, 0] = np.nan                  # a hidden box
    assert math.isnan(masked_mae(pred, truth, target))
    assert math.isnan(masked_rmse(pred, truth, target))
    per_var = per_variable_mae(pred, truth, target)
    assert math.isnan(per_var[0]) and per_var[1] == pytest.approx(2.5)

# --- 8: torch input ----------------------------------------------------------------------

def test_torch_tensors_give_the_same_numbers():
    pred, truth, target = random_case()
    t_pred, t_truth, t_target = (torch.from_numpy(a) for a in (pred, truth, target))
    assert masked_mae(t_pred, t_truth, t_target) == masked_mae(pred, truth, target)
    assert masked_rmse(t_pred, t_truth, t_target) == masked_rmse(pred, truth, target)
    assert (per_variable_mae(t_pred, t_truth, t_target) == per_variable_mae(pred, truth, target)).all()

def test_float32_input_is_summed_in_float64():
    pred, truth, target = random_case()
    exact = masked_mae(pred, truth, target)
    assert masked_mae(pred.astype(np.float32), truth.astype(np.float32), target) == pytest.approx(exact, rel=1e-6)

# --- 9 - 11: per-variable --------------------------------------------------------------

def test_per_variable_by_hand():
    truth, pred, target = np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), np.zeros((1, 3, 3))
    target[0, :, 0] = 1
    pred[0, :, 0] = [1, 2, 3]               # feature 0: errors 1, 2, 3 -> MAE 2
    target[0, 0, 1] = 1
    pred[0, 0, 1] = -4                      # feature 1: one error of 4 -> MAE 4
    # feature 2: nothing hidden -> NaN
    per_var = per_variable_mae(pred, truth, target)
    assert per_var[:2].tolist() == [2.0, 4.0]
    assert math.isnan(per_var[2])
    assert per_variable_mae(pred, truth, target, std=[10, 0.5, 1])[:2].tolist() == [20.0, 2.0]

def test_raw_units_match_undoing_the_z_scoring():
    pred, truth, target = random_case()
    mean, std = np.linspace(50, 150, 35), np.linspace(0.5, 30, 35)
    raw_pred, raw_truth = invert_scaler(pred, mean, std), invert_scaler(truth, mean, std)
    direct = per_variable_mae(raw_pred, raw_truth, target)
    assert per_variable_mae(pred, truth, target, std=std) == pytest.approx(direct)

def test_pooled_mae_is_the_count_weighted_average_of_per_variable():
    pred, truth, target = random_case()
    counts = target.sum(axis=(0, 1))
    weighted = (per_variable_mae(pred, truth, target) * counts).sum() / counts.sum()
    assert masked_mae(pred, truth, target) == pytest.approx(weighted)

# --- 12 + 13: one guess from K draws ----------------------------------------------------

def test_median_imputation_by_hand():
    odd = np.array([[5.0], [1.0], [3.0]])
    even = np.array([[4.0], [1.0], [3.0], [2.0]])          # sorted 1 2 | 3 4
    assert median_imputation(odd).tolist() == [3.0]
    assert median_imputation(even).tolist() == [2.0]        # the lower middle; np.median says 2.5
    assert median_imputation(even.T, axis=1).tolist() == [2.0]

def test_median_imputation_matches_upstreams_torch_median():
    for k in (1, 2, 5, 50, 100):                            # upstream scores samples.median(dim=...)
        samples = torch.randn(k, 3, 48, 35, generator=torch.Generator().manual_seed(k))
        assert np.array_equal(median_imputation(samples), samples.median(dim=0).values.numpy())
        swapped = samples.transpose(0, 1)                   # upstream's own layout: draws on axis 1
        assert np.array_equal(median_imputation(swapped, axis=1), swapped.median(dim=1).values.numpy())

def test_median_imputation_keeps_a_nan_draw_visible():
    samples = np.arange(8.0).reshape(4, 2)
    samples[0, 0] = np.nan
    guess = median_imputation(samples)
    assert math.isnan(guess[0]) and guess[1] == 3.0

# --- 14 + 15: dry run like M2-3 will use it --------------------------------------------

def test_dry_run_on_built_dataset(small_dataset):
    out = score_dataset(small_dataset, small_dataset["test_idx"], np.random.default_rng(0))
    hidden_truth = out["values"][out["target"] == 1]
    assert out["perfect_mae"] == 0
    assert out["mean_impute_mae"] == pytest.approx(np.abs(hidden_truth).mean())
    assert 0 < out["model_mae"] < out["model_rmse"] < out["mean_impute_mae"] * 2
    assert out["per_var_raw"].shape == (35,)
    assert np.isfinite(out["per_var_raw"]).all()

@needs_raw_data
def test_dry_run_on_set_a(set_a):
    out = score_dataset(set_a, set_a["test_idx"], np.random.default_rng(0))
    assert np.isfinite(out["mean_impute_mae"])
    assert out["per_var_raw"].shape == (35,)
    assert np.isfinite(out["per_var_raw"]).all()
