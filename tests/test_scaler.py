import pytest
import numpy as np
from data.physionet import fit_scaler, apply_scaler, invert_scaler

@pytest.fixture
def values() -> np.ndarray:
    rng = np.random.default_rng(0)
    values = rng.normal(loc=[80, 7.4, 200], scale=[15, 0.1, 60], size=(20, 48, 3))
    values[rng.random(values.shape) < 0.5] = np.nan
    return values

@pytest.fixture
def train_idx() -> np.ndarray:
    return np.arange(14)

def test_fit_scaler_by_hand_example():
    values = np.array([70., 90., 100., np.nan]).reshape(4, 1, 1)
    mean, std = fit_scaler(values, np.array([0, 1]))
    assert mean.tolist() == [80]
    assert std.tolist() == [10]

def test_apply_scaler_by_hand_example():
    values = np.array([70., 90., 100., np.nan]).reshape(4, 1, 1)
    scaled = apply_scaler(values, np.array([80.]), np.array([10.])).reshape(4)
    assert scaled[:3].tolist() == [-1, 1, 2]
    assert np.isnan(scaled[3])

def test_fit_scaler_gives_one_mean_and_std_per_variable(values: np.ndarray, train_idx: np.ndarray):
    mean, std = fit_scaler(values, train_idx)
    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert mean == pytest.approx([80, 7.4, 200], rel=0.05)
    assert std == pytest.approx([15, 0.1, 60], rel=0.1)

def test_fit_scaler_ignores_patients_outside_train(values: np.ndarray, train_idx: np.ndarray):
    mean, std = fit_scaler(values, train_idx)
    changed = values.copy()
    changed[14:] = changed[14:] * 1000 + 5
    mean_changed, std_changed = fit_scaler(changed, train_idx)
    assert (mean == mean_changed).all()
    assert (std == std_changed).all()

def test_apply_scaler_centres_the_train_patients(values: np.ndarray, train_idx: np.ndarray):
    mean, std = fit_scaler(values, train_idx)
    scaled = apply_scaler(values, mean, std)[train_idx]
    assert np.nanmean(scaled, axis=(0, 1)) == pytest.approx([0, 0, 0], abs=1e-9)
    assert np.nanstd(scaled, axis=(0, 1)) == pytest.approx([1, 1, 1])

def test_apply_scaler_keeps_shape_and_missing_cells(values: np.ndarray, train_idx: np.ndarray):
    mean, std = fit_scaler(values, train_idx)
    scaled = apply_scaler(values, mean, std)
    assert scaled.shape == values.shape
    assert (np.isnan(scaled) == np.isnan(values)).all()

def test_invert_scaler_by_hand_example():
    scaled = np.array([-1., 1., 2.]).reshape(3, 1, 1)
    restored = invert_scaler(scaled, np.array([80.]), np.array([10.]))
    assert restored.reshape(3).tolist() == [70, 90, 100]

def test_invert_scaler_undoes_apply_scaler(values: np.ndarray, train_idx: np.ndarray):
    mean, std = fit_scaler(values, train_idx)
    restored = invert_scaler(apply_scaler(values, mean, std), mean, std)
    observed = ~np.isnan(values)
    assert restored[observed] == pytest.approx(values[observed])
    assert (np.isnan(restored) == ~observed).all()
