import numpy as np
import pytest
from conftest import RAW_DIR
from data.physionet import FEATURES, KNOWN_BAD_RANGES, apply_scaler, known_bad_values

def one_patient(readings, mean=0.0, std=1.0):
    """A one-patient dataset from ``(hour, feature, raw value)`` readings, scaled as build_dataset does."""
    D = len(FEATURES)
    raw, mask = np.zeros((1, 48, D), np.float32), np.zeros((1, 48, D), np.float32)
    for hour, name, value in readings:
        raw[0, hour, FEATURES.index(name)], mask[0, hour, FEATURES.index(name)] = value, 1
    mean, std = np.full(D, mean, np.float32), np.full(D, std, np.float32)
    values = np.where(mask == 1, apply_scaler(raw, mean, std), 0).astype(np.float32)
    return {"values": values, "mask": mask, "mean": mean, "std": std, "features": np.array(FEATURES)}

def flagged(dataset):
    return [(int(t), FEATURES[d]) for _, t, d in zip(*np.nonzero(known_bad_values(dataset)))]

def test_the_ranges_are_the_readmes():
    assert KNOWN_BAD_RANGES == {"pH": (6.5, 8.0), "Temp": (25.0, None), "K": (None, 20.0), "PaO2": (20.0, None)}

def test_values_outside_their_range_are_flagged_and_the_bounds_are_not():
    readings = [(0, "pH", 735.0), (1, "pH", 1.0), (2, "pH", 6.5), (3, "pH", 8.0), (4, "pH", 7.4),
                (5, "Temp", -17.8), (6, "Temp", 25.0), (7, "K", 22.9), (8, "K", 20.0),
                (9, "PaO2", 0.0), (10, "PaO2", 20.0), (11, "HR", 900.0)]          # HR has no range
    assert flagged(one_patient(readings)) == [(0, "pH"), (1, "pH"), (5, "Temp"), (7, "K"), (9, "PaO2")]

def test_raw_units_are_judged_not_standardized_ones():
    # pH 7.0 is z = (7.0 - 100) / 10 = -9.3, well outside 6.5-8.0 in standardized units, and fine raw
    assert flagged(one_patient([(0, "pH", 7.0), (1, "pH", 735.0)], mean=100.0, std=10.0)) == [(1, "pH")]

def test_a_box_never_measured_is_never_flagged():
    dataset = one_patient([(0, "pH", 735.0)], mean=-50.0)          # an unmeasured pH box reads back raw as -50
    assert flagged(dataset) == [(0, "pH")]

@pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")
def test_set_a_has_the_readmes_bad_values(set_a):
    bad = known_bad_values(set_a)
    counts = {name: int(bad[..., FEATURES.index(name)].sum()) for name in KNOWN_BAD_RANGES}
    assert counts == {"pH": 10, "Temp": 73, "K": 1, "PaO2": 2}
    assert int(bad.sum()) == 86
