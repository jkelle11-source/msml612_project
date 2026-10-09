from pathlib import Path

import pytest
import numpy as np
from data.physionet import parse_record, compute_tslo, FEATURES

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "set-a"
needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")

@pytest.fixture
def random_mask() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((5, 48, 35)) < 0.2

@pytest.fixture(scope="module")
def real_tslo() -> dict[int, np.ndarray]:
    record_ids = [132539, 132540, 132543]
    mask = np.stack([~np.isnan(parse_record(RAW_DIR / f"{r}.txt")) for r in record_ids])
    return dict(zip(record_ids, compute_tslo(mask)))

def test_compute_tslo_known_row():
    mask = np.array([0, 0, 1, 0, 0, 1, 1, 0]).reshape(1, 8, 1)
    assert compute_tslo(mask).reshape(8).tolist() == [1, 2, 0, 1, 2, 0, 0, 1]

def test_compute_tslo_counts_each_patient_and_feature_separately():
    mask = np.array([
        [[1, 0], [0, 0], [0, 1], [1, 0]],
        [[0, 0], [1, 0], [1, 0], [0, 0]],
    ])
    expected = [
        [[0, 1], [1, 2], [2, 0], [0, 1]],
        [[1, 1], [0, 2], [0, 3], [1, 4]],
    ]
    assert compute_tslo(mask).tolist() == expected

def test_compute_tslo_is_zero_exactly_where_observed(random_mask: np.ndarray):
    assert ((compute_tslo(random_mask) == 0) == random_mask).all()

def test_compute_tslo_never_observed_feature_counts_up():
    mask = np.zeros((1, 48, 1))
    assert compute_tslo(mask).reshape(48).tolist() == list(range(1, 49))

def test_compute_tslo_keeps_shape_and_leaves_input_alone(random_mask: np.ndarray):
    before = random_mask.copy()
    assert compute_tslo(random_mask).shape == (5, 48, 35)
    assert (random_mask == before).all()

def test_compute_tslo_same_for_bool_and_float_masks(random_mask: np.ndarray):
    as_float = random_mask.astype(np.float32)
    assert (compute_tslo(random_mask) == compute_tslo(as_float)).all()

def test_compute_tslo_batch_matches_one_patient_at_a_time(random_mask: np.ndarray):
    together = compute_tslo(random_mask)
    for i in range(len(random_mask)):
        alone = compute_tslo(random_mask[i:i + 1])
        assert (together[i:i + 1] == alone).all()

# The expected rows below were worked out from the raw files with a separate
# shell script, not with compute_tslo.

@needs_raw_data
def test_real_132539_temp(real_tslo: dict[int, np.ndarray]):
    # measured at hour 0, then every 4 hours from hour 3 to hour 47
    expected = [0, 1, 2] + [0, 1, 2, 3] * 11 + [0]
    assert real_tslo[132539][:, FEATURES.index("Temp")].tolist() == expected

@needs_raw_data
def test_real_132540_gcs(real_tslo: dict[int, np.ndarray]):
    # measured at hours 1 3 4 7 11 15 19 23 27 31 35 37 40 43 47
    expected = [1, 0, 1, 0, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0,
                1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0,
                1, 2, 3, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0]
    assert real_tslo[132540][:, FEATURES.index("GCS")].tolist() == expected

@needs_raw_data
def test_real_132543_glucose(real_tslo: dict[int, np.ndarray]):
    # measured at hours 0, 8 and 32
    expected = list(range(8)) + list(range(24)) + list(range(16))
    assert real_tslo[132543][:, FEATURES.index("Glucose")].tolist() == expected

@needs_raw_data
def test_real_132539_cholesterol_never_measured(real_tslo: dict[int, np.ndarray]):
    expected = list(range(1, 49))
    assert real_tslo[132539][:, FEATURES.index("Cholesterol")].tolist() == expected
