from pathlib import Path

import pytest
import numpy as np
from data.physionet import parse_raw_data_folder, FEATURES

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "set-a"
needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")

@pytest.fixture
def small_folder(tmp_path: Path) -> Path:
    heart_rates = {"200": 72, "30": 65, "1000": 90}
    for record_id, hr in heart_rates.items():
        rows = [
            "Time,Parameter,Value",
            f"00:00,RecordID,{record_id}",
            f"00:10,HR,{hr}",
        ]
        (tmp_path / f"{record_id}.txt").write_text("\n".join(rows))
    (tmp_path / ".DS_Store").write_text("not a record")
    (tmp_path / "notes.csv").write_text("not a record")
    return tmp_path

@pytest.fixture(scope="module")
def set_a() -> tuple[np.ndarray, np.ndarray]:
    return parse_raw_data_folder(RAW_DIR)

def test_parse_raw_data_folder_shape_and_dtype(small_folder: Path):
    values, record_ids = parse_raw_data_folder(small_folder)
    assert values.shape == (3, 48, 35)
    assert values.dtype == np.float32
    assert record_ids.shape == (3,)
    assert record_ids.dtype == np.int64

def test_parse_raw_data_folder_ids_sorted_by_number(small_folder: Path):
    _, record_ids = parse_raw_data_folder(small_folder)
    assert record_ids.tolist() == [30, 200, 1000]

def test_parse_raw_data_folder_rows_line_up_with_ids(small_folder: Path):
    values, _ = parse_raw_data_folder(small_folder)
    assert values[:, 0, FEATURES.index("HR")].tolist() == [65, 72, 90]

def test_parse_raw_data_folder_missing_stays_nan(small_folder: Path):
    values, _ = parse_raw_data_folder(small_folder)
    assert (~np.isnan(values)).sum() == 3

@needs_raw_data
def test_set_a_shape(set_a: tuple[np.ndarray, np.ndarray]):
    values, record_ids = set_a
    assert values.shape == (4000, 48, 35)
    assert record_ids.shape == (4000,)

@needs_raw_data
def test_set_a_ids_sorted_and_unique(set_a: tuple[np.ndarray, np.ndarray]):
    _, record_ids = set_a
    assert record_ids[0] == 132539
    assert (np.diff(record_ids) > 0).all()

@needs_raw_data
def test_set_a_missing_rate(set_a: tuple[np.ndarray, np.ndarray]):
    values, _ = set_a
    filled = (~np.isnan(values)).sum()
    # counted independently from the raw files with a shell script
    assert filled == 1308246
    assert 1 - filled / values.size == pytest.approx(0.8053, abs=1e-4)
