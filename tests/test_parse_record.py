from pathlib import Path

import pytest
import numpy as np
from data.physionet import parse_record, FEATURES

REAL_RECORD = Path(__file__).resolve().parents[1] / "data" / "raw" / "set-a" / "132539.txt"
needs_raw_data = pytest.mark.skipif(not REAL_RECORD.exists(), reason="raw Set A not downloaded")

@pytest.fixture
def small_record(tmp_path: Path) -> Path:
    rows = [
        "Time,Parameter,Value",
        "00:00,RecordID,999",
        "00:00,Age,54",
        "00:07,HR,88",
        "00:20,GCS,15",
        "00:59,HR,67",
        "02:33,Weight,185",
        "03:22,MechVent,1",
        "04:10,FiO2,0",
        "05:23,Temp,65.6",
        "06:44,Temp,80.0",
        "07:11,FiO2,99",
        "08:33,Urine,60",
        "48:00,Urine,81",
    ]
    path = tmp_path / "small_record.txt"
    path.write_text("\n".join(rows))
    return path

def test_parse_record_fixture_two_readings_same_hour(small_record: Path):
    record = parse_record(small_record)
    assert record[0, FEATURES.index("HR")] == 67

def test_parse_record_fixture_last_hour_ignored(small_record: Path):
    record = parse_record(small_record)
    assert not (record == 81).any()

def test_parse_record_fixture_static_rows_ignored(small_record: Path):
    record = parse_record(small_record)
    assert sorted(record[~np.isnan(record)]) == [0, 15, 60, 65.6, 67, 80, 99]

def test_parse_record_fixture_non_present_feature_is_nan(small_record: Path):
    record = parse_record(small_record)
    assert np.isnan(record[:, FEATURES.index("MAP")]).all()

def test_parse_record_fixture_zero_stays_zero(small_record: Path):
    record = parse_record(small_record)
    assert record[4, FEATURES.index("FiO2")] == 0

def test_parse_record_fixture_shape(small_record: Path):
    record = parse_record(small_record)
    assert record.shape == (48, 35)

@needs_raw_data
def test_parse_record_hr_hour_0():
    record = parse_record(REAL_RECORD)
    assert record[0, FEATURES.index("HR")] == 77

@needs_raw_data
def test_parse_record_temp_hour_0():
    record = parse_record(REAL_RECORD)
    assert record[0, FEATURES.index("Temp")] == 35.6

@needs_raw_data
def test_parse_record_urine_hour_0():
    record = parse_record(REAL_RECORD)
    assert record[0, FEATURES.index("Urine")] == 60

@needs_raw_data
def test_parse_record_hr_hour_1():
    record = parse_record(REAL_RECORD)
    assert record[1, FEATURES.index("HR")] == 60

@needs_raw_data
def test_parse_record_temp_hour_1():
    record = parse_record(REAL_RECORD)
    assert np.isnan(record[1, FEATURES.index("Temp")])
