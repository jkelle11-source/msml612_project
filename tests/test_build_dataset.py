from pathlib import Path

import pytest
import numpy as np
import yaml
from data.physionet import (
    build_dataset, save_dataset, load_dataset, parse_raw_data_folder, compute_tslo,
    make_split, fit_scaler, apply_scaler, invert_scaler, FEATURES,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "set-a"
PROCESSED = ROOT / "data" / "processed" / "physionet_set_a.npz"
CONFIG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")
needs_processed = pytest.mark.skipif(not PROCESSED.exists(), reason="processed dataset not built")

CONTRACT = ["values", "mask", "delta_obs", "record_ids", "features",
            "train_idx", "val_idx", "test_idx", "mean", "std"]

@pytest.fixture(scope="module")
def small_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("records")
    rng = np.random.default_rng(0)
    for record_id in range(1, 21):
        rows = ["Time,Parameter,Value", f"00:00,RecordID,{record_id}"]
        for name in FEATURES:
            for hour in rng.choice(48, size=6, replace=False):
                rows.append(f"{hour:02d}:30,{name},{rng.normal(100, 20):.2f}")
        (folder / f"{record_id}.txt").write_text("\n".join(rows))
    return folder

@pytest.fixture(scope="module")
def raw(small_folder: Path) -> np.ndarray:
    values, _ = parse_raw_data_folder(small_folder)
    return values

@pytest.fixture(scope="module")
def dataset(small_folder: Path) -> dict[str, np.ndarray]:
    return build_dataset(small_folder, 0)

@pytest.fixture(scope="module")
def set_a() -> dict[str, np.ndarray]:
    return build_dataset(RAW_DIR, CONFIG["data"]["split_seed"])

def test_config_split_seed_is_its_own_seed():
    seed = CONFIG["data"]["split_seed"]
    assert isinstance(seed, int)
    assert seed != CONFIG["protocol"]["eval_masker_seed"]
    assert seed not in CONFIG["protocol"]["run_seeds"]

def test_build_dataset_returns_the_contract_names(dataset: dict[str, np.ndarray]):
    assert sorted(dataset) == sorted(CONTRACT)

def test_build_dataset_shapes_and_dtypes(dataset: dict[str, np.ndarray]):
    for name in ["values", "mask", "delta_obs"]:
        assert dataset[name].shape == (20, 48, 35)
        assert dataset[name].dtype == np.float32
    assert dataset["record_ids"].tolist() == list(range(1, 21))
    assert dataset["features"].tolist() == FEATURES
    assert dataset["mean"].shape == (35,)
    assert dataset["std"].shape == (35,)

def test_build_dataset_mask_marks_the_real_readings(dataset: dict[str, np.ndarray], raw: np.ndarray):
    assert (dataset["mask"] == ~np.isnan(raw)).all()

def test_build_dataset_values_are_zero_where_missing(dataset: dict[str, np.ndarray]):
    assert not np.isnan(dataset["values"]).any()
    assert (dataset["values"][dataset["mask"] == 0] == 0).all()

def test_build_dataset_delta_comes_from_the_mask(dataset: dict[str, np.ndarray]):
    assert (dataset["delta_obs"] == compute_tslo(dataset["mask"])).all()
    assert ((dataset["delta_obs"] == 0) == (dataset["mask"] == 1)).all()

def test_build_dataset_split_uses_the_seed(dataset: dict[str, np.ndarray], small_folder: Path):
    train, val, test = make_split(20, 0)
    assert (dataset["train_idx"] == train).all()
    assert (dataset["val_idx"] == val).all()
    assert (dataset["test_idx"] == test).all()
    other = build_dataset(small_folder, 1)
    assert not (other["train_idx"] == train).all()

def test_build_dataset_scaler_is_fitted_on_train_only(dataset: dict[str, np.ndarray], raw: np.ndarray):
    mean, std = fit_scaler(raw, dataset["train_idx"])
    assert dataset["mean"] == pytest.approx(mean)
    assert dataset["std"] == pytest.approx(std)
    observed = dataset["mask"] == 1
    assert dataset["values"][observed] == pytest.approx(apply_scaler(raw, mean, std)[observed])

def test_build_dataset_round_trips_to_raw_values(dataset: dict[str, np.ndarray], raw: np.ndarray):
    restored = invert_scaler(dataset["values"], dataset["mean"], dataset["std"])
    observed = dataset["mask"] == 1
    assert restored[observed] == pytest.approx(raw[observed], rel=1e-4)

def test_build_dataset_is_repeatable(dataset: dict[str, np.ndarray], small_folder: Path):
    again = build_dataset(small_folder, 0)
    for name in CONTRACT:
        assert (dataset[name] == again[name]).all()

def test_save_and_load_give_back_the_same_dataset(dataset: dict[str, np.ndarray], tmp_path: Path):
    path = tmp_path / "dataset.npz"
    save_dataset(dataset, path)
    loaded = load_dataset(path)
    assert sorted(loaded) == sorted(CONTRACT)
    for name in CONTRACT:
        assert loaded[name].dtype == dataset[name].dtype
        assert (loaded[name] == dataset[name]).all()

@needs_raw_data
@needs_processed
def test_saved_set_a_matches_a_fresh_build(set_a: dict[str, np.ndarray]):
    saved = load_dataset(PROCESSED)
    assert sorted(saved) == sorted(CONTRACT)
    for name in CONTRACT:
        assert (saved[name] == set_a[name]).all()

@needs_raw_data
def test_set_a_shapes_and_missing_rate(set_a: dict[str, np.ndarray]):
    for name in ["values", "mask", "delta_obs"]:
        assert set_a[name].shape == (4000, 48, 35)
    # filled-cell count was taken independently from the raw files with a shell script
    assert set_a["mask"].sum() == 1308246
    assert 1 - set_a["mask"].mean() == pytest.approx(0.8053, abs=1e-4)

@needs_raw_data
def test_set_a_split_is_70_10_20_with_no_shared_patients(set_a: dict[str, np.ndarray]):
    train, val, test = set_a["train_idx"], set_a["val_idx"], set_a["test_idx"]
    assert (len(train), len(val), len(test)) == (2800, 400, 800)
    assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(4000))

@needs_raw_data
def test_set_a_has_no_bad_numbers(set_a: dict[str, np.ndarray]):
    for name in ["values", "delta_obs", "mean", "std"]:
        assert np.isfinite(set_a[name]).all()
    assert (set_a["std"] > 0).all()

@needs_raw_data
def test_set_a_train_patients_are_standardized(set_a: dict[str, np.ndarray]):
    train = set_a["train_idx"]
    values = np.where(set_a["mask"][train] == 1, set_a["values"][train], np.nan).astype(np.float64)
    assert np.nanmean(values, axis=(0, 1)) == pytest.approx(np.zeros(35), abs=1e-3)
    assert np.nanstd(values, axis=(0, 1)) == pytest.approx(np.ones(35), abs=1e-3)
