from pathlib import Path

import numpy as np
import pytest
import yaml
from data.physionet import build_dataset, FEATURES

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "set-a"
CONFIG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())

@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, np.ndarray]:
    """40 fake patients (record ids 101-140) run through the real ``build_dataset``."""
    folder = tmp_path_factory.mktemp("records")
    rng = np.random.default_rng(0)
    for record_id in range(101, 141):
        rows = ["Time,Parameter,Value", f"00:00,RecordID,{record_id}"]
        for name in FEATURES:
            for hour in rng.choice(48, size=8, replace=False):
                rows.append(f"{hour:02d}:30,{name},{rng.normal(100, 20):.2f}")
        (folder / f"{record_id}.txt").write_text("\n".join(rows))
    return build_dataset(folder, 0)

@pytest.fixture(scope="session")
def set_a() -> dict[str, np.ndarray]:
    return build_dataset(RAW_DIR, CONFIG["data"]["split_seed"])
