from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from data.physionet import build_dataset, FEATURES

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "set-a"
CONFIG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())

def csdi_config(target_strategy="external", num_steps=50):
    """Upstream config/base.yaml at the pinned commit; "external" is our added strategy (M2-1)."""
    return {
        "diffusion": {
            "layers": 4,
            "channels": 64,
            "nheads": 8,
            "diffusion_embedding_dim": 128,
            "beta_start": 0.0001,
            "beta_end": 0.5,
            "num_steps": num_steps,
            "schedule": "quad",
            "is_linear": False,
        },
        "model": {
            "is_unconditional": 0,
            "timeemb": 128,
            "featureemb": 16,
            "target_strategy": target_strategy,
        },
    }

def nonzero_output_layer(model):
    """Give the denoiser's last layer the non-zero weights it has after its first update.

    Upstream zero-initialises that layer, so a freshly built model's output ignores everything
    before it (the timestep, dropout, any conditioning) and a test on it would pass vacuously.
    """
    with torch.no_grad():
        weight = model.diffmodel.output_projection2.weight
        weight.copy_(torch.randn(weight.shape, generator=torch.Generator().manual_seed(0)) * 0.5)
    return model

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
