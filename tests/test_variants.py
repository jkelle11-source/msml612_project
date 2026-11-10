import copy

import pytest
import torch
import yaml
from conftest import CONFIG, RAW_DIR
from diffusion.data import to_batch
from diffusion.run import VARIANT_FLAGS, main, variant_name
from diffusion.streams import RunStreams
from eval import EvalMasker, EvalSampler, TrainMasker
from models.build import build_model

needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")

T, D = 48, 35
C = CONFIG["diffusion"]["channels"]     # upstream's name for d_model

def config_of(name, **diffusion):
    """The repo's config with the toggles VARIANT_FLAGS gives ``name``."""
    config = copy.deepcopy(CONFIG)
    config["model"].update(use_mask_embedding="--use_mask_embedding" in VARIANT_FLAGS[name],
                           use_time_encoding="--use_time_encoding" in VARIANT_FLAGS[name])
    config["diffusion"].update(diffusion)
    return config

def built(name, **diffusion):
    RunStreams(42).seed_model_init()                                # as train() does, right before
    return build_model(config_of(name, **diffusion), "cpu")

def generator():
    return torch.Generator().manual_seed(7)

@pytest.fixture(scope="module")
def real(set_a) -> dict[str, torch.Tensor]:
    """16 Set A training patients, a batch the size of the real loop's, with TrainMasker's split."""
    rows = set_a["train_idx"][:16]
    values, mask = torch.from_numpy(set_a["values"][rows]), torch.from_numpy(set_a["mask"][rows])
    return {"values": values, "mask": mask, "cond_mask": TrainMasker(42)(mask)[0]}

def batch_of(inputs):
    return to_batch(inputs["values"], inputs["mask"], inputs["cond_mask"])

def train_steps(model, inputs, steps):
    """Adam steps on one batch. Upstream's zero last layer and the encoders' zero gates mean
    gradients reach every encoder weight only from the third step (tests/test_time_encoding.py)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        model(batch_of(inputs), generator=generator()).backward()
        optimizer.step()
        optimizer.zero_grad()
    return model

# --- the variant -> flags mapping the harness reads ---------------------------------------

def test_there_are_exactly_the_four_variants():
    assert list(VARIANT_FLAGS) == ["base", "ext1", "ext2", "ext1+2"]

@pytest.mark.parametrize("name", list(VARIANT_FLAGS))
def test_each_variants_flags_train_that_variant(small_dataset, tmp_path, name):
    from data.physionet import save_dataset
    config = copy.deepcopy(CONFIG)
    config["diffusion"].update(channels=16, layers=1, num_steps=10)
    config["train"].update(epochs=1, batch_size=8)
    config["validation"].update(every_n_epochs=1, n_samples=2, batch_size=8)
    save_dataset(small_dataset, tmp_path / "data.npz")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    main(["--seed", "42", "--dataset", str(tmp_path / "data.npz"), "--device", "cpu",
          "--config", str(tmp_path / "config.yaml"), "--out", str(tmp_path / "run"), *VARIANT_FLAGS[name]])
    saved = yaml.safe_load((tmp_path / "run" / "config.yaml").read_text())
    weights = torch.load(tmp_path / "run" / "model.pt")
    assert variant_name(saved) == name
    assert any(k.startswith("mask_encoder.") for k in weights) == ("ext1" in name)
    assert any(k.startswith("time_encoder.") for k in weights) == (name in ("ext2", "ext1+2"))

# --- every variant trains and samples on real data, at the gate's width and depth ----------

@needs_raw_data
@pytest.mark.parametrize("name", list(VARIANT_FLAGS))
def test_every_variant_runs_forward_backward_and_samples_on_set_a(real, name):
    # The gate's width and depth; 5 diffusion steps, not 50, as in the other sampler tests (the
    # step count changes neither a shape nor a path, only how long sampling takes).
    model = train_steps(built(name, num_steps=5).train(), real, steps=2)
    loss = model(batch_of(real), generator=generator())
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)

    sampler, (values, mask) = EvalSampler(2021, 2, 16), (real["values"].numpy(), real["mask"].numpy())
    cond, _ = EvalMasker.from_config()(mask, list(range(len(mask))))
    samples = sampler(model, values, mask, cond)
    assert samples.shape == (2, 16, T, D) and torch.isfinite(torch.from_numpy(samples)).all()

@needs_raw_data
def test_with_both_on_nonzero_gradients_reach_every_weight_of_both_encoders(real):
    model = train_steps(built("ext1+2"), real, steps=2)
    model(batch_of(real), generator=generator()).backward()
    for encoder in (model.mask_encoder, model.time_encoder):
        grads = {name: p.grad for name, p in encoder.named_parameters()}
        assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads.values())
    assert model.mask_encoder.gate.item() != 0 and model.time_encoder.gate.item() != 0     # both opened

def test_base_has_neither_encoder_and_upstreams_weights_only():
    base = built("base")
    assert base.mask_encoder is None and base.time_encoder is None
    assert not any(k.startswith(("mask_encoder.", "time_encoder.")) for k in base.state_dict())
