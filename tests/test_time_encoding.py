import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import models.csdi.main_model as main_model
from conftest import CONFIG, nonzero_output_layer
from data.physionet import FEATURES, compute_tslo
from diffusion.data import to_batch
from eval import BlockEvalMasker, EvalMasker, EvalSampler, TrainMasker
from models.build import build_model
from models.time_encoding import tslo

ACCELERATOR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None

T, D = 48, 35
C = CONFIG["diffusion"]["channels"]     # upstream's name for d_model

def as_mask(rows):
    return torch.tensor(rows, dtype=torch.float32)

@pytest.fixture
def random_mask() -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.from_numpy((rng.random((5, 48, 35)) < 0.2).astype(np.float32))

# --- tslo: compute_tslo's rule, in torch --------------------------------------------------

def test_tslo_known_row():
    mask = as_mask([0, 0, 1, 0, 0, 1, 1, 0]).reshape(1, 8, 1)
    assert tslo(mask).reshape(8).tolist() == [1, 2, 0, 1, 2, 0, 0, 1]

def test_tslo_counts_each_patient_and_feature_separately():
    # (B, T, D) = (2, 4, 2): T != D, so walking the feature axis or stacking on the wrong
    # axis gives a different list.
    mask = as_mask([
        [[1, 0], [0, 0], [0, 1], [1, 0]],
        [[0, 0], [1, 0], [1, 0], [0, 0]],
    ])
    expected = [
        [[0, 1], [1, 2], [2, 0], [0, 1]],
        [[1, 1], [0, 2], [0, 3], [1, 4]],
    ]
    assert tslo(mask).tolist() == expected

def test_tslo_held_out_target_is_not_an_observation():
    # Test patient 139891, HR, hours 19-24: EvalMasker holds out hour 21. delta_obs, from the
    # observed mask, reads 0 there and so marks the target; δ from the conditioning mask
    # counts on from hour 20, the last value the model can see.
    observed = as_mask([1, 1, 1, 0, 0, 1]).reshape(1, 6, 1)
    cond = as_mask([1, 1, 0, 0, 0, 1]).reshape(1, 6, 1)
    assert compute_tslo(observed.numpy()).reshape(6).tolist() == [0, 0, 0, 1, 2, 0]
    assert tslo(cond).reshape(6).tolist() == [0, 0, 1, 2, 3, 0]

def test_tslo_matches_compute_tslo(random_mask):
    assert np.array_equal(tslo(random_mask).numpy(), compute_tslo(random_mask.numpy()))

def test_tslo_leaves_cond_mask_alone(random_mask):
    before = random_mask.clone()
    tslo(random_mask)
    assert torch.equal(random_mask, before)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tslo_keeps_shape_and_dtype(random_mask, dtype):
    delta = tslo(random_mask.to(dtype))
    assert delta.shape == (5, 48, 35)
    assert delta.dtype == dtype

@pytest.mark.skipif(ACCELERATOR is None, reason="no GPU available")
def test_tslo_runs_on_the_mask_device(random_mask):
    delta = tslo(random_mask.to(ACCELERATOR))
    assert delta.device.type == ACCELERATOR
    assert torch.equal(delta.cpu(), tslo(random_mask))

# --- δ inside the model: from the conditioning mask, on every path (M5-1) -----------------

class DeltaProbe(nn.Module):
    """Stands in for M5-2's ContinuousTimeEncoding: keeps every δ the model gives it and
    returns a time_enc built from δ, so the samples depend on δ too."""
    def __init__(self):
        super().__init__()
        self.deltas = []

    def forward(self, delta):
        self.deltas.append(delta.clone())
        return (delta / T)[..., None].expand(-1, -1, -1, C)

def model_with(probe=True, device="cpu", num_steps=3):
    config = copy.deepcopy(CONFIG)
    config["diffusion"]["num_steps"] = num_steps
    torch.manual_seed(0)
    model = build_model(config, device).eval()                      # eval: no dropout
    if probe:
        model.time_encoder = DeltaProbe()
    return model

def generator():
    return torch.Generator().manual_seed(7)

PATHS = ["training", "validation", "sampler"]

def run_path(model, batch, path):
    """One call on the training-loss path, the validation-loss path (every diffusion step), or
    the sampler path (what EvalSampler scores)."""
    if path == "training":
        return model(batch, generator=generator())
    if path == "validation":
        with torch.no_grad():
            return model(batch, is_train=0, generator=generator())
    return model.evaluate(batch, 2, generator=generator())[0]

@pytest.fixture
def inputs() -> dict[str, torch.Tensor]:
    # Same stand-in for the pipeline output as the other model tests: (B, 48, 35), about 80% missing.
    g = torch.Generator().manual_seed(0)
    observed_mask = (torch.rand(4, T, D, generator=g) < 0.2).float()
    cond_mask, _ = TrainMasker(42)(observed_mask)
    return {"values": torch.randn(4, T, D, generator=g) * observed_mask, "mask": observed_mask, "cond_mask": cond_mask}

@pytest.fixture
def held_out_hr() -> dict[str, torch.Tensor]:
    """One patient with only HR observed, at hours 2, 3, 5 and 6; hour 3 is held out."""
    hr = list(FEATURES).index("HR")
    observed, cond = torch.zeros(1, T, D), torch.zeros(1, T, D)
    observed[0, [2, 3, 5, 6], hr] = 1
    cond[0, [2, 5, 6], hr] = 1
    return {"values": observed.clone(), "mask": observed, "cond_mask": cond}

def batch_of(inputs, **replace):
    inputs = {**inputs, **replace}
    return to_batch(inputs["values"], inputs["mask"], inputs["cond_mask"])

@pytest.mark.parametrize("path", PATHS)
def test_model_delta_counts_past_a_held_out_target(held_out_hr, path):
    hr = list(FEATURES).index("HR")
    model = model_with()
    run_path(model, batch_of(held_out_hr), path)
    (delta,) = model.time_encoder.deltas
    assert compute_tslo(held_out_hr["mask"].numpy())[0, 3, hr] == 0     # delta_obs: "observed"
    assert delta[0, :, hr].tolist() == [1, 2, 0, 1, 2, 0, 0] + list(range(1, 42))
    others = [d for d in range(D) if d != hr]                           # never visible: 1 ... 48
    assert torch.equal(delta[0][:, others], torch.arange(1.0, T + 1)[:, None].expand(T, D - 1))

@pytest.mark.parametrize("path", PATHS)
def test_model_delta_is_tslo_of_the_conditioning_mask_once_per_call(inputs, path):
    model = model_with()
    run_path(model, batch_of(inputs), path)
    assert len(model.time_encoder.deltas) == 1
    assert np.array_equal(model.time_encoder.deltas[0].numpy(), compute_tslo(inputs["cond_mask"].numpy()))

@pytest.mark.parametrize("path", PATHS)
def test_model_delta_ignores_observed_mask_and_follows_the_conditioning_mask(inputs, path):
    model = model_with()
    deltas = model.time_encoder.deltas
    run_path(model, batch_of(inputs), path)
    run_path(model, batch_of(inputs, mask=torch.ones_like(inputs["mask"])), path)  # targets everywhere
    assert torch.equal(deltas[0], deltas[1])

    flipped = inputs["cond_mask"].clone()
    flipped[0, 0, 0] = 1 - flipped[0, 0, 0]                         # ...while the conditioning mask matters
    run_path(model, batch_of(inputs, cond_mask=flipped), path)
    assert not torch.equal(deltas[0], deltas[2])

def test_time_enc_reaches_the_denoiser_on_every_path(inputs):
    model = model_with(num_steps=3)
    outputs, received = [], []
    model.time_encoder.register_forward_hook(lambda module, args, output: outputs.append(output))
    model.diffmodel.register_forward_pre_hook(
        lambda module, args, kwargs: received.append(kwargs["time_enc"]), with_kwargs=True)
    for path in PATHS:
        run_path(model, batch_of(inputs), path)
    assert len(outputs) == 3                                        # once per call, not per step
    expected = [outputs[0]] + [outputs[1]] * 3 + [outputs[2]] * 2 * 3
    assert len(received) == len(expected) and all(r is e for r, e in zip(received, expected))

def test_base_model_computes_no_delta(inputs, monkeypatch):
    def no_tslo(mask):
        raise AssertionError("the base model computed δ")
    monkeypatch.setattr(main_model, "tslo", no_tslo)
    model = model_with(probe=False)
    assert model.time_encoder is None
    assert model.get_time_enc(inputs["cond_mask"].permute(0, 2, 1)) is None
    for path in PATHS:
        run_path(model, batch_of(inputs), path)

def test_sampling_with_delta_present_is_finite_and_depends_on_it(inputs):
    # Upstream zero-initialises the denoiser's last layer, which would make the samples ignore
    # time_enc; nonzero_output_layer gives it the weights it has after its first update.
    model = nonzero_output_layer(model_with(num_steps=5))
    with_delta = run_path(model, batch_of(inputs), "sampler")
    model.time_encoder = None
    without = run_path(model, batch_of(inputs), "sampler")
    assert with_delta.shape == (4, 2, D, T) and torch.isfinite(with_delta).all()
    assert not torch.equal(with_delta, without)

@pytest.mark.parametrize("masker", [EvalMasker, BlockEvalMasker])
def test_eval_sampler_gives_the_model_delta_of_the_holdouts_conditioning_mask(small_dataset, masker):
    rows = small_dataset["test_idx"][:5]
    values, mask = small_dataset["values"][rows], small_dataset["mask"][rows]
    cond_mask, target = masker.from_config()(mask, small_dataset["record_ids"][rows])
    model = model_with()
    EvalSampler(seed=0, n_samples=2, batch_size=2)(model, values, mask, cond_mask)
    assert len(model.time_encoder.deltas) == 3                      # batches of 2, 2 and 1
    delta = torch.cat(model.time_encoder.deltas).numpy()
    assert np.array_equal(delta, compute_tslo(cond_mask))
    assert target.any() and (delta[target == 1] > 0).all()          # no held-out target reads "observed"

@pytest.mark.skipif(ACCELERATOR is None, reason="no GPU available")
@pytest.mark.parametrize("path", ["training", "sampler"])
def test_model_delta_is_computed_on_the_batch_device(inputs, path):
    model = model_with(device=ACCELERATOR)
    run_path(model, batch_of(inputs), path)
    (delta,) = model.time_encoder.deltas
    assert delta.device.type == ACCELERATOR
    assert np.array_equal(delta.cpu().numpy(), compute_tslo(inputs["cond_mask"].numpy()))
