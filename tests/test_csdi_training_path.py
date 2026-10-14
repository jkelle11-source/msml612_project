import numpy as np
import pytest
import torch
import torch.nn as nn
from conftest import csdi_config, nonzero_output_layer
from eval import TrainMasker
from models.csdi import CSDI_Physio

B, T, D = 4, 48, 35
ACCELERATOR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None

def build_model(target_strategy="external", device="cpu"):
    torch.manual_seed(0)
    return CSDI_Physio(csdi_config(target_strategy), device=device, target_dim=D).to(device).eval()   # eval: no dropout

def generator(seed=7):
    return torch.Generator().manual_seed(seed)

class FixedDenoiser(nn.Module):
    """Stands in for the denoiser: remembers what it was given, returns a tensor we control."""

    def __init__(self, device="cpu"):
        super().__init__()
        self.predicted = torch.zeros(B, D, T, device=device, requires_grad=True)

    def forward(self, total_input, side_info, diffusion_step):
        self.total_input, self.side_info, self.diffusion_step = total_input, side_info, diffusion_step
        return self.predicted

@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    # Same stand-in for the pipeline output as the smoke test: (B, 48, 35), about 80% missing.
    generator = torch.Generator().manual_seed(0)
    observed_mask = (torch.rand(B, T, D, generator=generator) < 0.2).float()
    return {
        "observed_data": torch.randn(B, T, D, generator=generator) * observed_mask,
        "observed_mask": observed_mask,
        "timepoints": torch.arange(T).float().expand(B, T),
    }

def with_mask(batch, cond_mask):
    return {**batch, "gt_mask": cond_mask}

def loss_with(model, batch, cond_mask, seed=1):
    torch.manual_seed(seed)             # same timestep and noise draws for every call
    np.random.seed(seed)                # upstream's get_randmask also draws from NumPy
    return model(with_mask(batch, cond_mask)).item()

# --- the mask we hand over is the one that gets used -----------------------------------

def test_training_loss_follows_the_mask_it_is_given(batch):
    model = build_model()
    cond_a, _ = TrainMasker(42)(batch["observed_mask"])
    cond_b, _ = TrainMasker(43)(batch["observed_mask"])
    assert loss_with(model, batch, cond_a) == loss_with(model, batch, cond_a)
    assert loss_with(model, batch, cond_a) != loss_with(model, batch, cond_b)

def test_upstreams_own_masking_is_off_the_training_path(batch, monkeypatch):
    model = build_model()

    def forbidden(*args, **kwargs):
        raise AssertionError("upstream masking ran on the training path")

    monkeypatch.setattr(model, "get_randmask", forbidden)
    monkeypatch.setattr(model, "get_hist_mask", forbidden)
    cond, _ = TrainMasker(42)(batch["observed_mask"])
    assert torch.isfinite(model(with_mask(batch, cond)))

def test_loss_is_scored_on_exactly_the_train_masker_targets(batch):
    model = build_model()
    model.diffmodel = FixedDenoiser()
    cond, target = TrainMasker(42)(batch["observed_mask"])
    model(with_mask(batch, cond)).backward()
    scored = (model.diffmodel.predicted.grad != 0).float().permute(0, 2, 1)     # back to (B, T, D)
    assert torch.equal(scored, target)

def test_denoiser_is_shown_only_the_conditioning_values(batch):
    model = build_model()
    model.diffmodel = FixedDenoiser()
    cond, target = TrainMasker(42)(batch["observed_mask"])
    model(with_mask(batch, cond))
    values_channel = model.diffmodel.total_input[:, 0].permute(0, 2, 1)         # (B, T, D)
    mask_channel = model.diffmodel.side_info[:, -1].permute(0, 2, 1)
    assert torch.equal(values_channel, batch["observed_data"] * cond)
    assert torch.equal(mask_channel, cond)
    assert (values_channel[target == 1] == 0).all() and batch["observed_data"][target == 1].abs().sum() > 0

def test_validation_still_conditions_on_the_mask_it_is_given(batch):
    model = build_model()
    model.diffmodel = FixedDenoiser()
    cond, _ = TrainMasker(42)(batch["observed_mask"])
    with torch.no_grad():
        model(with_mask(batch, cond), is_train=0)
    assert torch.equal(model.diffmodel.side_info[:, -1].permute(0, 2, 1), cond)

# --- timestep and noise come from the generator we pass (seed pairing, §3.3) -------------

def test_generator_decides_the_draws_and_the_global_rng_does_not(batch):
    model = nonzero_output_layer(build_model())
    cond, _ = TrainMasker(42)(batch["observed_mask"])

    def loss(global_seed, generator_seed):
        torch.manual_seed(global_seed)
        return model(with_mask(batch, cond), generator=generator(generator_seed)).item()

    assert loss(0, 7) == loss(1, 7)
    assert loss(0, 7) != loss(0, 8)

def test_timestep_then_noise_are_drawn_from_the_generator(batch):
    model = build_model()
    model.diffmodel = FixedDenoiser()                           # predicts 0, so loss = mean noise^2 on targets
    cond, target = TrainMasker(42)(batch["observed_mask"])
    loss = model(with_mask(batch, cond), generator=generator())

    expected = generator()
    timestep = torch.randint(0, 50, [B], generator=expected)
    noise = torch.randn(B, D, T, generator=expected)
    assert torch.equal(model.diffmodel.diffusion_step, timestep)
    by_hand = (noise ** 2 * target.permute(0, 2, 1)).sum() / target.sum()
    assert loss.item() == pytest.approx(by_hand.item(), rel=1e-6)

def test_validation_noise_comes_from_the_generator_too(batch):
    model = nonzero_output_layer(build_model())
    cond, _ = TrainMasker(42)(batch["observed_mask"])
    losses = []
    for global_seed in (0, 1):
        torch.manual_seed(global_seed)
        with torch.no_grad():
            losses.append(model(with_mask(batch, cond), is_train=0, generator=generator()).item())
    assert losses[0] == losses[1]

def test_without_a_generator_the_draws_are_upstreams_global_ones(batch):
    model = nonzero_output_layer(build_model())
    cond, _ = TrainMasker(42)(batch["observed_mask"])
    assert loss_with(model, batch, cond, seed=1) == loss_with(model, batch, cond, seed=1)
    assert loss_with(model, batch, cond, seed=1) != loss_with(model, batch, cond, seed=2)

@pytest.mark.skipif(ACCELERATOR is None, reason="no GPU available")
def test_a_cpu_generator_gives_the_gpu_the_same_draws(batch):
    cond, _ = TrainMasker(42)(batch["observed_mask"])
    seen = {}
    for device in ("cpu", ACCELERATOR):
        model = build_model(device=device)
        model.diffmodel = FixedDenoiser(device)
        loss = model(with_mask(batch, cond), generator=generator())
        assert loss.device.type == device
        seen[device] = (model.diffmodel.diffusion_step.cpu(), loss.item())
    assert torch.equal(seen["cpu"][0], seen[ACCELERATOR][0])
    assert seen["cpu"][1] == pytest.approx(seen[ACCELERATOR][1], rel=1e-5)

# --- upstream's own strategy is left as published ---------------------------------------

def test_random_strategy_still_ignores_the_mask_as_upstream_does(batch):
    model = build_model("random")
    cond_a, _ = TrainMasker(42)(batch["observed_mask"])
    cond_b, _ = TrainMasker(43)(batch["observed_mask"])
    assert loss_with(model, batch, cond_a) == loss_with(model, batch, cond_b)
