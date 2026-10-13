import pytest
import torch
from models.csdi import CSDI_Physio, diff_CSDI

B, T, D = 4, 48, 35

@pytest.fixture
def config() -> dict:
    # Upstream config/base.yaml at the pinned commit.
    return {
        "diffusion": {
            "layers": 4,
            "channels": 64,
            "nheads": 8,
            "diffusion_embedding_dim": 128,
            "beta_start": 0.0001,
            "beta_end": 0.5,
            "num_steps": 50,
            "schedule": "quad",
            "is_linear": False,
        },
        "model": {
            "is_unconditional": 0,
            "timeemb": 128,
            "featureemb": 16,
            "target_strategy": "random",
        },
    }

@pytest.fixture
def toy_batch() -> dict[str, torch.Tensor]:
    # Synthetic stand-in for the pipeline output: (B, 48, 35), about 80% missing,
    # values zeroed where unobserved.
    generator = torch.Generator().manual_seed(0)
    observed_mask = (torch.rand(B, T, D, generator=generator) < 0.2).float()
    observed_data = torch.randn(B, T, D, generator=generator) * observed_mask
    gt_mask = observed_mask * (torch.rand(B, T, D, generator=generator) < 0.9).float()
    return {
        "observed_data": observed_data,
        "observed_mask": observed_mask,
        "gt_mask": gt_mask,
        "timepoints": torch.arange(T).float().expand(B, T),
    }

def test_denoiser_forward_shape(config):
    denoiser = diff_CSDI({**config["diffusion"], "side_dim": 145}, inputdim=2)
    x = torch.randn(B, 2, D, T)
    cond_info = torch.randn(B, 145, D, T)
    diffusion_step = torch.randint(0, 50, (B,))
    predicted = denoiser(x, cond_info, diffusion_step)
    assert predicted.shape == (B, D, T)
    assert torch.isfinite(predicted).all()

def test_toy_batch_trains_without_nans(config, toy_batch):
    torch.manual_seed(0)
    model = CSDI_Physio(config, device="cpu", target_dim=D)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    for _ in range(5):
        optimizer.zero_grad()
        loss = model(toy_batch)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    # After the last step every parameter that received a gradient has a finite one,
    # and the gradient reached the first layer of the denoiser.
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
        assert torch.isfinite(p).all(), name
    assert model.diffmodel.input_projection.weight.grad.abs().sum() > 0
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters())

def test_validation_loss_and_imputation_run(config, toy_batch):
    torch.manual_seed(0)
    config["diffusion"]["num_steps"] = 5
    model = CSDI_Physio(config, device="cpu", target_dim=D).eval()
    with torch.no_grad():
        assert torch.isfinite(model(toy_batch, is_train=0))
    samples, observed_data, target_mask, _, _ = model.evaluate(toy_batch, n_samples=2)
    assert samples.shape == (B, 2, D, T)
    assert observed_data.shape == target_mask.shape == (B, D, T)
    assert torch.isfinite(samples).all()
