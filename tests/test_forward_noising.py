import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
from conftest import CONFIG
from diffusion.data import to_batch
from models.build import build_model

B, T, D = 64, 48, 35        # 64 * 48 * 35 = 107,520 noised values per diffusion step
X0 = 1.5                    # nonzero, so a wrongly scaled signal shows in the mean

class RecordingDenoiser(nn.Module):
    """Stands in for the denoiser: records the noisy values it is shown at each step, predicts 0."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, total_input, side_info, diffusion_step, *, mask_emb=None, time_enc=None):
        # Channel 1 is (1 - cond_mask) * x_t; with an all-zero conditioning mask it is all of x_t.
        self.seen.append((diffusion_step.clone(), total_input[:, 1].clone()))
        return torch.zeros_like(total_input[:, 1])

def alpha_bar_by_hand(diffusion):
    """The quad schedule written out: beta_t on a straight line in sqrt space, alpha_bar_t = prod(1 - beta_s)."""
    root_start, root_end, steps = diffusion["beta_start"] ** 0.5, diffusion["beta_end"] ** 0.5, diffusion["num_steps"]
    alpha_bar, running = [], 1.0
    for t in range(steps):
        beta = (root_start + t / (steps - 1) * (root_end - root_start)) ** 2
        running *= 1 - beta
        alpha_bar.append(running)
    return np.array(alpha_bar)

@pytest.fixture(scope="module")
def noised():
    """Our config's model, its denoiser swapped for the recorder, run once through every diffusion step."""
    model = build_model(copy.deepcopy(CONFIG), "cpu")
    model.diffmodel = RecordingDenoiser()
    x0 = torch.full((B, T, D), X0)
    batch = to_batch(x0, torch.ones(B, T, D), torch.zeros(B, T, D))
    with torch.no_grad():
        model(batch, is_train=0, generator=torch.Generator().manual_seed(0))   # validation loss: every t once
    return model, model.diffmodel.seen

# --- the schedule ------------------------------------------------------------------------

def test_schedule_is_the_quad_schedule_from_the_config(noised):
    model, _ = noised
    diffusion = CONFIG["diffusion"]
    alpha_bar = alpha_bar_by_hand(diffusion)
    assert model.beta[0] == pytest.approx(diffusion["beta_start"])
    assert model.beta[-1] == pytest.approx(diffusion["beta_end"])
    # Upstream's names are swapped from the DDPM paper's: model.alpha is the cumulative
    # product (alpha_bar), model.alpha_hat the per-step 1 - beta.
    np.testing.assert_allclose(model.alpha, alpha_bar, rtol=1e-12)
    # alpha_torch is the float32 copy the noising actually multiplies by.
    np.testing.assert_allclose(model.alpha_torch.flatten().numpy(), alpha_bar, rtol=1e-6)

# --- the noising -------------------------------------------------------------------------

def test_every_diffusion_step_is_noised_once_in_order(noised):
    _, seen = noised
    assert [int(step[0]) for step, _ in seen] == list(range(CONFIG["diffusion"]["num_steps"]))
    assert all((step == step[0]).all() for step, _ in seen)

def test_noised_values_have_the_closed_form_mean_and_variance_at_every_step(noised):
    # q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, 1 - alpha_bar_t), so the residual below is pure
    # noise with mean 0 and variance 1 - alpha_bar_t. Tolerances are ~5 standard errors of each
    # estimate at n = 107,520, so they would hold for almost any seed, not just this one.
    _, seen = noised
    alpha_bar = alpha_bar_by_hand(CONFIG["diffusion"])
    n = B * T * D
    for t, (_, x_t) in enumerate(seen):
        variance = 1 - alpha_bar[t]
        residual = x_t.double() - alpha_bar[t] ** 0.5 * X0
        assert abs(residual.mean().item()) < 5 * (variance / n) ** 0.5, f"t={t}: mean off"
        assert residual.var().item() / variance == pytest.approx(1, abs=0.02), f"t={t}: variance off"
