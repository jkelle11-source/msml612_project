import numpy as np
import pytest
import torch
import torch.nn as nn
from conftest import CONFIG, csdi_config, nonzero_output_layer
from eval import EvalMasker, EvalSampler
from models.csdi import CSDI_Physio

N, T, D = 6, 48, 35
STEPS = 4                               # reverse steps; upstream's 50 would only make these slower
FROZEN_SEED = CONFIG["protocol"]["eval_sampling_seed"]
ACCELERATOR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None

def build_model(device="cpu", num_steps=STEPS):
    torch.manual_seed(0)
    model = CSDI_Physio(csdi_config(num_steps=num_steps), device=device, target_dim=D).to(device)
    return nonzero_output_layer(model).eval()

def sampler(seed=FROZEN_SEED, n_samples=2, batch_size=4):
    return EvalSampler(seed, n_samples, batch_size)

def as_batch(values, mask, cond, rows=slice(None)):
    return {
        "observed_data": torch.from_numpy(values[rows]),
        "observed_mask": torch.from_numpy(mask[rows]),
        "gt_mask": torch.from_numpy(cond[rows]),
        "timepoints": torch.arange(T).float().expand(len(values[rows]), T),
    }

class ZeroDenoiser(nn.Module):
    """Predicts no noise at all, which leaves the sampler's own arithmetic easy to do by hand."""

    def forward(self, total_input, side_info, diffusion_step, *, mask_emb=None, time_enc=None):
        return torch.zeros(total_input.shape[0], *total_input.shape[2:], device=total_input.device)

@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    mask = (rng.random((N, T, D)) < 0.2).astype(np.float32)
    values = (rng.normal(size=(N, T, D)) * mask).astype(np.float32)
    cond, target = EvalMasker.from_config()(mask, np.arange(N))
    return values, mask, cond, target

# --- config ---------------------------------------------------------------------------

def test_from_config_reads_the_frozen_protocol():
    from_config = EvalSampler.from_config()
    assert from_config.seed == CONFIG["protocol"]["eval_sampling_seed"] == 2021
    assert from_config.n_samples == CONFIG["protocol"]["eval_n_samples"] == 100
    assert from_config.batch_size == CONFIG["protocol"]["eval_batch_size"] == 16

def test_sampling_seed_is_not_any_other_seed():
    others = [*CONFIG["protocol"]["run_seeds"], CONFIG["protocol"]["eval_masker_seed"], CONFIG["data"]["split_seed"]]
    assert FROZEN_SEED not in others

# --- what comes back ------------------------------------------------------------------

def test_samples_come_back_as_k_n_t_d(data):
    values, mask, cond, _ = data
    model = build_model()
    samples = sampler(n_samples=3)(model, values, mask, cond)
    assert samples.shape == (3, N, T, D) and samples.dtype == np.float32
    assert np.isfinite(samples).all()

    generator = torch.Generator().manual_seed(FROZEN_SEED)      # the same pass, straight through upstream
    upstream = torch.cat([model.evaluate(as_batch(values, mask, cond, rows), 3, generator=generator)[0]
                          for rows in (slice(0, 4), slice(4, 6))])                  # (N, K, D, T)
    for k, n, t, d in [(0, 0, 0, 0), (2, 5, 47, 34), (1, 4, 10, 3)]:
        assert samples[k, n, t, d] == upstream[n, k, d, t]
    assert np.array_equal(samples, upstream.permute(1, 0, 3, 2).numpy())

def test_torch_tensors_are_accepted_too(data):
    values, mask, cond, _ = data
    model = build_model()
    from_tensors = sampler()(model, *(torch.from_numpy(a) for a in (values, mask, cond)))
    assert np.array_equal(from_tensors, sampler()(model, values, mask, cond))

# --- the frozen-noise guarantees --------------------------------------------------------

def test_every_pass_gives_the_same_samples(data):
    values, mask, cond, _ = data
    model, one_sampler = build_model(), sampler()
    passes = []
    for global_seed in (0, 1, 2):
        torch.manual_seed(global_seed)                          # run seed, epoch, whatever came before
        np.random.seed(global_seed)
        passes.append(one_sampler(model, values, mask, cond))   # the same object, pass after pass
    passes.append(sampler()(model, values, mask, cond))
    for later in passes[1:]:
        assert later.tobytes() == passes[0].tobytes()

def test_the_frozen_seed_is_what_decides_the_noise(data):
    values, mask, cond, _ = data
    model = build_model()
    assert not np.array_equal(sampler(2021)(model, values, mask, cond), sampler(2022)(model, values, mask, cond))

def test_a_pass_leaves_the_global_rng_alone(data):
    values, mask, cond, _ = data
    model = build_model().train()                               # as when validating mid-training
    torch.manual_seed(5)
    before = torch.get_rng_state()
    sampler()(model, values, mask, cond)
    assert torch.equal(torch.get_rng_state(), before)           # training's dropout stream is not shifted

def test_a_pass_runs_in_eval_mode_and_hands_the_model_back_as_it_was(data):
    values, mask, cond, _ = data
    model = build_model()
    in_eval_mode = sampler()(model, values, mask, cond)
    assert not model.training
    model.train()
    assert np.array_equal(sampler()(model, values, mask, cond), in_eval_mode)       # so dropout was off
    assert model.training

def test_the_model_is_handed_back_even_if_the_pass_fails(data, monkeypatch):
    values, mask, cond, _ = data
    model = build_model().train()

    def broken(*args, **kwargs):
        raise RuntimeError("sampler failed")

    monkeypatch.setattr(model, "evaluate", broken)
    with pytest.raises(RuntimeError):
        sampler()(model, values, mask, cond)
    assert model.training

# --- no leakage -------------------------------------------------------------------------

def test_held_out_values_cannot_reach_the_samples(data):
    values, mask, cond, target = data
    model = build_model()
    samples = sampler()(model, values, mask, cond)

    tampered = values.copy()
    tampered[target == 1] = 1e3                                 # change every value the model must not see
    assert np.array_equal(sampler()(model, tampered, mask, cond), samples)

    tampered = values.copy()
    tampered[cond == 1] += 1.0                                  # ...while what it does see matters
    assert not np.array_equal(sampler()(model, tampered, mask, cond), samples)

# --- the vendored sampler: every draw comes from the generator ---------------------------

def test_sampler_noise_by_hand_from_the_generator(data):
    values, mask, cond, _ = data
    model = build_model(num_steps=3)
    model.diffmodel = ZeroDenoiser()
    batch = as_batch(values, mask, cond, slice(0, 4))
    torch.manual_seed(0)
    samples = model.evaluate(batch, 2, generator=torch.Generator().manual_seed(7))[0]    # (B, K, D, T)

    generator, by_hand = torch.Generator().manual_seed(7), []
    for _ in range(2):
        x = torch.randn(4, D, T, generator=generator)           # each sample starts from pure noise
        for t in (2, 1, 0):
            x = x / model.alpha_hat[t] ** 0.5                   # the denoiser predicted 0
            if t > 0:
                sigma = ((1 - model.alpha[t - 1]) / (1 - model.alpha[t]) * model.beta[t]) ** 0.5
                x = x + sigma * torch.randn(4, D, T, generator=generator)
        by_hand.append(x)
    assert torch.allclose(samples, torch.stack(by_hand, dim=1).float(), rtol=1e-5, atol=1e-6)

def test_without_a_generator_sampling_draws_from_the_global_rng_as_upstream(data):
    values, mask, cond, _ = data
    model, batch = build_model(), as_batch(values, mask, cond, slice(0, 4))

    def draw(global_seed):
        torch.manual_seed(global_seed)
        return model.evaluate(batch, 2)[0]

    assert torch.equal(draw(1), draw(1))
    assert not torch.equal(draw(1), draw(2))

@pytest.mark.skipif(ACCELERATOR is None, reason="no GPU available")
def test_a_cpu_generator_gives_the_gpu_the_same_samples(data):
    values, mask, cond, _ = data
    seen = {}
    for device in ("cpu", ACCELERATOR):
        model = build_model(device=device)
        model.diffmodel = ZeroDenoiser()
        seen[device] = sampler()(model, values, mask, cond)
    assert np.allclose(seen["cpu"], seen[ACCELERATOR], rtol=1e-4, atol=1e-5)
