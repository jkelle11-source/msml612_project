import pytest
import torch
from conftest import CONFIG, RAW_DIR, nonzero_output_layer
from diffusion.data import to_batch
from eval import TrainMasker
from models.build import build_model

B, T, D = 4, 48, 35
C = CONFIG["diffusion"]["channels"]     # upstream's name for d_model
needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")

def model_with(num_steps=50):
    torch.manual_seed(0)
    config = {**CONFIG, "diffusion": {**CONFIG["diffusion"], "num_steps": num_steps}}
    return nonzero_output_layer(build_model(config, "cpu")).eval()     # eval: no dropout

def hook(seed):
    return torch.randn(B, T, D, C, generator=torch.Generator().manual_seed(seed))

def generator():
    return torch.Generator().manual_seed(7)

@pytest.fixture
def inputs() -> dict[str, torch.Tensor]:
    # Same stand-in for the pipeline output as the other model tests: (B, 48, 35), about 80% missing.
    g = torch.Generator().manual_seed(0)
    observed_mask = (torch.rand(B, T, D, generator=g) < 0.2).float()
    cond_mask, _ = TrainMasker(42)(observed_mask)
    return {"values": torch.randn(B, T, D, generator=g) * observed_mask, "mask": observed_mask, "cond_mask": cond_mask}

def model_view(model, inputs):
    """What CSDI_base works on internally: data and masks as (B, D, T), plus the side information."""
    observed_data, observed_mask, timepoints, cond_mask, _, _ = model.process_data(
        to_batch(inputs["values"], inputs["mask"], inputs["cond_mask"])
    )
    return observed_data, cond_mask, observed_mask, model.get_side_info(timepoints, cond_mask)

def loss(model, inputs, is_train=1, **hooks):
    observed_data, cond_mask, observed_mask, side_info = model_view(model, inputs)
    loss_func = model.calc_loss if is_train == 1 else model.calc_loss_valid
    return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train, generator=generator(), **hooks)

def samples(model, inputs, **hooks):
    observed_data, cond_mask, _, side_info = model_view(model, inputs)
    with torch.no_grad():
        return model.impute(observed_data, cond_mask, side_info, n_samples=2, generator=generator(), **hooks)

# --- None is the base model, exactly -----------------------------------------------------

def test_without_hooks_the_loss_is_the_frozen_training_calls(inputs):
    model = model_with()
    batch = to_batch(inputs["values"], inputs["mask"], inputs["cond_mask"])
    assert torch.equal(loss(model, inputs, mask_emb=None, time_enc=None), model(batch, generator=generator()))

def test_without_hooks_the_samples_are_the_frozen_evaluation_calls(inputs):
    model = model_with(num_steps=5)
    batch = to_batch(inputs["values"], inputs["mask"], inputs["cond_mask"])
    assert torch.equal(samples(model, inputs, mask_emb=None, time_enc=None), model.evaluate(batch, 2, generator=generator())[0])

def test_zero_hooks_change_nothing(inputs):
    model = model_with()
    zeros = torch.zeros(B, T, D, C)
    assert torch.equal(loss(model, inputs, mask_emb=zeros, time_enc=zeros), loss(model, inputs))

# --- a hook reaches the denoiser on every path ------------------------------------------

@pytest.mark.parametrize("name", ["mask_emb", "time_enc"])
def test_a_nonzero_hook_changes_the_training_loss(inputs, name):
    model = model_with()
    assert loss(model, inputs, **{name: hook(1)}) != loss(model, inputs)

@pytest.mark.parametrize("name", ["mask_emb", "time_enc"])
def test_a_nonzero_hook_changes_the_validation_loss(inputs, name):
    model = model_with(num_steps=5)
    with torch.no_grad():
        assert loss(model, inputs, is_train=0, **{name: hook(1)}) != loss(model, inputs, is_train=0)

@pytest.mark.parametrize("name", ["mask_emb", "time_enc"])
def test_a_nonzero_hook_changes_the_samples(inputs, name):
    model = model_with(num_steps=5)
    assert not torch.equal(samples(model, inputs, **{name: hook(1)}), samples(model, inputs))

def test_every_denoiser_call_receives_the_hooks(inputs, monkeypatch):
    model = model_with(num_steps=5)
    received = []
    denoise = model.diffmodel.forward

    def spy(*args, **kwargs):
        received.append((kwargs["mask_emb"], kwargs["time_enc"]))
        return denoise(*args, **kwargs)

    monkeypatch.setattr(model.diffmodel, "forward", spy)
    mask_emb, time_enc = hook(1), hook(2)
    loss(model, inputs, mask_emb=mask_emb, time_enc=time_enc)
    with torch.no_grad():
        loss(model, inputs, is_train=0, mask_emb=mask_emb, time_enc=time_enc)
    samples(model, inputs, mask_emb=mask_emb, time_enc=time_enc)
    assert len(received) == 1 + 5 + 2 * 5     # one training step, 5 validation steps, 5 reverse steps per sample
    assert all(m is mask_emb and t is time_enc for m, t in received)

# --- how a hook enters: added, each vector at its own hour and feature ------------------

def test_the_two_hooks_add_together(inputs):
    model = model_with()
    a, b = hook(1), hook(2)
    both = loss(model, inputs, mask_emb=a, time_enc=b)
    assert both.item() == pytest.approx(loss(model, inputs, mask_emb=a + b).item(), rel=1e-5)
    assert both.item() == pytest.approx(loss(model, inputs, time_enc=a + b).item(), rel=1e-5)

def test_a_hook_vector_lands_at_its_own_hour_and_feature():
    denoiser = model_with().diffmodel
    torch.manual_seed(0)
    x, step = torch.randn(B, 2, D, T), torch.randint(0, 50, (B,))
    cond_info = torch.randn(B, denoiser.residual_layers[0].cond_projection.in_channels, D, T)
    seen = []                                                   # the stream entering the first residual block
    denoiser.residual_layers[0].register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    hour, feature = 5, 3
    one_cell = torch.zeros(B, T, D, C)
    one_cell[:, hour, feature] = hook(1)[:, hour, feature]      # a different vector for each patient
    denoiser(x, cond_info, step)
    denoiser(x, cond_info, step, mask_emb=one_cell)
    added = seen[1] - seen[0]                                   # (B, C, D, T)
    expected = torch.zeros(B, C, D, T)
    expected[:, :, feature, hour] = one_cell[:, hour, feature]
    assert torch.allclose(added, expected, atol=1e-6)

# (B, T, C) is one vector per hour; the next two have the hour and feature axes swapped.
@pytest.mark.parametrize("shape", [(B, T, C), (B, D, T, C), (B, C, D, T), (B, T, D, C + 1)])
@pytest.mark.parametrize("name", ["mask_emb", "time_enc"])
def test_a_hook_of_the_wrong_shape_is_refused(inputs, name, shape):
    model = model_with()
    with pytest.raises(ValueError, match=f"{name} must be"):
        loss(model, inputs, **{name: torch.ones(shape)})

# --- a real Set A batch -----------------------------------------------------------------

@needs_raw_data
def test_real_batch_trains_through_the_hooks(set_a):
    rows = set_a["train_idx"][:8]
    values, mask = torch.from_numpy(set_a["values"][rows]), torch.from_numpy(set_a["mask"][rows])
    assert values.shape == (8, T, D)
    real = {"values": values, "mask": mask, "cond_mask": TrainMasker(42)(mask)[0]}
    model = model_with().train()

    # The frozen training call, no hooks.
    model(to_batch(real["values"], real["mask"], real["cond_mask"]), generator=generator()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    # The same batch through the hooks: the gradient reaches them, as an extension's weights will need.
    model.zero_grad()
    mask_emb = torch.randn(8, T, D, C, requires_grad=True)
    time_enc = torch.randn(8, T, D, C, requires_grad=True)
    hooked = loss(model, real, mask_emb=mask_emb, time_enc=time_enc)
    assert torch.isfinite(hooked)
    hooked.backward()
    for h in (mask_emb, time_enc):
        assert torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0
