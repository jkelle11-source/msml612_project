import copy

import pytest
import torch
from conftest import CONFIG
from diffusion.data import to_batch
from diffusion.streams import RunStreams
from diffusion.train import train
from eval import TrainMasker
from models.build import build_model
from models.missingness import MissingnessEncoder

B, T, D = 4, 48, 35
C = CONFIG["diffusion"]["channels"]     # upstream's name for d_model

def config_with(ext1, **diffusion):
    config = copy.deepcopy(CONFIG)
    config["model"]["use_mask_embedding"] = ext1
    config["diffusion"].update(diffusion)
    return config

def model_with(ext1=True, seed=0, **diffusion):
    torch.manual_seed(seed)
    return build_model(config_with(ext1, **diffusion), "cpu")

def open_gate(enc):
    """The gate starts at 0 (Ext 1 starts as base), which makes every mask_emb zero. Tests of
    what the layers compute open it; the gate itself is tested on its own."""
    with torch.no_grad():
        enc.gate.fill_(1.0)
    return enc

def encoder(n_layers=2):
    torch.manual_seed(0)
    return open_gate(MissingnessEncoder(D, T, C, n_layers=n_layers)).eval()     # eval: no dropout

def generator():
    return torch.Generator().manual_seed(7)

@pytest.fixture
def inputs() -> dict[str, torch.Tensor]:
    # Same stand-in for the pipeline output as the other model tests: (B, 48, 35), about 80% missing.
    g = torch.Generator().manual_seed(0)
    observed_mask = (torch.rand(B, T, D, generator=g) < 0.2).float()
    cond_mask, _ = TrainMasker(42)(observed_mask)
    return {"values": torch.randn(B, T, D, generator=g) * observed_mask, "mask": observed_mask, "cond_mask": cond_mask}

def batch_of(inputs, **replace):
    inputs = {**inputs, **replace}
    return to_batch(inputs["values"], inputs["mask"], inputs["cond_mask"])

def warm_up(model, inputs, steps=2):
    """Training steps until gradients reach the whole encoder. Upstream zero-initialises the
    denoiser's last layer, so step 1 trains only that layer; the encoder's gate starts at 0, so
    step 2 moves only the gate among the encoder's weights; from step 3 all of them train."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        model(batch_of(inputs), generator=generator()).backward()
        optimizer.step()
        optimizer.zero_grad()
    return model

def run_path(model, batch, path):
    """One call on the model's training-loss path, or on its sampler path (what EvalSampler scores)."""
    if path == "training":
        return model(batch, generator=generator())
    return model.evaluate(batch, 2, generator=generator())[0]

class Recorder:
    """Keeps every mask the model's encoder is given and every mask_emb it returns."""
    def __init__(self, model):
        self.masks, self.outputs = [], []
        model.mask_encoder.register_forward_hook(self.record)

    def record(self, module, args, output):
        self.masks.append(args[0].clone())
        self.outputs.append(output)

# --- the encoder: one token per (hour, feature), attending over both axes ----------------

def test_one_d_model_vector_per_hour_and_feature(inputs):
    out = encoder()(inputs["cond_mask"])
    assert out.shape == (B, T, D, C) and torch.isfinite(out).all()

def test_layers_attend_over_time_within_a_feature_then_over_features_within_an_hour(inputs):
    # The reshapes and transposes, checked against the same layers run one sequence at a time.
    enc, mask = encoder(), inputs["cond_mask"]
    with torch.no_grad():
        x = enc.input_proj(mask.unsqueeze(-1)) + enc.feature_emb.weight + enc.time_emb.weight[:, None]
        for t_layer, f_layer in zip(enc.temporal, enc.feature):
            x = torch.stack([t_layer(x[:, :, d]) for d in range(D)], dim=2)     # each feature's 48 hours
            x = torch.stack([f_layer(x[:, t]) for t in range(T)], dim=1)        # each hour's 35 features
        assert torch.allclose(enc(mask), x, rtol=1e-4, atol=1e-5)

def test_one_value_reaches_its_feature_at_other_hours_and_its_hour_in_other_features():
    # One layer pair. A temporal-only encoder would leave other features untouched; a
    # feature-only one, other hours.
    enc, mask = encoder(n_layers=1), torch.zeros(1, T, D)
    flipped = mask.clone()
    flipped[0, 10, 5] = 1
    with torch.no_grad():
        moved = (enc(flipped) - enc(mask)).abs().sum(-1)[0]          # (T, D)
    assert moved[30, 5] > 0             # same feature, another hour: temporal attention
    assert moved[10, 20] > 0            # same hour, another feature: feature attention
    assert moved[30, 20] > 0            # both: time, then features

def test_hour_and_feature_embeddings_tell_tokens_apart():
    out = encoder()(torch.zeros(1, T, D))[0]                        # nothing but position to go on
    assert not torch.allclose(out[0, 0], out[0, 1]) and not torch.allclose(out[0, 0], out[1, 0])

# --- mask_emb comes from the conditioning mask alone ---------------------------------------

# Both paths: forward and evaluate each call get_mask_emb themselves, and evaluate's is the one
# every reported score comes from (M4-2 review: a leak there passed every test).
@pytest.mark.parametrize("path", ["training", "sampler"])
def test_the_encoder_is_given_the_conditioning_mask_in_our_axis_order(inputs, path):
    model = model_with(num_steps=3).eval()
    seen = Recorder(model)
    run_path(model, batch_of(inputs), path)
    assert len(seen.masks) == 1 and torch.equal(seen.masks[0], inputs["cond_mask"])

@pytest.mark.parametrize("path", ["training", "sampler"])
def test_changing_observed_mask_leaves_mask_emb_unchanged(inputs, path):
    model = model_with(num_steps=3).eval()
    open_gate(model.mask_encoder)
    seen = Recorder(model)
    more_observed = torch.ones_like(inputs["mask"])                 # every value "observed": targets everywhere
    out = [run_path(model, batch_of(inputs, mask=mask), path) for mask in (inputs["mask"], more_observed)]
    assert torch.equal(seen.outputs[0], seen.outputs[1])
    if path == "sampler":                                           # nothing else on the scored path reads it
        assert torch.equal(out[0], out[1])                          # either (the loss does, to pick its targets)

    flipped = inputs["cond_mask"].clone()
    flipped[0, 0, 0] = 1 - flipped[0, 0, 0]                         # ...while the conditioning mask matters
    run_path(model, batch_of(inputs, cond_mask=flipped), path)
    assert not torch.equal(seen.outputs[0], seen.outputs[2])

def test_mask_emb_reaches_the_denoiser_on_every_path(inputs):
    model = model_with(num_steps=3).eval()
    seen = Recorder(model)
    received = []
    model.diffmodel.register_forward_pre_hook(
        lambda module, args, kwargs: received.append(kwargs["mask_emb"]), with_kwargs=True)

    model(batch_of(inputs), generator=generator())                  # training
    with torch.no_grad():
        model(batch_of(inputs), is_train=0, generator=generator())  # validation: every step
    model.evaluate(batch_of(inputs), 2, generator=generator())      # sampler: every reverse step
    assert len(seen.outputs) == 3                                   # once per call, not per step
    assert len(received) == 1 + 3 + 2 * 3
    expected = [seen.outputs[0]] + [seen.outputs[1]] * 3 + [seen.outputs[2]] * 6
    assert all(r is e for r, e in zip(received, expected))

# --- the toggle: off is the base model, on adds the encoder and nothing else -------------

def test_off_builds_no_encoder_and_computes_no_mask_emb(inputs):
    model = model_with(ext1=False)
    assert model.mask_encoder is None and model.get_mask_emb(inputs["cond_mask"]) is None
    assert not any(name.startswith("mask_encoder") for name, _ in model.named_parameters())

def test_on_builds_the_encoder_at_the_denoisers_width():
    model = model_with(channels=32)
    assert isinstance(model.mask_encoder, MissingnessEncoder)
    assert model.mask_encoder.input_proj.out_features == model.diffmodel.channels == 32
    assert model.mask_encoder.feature_emb.num_embeddings == CONFIG["data"]["n_features"]
    assert model.mask_encoder.time_emb.num_embeddings == CONFIG["data"]["n_timesteps"]

def test_shared_weights_start_identical_in_both_variants():
    # The encoder is built after the shared parts, so paired seeds stay paired (README, Random streams).
    base, ext1 = model_with(ext1=False).state_dict(), model_with().state_dict()
    assert set(base) < set(ext1)
    assert set(ext1) - set(base) == {k for k in ext1 if k.startswith("mask_encoder.")}
    assert all(torch.equal(base[k], ext1[k]) for k in base)

def test_with_the_encoder_removed_the_loss_and_samples_are_the_base_models(inputs):
    base, ext1 = model_with(ext1=False, num_steps=5).eval(), model_with(num_steps=5).eval()
    ext1.mask_encoder = None
    assert torch.equal(ext1(batch_of(inputs), generator=generator()), base(batch_of(inputs), generator=generator()))
    assert torch.equal(ext1.evaluate(batch_of(inputs), 2, generator=generator())[0],
                       base.evaluate(batch_of(inputs), 2, generator=generator())[0])

def test_the_gate_starts_closed_so_ext1_starts_as_the_base_model(inputs):
    # eval: in training the encoder's dropout draws from the global RNG and shifts the
    # denoiser's dropout masks, the one stream that is not paired (README, Random streams).
    base, ext1 = model_with(ext1=False, num_steps=5).eval(), model_with(num_steps=5).eval()
    assert ext1.mask_encoder.gate.item() == 0
    assert not ext1.get_mask_emb(inputs["cond_mask"].permute(0, 2, 1)).any()
    assert torch.equal(ext1(batch_of(inputs), generator=generator()), base(batch_of(inputs), generator=generator()))
    assert torch.equal(ext1.evaluate(batch_of(inputs), 2, generator=generator())[0],
                       base.evaluate(batch_of(inputs), 2, generator=generator())[0])

def test_once_trained_the_encoder_changes_the_loss(inputs):
    model = warm_up(model_with().eval(), inputs)
    with_encoder = model(batch_of(inputs), generator=generator())
    model.mask_encoder = None
    assert with_encoder != model(batch_of(inputs), generator=generator())

# --- trained end-to-end -----------------------------------------------------------------

def test_the_gate_trains_first_then_the_layers_behind_it(inputs):
    # Step 2's gradients: the gate's is (upstream gradient) x (encoder output), nonzero at gate 0;
    # every other encoder weight's is multiplied by the gate, so zero until the gate moves.
    model = warm_up(model_with().eval(), inputs, steps=1)
    model(batch_of(inputs), generator=generator()).backward()
    grads = dict(model.mask_encoder.named_parameters())
    assert grads.pop("gate").grad.abs() > 0
    assert all(p.grad is None or not p.grad.any() for p in grads.values())

def test_nonzero_gradients_reach_every_encoder_weight(inputs):
    model = warm_up(model_with(), inputs)
    model(batch_of(inputs), generator=generator()).backward()
    grads = {name: p.grad for name, p in model.mask_encoder.named_parameters()}
    assert grads and all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads.values())

def test_the_unchanged_training_loop_trains_the_encoder(small_dataset):
    config = config_with(True, channels=16, layers=1, num_steps=10)
    config["train"].update(epochs=2, batch_size=8)
    RunStreams(42).seed_model_init()                                # as train() does, so this is its start
    initial = copy.deepcopy(build_model(config, "cpu").mask_encoder.state_dict())
    model, history = train(config, small_dataset, 42, "cpu")
    trained = model.mask_encoder.state_dict()
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in history)
    assert all(torch.isfinite(trained[k]).all() for k in trained)
    assert all(not torch.equal(initial[k], trained[k]) for k in initial)
