import copy
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import models.csdi.main_model as main_model
from conftest import CONFIG, nonzero_output_layer
from data.physionet import FEATURES, compute_tslo
from diffusion.data import to_batch
from diffusion.streams import RunStreams
from diffusion.train import train
from eval import BlockEvalMasker, EvalMasker, EvalSampler, TrainMasker
from models.build import build_model
from models.missingness import MissingnessEncoder
from models.time_encoding import ContinuousTimeEncoding, tslo

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

# --- ContinuousTimeEncoding: true Time2Vec of δ, gated from 0 (M5-2) ----------------------

def time_encoder(d_k=16, d_model=C, seed=0, gate=1.0):
    """Gate opened to 1 by default: tests of what the layers compute need it open; the gate
    itself is tested on its own."""
    torch.manual_seed(seed)
    enc = ContinuousTimeEncoding(d_k, d_model)
    with torch.no_grad():
        enc.gate.fill_(gate)
    return enc

def hand_set(enc):
    """Lesson 3's weights: linear 0.1·δ, waves sin(π/2·δ) and sin(π/6·δ), all φ = 0; the
    projection keeps the linear feature and adds the two waves."""
    with torch.no_grad():
        enc.w0.weight.fill_(0.1)
        enc.w0.bias.zero_()
        enc.w.weight.copy_(torch.tensor([[math.pi / 2], [math.pi / 6]]))
        enc.w.bias.zero_()
        enc.proj.weight.copy_(torch.tensor([[1.0, 0, 0], [0, 1, 1]]))
        enc.proj.bias.zero_()
    return enc

@pytest.fixture
def real_delta(random_mask) -> torch.Tensor:
    return tslo(random_mask)                                        # (5, 48, 35), 0 ... 48 hours

def test_encoder_hand_case():
    # δ = 1: features [0.1, 1, 0.5] -> [0.1, 1.5]; δ = 3: [0.3, -1, 1] -> [0.3, 0].
    enc = hand_set(time_encoder(d_k=3, d_model=2))
    out = enc(torch.tensor([[[1.0], [3.0]]]))                       # (B, T, D) = (1, 2, 1)
    assert out.shape == (1, 2, 1, 2)
    assert torch.allclose(out[0, :, 0], torch.tensor([[0.1, 1.5], [0.3, 0.0]]), atol=1e-6)
    with torch.no_grad():
        enc.gate.zero_()
    assert not enc(torch.tensor([[[1.0], [3.0]]])).any()

def test_encoder_layers_have_time2vecs_shapes():
    # One linear feature and d_k - 1 waves, concatenated to d_k and projected to d_model.
    enc = time_encoder(d_k=16, d_model=64)
    shapes = {name: tuple(p.shape) for name, p in enc.named_parameters()}
    assert shapes == {"w0.weight": (1, 1), "w0.bias": (1,), "w.weight": (15, 1), "w.bias": (15,),
                      "proj.weight": (64, 16), "proj.bias": (64,), "gate": ()}
    assert sum(p.numel() for p in enc.parameters()) == 1121

def test_encoder_features_are_one_linear_then_waves_concatenated(real_delta):
    # With the projection made the identity, the output is the Time2Vec features themselves,
    # checked against the formula worked out element by element.
    enc = time_encoder(d_k=16, d_model=16)
    with torch.no_grad():
        enc.proj.weight.copy_(torch.eye(16))
        enc.proj.bias.zero_()
        features = enc(real_delta)
    d = real_delta[..., None]
    linear = enc.w0.weight[0, 0] * d + enc.w0.bias[0]
    waves = torch.sin(enc.w.weight[:, 0] * d + enc.w.bias)
    # atol: nn.Linear and the formula round ω·δ + φ differently in float32 (up to 2e-6 here).
    assert torch.allclose(features, torch.cat([linear, waves], dim=-1).detach(), atol=1e-5)

def test_encoder_linear_feature_keeps_growing_and_waves_stay_bounded():
    enc = time_encoder(d_k=16, d_model=16)
    with torch.no_grad():
        enc.proj.weight.copy_(torch.eye(16))
        enc.proj.bias.zero_()
        enc.w0.weight.fill_(0.1)
        enc.w0.bias.zero_()
        features = enc(torch.tensor([[[0.0], [48.0], [480.0]]]))[0, :, 0]
    assert features[:, 0].tolist() == pytest.approx([0.0, 4.8, 48.0])     # not wrapped by a sine
    assert features[:, 1:].abs().max() <= 1

def test_encoder_gate_starts_at_zero_so_time_enc_starts_at_zero(real_delta):
    torch.manual_seed(0)
    enc = ContinuousTimeEncoding(16, C)
    assert isinstance(enc.gate, nn.Parameter) and enc.gate.shape == () and enc.gate.item() == 0
    assert "gate" in enc.state_dict()
    assert enc.proj.bias.abs().sum() > 0                            # so gating before proj would show
    assert not enc(real_delta).any()

def test_encoder_each_value_gets_its_own_vector_from_shared_weights():
    # Equal δ at any (patient, hour, feature) gives equal vectors; changing one δ changes only
    # that position's vector.
    enc = time_encoder()
    delta = torch.full((2, T, D), 5.0)
    out = enc(delta)
    assert torch.equal(out, out[0, 0, 0].expand_as(out))
    changed = delta.clone()
    changed[1, 30, 7] = 6.0
    moved = (enc(changed) - out).abs().sum(-1)
    assert moved[1, 30, 7] > 0 and moved.count_nonzero() == 1

def test_encoder_output_shape_and_values(real_delta):
    out = time_encoder()(real_delta)
    assert out.shape == (5, T, D, C) and out.dtype == torch.float32
    assert torch.isfinite(out).all()

def test_encoder_at_gate_zero_only_the_gate_has_a_gradient(real_delta):
    # Lesson 4: d out / d gate = x, nonzero; every weight behind the gate gets gate · (...) = 0.
    weights = torch.randn(5, T, D, C, generator=torch.Generator().manual_seed(0))
    enc = time_encoder(gate=0.0)
    (enc(real_delta) * weights).sum().backward()
    grads = dict(enc.named_parameters())
    assert grads.pop("gate").grad.abs() > 0
    assert all(not p.grad.any() for p in grads.values())

    enc = time_encoder(gate=0.5)                                    # once the gate has moved
    (enc(real_delta) * weights).sum().backward()
    assert all(p.grad.abs().sum() > 0 for p in enc.parameters())

@pytest.mark.skipif(ACCELERATOR is None, reason="no GPU available")
def test_encoder_runs_on_the_delta_device(real_delta):
    enc = time_encoder()
    on_cpu = enc(real_delta)
    on_device = enc.to(ACCELERATOR)(real_delta.to(ACCELERATOR))
    assert on_device.device.type == ACCELERATOR
    assert torch.allclose(on_device.cpu(), on_cpu, atol=1e-5)

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

# --- the toggle: Ext 2 built into the model and paired with every variant (M5-2) ---------

VARIANTS = ((False, False), (True, False), (False, True), (True, True))     # base, ext1, ext2, ext1+2

def config_with(ext1=False, ext2=True, **diffusion):
    config = copy.deepcopy(CONFIG)
    config["model"].update(use_mask_embedding=ext1, use_time_encoding=ext2)
    config["diffusion"].update(diffusion)
    return config

def variant(ext1, ext2, seed=42, **diffusion):
    RunStreams(seed).seed_model_init()                              # as train() does, right before
    return build_model(config_with(ext1, ext2, **diffusion), "cpu")

def warm_up(model, inputs, steps):
    """Training steps before gradients reach the whole encoder: step 1 trains only upstream's
    zero-initialised last layer, step 2 moves the gate, and from step 3 every weight trains."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        model(batch_of(inputs), generator=generator()).backward()
        optimizer.step()
        optimizer.zero_grad()
    return model

def test_off_builds_no_time_encoder():
    model = build_model(config_with(ext2=False), "cpu")
    assert model.time_encoder is None
    assert not any(name.startswith("time_encoder") for name, _ in model.named_parameters())

def test_on_builds_the_encoder_at_the_configs_d_k_and_the_denoisers_width():
    config = config_with(channels=32)
    config["model"]["time_encoding_dk"] = 8
    enc = build_model(config, "cpu").time_encoder
    assert isinstance(enc, ContinuousTimeEncoding)
    assert enc.w.out_features == 7 and enc.proj.in_features == 8 and enc.proj.out_features == 32

@pytest.mark.skipif(ACCELERATOR is None, reason="no GPU available")
def test_time_encoder_is_on_the_requested_device():
    model = build_model(config_with(ext1=True), ACCELERATOR)
    assert all(p.device.type == ACCELERATOR for p in model.time_encoder.parameters())

def test_variants_are_paired_at_a_seed():
    # Shared weights equal base's in every variant; each encoder is the same in every variant
    # that has it (PROJECT_PLAN §3.3).
    base, ext1, ext2, both = (variant(*v).state_dict() for v in VARIANTS)
    for model in (ext1, ext2, both):
        assert all(torch.equal(base[k], model[k]) for k in base)
    mask = {k for k in both if k.startswith("mask_encoder.")}
    time = {k for k in both if k.startswith("time_encoder.")}
    assert mask and time and set(both) == set(base) | mask | time
    assert all(torch.equal(ext1[k], both[k]) for k in mask)
    assert all(torch.equal(ext2[k], both[k]) for k in time)

def test_time_encoder_starts_from_the_init_stream_right_after_the_shared_parts():
    # Seed 42's numbers, from an encoder built straight after the shared parts (lesson 5).
    # Built after the mask encoder without the restore, Ext 1+2's would start at w0 = -0.9452.
    for ext1 in (False, True):
        enc = variant(ext1, True).time_encoder
        assert round(enc.w0.weight.item(), 4) == -0.5433
        assert [round(v, 4) for v in enc.w.weight[:3, 0].tolist()] == [-0.7165, 0.8479, 0.5076]

def test_mask_encoder_starts_where_it_did_before_ext2():
    # M4-2 built it straight after the shared parts, with no restore; Ext 1's runs keep those weights.
    variant(False, False)
    before = MissingnessEncoder(D, T, C).state_dict()
    for ext2 in (False, True):
        now = variant(True, ext2).mask_encoder.state_dict()
        assert all(torch.equal(before[k], now[k]) for k in before)

@pytest.mark.parametrize("ext1", [False, True])
def test_the_gate_starts_closed_so_ext2_starts_as_the_variant_without_it(inputs, ext1):
    # eval: no dropout. nonzero_output_layer, or upstream's zero last layer would make every
    # prediction 0 and the comparison vacuous.
    without = nonzero_output_layer(variant(ext1, False, num_steps=5).eval())
    with_ext2 = nonzero_output_layer(variant(ext1, True, num_steps=5).eval())
    assert with_ext2.time_encoder.gate.item() == 0
    for path in PATHS:
        assert torch.equal(run_path(with_ext2, batch_of(inputs), path), run_path(without, batch_of(inputs), path))

def without_and_with_ext1(ext2):
    """Ext 2 and Ext 1+2 (or base and Ext 1) at seed 42, Ext 2's gate open in both so that Ext 1+2
    is held to a working Ext 2. eval: Ext 1's dropout draws from the global RNG (README, Random
    streams). nonzero_output_layer, or upstream's zero last layer makes every comparison vacuous."""
    without = nonzero_output_layer(variant(False, ext2, num_steps=5).eval())
    with_ext1 = nonzero_output_layer(variant(True, ext2, num_steps=5).eval())
    if ext2:
        with torch.no_grad():
            without.time_encoder.gate.fill_(1.0)
            with_ext1.time_encoder.gate.fill_(1.0)
    return without, with_ext1

# Ext 1's side of composing (M5-3): Ext 1 changes Ext 2 exactly as it changes base.
@pytest.mark.parametrize("ext2", [False, True])
def test_the_mask_gate_starts_closed_so_ext1_starts_as_the_variant_without_it(inputs, ext2):
    without, with_ext1 = without_and_with_ext1(ext2)
    assert with_ext1.mask_encoder.gate.item() == 0
    for path in PATHS:
        assert torch.equal(run_path(with_ext1, batch_of(inputs), path), run_path(without, batch_of(inputs), path))

@pytest.mark.parametrize("ext2", [False, True])
def test_with_the_mask_gate_open_ext1_changes_the_loss_and_the_samples(inputs, ext2):
    without, with_ext1 = without_and_with_ext1(ext2)
    with torch.no_grad():
        with_ext1.mask_encoder.gate.fill_(1.0)
    for path in PATHS:
        assert not torch.equal(run_path(with_ext1, batch_of(inputs), path), run_path(without, batch_of(inputs), path))

def test_with_the_gate_open_ext2_changes_the_loss_and_the_samples(inputs):
    without = nonzero_output_layer(variant(False, False, num_steps=5).eval())
    with_ext2 = nonzero_output_layer(variant(False, True, num_steps=5).eval())
    with torch.no_grad():
        with_ext2.time_encoder.gate.fill_(1.0)
    for path in PATHS:
        assert not torch.equal(run_path(with_ext2, batch_of(inputs), path), run_path(without, batch_of(inputs), path))

def test_the_gate_trains_first_then_the_layers_behind_it(inputs):
    model = warm_up(variant(False, True).eval(), inputs, steps=1)
    model(batch_of(inputs), generator=generator()).backward()
    grads = dict(model.time_encoder.named_parameters())
    assert grads.pop("gate").grad.abs() > 0
    assert all(p.grad is None or not p.grad.any() for p in grads.values())

def test_nonzero_gradients_reach_every_time_encoder_weight(inputs):
    model = warm_up(variant(False, True), inputs, steps=2)
    model(batch_of(inputs), generator=generator()).backward()
    grads = {name: p.grad for name, p in model.time_encoder.named_parameters()}
    assert len(grads) == 7
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads.values())

def test_the_unchanged_training_loop_trains_the_time_encoder(small_dataset):
    config = config_with(channels=16, layers=1, num_steps=10)
    config["train"].update(epochs=2, batch_size=8)
    RunStreams(42).seed_model_init()                                # as train() does, so this is its start
    initial = copy.deepcopy(build_model(config, "cpu").time_encoder.state_dict())
    model, history = train(config, small_dataset, 42, "cpu")
    trained = model.time_encoder.state_dict()
    assert all(math.isfinite(r["loss"]) for r in history)
    assert all(torch.isfinite(trained[k]).all() for k in trained)
    assert all(not torch.equal(initial[k], trained[k]) for k in initial)

def test_ext2_draws_nothing_from_the_global_rng_so_dropout_stays_paired(inputs):
    # start_epoch re-seeds the global RNG, and the time encoder has no dropout, so a training
    # step leaves the global RNG where base's does: every dropout mask matches base's. Ext 1's
    # encoder has dropout layers and moves it (README, Random streams).
    def after_one_step(ext1, ext2):
        streams = RunStreams(42)
        streams.seed_model_init()
        model = build_model(config_with(ext1, ext2), "cpu").train()
        streams.start_epoch(0)
        model(batch_of(inputs), generator=streams.diffusion).backward()
        return torch.get_rng_state()
    base, ext1, ext2, both = (after_one_step(*v) for v in VARIANTS)
    assert torch.equal(ext2, base) and torch.equal(both, ext1)
    assert not torch.equal(ext1, base)
