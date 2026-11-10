import copy

import numpy as np
import pytest
import torch
import yaml
from conftest import CONFIG, RAW_DIR
from data.physionet import save_dataset
from diffusion.data import to_batch
from diffusion.run import VARIANT_FLAGS, VARIANTS, main, variant_name, with_variant
from diffusion.streams import RunStreams
from diffusion.train import train
from eval import EvalMasker, EvalSampler, TrainMasker, masked_mae, median_imputation
from models.build import build_model

needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")

T, D = 48, 35
NAMES = ["base", "ext1", "ext2", "ext1+2"]

def config_of(name, **diffusion):
    """The repo's config set to variant ``name``."""
    config = with_variant(CONFIG, name)
    config["diffusion"].update(diffusion)
    return config

def built(name, **diffusion):
    RunStreams(42).seed_model_init()                                # as train() does, right before
    return build_model(config_of(name, **diffusion), "cpu")

def generator():
    return torch.Generator().manual_seed(7)

def launch(tmp_path, dataset, config, *args):
    """``main`` at seed 42 on ``dataset`` and a tiny 1-epoch version of ``config``; returns the
    run's saved config and weights."""
    config = copy.deepcopy(config)
    config["diffusion"].update(channels=16, layers=1, num_steps=10)
    config["train"].update(epochs=1, batch_size=8)
    config["validation"].update(every_n_epochs=1, n_samples=2, batch_size=8)
    save_dataset(dataset, tmp_path / "data.npz")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    main(["--seed", "42", "--dataset", str(tmp_path / "data.npz"), "--device", "cpu",
          "--config", str(tmp_path / "config.yaml"), "--out", str(tmp_path / "run"), *args])
    return (yaml.safe_load((tmp_path / "run" / "config.yaml").read_text()),
            torch.load(tmp_path / "run" / "model.pt"))

def encoders_saved(weights):
    return {name for name in ("mask_encoder", "time_encoder") if any(k.startswith(name + ".") for k in weights)}

ENCODERS = {"base": set(), "ext1": {"mask_encoder"}, "ext2": {"time_encoder"},
            "ext1+2": {"mask_encoder", "time_encoder"}}

@pytest.fixture(scope="module")
def real(set_a) -> dict[str, torch.Tensor]:
    """4 Set A training patients with TrainMasker's split. The batch size changes no shape the
    model checks, and 4 keeps each test near a second (16, the real loop's, took 20 s or more)."""
    rows = set_a["train_idx"][:4]
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

# --- the variants and how a run is set to one ------------------------------------------------

def test_there_are_exactly_the_four_variants():
    # M6-3 adds ext1blind here, with its toggle written in every variant.
    assert VARIANTS == {
        "base": {"use_mask_embedding": False, "use_time_encoding": False},
        "ext1": {"use_mask_embedding": True, "use_time_encoding": False},
        "ext2": {"use_mask_embedding": False, "use_time_encoding": True},
        "ext1+2": {"use_mask_embedding": True, "use_time_encoding": True},
    }
    assert list(VARIANTS) == NAMES
    assert VARIANT_FLAGS == {"base": [], "ext1": ["--use_mask_embedding"], "ext2": ["--use_time_encoding"],
                             "ext1+2": ["--use_mask_embedding", "--use_time_encoding"]}

@pytest.mark.parametrize("start", [False, True], ids=["from all off", "from all on"])
@pytest.mark.parametrize("name", NAMES)
def test_with_variant_writes_every_toggle_whatever_the_config_held(name, start):
    config = copy.deepcopy(CONFIG)
    config["model"].update(use_mask_embedding=start, use_time_encoding=start)
    before = copy.deepcopy(config)
    set_to = with_variant(config, name)
    assert variant_name(set_to) == name
    assert all(set_to["model"][toggle] is on for toggle, on in VARIANTS[name].items())
    assert config == before                                         # a copy; the caller's is untouched

@pytest.mark.parametrize("name", NAMES)
def test_each_variants_flags_train_that_variant(small_dataset, tmp_path, name):
    saved, weights = launch(tmp_path, small_dataset, CONFIG, *VARIANT_FLAGS[name])
    assert variant_name(saved) == name
    assert encoders_saved(weights) == ENCODERS[name]

@pytest.mark.parametrize("name", NAMES)
def test_variant_trains_that_variant_even_from_a_config_with_every_toggle_on(small_dataset, tmp_path, name):
    config = copy.deepcopy(CONFIG)
    config["model"].update(use_mask_embedding=True, use_time_encoding=True)
    saved, weights = launch(tmp_path, small_dataset, config, "--variant", name)
    assert variant_name(saved) == name
    assert all(saved["model"][toggle] is on for toggle, on in VARIANTS[name].items())
    assert encoders_saved(weights) == ENCODERS[name]

@pytest.mark.parametrize("flag", ["--use_mask_embedding", "--use_time_encoding"])
def test_variant_and_a_toggle_flag_together_are_refused(small_dataset, tmp_path, flag):
    with pytest.raises(SystemExit):
        launch(tmp_path, small_dataset, CONFIG, "--variant", "base", flag)
    assert not (tmp_path / "run").exists()                          # refused before anything ran

# --- every variant trains and samples on real data, at the gate's width and depth ----------

@needs_raw_data
@pytest.mark.parametrize("name", NAMES)
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
    assert samples.shape == (2, len(mask), T, D) and torch.isfinite(torch.from_numpy(samples)).all()

@needs_raw_data
def test_with_both_on_nonzero_gradients_reach_every_weight_of_both_encoders(real):
    model = train_steps(built("ext1+2"), real, steps=2)
    model(batch_of(real), generator=generator()).backward()
    for encoder in (model.mask_encoder, model.time_encoder):
        grads = {name: p.grad for name, p in encoder.named_parameters()}
        assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads.values())
    assert model.mask_encoder.gate.item() != 0 and model.time_encoder.gate.item() != 0     # both opened

# --- base is the gate model -------------------------------------------------------------------

# Base at seed 42 on the 40 fake patients, on CPU, at width 16 with 2 layers and 10 diffusion
# steps: 2 epochs (8 steps), then 2 frozen-noise draws on the 4 val patients. Worked out at the gate
# runs' commit, 9314475, in a copy of it (M6-1 review), not by this code. 2 layers, not 1: the skip
# connections are summed and divided by sqrt(layers), which is 1 for one layer.
GATE_LOSSES = [1.0110571384429932, 1.0674283504486084, 1.108137607574463, 0.9675644040107727,
               1.022009015083313, 1.1213350296020508, 1.0342506170272827, 1.014701247215271]
GATE_GRAD_NORMS = [1.9010037183761597, 1.826764464378357, 2.1411895751953125, 2.7812206745147705,
                   1.8262090682983398, 2.5134270191192627, 2.3995089530944824, 2.101351261138916]
GATE_WEIGHT_ABS_SUM = 5117.809426162689     # sum of |w| over every weight after the 8 steps
GATE_SAMPLE_ABS_SUM = 37548.85115848812     # sum of |draw| over both draws
GATE_VAL_MAE = 3.235771840353014            # the lower of the 2 draws, on EvalMasker's 112 targets

def test_base_computes_what_it_did_at_the_gate(small_dataset):
    # To 1e-4: another CPU's float rounding stays far inside it, a change to base does not. The two
    # tried in the M6-1 review (the skip connections divided by layers, not sqrt(layers); a leaky
    # ReLU after the input projection) moved the losses by 26% and 1%, which the suite missed.
    config = copy.deepcopy(CONFIG)
    config["diffusion"].update(channels=16, layers=2, num_steps=10)
    config["train"].update(epochs=2, batch_size=8)
    model, history = train(config, small_dataset, 42, "cpu")
    assert [r["loss"] for r in history] == pytest.approx(GATE_LOSSES, rel=1e-4)
    assert [r["grad_norm"] for r in history] == pytest.approx(GATE_GRAD_NORMS, rel=1e-4)
    weight_abs_sum = sum(p.detach().double().abs().sum().item() for p in model.parameters())
    assert weight_abs_sum == pytest.approx(GATE_WEIGHT_ABS_SUM, rel=1e-4)

    idx = small_dataset["val_idx"]
    values, mask = small_dataset["values"][idx], small_dataset["mask"][idx]
    cond, target = EvalMasker.from_config()(mask, small_dataset["record_ids"][idx])
    samples = EvalSampler(CONFIG["protocol"]["eval_sampling_seed"], 2, 8)(model, values, mask, cond)
    assert np.abs(samples.astype(np.float64)).sum() == pytest.approx(GATE_SAMPLE_ABS_SUM, rel=1e-4)
    assert masked_mae(median_imputation(samples), values, target) == pytest.approx(GATE_VAL_MAE, rel=1e-4)
