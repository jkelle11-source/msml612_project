import copy
import json

import numpy as np
import pytest
import torch
import yaml
from conftest import CONFIG
from data.physionet import save_dataset
from diffusion import score as scoring
from diffusion.data import to_batch
from diffusion.run import RUN_FILES, VARIANT_FLAGS, main, run, run_name, variant_name, weight_hashes, with_variant
from diffusion.streams import RunStreams
from diffusion.train import train
from eval import EvalMasker, EvalSampler, OffGridBlockEvalMasker, TrainMasker, masked_mae, median_imputation
from models.build import build_model

B, T, D = 4, 48, 35

def built(name, seed=42, **diffusion):
    RunStreams(seed).seed_model_init()                              # as train() does, right before
    config = with_variant(CONFIG, name)
    config["diffusion"].update(diffusion)
    return build_model(config, "cpu")

def open_gate(model):
    """The gate starts at 0, which makes every mask_emb zero, blind or not: open it to compare them."""
    with torch.no_grad():
        model.mask_encoder.gate.fill_(1.0)
    return model

def generator():
    return torch.Generator().manual_seed(7)

@pytest.fixture
def inputs() -> dict[str, torch.Tensor]:
    # The other model tests' stand-in for the pipeline output: (B, 48, 35), about 80% missing.
    g = torch.Generator().manual_seed(0)
    observed_mask = (torch.rand(B, T, D, generator=g) < 0.2).float()
    cond_mask, _ = TrainMasker(42)(observed_mask)
    return {"values": torch.randn(B, T, D, generator=g) * observed_mask, "mask": observed_mask, "cond_mask": cond_mask}

def batch_of(inputs, **replace):
    inputs = {**inputs, **replace}
    return to_batch(inputs["values"], inputs["mask"], inputs["cond_mask"])

def run_path(model, batch, path):
    """One call on the training-loss path, the validation-loss path, or the sampler path."""
    if path == "training":
        return model(batch, generator=generator())
    if path == "validation":
        with torch.no_grad():
            return model(batch, is_train=0, generator=generator())
    return model.evaluate(batch, 2, generator=generator())[0]

def mask_embs(model, inputs, cond_masks, path):
    """The mask_emb the model computes on ``path`` for each conditioning mask."""
    seen = []
    model.mask_encoder.register_forward_hook(lambda module, args, output: seen.append(output.detach().clone()))
    for cond_mask in cond_masks:
        run_path(model, batch_of(inputs, cond_mask=cond_mask), path)
    return seen

def other_masks(inputs):
    """Five conditioning masks unlike each other: the real one, none, every observed value, another
    draw, and the real one with one bit flipped."""
    flipped = inputs["cond_mask"].clone()
    flipped[0, 0, 0] = 1 - flipped[0, 0, 0]
    return [inputs["cond_mask"], torch.zeros_like(inputs["mask"]), inputs["mask"],
            TrainMasker(43)(inputs["mask"])[0], flipped]

# --- the toggle and its name ------------------------------------------------------------

def test_ext1blind_is_ext1_with_the_blind_toggle_on():
    config = with_variant(CONFIG, "ext1blind")
    assert config["model"]["use_mask_embedding"] and config["model"]["mask_embedding_blind"]
    assert not config["model"]["use_time_encoding"]
    assert variant_name(config) == "ext1blind" and variant_name(with_variant(CONFIG, "ext1")) == "ext1"
    assert run_name(config, 42) == "ext1blind_random_seed42" != run_name(with_variant(CONFIG, "ext1"), 42)
    assert VARIANT_FLAGS["ext1blind"] == ["--use_mask_embedding", "--mask_embedding_blind"]
    assert CONFIG["model"]["mask_embedding_blind"] is False                # off by default

def test_blind_without_ext1_is_refused(small_dataset, tmp_path):
    for config in (with_variant(CONFIG, "base"), with_variant(CONFIG, "ext2")):
        config["model"]["mask_embedding_blind"] = True
        with pytest.raises(ValueError, match="needs model.use_mask_embedding"):
            build_model(config, "cpu")
        with pytest.raises(ValueError, match="none of the variants"):
            variant_name(config)
    save_dataset(small_dataset, tmp_path / "data.npz")
    with pytest.raises(SystemExit):
        main(["--seed", "42", "--dataset", str(tmp_path / "data.npz"), "--device", "cpu",
              "--out", str(tmp_path / "run"), "--mask_embedding_blind"])
    assert not (tmp_path / "run").exists()                          # refused before anything ran

def test_run_refuses_a_config_that_is_no_variant_before_touching_its_folder(small_dataset, tmp_path):
    # Before M6-3 every toggle combination was a variant; blind without Ext 1 is not. run() must
    # refuse it before --overwrite deletes an earlier run's files, not after.
    config = with_variant(CONFIG, "base")
    config["model"]["mask_embedding_blind"] = True
    out = tmp_path / "base_random_seed42"
    out.mkdir()
    for name in RUN_FILES:
        (out / name).write_text("an earlier run's file")
    with pytest.raises(ValueError, match="none of the variants"):
        run(config, small_dataset, 42, "cpu", out, overwrite=True, wandb_mode="disabled")
    assert all((out / name).read_text() == "an earlier run's file" for name in RUN_FILES)

def test_a_config_saved_before_the_toggle_is_not_blind():
    config = with_variant(CONFIG, "ext1")
    del config["model"]["mask_embedding_blind"]
    assert variant_name(config) == "ext1"
    RunStreams(42).seed_model_init()
    assert build_model(config, "cpu").mask_encoder.blind is False

# --- mask_emb carries no mask information ---------------------------------------------------

@pytest.mark.parametrize("path", ["training", "validation", "sampler"])
def test_blind_mask_emb_is_identical_for_any_two_conditioning_masks(inputs, path):
    blind = open_gate(built("ext1blind", num_steps=3)).eval()
    embs = mask_embs(blind, inputs, other_masks(inputs), path)
    assert all(torch.equal(embs[0], emb) for emb in embs[1:])
    # It is Ext 1's mask_emb for an all-ones mask, from the same weights, and Ext 1's does change.
    ext1 = open_gate(built("ext1", num_steps=3)).eval()
    assert torch.equal(embs[0], mask_embs(ext1, inputs, [torch.ones_like(inputs["mask"])], path)[0])
    seen = mask_embs(ext1, inputs, other_masks(inputs), path)
    assert all(not torch.equal(seen[0], emb) for emb in seen[1:])

def dropout_on_mask_embs(model, inputs, cond_masks, seed=0):
    """The mask_emb of a training-path call in training mode (dropout on), the global RNG re-seeded
    to ``seed`` before each call, so each call's encoder draws the same dropout masks."""
    seen = []
    model.mask_encoder.register_forward_hook(lambda module, args, output: seen.append(output.detach().clone()))
    model.train()
    for cond_mask in cond_masks:
        torch.manual_seed(seed)
        model(batch_of(inputs, cond_mask=cond_mask), generator=generator())
    return seen

def test_blind_mask_emb_ignores_the_mask_with_dropout_on(inputs):
    # The training path as training runs it, dropout on (the test above turns it off).
    blind = open_gate(built("ext1blind", num_steps=3))
    embs = dropout_on_mask_embs(blind, inputs, other_masks(inputs))
    assert all(torch.equal(embs[0], emb) for emb in embs[1:])
    # Not vacuous: dropout is on (another seed, another mask_emb), and Ext 1's does change.
    assert not torch.equal(embs[0], dropout_on_mask_embs(blind, inputs, other_masks(inputs)[:1], seed=1)[0])
    seen = dropout_on_mask_embs(open_gate(built("ext1", num_steps=3)), inputs, other_masks(inputs))
    assert all(not torch.equal(seen[0], emb) for emb in seen[1:])

def test_blind_draws_ext1s_dropout_masks_at_every_step(small_dataset):
    # Ext 1-blind has Ext 1's layers, so it draws from the global RNG exactly as Ext 1 does: the
    # same dropout masks at every step, so the two are paired in every stream, dropout included.
    # Base and Ext 1 are not (PROJECT_PLAN §3.3): Ext 1's encoder has dropout layers base lacks.
    def rng_states(name):
        config = with_variant(CONFIG, name)
        config["diffusion"].update(channels=16, layers=2, num_steps=10)
        config["train"].update(epochs=2, batch_size=8)
        states = []
        train(config, small_dataset, 42, "cpu", on_step_end=lambda record: states.append(torch.get_rng_state()))
        return states
    ext1, blind, base = rng_states("ext1"), rng_states("ext1blind"), rng_states("base")
    assert len(ext1) == len(blind) == 8 and all(torch.equal(a, b) for a, b in zip(ext1, blind))
    assert not all(torch.equal(a, b) for a, b in zip(ext1, base))

def test_blind_reaches_the_denoiser_once_per_call_on_every_path(inputs):
    model = open_gate(built("ext1blind", num_steps=3)).eval()
    received = []
    model.diffmodel.register_forward_pre_hook(
        lambda module, args, kwargs: received.append(kwargs["mask_emb"]), with_kwargs=True)
    for path in ("training", "validation", "sampler"):
        run_path(model, batch_of(inputs), path)
    assert len(received) == 1 + 3 + 2 * 3                           # once per call, reused at every step
    assert all(r is received[1] for r in received[1:4]) and all(r is received[4] for r in received[4:])
    # Close, not equal, across paths: without gradients torch's encoder layers take their fused
    # fast path, which rounds differently. Each path's is exactly mask-independent (above).
    assert all(torch.allclose(received[0], r, atol=1e-5) for r in received)

# --- the weights: Ext 1's, at every seed -----------------------------------------------------

@pytest.mark.parametrize("seed", [42, 43])
def test_blind_starts_from_ext1s_weights_at_a_seed(seed):
    ext1, blind = built("ext1", seed).state_dict(), built("ext1blind", seed).state_dict()
    assert list(ext1) == list(blind) and all(torch.equal(ext1[key], blind[key]) for key in ext1)
    assert weight_hashes(ext1) == weight_hashes(blind)              # what diffusion.results pairs them by

def test_blind_trains_differently_from_ext1_once_the_gate_opens(small_dataset):
    def history(name):
        config = with_variant(CONFIG, name)
        config["diffusion"].update(channels=16, layers=2, num_steps=10)
        config["train"].update(epochs=2, batch_size=8)
        return [r["loss"] for r in train(config, small_dataset, 42, "cpu")[1]]
    ext1, blind = history("ext1"), history("ext1blind")
    # Steps 1 and 2: the gate is still 0, so mask_emb is 0 in both; from step 3 Ext 1 sees the mask.
    assert ext1[:2] == blind[:2] and ext1[2:] != blind[2:]

# --- with the toggle off, Ext 1 is unchanged --------------------------------------------------

# Ext 1 at seed 42 on the 40 fake patients, on CPU, at width 16 with 2 layers and 10 diffusion steps:
# 2 epochs (8 steps), then 2 frozen-noise draws on the 4 val patients, as tests/test_variants.py's
# gate test. Worked out at 6b8a298, the commit before M6-3 added the toggle, in a copy of it.
EXT1_LOSSES = [1.0110571384429932, 1.0674759149551392, 1.10837984085083, 0.9676856994628906,
               1.0219664573669434, 1.1218671798706055, 1.0354413986206055, 1.0149375200271606]
EXT1_GRAD_NORMS = [1.9175783395767212, 1.8134101629257202, 2.10922908782959, 2.7648398876190186,
                   1.8437892198562622, 2.4753527641296387, 2.313598871231079, 2.0847513675689697]
EXT1_WEIGHT_ABS_SUM = 7711.944107683487
EXT1_SAMPLE_ABS_SUM = 37548.971559995814
EXT1_VAL_MAE = 3.2348967463440204

def test_ext1_computes_what_it_did_before_the_toggle(small_dataset):
    config = with_variant(CONFIG, "ext1")
    assert config["model"]["mask_embedding_blind"] is False
    config["diffusion"].update(channels=16, layers=2, num_steps=10)
    config["train"].update(epochs=2, batch_size=8)
    model, history = train(config, small_dataset, 42, "cpu")
    assert model.mask_encoder.blind is False
    assert [r["loss"] for r in history] == pytest.approx(EXT1_LOSSES, rel=1e-4)
    assert [r["grad_norm"] for r in history] == pytest.approx(EXT1_GRAD_NORMS, rel=1e-4)
    assert sum(p.detach().double().abs().sum().item() for p in model.parameters()) == pytest.approx(
        EXT1_WEIGHT_ABS_SUM, rel=1e-4)

    idx = small_dataset["val_idx"]
    values, mask = small_dataset["values"][idx], small_dataset["mask"][idx]
    cond, target = EvalMasker.from_config()(mask, small_dataset["record_ids"][idx])
    samples = EvalSampler(CONFIG["protocol"]["eval_sampling_seed"], 2, 8)(model, values, mask, cond)
    assert np.abs(samples.astype(np.float64)).sum() == pytest.approx(EXT1_SAMPLE_ABS_SUM, rel=1e-4)
    assert masked_mae(median_imputation(samples), values, target) == pytest.approx(EXT1_VAL_MAE, rel=1e-4)

# --- end to end: diffusion.run, then diffusion.score on the block and off-grid holdouts -------

@pytest.fixture(scope="module")
def blind_run(small_dataset, tmp_path_factory):
    """Ext 1-blind trained by ``diffusion.run``'s command line, tiny, at seed 42; the repo's config
    with 2 draws for scoring beside it."""
    tmp = tmp_path_factory.mktemp("blind")
    config = copy.deepcopy(CONFIG)
    config["diffusion"].update(channels=16, layers=1, num_steps=10)
    config["train"].update(epochs=1, batch_size=8)
    config["validation"].update(every_n_epochs=1, n_samples=2, batch_size=8)
    (tmp / "config.yaml").write_text(yaml.safe_dump(config))
    quick = copy.deepcopy(CONFIG)
    quick["protocol"]["eval_n_samples"] = 2
    (tmp / "quick.yaml").write_text(yaml.safe_dump(quick))
    save_dataset(small_dataset, tmp / "data.npz")
    main(["--seed", "42", "--dataset", str(tmp / "data.npz"), "--device", "cpu",
          "--config", str(tmp / "config.yaml"), "--variant", "ext1blind", "--out", str(tmp / "ext1blind_random_seed42")])
    return tmp

def test_diffusion_run_trains_ext1blind(blind_run):
    run_dir = blind_run / "ext1blind_random_seed42"
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert variant_name(saved) == summary["variant"] == "ext1blind" and saved["model"]["mask_embedding_blind"]
    weights = torch.load(run_dir / "model.pt")
    assert any(key.startswith("mask_encoder.") for key in weights)
    assert build_model(saved, "cpu").mask_encoder.blind                    # how diffusion.score rebuilds it

@pytest.mark.parametrize("holdout", ["block", "offgrid"])
def test_diffusion_score_scores_ext1blind(blind_run, small_dataset, holdout, capsys):
    run_dir = blind_run / "ext1blind_random_seed42"
    scoring.main([str(run_dir), "--holdout", holdout, "--config", str(blind_run / "quick.yaml"),
                  "--dataset", str(blind_run / "data.npz"), "--device", "cpu"])
    scores = json.loads((run_dir / f"scores_test_{holdout}_0.1.json").read_text())
    assert (scores["variant"], scores["holdout"], scores["fraction"]) == ("ext1blind", holdout, 0.1)
    assert np.isfinite(scores["mae"]) and (run_dir / f"median_test_{holdout}_0.1.npy").exists()
    assert scores["holdout_settings"] == scoring.holdout_settings(CONFIG["protocol"], holdout)
    if holdout == "offgrid":
        idx = small_dataset["test_idx"]
        _, target = OffGridBlockEvalMasker.from_config()(small_dataset["mask"][idx], small_dataset["record_ids"][idx])
        assert scores["n_targets"] == int(target.sum())
