import copy
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import CONFIG
from data.physionet import load_dataset
from diffusion.data import to_batch, train_loader
from diffusion.streams import RunStreams
from diffusion.train import train
from models.build import build_model
from models.csdi import CSDI_base

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed" / "physionet_set_a.npz"
needs_processed = pytest.mark.skipif(not PROCESSED.exists(), reason="processed dataset not built")

def tiny_config(**train):
    """A model small enough to train for a few epochs in well under a second."""
    config = copy.deepcopy(CONFIG)
    config["diffusion"].update(channels=16, layers=1, num_steps=10)
    config["train"].update({"epochs": 3, "batch_size": 8, **train})    # 28 training patients -> 4 steps/epoch
    return config

def weights(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def run(dataset, seed=42, **train_settings):
    model, history = train(tiny_config(**train_settings), dataset, seed, "cpu")
    return weights(model), history

# --- one step on real data (M2-1 Definition of Done) -----------------------------------

@needs_processed
def test_one_training_step_on_a_real_batch():
    dataset = load_dataset(PROCESSED)
    dataset["train_idx"] = dataset["train_idx"][:16]                   # one batch -> one step
    config = copy.deepcopy(CONFIG)                                     # upstream's full model
    config["train"]["epochs"] = 1

    model, history = train(config, dataset, 42, "cpu")

    streams = RunStreams(42)
    streams.seed_model_init()
    initial = weights(build_model(config, "cpu"))                      # what train() started from
    assert len(history) == 1
    assert math.isfinite(history[0]["loss"]) and math.isfinite(history[0]["grad_norm"])
    assert torch.isfinite(weights(model)).all()
    assert not torch.equal(weights(model), initial)

# --- the loss is scored on TrainMasker's targets; upstream's masking never runs ----------

def test_upstreams_masking_never_runs(small_dataset, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("upstream masking ran on the training path")

    monkeypatch.setattr(CSDI_base, "get_randmask", forbidden)
    monkeypatch.setattr(CSDI_base, "get_hist_mask", forbidden)
    _, history = run(small_dataset, epochs=1)
    assert len(history) == 4

def test_every_step_scores_exactly_the_train_masker_targets(small_dataset, monkeypatch):
    scored = []
    calc_loss = CSDI_base.calc_loss

    def spy(self, observed_data, cond_mask, observed_mask, *args, **kwargs):
        scored.append((observed_mask - cond_mask).permute(0, 2, 1))       # back to (B, T, D)
        return calc_loss(self, observed_data, cond_mask, observed_mask, *args, **kwargs)

    monkeypatch.setattr(CSDI_base, "calc_loss", spy)
    run(small_dataset, seed=42, epochs=2)

    # Replay both epochs' shuffles and masks from a fresh copy of the run's streams.
    streams = RunStreams(42)
    loader = train_loader(small_dataset, 8, streams.data_order)
    expected = []
    for epoch in range(2):
        streams.start_epoch(epoch)
        expected += [streams.train_masker(mask)[1] for _, mask in loader]
    assert len(scored) == len(expected)
    assert all(torch.equal(s, e) for s, e in zip(scored, expected))

def test_each_step_records_the_loss_and_gradient_of_its_own_batch_alone(small_dataset):
    # lr = 0 freezes the weights, so every step can be replayed from scratch. Catches a
    # missing zero_grad (gradients pile up), noise from the wrong RNG, and a misordered step.
    config = tiny_config(lr=0.0, epochs=2)
    _, history = train(config, small_dataset, 42, "cpu")

    streams = RunStreams(42)
    streams.seed_model_init()
    model = build_model(config, "cpu").train()
    loader = train_loader(small_dataset, 8, streams.data_order)
    expected = []
    for epoch in range(2):
        streams.start_epoch(epoch)
        for values, mask in loader:
            cond_mask, _ = streams.train_masker(mask)
            loss = model(to_batch(values, mask, cond_mask), generator=streams.diffusion)
            grads = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
            norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads if g is not None]))
            expected += [loss.item(), norm.item()]
    assert [v for h in history for v in (h["loss"], h["grad_norm"])] == pytest.approx(expected, rel=1e-5)

# --- reproducibility ---------------------------------------------------------------------

def test_same_seed_reproduces_the_run_and_another_seed_does_not(small_dataset):
    weights_a, history_a = run(small_dataset, seed=42)
    weights_b, history_b = run(small_dataset, seed=42)
    weights_c, _ = run(small_dataset, seed=43)
    assert torch.equal(weights_a, weights_b) and history_a == history_b
    assert not torch.equal(weights_a, weights_c)

def test_training_switches_on_deterministic_kernels(small_dataset, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.use_deterministic_algorithms(False)
    run(small_dataset, epochs=1)
    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"    # needed by cuBLAS for deterministic CUDA

def test_a_cublas_workspace_that_is_already_set_is_kept(small_dataset, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    run(small_dataset, epochs=1)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"

def test_an_epoch_end_hook_cannot_shift_the_training_streams(small_dataset):
    # M3's validation runs here; whatever it draws or switches must not change training.
    def careless_hook(epoch, model):
        model.eval()
        torch.rand(1000)
        np.random.rand(1000)

    plain, _ = run(small_dataset)
    model, _ = train(tiny_config(), small_dataset, 42, "cpu", on_epoch_end=careless_hook)
    assert torch.equal(weights(model), plain)

# --- clipping, schedule, history, hook ---------------------------------------------------

def test_clipping_is_off_by_default_and_on_when_configured(small_dataset):
    default, history = run(small_dataset)                              # clip_grad_norm: null
    huge, _ = run(small_dataset, clip_grad_norm=1.0e9)
    tiny, _ = run(small_dataset, clip_grad_norm=1.0e-3)
    assert torch.equal(default, huge)
    assert not torch.equal(default, tiny)
    assert all(math.isfinite(h["grad_norm"]) and h["grad_norm"] > 0 for h in history)

def test_learning_rate_drops_by_gamma_at_each_milestone_epoch(small_dataset):
    _, history = run(small_dataset, epochs=4, lr_milestones=[0.5, 0.75])   # epochs 2 and 3
    lr_by_epoch = {h["epoch"]: h["lr"] for h in history}
    assert [lr_by_epoch[e] for e in range(4)] == pytest.approx([1.0e-3, 1.0e-3, 1.0e-4, 1.0e-5])

def test_no_drop_comes_before_the_first_epoch_has_trained(small_dataset):
    # Upstream's int(0.75 * 1) = int(0.9 * 1) = 0 would train a 1-epoch run at 1e-5 throughout
    # (M3-2 review); from 2 epochs on the milestones are upstream's, both at epoch 1 here.
    _, one = run(small_dataset, epochs=1)
    _, two = run(small_dataset, epochs=2)
    assert [h["lr"] for h in one] == pytest.approx([1.0e-3] * 4)
    lr_by_epoch = {h["epoch"]: h["lr"] for h in two}
    assert [lr_by_epoch[e] for e in range(2)] == pytest.approx([1.0e-3, 1.0e-5])

def test_history_has_one_plain_number_record_per_step(small_dataset):
    _, history = run(small_dataset)
    assert len(history) == 3 * 4
    assert [h["epoch"] for h in history] == [e for e in range(3) for _ in range(4)]
    for record in history:
        assert set(record) == {"epoch", "loss", "grad_norm", "lr"}
        assert all(isinstance(v, (int, float)) for v in record.values())   # not tensors

def test_epoch_end_hook_gets_each_epoch_in_order_with_the_model(small_dataset):
    calls = []
    model, _ = train(tiny_config(), small_dataset, 42, "cpu",
                     on_epoch_end=lambda epoch, model: calls.append((epoch, model)))
    assert [epoch for epoch, _ in calls] == [0, 1, 2]
    assert all(seen is model for _, seen in calls)
