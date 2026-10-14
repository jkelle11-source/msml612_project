import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from conftest import CONFIG, csdi_config, nonzero_output_layer
from diffusion.streams import STREAMS, RunStreams, derive_seed
from eval import TrainMasker
from models.csdi import CSDI_Physio

RUN_SEEDS = CONFIG["protocol"]["run_seeds"]
B, T, D = 4, 48, 35

def random_mask(n=B, seed=0, observed_rate=0.2):
    rng = np.random.default_rng(seed)
    return torch.from_numpy((rng.random((n, T, D)) < observed_rate).astype(np.float32))

def toy_batch():
    generator = torch.Generator().manual_seed(0)
    observed_mask = random_mask()
    return {
        "observed_data": torch.randn(B, T, D, generator=generator) * observed_mask,
        "observed_mask": observed_mask,
        "timepoints": torch.arange(T).float().expand(B, T),
    }

def draws(streams, steps=3):
    """What a few training steps take from each of the three streams, in order."""
    mask = random_mask()
    return (
        torch.randperm(100, generator=streams.data_order),
        torch.stack([streams.train_masker(mask)[1] for _ in range(steps)]),
        torch.randint(0, 50, [steps, B], generator=streams.diffusion),
        torch.randn(steps, B, generator=streams.diffusion),
    )

def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))

def at_epoch(run_seed, epoch):
    streams = RunStreams(run_seed)
    streams.start_epoch(epoch)
    return streams

# --- seeds ------------------------------------------------------------------------------

def test_every_stream_of_every_epoch_gets_its_own_seed():
    seeds = [derive_seed(run_seed, stream, epoch) for run_seed in RUN_SEEDS for stream in STREAMS for epoch in range(200)]
    assert len(set(seeds)) == len(seeds)                                    # no two streams share a sequence
    assert all(seed not in RUN_SEEDS for seed in seeds)
    assert seeds == [derive_seed(run_seed, stream, epoch) for run_seed in RUN_SEEDS for stream in STREAMS for epoch in range(200)]

def test_unknown_stream_raises():
    with pytest.raises(ValueError):
        derive_seed(42, "not a stream")

# --- reproducible, separate, private ----------------------------------------------------

def test_same_seed_and_epoch_give_the_same_draws():
    assert same(draws(at_epoch(42, 0)), draws(at_epoch(42, 0)))
    assert same(draws(RunStreams(42)), draws(at_epoch(42, 0)))             # a new run starts at epoch 0

def test_other_seeds_and_epochs_give_other_draws():
    base = draws(at_epoch(42, 0))
    for other in (draws(at_epoch(43, 0)), draws(at_epoch(42, 1))):
        assert not any(torch.equal(x, y) for x, y in zip(base, other))

def test_the_streams_are_not_copies_of_one_sequence():
    streams = at_epoch(42, 0)
    first = [torch.rand(8, generator=g) for g in (streams.data_order, streams.train_masker.generator, streams.diffusion)]
    assert not torch.equal(first[0], first[1]) and not torch.equal(first[1], first[2]) and not torch.equal(first[0], first[2])

def test_other_random_draws_do_not_shift_the_streams():
    expected = draws(at_epoch(42, 0))
    streams = at_epoch(42, 0)
    nn.Linear(35, 128)                      # like an extension's extra layers drawing init weights
    torch.rand(1000)                        # like dropout
    np.random.rand(1000)
    assert same(draws(streams), expected)

# --- start_epoch ------------------------------------------------------------------------

def test_start_epoch_pins_the_global_rng_whatever_came_before():
    seen = []
    for junk in (0, 1):
        streams = RunStreams(42)
        torch.manual_seed(junk)
        torch.rand(junk * 100 + 1)
        streams.start_epoch(3)
        seen.append(torch.rand(4))          # the next thing dropout would draw
    assert torch.equal(seen[0], seen[1])
    RunStreams(42).start_epoch(4)
    assert not torch.equal(torch.rand(4), seen[0])

def test_start_epoch_reseeds_in_place_so_a_loader_built_once_follows_it():
    def epoch_orders(epochs):
        streams = RunStreams(42)
        held = (streams.data_order, streams.train_masker, streams.diffusion)
        loader = DataLoader(TensorDataset(torch.arange(50)), batch_size=8, shuffle=True, generator=streams.data_order)
        orders = []
        for epoch in epochs:
            streams.start_epoch(epoch)
            orders.append(torch.cat([batch[0] for batch in loader]))
        assert held == (streams.data_order, streams.train_masker, streams.diffusion)     # same objects throughout
        return orders

    first, again = epoch_orders(range(3)), epoch_orders(range(3))
    assert same(first, again)
    assert not torch.equal(first[0], first[1]) and not torch.equal(first[1], first[2])
    assert sorted(first[0].tolist()) == list(range(50))

def test_train_masker_stream_matches_a_masker_seeded_directly():
    mask = random_mask()
    streams = at_epoch(42, 5)
    direct = TrainMasker(derive_seed(42, "train_mask", 5))
    for _ in range(3):
        assert torch.equal(streams.train_masker(mask)[1], direct(mask)[1])

# --- model init -------------------------------------------------------------------------

def build(run_seed, junk=0, extra_layers=False):
    torch.manual_seed(junk)                 # whatever state the process happens to be in
    streams = RunStreams(run_seed)
    streams.seed_model_init()
    model = CSDI_Physio(csdi_config(), device="cpu", target_dim=D)
    if extra_layers:
        nn.TransformerEncoderLayer(64, 2, batch_first=True)     # like Ext 1's encoder, built after the base
    return streams, model

def weights(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def test_model_init_depends_only_on_the_run_seed():
    assert torch.equal(weights(build(42, junk=0)[1]), weights(build(42, junk=1)[1]))
    assert torch.equal(weights(build(42)[1]), weights(build(42, extra_layers=True)[1]))
    assert not torch.equal(weights(build(42)[1]), weights(build(43)[1]))

# --- the point of it all: variants stay paired on a seed --------------------------------

def training_losses(extra_layers, steps=3):
    streams, model = build(42, junk=int(extra_layers), extra_layers=extra_layers)
    nonzero_output_layer(model).train()                         # dropout on
    batch = toy_batch()
    streams.start_epoch(0)
    losses = []
    with torch.no_grad():
        for _ in range(steps):
            cond, _ = streams.train_masker(batch["observed_mask"])
            losses.append(model({**batch, "gt_mask": cond}, generator=streams.diffusion).item())
    return losses

def test_a_variant_that_builds_extra_layers_sees_the_same_training_draws():
    assert training_losses(extra_layers=False) == training_losses(extra_layers=True)

# --- resuming at an epoch boundary, with no RNG state in the checkpoint ------------------

def test_resumed_streams_continue_exactly():
    running = RunStreams(42)
    uninterrupted = []
    for epoch in range(4):
        running.start_epoch(epoch)
        uninterrupted.append(draws(running, steps=3 + epoch))       # epochs need not draw equally often
    torch.manual_seed(123)                  # a new process after a crash: nothing carried over
    resumed = RunStreams(42)
    for epoch in (2, 3):
        resumed.start_epoch(epoch)
        assert same(draws(resumed, steps=3 + epoch), uninterrupted[epoch])

def train(model, optimizer, streams, epochs, n_patients=8):
    """A minimal stand-in for the M2-1 loop: shuffle, mask, noise, step."""
    generator = torch.Generator().manual_seed(0)
    observed_mask = random_mask(n_patients)
    observed_data = torch.randn(n_patients, T, D, generator=generator) * observed_mask
    loader = DataLoader(TensorDataset(torch.arange(n_patients)), batch_size=B, shuffle=True, generator=streams.data_order)
    model.train()                           # dropout on
    for epoch in epochs:
        streams.start_epoch(epoch)
        for (rows,) in loader:
            cond, _ = streams.train_masker(observed_mask[rows])
            batch = {
                "observed_data": observed_data[rows],
                "observed_mask": observed_mask[rows],
                "gt_mask": cond,
                "timepoints": torch.arange(T).float().expand(len(rows), T),
            }
            optimizer.zero_grad()
            model(batch, generator=streams.diffusion).backward()
            optimizer.step()

def test_a_run_resumed_from_a_checkpoint_ends_with_the_same_weights(tmp_path):
    streams, model = build(42)
    train(model, torch.optim.Adam(model.parameters(), lr=1e-3), streams, epochs=range(3))

    streams, interrupted = build(42)
    optimizer = torch.optim.Adam(interrupted.parameters(), lr=1e-3)
    train(interrupted, optimizer, streams, epochs=range(1))
    torch.save({"model": interrupted.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 0}, tmp_path / "checkpoint.pt")
    assert not torch.equal(weights(interrupted), weights(model))

    _, resumed = build(7, junk=5)           # a new process: other weights, other RNG state
    checkpoint = torch.load(tmp_path / "checkpoint.pt")
    resumed.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(resumed.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint["optimizer"])
    train(resumed, optimizer, RunStreams(42), epochs=range(checkpoint["epoch"] + 1, 3))
    assert torch.equal(weights(resumed), weights(model))
