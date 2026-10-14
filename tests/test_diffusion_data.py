import copy

import torch
from conftest import CONFIG
from diffusion.data import to_batch, train_loader
from diffusion.streams import RunStreams
from models.build import build_model

T, D = 48, 35

def rows_of(loader):
    """One full pass, concatenated: (values, mask), each (n, T, D)."""
    values, masks = zip(*loader)
    return torch.cat(values), torch.cat(masks)

def patient_of_each_row(dataset, values):
    """Which patient (index into the full dataset) each row of ``values`` is."""
    full = torch.from_numpy(dataset["values"]).flatten(1)
    matches = (values.flatten(1)[:, None] == full[None]).all(-1)       # (rows, N)
    assert (matches.sum(1) == 1).all(), "a row matches no patient, or several"
    return matches.int().argmax(1)

# --- train_loader ------------------------------------------------------------------------

def test_one_pass_yields_every_training_patient_once_and_nothing_else(small_dataset):
    values, _ = rows_of(train_loader(small_dataset, 8, torch.Generator().manual_seed(0)))
    patients = patient_of_each_row(small_dataset, values)
    assert sorted(patients.tolist()) == sorted(small_dataset["train_idx"].tolist())

def test_each_patients_mask_travels_with_their_values(small_dataset):
    values, mask = rows_of(train_loader(small_dataset, 8, torch.Generator().manual_seed(0)))
    patients = patient_of_each_row(small_dataset, values)
    assert torch.equal(mask, torch.from_numpy(small_dataset["mask"][patients.numpy()]))

def test_batches_are_float32_in_our_layout(small_dataset):
    sizes = []
    for values, mask in train_loader(small_dataset, 8, torch.Generator().manual_seed(0)):
        assert values.shape == mask.shape == (len(values), T, D)
        assert values.dtype == mask.dtype == torch.float32
        sizes.append(len(values))
    assert sizes == [8, 8, 8, 4]                                        # 28 training patients

def test_order_is_a_function_of_run_seed_and_epoch(small_dataset):
    # The real usage: built once on streams.data_order, re-seeded in place by start_epoch.
    streams = RunStreams(42)
    loader = train_loader(small_dataset, 8, streams.data_order)

    def order(epoch):
        streams.start_epoch(epoch)
        return patient_of_each_row(small_dataset, rows_of(loader)[0])

    assert torch.equal(order(0), order(0))
    assert not torch.equal(order(0), order(1))

# --- to_batch ----------------------------------------------------------------------------

def test_batch_holds_exactly_upstreams_four_keys():
    values, mask, cond_mask = torch.randn(3, T, D), torch.ones(3, T, D), torch.zeros(3, T, D)
    batch = to_batch(values, mask, cond_mask)
    assert set(batch) == {"observed_data", "observed_mask", "gt_mask", "timepoints"}
    assert batch["observed_data"] is values
    assert batch["observed_mask"] is mask
    assert batch["gt_mask"] is cond_mask

def test_timepoints_are_the_hour_index_for_every_patient():
    # The same timepoints eval.EvalSampler gives the model, so training and scoring agree.
    for n in (1, 5):                                                    # odd sizes too
        timepoints = to_batch(torch.zeros(n, T, D), torch.zeros(n, T, D), torch.zeros(n, T, D))["timepoints"]
        assert timepoints.dtype == torch.float32
        assert torch.equal(timepoints, torch.arange(T).float().expand(n, T))

def test_the_model_trains_on_a_batch_from_the_feed(small_dataset):
    streams = RunStreams(42)
    streams.seed_model_init()
    model = build_model(copy.deepcopy(CONFIG), "cpu")
    streams.start_epoch(0)
    values, mask = next(iter(train_loader(small_dataset, 8, streams.data_order)))
    cond_mask, _ = streams.train_masker(mask)
    loss = model(to_batch(values, mask, cond_mask), generator=streams.diffusion)
    loss.backward()
    assert torch.isfinite(loss)
