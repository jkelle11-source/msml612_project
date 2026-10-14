"""M2-3: the reused reverse sampler on real Set A patients, driven through ``EvalMasker``.

``tests/test_eval_sampler.py`` covers ``EvalSampler`` itself on synthetic data with a 4-step
schedule. This is the end-to-end check the issue asks for: an untrained model built from our
config (upstream's full 50 steps), real patients, and ``EvalMasker``'s conditioning mask
passed as upstream's ``gt_mask``, so shape, NaN and mask hand-off bugs show up before training.
"""
import numpy as np
import pytest
from conftest import CONFIG, RAW_DIR, nonzero_output_layer
from eval import EvalMasker, EvalSampler, masked_mae, median_imputation
from models.build import build_model

needs_raw_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="raw Set A not downloaded")

N, K = 4, 3                  # 4 patients in batches of 2, so the pass spans more than one batch

@pytest.fixture(scope="module")
def real_batch(set_a):
    idx = set_a["test_idx"][:N]
    mask, values = set_a["mask"][idx], set_a["values"][idx]
    cond, target = EvalMasker.from_config()(mask, set_a["record_ids"][idx])
    model = nonzero_output_layer(build_model(CONFIG, "cpu"))   # untrained, but not a zero output
    sampler = EvalSampler(CONFIG["protocol"]["eval_sampling_seed"], K, batch_size=2)
    return model, sampler, values, mask, cond, target, sampler(model, values, mask, cond)

@needs_raw_data
def test_the_model_runs_upstreams_full_schedule(real_batch):
    model = real_batch[0]
    assert model.num_steps == CONFIG["diffusion"]["num_steps"] == 50

@needs_raw_data
def test_samples_come_back_as_k_b_48_d_with_no_nans(real_batch):
    samples = real_batch[-1]
    assert samples.shape == (K, N, 48, 35) and samples.dtype == np.float32
    assert np.isfinite(samples).all()

@needs_raw_data
def test_the_k_draws_are_separate_samples(real_batch):
    samples = real_batch[-1]
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert not np.array_equal(samples[a], samples[b])

@needs_raw_data
def test_held_out_values_cannot_reach_the_samples(real_batch):
    model, sampler, values, mask, cond, target, samples = real_batch
    tampered = np.where(target == 1, values + 100, values).astype(np.float32)
    assert np.array_equal(sampler(model, tampered, mask, cond), samples)

@needs_raw_data
def test_samples_score_through_the_metrics(real_batch):
    values, target, samples = real_batch[2], real_batch[5], real_batch[-1]
    assert target.sum() > 0
    assert np.isfinite(masked_mae(median_imputation(samples), values, target))
