"""The GPU timing test (M6-2): the estimate against a hand count, and a timing run end to end."""
import copy
import json
import math
import types

import pytest
import torch
import yaml
from conftest import CONFIG
from diffusion import timing
from diffusion.timing import estimate_hours, main, n_validations, record_name, time_gpu
from diffusion.train import use_tf32
from eval import EvalSampler

# --- the estimate -----------------------------------------------------------------------

def test_a_run_validates_every_nth_epoch_and_on_the_last():
    assert [n_validations(e, 10) for e in (200, 20, 25, 2)] == [20, 2, 3, 1]

TIMINGS = {"base": {"epoch_seconds": 10, "validation_seconds": 20, "scoring_batch_seconds": 30},
           "ext1+2": {"epoch_seconds": 12, "validation_seconds": 25, "scoring_batch_seconds": 40}}
CELLS = CONFIG["sweep"]["regimes"]                         # the sweep's: what diffusion.sweep trains

def test_the_sweep_estimate_matches_a_hand_count():
    # 30 runs. At base's cost, base and ext2 in the random regime and base block-aware, 5 seeds each:
    # 15 runs of 200*10 + 20 validations*20 + 2 holdouts*50 batches*30 = 5,400 s. At ext1+2's cost,
    # ext1 and ext1+2 random and ext1 block-aware: 15 runs of 200*12 + 20*25 + 2*50*40 = 6,900 s.
    # Then base's 50% and 90% at 5 seeds: 10 passes of 50*30 = 1,500 s.
    assert estimate_hours(TIMINGS, CELLS, epochs=200, every=10, n_seeds=5, n_batches=50, robustness_passes=10) \
        == pytest.approx((15 * 5_400 + 15 * 6_900 + 10 * 1_500) / 3600)            # 55.4 h

def test_the_dry_run_estimate_matches_a_hand_count():
    # The same 6 cells at 2 seeds and 20 epochs, scored as the sweep is, no robustness passes:
    # 6 runs of 20*10 + 2*20 + 2*50*30 = 3,240 s and 6 of 20*12 + 2*25 + 2*50*40 = 4,290 s.
    assert estimate_hours(TIMINGS, CELLS, epochs=20, every=10, n_seeds=2, n_batches=50, robustness_passes=0) \
        == pytest.approx((6 * 3_240 + 6 * 4_290) / 3600)                           # 12.6 h

def test_the_estimate_costs_the_cells_it_is_given():
    # Only base random and ext1blind block-aware (at ext1+2's cost), 3 seeds, no robustness passes:
    # 3 runs of 5,400 s and 3 of 6,900 s, as in the sweep's hand count.
    cells = {"random": ["base"], "block-aware": ["ext1blind"]}
    assert estimate_hours(TIMINGS, cells, epochs=200, every=10, n_seeds=3, n_batches=50, robustness_passes=0) \
        == pytest.approx((3 * 5_400 + 3 * 6_900) / 3600)                           # 10.25 h

# --- a timing run -----------------------------------------------------------------------

def tiny():
    """The repo's config with a model, validation and sampler small enough for the 40 fake patients."""
    config = copy.deepcopy(CONFIG)
    config["diffusion"].update(channels=16, layers=1, num_steps=10)
    config["train"].update(batch_size=8)
    config["validation"].update(n_samples=2, batch_size=8)
    config["protocol"].update(eval_n_samples=2, eval_batch_size=2)          # several batches of test patients
    return config

def test_a_timing_trains_two_epochs_and_times_two_scoring_batches_of_each_variant(small_dataset, tmp_path, monkeypatch):
    drawn, real = [], EvalSampler.batches

    def counting(self, *args):
        for batch in real(self, *args):
            drawn.append((self.n_samples, self.batch_size))
            yield batch

    monkeypatch.setattr(EvalSampler, "batches", counting)
    monkeypatch.setattr(timing, "RUNS", tmp_path / "runs")
    real_scoring = timing.time_scoring                      # timed for real, reported as 7 s then 3 s
    monkeypatch.setattr(timing, "time_scoring", lambda *args: real_scoring(*args) and [7.0, 3.0])
    config = tiny()
    cells = config["sweep"]["regimes"] = {"random": ["base", "ext2"], "block-aware": ["ext1"]}   # not the repo's
    record = time_gpu(config, small_dataset, "cpu", log=lambda line: None)
    scoring = [batch for batch in drawn if batch != (2, 8)]     # the validation passes (batch 8) aside
    assert scoring == [(2, 2)] * 4                          # two batches of each variant's scoring pass, not the pass
    for variant in ("base", "ext1+2"):
        folder = tmp_path / "runs" / variant
        curve = json.loads((folder / "curve.json").read_text())
        assert len(curve) == 2                              # 2 epochs, whatever train.epochs says
        times = record["timings"][variant]
        assert (times["first_epoch_seconds"], times["epoch_seconds"], times["validation_seconds"]) \
            == (curve[0]["seconds"], curve[1]["seconds"], curve[1]["val_seconds"])
        assert (times["first_scoring_batch_seconds"], times["scoring_batch_seconds"]) == (7.0, 3.0)  # the second
        assert not list(folder.glob("scores_*"))                                                       # nothing scored
    n_batches = math.ceil(len(small_dataset["test_idx"]) / 2)
    expected = {"train_epochs": 200, "validate_every": 10, "seeds": 5,      # the sweep's, from the config
                "scoring_batches_per_pass": n_batches, "robustness_passes": 10}
    assert {key: record["assumes"][key] for key in expected} == expected
    assert record["assumes"]["sweep_cells"] == cells                        # the config's sweep block
    assert record["estimate_hours"] == {
        "sweep": round(estimate_hours(record["timings"], cells, 200, 10, 5, n_batches, 10), 1),
        "dry_run": round(estimate_hours(record["timings"], cells, 20, 10, 2, n_batches, 0), 1)}
    assert (record["gpu"], record["device"], record["tf32"]) == (None, "cpu", False)

def test_scoring_batches_are_timed_one_by_one_and_only_the_first_two(small_dataset, monkeypatch):
    clock, yielded = {"now": 100.0}, []
    monkeypatch.setattr(timing, "time", types.SimpleNamespace(time=lambda: clock["now"]))

    def batches(self, model, values, mask, cond_mask):
        for seconds in (7.0, 3.0, 3.0, 3.0):                # the first batch also warms the GPU up
            clock["now"] += seconds
            yielded.append(seconds)
            yield slice(0, 1), None

    monkeypatch.setattr(EvalSampler, "batches", batches)
    assert timing.time_scoring(None, small_dataset, tiny()["protocol"]) == [7.0, 3.0]
    assert yielded == [7.0, 3.0]                            # the rest of the pass is never drawn

def test_with_tf32_the_runs_and_the_scoring_batches_are_timed_with_it_on(small_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "RUNS", tmp_path / "runs")
    try:
        record = time_gpu(tiny(), small_dataset, "cpu", tf32=True, log=lambda line: None)
        assert torch.backends.cudnn.conv.fp32_precision == torch.backends.cuda.matmul.fp32_precision == "tf32"
    finally:
        use_tf32(False)
    assert record["tf32"] is True
    for variant in ("base", "ext1+2"):
        assert yaml.safe_load((tmp_path / "runs" / variant / "config.yaml").read_text())["protocol"]["tf32"] is True

# --- the record -------------------------------------------------------------------------

@pytest.mark.parametrize("gpu, device, tf32, name", [("Tesla T4", "cuda", False, "tesla_t4"),
                                                     ("NVIDIA A100-SXM4-40GB", "cuda", True, "nvidia_a100_sxm4_40gb_tf32"),
                                                     (None, "mps", False, "mps")])
def test_a_record_is_named_for_its_gpu_and_tf32_setting(gpu, device, tf32, name):
    assert record_name({"gpu": gpu, "device": device, "tf32": tf32}) == name

def test_the_command_line_writes_the_record(tmp_path, monkeypatch):
    asked = {}

    def fake_time_gpu(config, dataset, device, tf32=False, log=print):
        asked.update(device=device, tf32=tf32, epochs=config["train"]["epochs"])
        return {"gpu": "NVIDIA L4", "device": device, "tf32": tf32, "estimate_hours": {"sweep": 80.0, "dry_run": 20.0}}

    monkeypatch.setattr(timing, "TIMING", tmp_path)
    monkeypatch.setattr(timing, "load_dataset", lambda path: None)
    monkeypatch.setattr(timing, "time_gpu", fake_time_gpu)
    record = main(["--device", "cpu", "--tf32"])
    assert asked == {"device": "cpu", "tf32": True, "epochs": CONFIG["train"]["epochs"]}     # the repo's config
    assert json.loads((tmp_path / "nvidia_l4_tf32.json").read_text()) == record
