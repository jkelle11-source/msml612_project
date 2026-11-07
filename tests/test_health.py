import copy
import json
import random

import pytest
import torch
from conftest import CONFIG, ROOT
from diffusion.health import main, record, record_path

APPLE_M4_PRO = {"apple": {"name": "Apple M4 Pro", "gpuCores": 20, "memoryGb": 24}, "gpu": None}
APPLE_M4 = {"apple": {"name": "Apple M4", "gpuCores": 8, "memoryGb": 16}, "gpu": None}
COLAB_T4 = {"gpu": "Tesla T4", "gpu_count": 1}

# The health report of the fake runs' first 500 steps, worked out by hand: loss 1.0 on steps 0-49,
# 0.5 on 50-449 and 0.25 from 450 on, so the first tenth (50 steps) averages 1.0 and the last 0.25;
# every gradient norm is 1.0 but step 7's 3.0, so the median is 1.0 and 3.0 is no spike (< 10x).
HEALTH = {"n_steps": 500, "finite": True, "loss_first_tenth": 1.0, "loss_last_tenth": 0.25,
          "grad_norm_median": 1.0, "grad_norm_max": 3.0, "grad_norm_max_step": 7, "spikes": [],
          "learning": True, "collapsed": False, "exploded": False, "complete": True, "healthy": True}
EPOCH_SECONDS = [30.0, 31.0, 90.0, 33.0]        # epoch 2 stalls: 184.0 in all, median 32.0
STEPS_PER_EPOCH = 150                           # 4 epochs, 600 steps: more than the 500 checked

def fake_run(variant="base", every=2, val_mae=(0.30, 0.25), metadata=APPLE_M4_PRO, run_id="run1"):
    """A finished 4-epoch run as ``diffusion.health.fetch`` returns it, with wandb's flattened summary."""
    config = copy.deepcopy({k: v for k, v in CONFIG.items() if k != "tracking"})
    config["train"]["epochs"] = 4
    config["validation"]["every_n_epochs"] = every
    config["model"]["use_mask_embedding"] = variant == "ext1"
    config |= {"seed": 42, "variant": variant, "regime": "random", "device": "mps", "out_dir": f"/runs/{variant}"}
    history, maes = [], iter(val_mae)
    for step in range(4 * STEPS_PER_EPOCH):
        epoch = step // STEPS_PER_EPOCH
        row = {"_step": step, "epoch": epoch, "train/lr": 1e-3,
               "train/loss": 1.0 if step < 50 else 0.5 if step < 450 else 0.25,
               "train/grad_norm": 3.0 if step == 7 else 1.0}
        if (step + 1) % STEPS_PER_EPOCH == 0:
            row["epoch/seconds"] = EPOCH_SECONDS[epoch]
            if (epoch + 1) % every == 0:
                row |= {"val/mae": next(maes), "val/rmse": 1.0, "val/seconds": 300.0}
        history.append(row)
    summary = {f"health.{k}": v for k, v in HEALTH.items()} | {
        "final_val_mae": val_mae[-1], "forward_fill_val_mae": 0.436614, "below_forward_fill": True}
    return {"path": f"DATA612/msml612-csdi/{run_id}", "name": f"{variant}_random_seed42", "state": "finished",
            "tags": [variant, "health"], "config": config, "summary": summary,
            "metadata": {"git": {"commit": f"commit-of-{variant}"}, **metadata},
            "history": history, "weights": None}

def ext1_run(**kwargs):
    return fake_run("ext1", every=1, val_mae=(0.31, 0.29, 0.27, 0.24), run_id="run2", **kwargs)

# --- the record ---------------------------------------------------------------------------

def test_the_record_by_hand():
    result = record([fake_run(), ext1_run(metadata=APPLE_M4)], notes={"base": "the gate run, reused"})
    assert result["status"] == "health only, not an effect measurement"
    assert (result["ticket"], result["seed"], result["regime"]) == ("M5-3", 42, "random")
    assert result["validation"] == {"split": "val", "holdout": "random", "fraction": 0.1,
                                    "n_samples": 5, "batch_size": 64}
    base, ext1 = result["runs"]["base"], result["runs"]["ext1"]
    assert base == {
        "run": "base_random_seed42", "wandb_run": "DATA612/msml612-csdi/run1", "tags": ["base", "health"],
        "commit": "commit-of-base", "device": "mps", "machine": "Apple M4 Pro (20-core GPU, 24 GB)",
        "note": "the gate run, reused", "health": HEALTH,
        "final_val_mae": 0.25, "forward_fill_val_mae": 0.436614, "below_forward_fill": True,
        "val_mae_by_epoch": {"1": 0.30, "3": 0.25},     # every 2 epochs, 0-based: epochs 1 and 3
        "train_seconds": 184.0, "val_seconds": 600.0, "val_passes": 2,
        "epoch_seconds_median": 32.0, "slowest_epoch": {"epoch": 2, "seconds": 90.0},
    }                                                   # no gate: base has no encoder
    assert ext1["machine"] == "Apple M4 (8-core GPU, 16 GB)" and "note" not in ext1
    assert ext1["val_mae_by_epoch"] == {"0": 0.31, "1": 0.29, "2": 0.27, "3": 0.24}
    assert (ext1["final_val_mae"], ext1["val_seconds"], ext1["val_passes"]) == (0.24, 1200.0, 4)
    assert ext1["mask_encoder_gate"] is None            # no model.pt on the run, none given
    assert result["config_differences"] == {"model.use_mask_embedding": {"base": False, "ext1": True},
                                            "validation.every_n_epochs": {"base": 2, "ext1": 1}}

def test_timings_compare_only_on_one_machine():
    assert record([fake_run(), ext1_run()])["timings"] == \
        "comparable: every run is on Apple M4 Pro (20-core GPU, 24 GB)"
    assert record([fake_run(), ext1_run(metadata=APPLE_M4)])["timings"].startswith("not comparable")
    t4 = record([fake_run(metadata=COLAB_T4), ext1_run(metadata=COLAB_T4)])
    assert t4["runs"]["base"]["machine"] == "Tesla T4" and t4["timings"] == "comparable: every run is on Tesla T4"
    assert record([fake_run(metadata={}), ext1_run()])["timings"].startswith("unknown")

def test_the_order_wandb_returns_rows_in_does_not_matter():
    shuffled = fake_run()
    random.Random(0).shuffle(shuffled["history"])
    assert record([shuffled]) == record([fake_run()])

def test_a_nested_summary_reads_like_a_flattened_one():
    nested = fake_run()
    nested["summary"] = {k: v for k, v in nested["summary"].items() if not k.startswith("health.")} | {"health": HEALTH}
    assert record([nested]) == record([fake_run()])

def test_a_float_wandb_rounded_in_its_last_digit_is_not_a_mismatch():
    run = fake_run()
    run["summary"]["final_val_mae"] = 0.25 * (1 + 1e-15)
    assert record([run])["runs"]["base"]["final_val_mae"] == 0.25      # the logged value, not the summary's

# --- refusals -----------------------------------------------------------------------------

def unfinished(runs): runs[0]["state"] = "running"
def short(runs): runs[0]["history"] = runs[0]["history"][:3 * STEPS_PER_EPOCH]
def health_differs(runs): runs[0]["summary"]["health.loss_first_tenth"] = 0.9
def final_mae_differs(runs): runs[0]["summary"]["final_val_mae"] = 0.26
def other_seed(runs): runs[1]["config"]["seed"] = 43
def other_regime(runs): runs[1]["config"]["regime"] = "block"
def other_draws(runs): runs[1]["config"]["validation"]["n_samples"] = 10
def other_holdout_seed(runs): runs[1]["config"]["protocol"]["eval_masker_seed"] = 1
def same_variant(runs): runs[1]["config"]["variant"] = "base"
def unknown_regime(runs):
    for run in runs:
        run["config"]["regime"] = "block"

@pytest.mark.parametrize("break_it, message", [
    (unfinished, "not finished"),
    (short, "3 epochs logged, not the config's 4"),
    (health_differs, "health report"),
    (final_mae_differs, "final_val_mae"),
    (other_seed, "different seeds"),
    (other_regime, "different regimes"),
    (other_draws, "validation.n_samples"),
    (other_holdout_seed, "protocol.eval_masker_seed"),
    (same_variant, "two runs of one variant"),
    (unknown_regime, "no validation holdout known"),
])
def test_runs_that_would_make_the_record_mean_something_else_are_refused(break_it, message):
    runs = [fake_run(), ext1_run()]
    break_it(runs)
    with pytest.raises(ValueError, match=message):
        record(runs)

def test_a_note_for_a_variant_not_given_is_refused():
    with pytest.raises(ValueError, match="notes for variants not given"):
        record([fake_run()], notes={"ext2": "pending"})

# --- Ext 1's gate -------------------------------------------------------------------------

def test_the_gate_is_read_from_the_runs_weights():
    run = ext1_run()
    run["weights"] = {"mask_encoder.gate": torch.tensor(-0.15579)}
    assert record([fake_run(), run])["runs"]["ext1"]["mask_encoder_gate"] == -0.1558

def test_weights_for_a_run_without_the_encoder_are_refused():
    run = fake_run()
    run["weights"] = {"mask_encoder.gate": torch.tensor(0.0)}
    with pytest.raises(ValueError, match="without Ext 1's encoder"):
        record([run])

# --- the command line ---------------------------------------------------------------------

def test_the_command_line_writes_the_record(tmp_path):
    runs = {"DATA612/msml612-csdi/run1": fake_run(), "DATA612/msml612-csdi/run2": ext1_run()}
    fetched = []
    def fetch(path):
        fetched.append(path)
        return copy.deepcopy(runs[path])
    weights = tmp_path / "model.pt"
    torch.save({"mask_encoder.gate": torch.tensor(-0.15579)}, weights)
    out = tmp_path / "health.json"
    main(["run1", "DATA612/msml612-csdi/run2", "--weights", f"ext1={weights}",
          "--note", "base=the gate run, reused", "--out", str(out)], fetch=fetch)
    assert fetched == ["DATA612/msml612-csdi/run1", "DATA612/msml612-csdi/run2"]   # bare ids: the tracking project
    written = json.loads(out.read_text())
    assert written["runs"]["ext1"]["mask_encoder_gate"] == -0.1558
    assert written["runs"]["base"]["note"] == "the gate run, reused"
    assert record_path(42) == ROOT / "results" / "health_seed42.json"

def test_the_command_line_refuses_weights_for_a_variant_not_given(tmp_path):
    with pytest.raises(SystemExit, match="not among the runs given"):
        main(["run1", "--weights", f"ext1={tmp_path / 'model.pt'}"], fetch=lambda path: fake_run())
