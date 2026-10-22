"""BRITS, the learned baseline (M3-3): the model, what it may see, training, scoring and the record."""
import json

import numpy as np
import pytest
import torch
import wandb
from data.physionet import save_dataset
from eval import EvalMasker
from eval.baselines import RESULTS_PATH, baseline_results, wandb_summary
from models import brits
from models.brits import BRITS, LEARNED_PATH, FeatureRegression, impute, mean_se, time_gaps, train

def random_batch(n=4, seed=0):
    rng = np.random.default_rng(seed)
    mask = (rng.random((n, 48, 35)) < 0.3).astype(np.float32)
    return (rng.normal(size=(n, 48, 35)) * mask).astype(np.float32), mask

def untrained(seed=0):
    torch.manual_seed(seed)
    return BRITS()

# --- the time gaps ----------------------------------------------------------------------

def test_time_gaps_by_hand():
    seen = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0], dtype=torch.float32)        # one variable, 8 hours
    mask = seen.reshape(1, 8, 1)
    assert time_gaps(mask).flatten().tolist() == [0, 1, 2, 3, 1, 2, 1, 1]
    assert time_gaps(mask.flip(1)).flatten().tolist() == [0, 1, 1, 1, 2, 1, 2, 3]   # read backwards: 0,1,1,0,1,0,0,1
    assert time_gaps(torch.zeros(1, 4, 1)).flatten().tolist() == [0, 1, 2, 3]   # never visible

def test_time_gaps_come_from_the_mask_brits_is_given(monkeypatch):
    values, mask = random_batch()
    cond, _ = EvalMasker.from_config()(mask, np.arange(len(mask)))
    used = []
    real = brits.time_gaps
    monkeypatch.setattr(brits, "time_gaps", lambda m: used.append(m.clone()) or real(m))
    impute(untrained(), values, cond, "cpu")
    assert len(used) == 2                                   # once per direction
    assert torch.equal(used[0], torch.as_tensor(cond)) and torch.equal(used[1], torch.as_tensor(cond).flip(1))

# --- what the model may see -------------------------------------------------------------

def test_output_shape_and_visible_values_kept():
    values, mask = random_batch()
    guess = impute(untrained(), values, mask, "cpu")
    assert guess.shape == (4, 48, 35) and guess.dtype == np.float32 and np.isfinite(guess).all()
    assert np.array_equal(guess[mask == 1], values[mask == 1])

def test_never_reads_hidden_values():
    values, mask = random_batch()
    cond, _ = EvalMasker.from_config()(mask, np.arange(len(mask)))
    model = untrained()
    expected = impute(model, values, cond, "cpu")
    hidden = cond == 0                                      # held-out targets and never-observed boxes
    for changed in (values * 100, values + 5, np.full_like(values, np.nan)):
        assert np.array_equal(impute(model, np.where(hidden, changed, values), cond, "cpu"), expected)

def test_a_variable_cannot_see_itself():
    regression = FeatureRegression(35)
    x = torch.randn(3, 35)
    changed = x.clone()
    changed[:, 7] += 50.0
    before, after = regression(x), regression(changed)
    assert torch.equal(before[:, 7], after[:, 7])           # its own guess ignores it
    assert not torch.equal(before[:, 8], after[:, 8])       # the others use it

def test_patients_do_not_affect_each_other():
    values, mask = random_batch()
    model = untrained()
    expected = impute(model, values, mask, "cpu")
    other = values.copy()
    other[1] += 3.0
    assert np.array_equal(impute(model, other, mask, "cpu")[0], expected[0])

# --- training ---------------------------------------------------------------------------

def test_training_lowers_the_loss_and_keeps_the_best_val_weights(small_dataset):
    model, curve, best_epoch = train(small_dataset, 42, "cpu", max_epochs=6)
    assert curve[-1]["loss"] < curve[0]["loss"]
    assert best_epoch == int(np.argmin([row["val_mae"] for row in curve]))
    val = small_dataset["val_idx"]
    values = small_dataset["values"][val]
    cond, target = EvalMasker.from_config()(small_dataset["mask"][val], small_dataset["record_ids"][val])
    from eval import masked_mae
    assert masked_mae(impute(model, values, cond, "cpu"), values, target) == pytest.approx(curve[best_epoch]["val_mae"])

def test_training_stops_after_patience_epochs_without_improvement(small_dataset, monkeypatch):
    scores = iter([0.5, 0.4, 0.45, 0.41, 0.3])              # best at epoch 1, then two worse
    monkeypatch.setattr("eval.masked_mae", lambda *args: next(scores))
    _, curve, best_epoch = train(small_dataset, 42, "cpu", max_epochs=5, patience=2)
    assert best_epoch == 1 and len(curve) == 4

def test_same_seed_same_result_different_seed_different_result(small_dataset):
    runs = [train(small_dataset, seed, "cpu", max_epochs=2)[0].state_dict() for seed in (42, 42, 43)]
    assert all(torch.equal(runs[0][k], runs[1][k]) for k in runs[0])
    assert not all(torch.equal(runs[0][k], runs[2][k]) for k in runs[0])

# --- scoring and the record -------------------------------------------------------------

def test_scores_have_the_baselines_layout(small_dataset):
    model = untrained()
    ours = baseline_results(small_dataset, {"brits": lambda v, c: impute(model, v, c, "cpu")})
    theirs = baseline_results(small_dataset)
    for holdout in ("random", "block"):
        assert ours[holdout]["fractions"].keys() == theirs[holdout]["fractions"].keys()
        for fraction, rows in ours[holdout]["fractions"].items():
            for split, row in rows.items():
                assert set(row) == {"n_patients", "n_targets", "brits"} and set(row["brits"]) == {"mae", "rmse"}
                assert row["n_targets"] == theirs[holdout]["fractions"][fraction][split]["n_targets"]

def test_mean_and_se_by_hand(small_dataset):
    layout = baseline_results(small_dataset, {"brits": lambda v, c: np.zeros_like(v)})
    per_seed = []
    for mae in (0.30, 0.32, 0.34, 0.31, 0.33):
        scores = json.loads(json.dumps(layout))
        scores["random"]["fractions"]["0.1"]["test"]["brits"]["mae"] = mae
        per_seed.append(scores)
    summary = mean_se(per_seed)["random"]["fractions"]["0.1"]["test"]["brits"]["mae"]
    assert summary["mean"] == pytest.approx(0.32)
    assert summary["se"] == pytest.approx(np.std([0.30, 0.32, 0.34, 0.31, 0.33], ddof=1) / np.sqrt(5), abs=1e-6)
    assert mean_se(per_seed[:1])["random"]["fractions"]["0.1"]["test"]["brits"]["mae"]["se"] is None

def test_the_command_line_writes_the_record_and_logs_each_seed(small_dataset, tmp_path, monkeypatch):
    runs = []

    class FakeRun:
        def __init__(self, **kwargs):
            self.init, self.logs, self.summary = kwargs, [], {}
            runs.append(self)

        def log(self, data, step):
            self.logs.append((step, data))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(wandb, "init", FakeRun)
    save_dataset(small_dataset, tmp_path / "data.npz")
    brits.main(["--seeds", "42", "43", "--max-epochs", "2", "--dataset", str(tmp_path / "data.npz"),
                "--out", str(tmp_path / "learned.json"), "--wandb", "offline"])
    record = json.loads((tmp_path / "learned.json").read_text())
    assert record["model"] == brits.MODEL_NAME and record["fallback"] is False and record["seeds"] == [42, 43]
    assert [r.init["name"] for r in runs] == ["brits_seed42", "brits_seed43"]
    assert {r.init["job_type"] for r in runs} == {"baseline"} and {r.init["mode"] for r in runs} == {"offline"}
    for run, seed in zip(runs, ("42", "43")):
        scores = record["per_seed"][seed]["scores"]
        assert [step for step, _ in run.logs] == [0, 1] and set(run.logs[0][1]) == {"epoch/loss_mean", "epoch/val_mae"}
        assert not any(key in data for _, data in run.logs for key in run.summary)   # wandb would overwrite them
        assert run.summary == wandb_summary(scores, "brits") | {
            "best_epoch": record["per_seed"][seed]["best_epoch"], "epochs_run": 2}
    assert record["mean_se"] == mean_se([record["per_seed"][s]["scores"] for s in ("42", "43")])

@pytest.mark.skipif(not LEARNED_PATH.exists(), reason="results/learned_baseline.json not written yet")
def test_recorded_learned_baseline_beats_forward_fill():
    record = json.loads(LEARNED_PATH.read_text())
    floor = json.loads(RESULTS_PATH.read_text())["random"]["fractions"]
    assert record["model"] and record["seeds"] == [42, 43, 44, 45, 46]
    for fraction in ("0.1", "0.5", "0.9"):
        ours = record["mean_se"]["random"]["fractions"][fraction]["val"]["brits"]["mae"]["mean"]
        assert ours < floor[fraction]["val"]["forward_fill"]["mae"]
