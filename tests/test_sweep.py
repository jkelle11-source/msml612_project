import copy
import hashlib
import json
import subprocess

import pytest
import torch
import yaml
from conftest import CONFIG
from data.physionet import save_dataset
from diffusion.run import train, variant_name
from diffusion.score import score
from diffusion.sweep import cells, compare, current_commit, main, reproduces, sweep, uncommitted

SOME = [("random", "base", 42), ("random", "ext1", 42), ("block-aware", "base", 42)]
NAMES = ["base_random_seed42", "ext1_random_seed42", "base_block-aware_seed42"]

def quick_config():
    """A tiny model, 2 epochs and 2 scoring draws: a cell trains and is scored in seconds on the fake patients."""
    config = copy.deepcopy(CONFIG)
    config["diffusion"].update(channels=16, layers=1, num_steps=10)
    config["train"].update(epochs=2, batch_size=8)
    config["validation"].update(every_n_epochs=1, n_samples=2, batch_size=8)
    config["protocol"]["eval_n_samples"] = 2
    return config

@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(quick_config()))
    return path

def quiet(line):
    pass

def go(config_path, dataset, out, todo, tag="test", **kwargs):
    """``sweep`` as the command line runs it, unpaired and untracked, on the CPU."""
    config = yaml.safe_load(config_path.read_text())
    return sweep(config, dataset, out, tag, "cpu", todo, config["sweep"]["holdouts"], paired=False,
                 wandb_mode="disabled", config_path=config_path, log=quiet, **kwargs)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class SessionLost(Exception):
    """A Colab session ending in the middle of the sweep."""

# --- which cells, in which order --------------------------------------------------------

def test_the_cells_run_regime_by_regime_then_seed_by_seed():
    todo = cells(CONFIG)
    assert len(todo) == 30                                                  # 20 random-regime + 10 block-aware
    assert todo[:5] == [("random", "base", 42), ("random", "ext1", 42), ("random", "ext2", 42),
                        ("random", "ext1+2", 42), ("random", "base", 43)]
    assert todo[19:22] == [("random", "ext1+2", 46), ("block-aware", "base", 42), ("block-aware", "ext1", 42)]
    assert todo[-1] == ("block-aware", "ext1", 46)
    assert len(cells(CONFIG, seeds=[42, 43])) == 12                         # the dry run: 8 + 4
    assert cells(CONFIG, ["block-aware"], ["ext1blind"], [42, 43, 44]) == [   # M8-4's control
        ("block-aware", "ext1blind", 42), ("block-aware", "ext1blind", 43), ("block-aware", "ext1blind", 44)]
    for bad in ({"regimes": ["block"]}, {"variants": ["ext3"]}):
        with pytest.raises(ValueError):
            cells(CONFIG, **bad)

# --- each cell: trained, checked, scored, attached --------------------------------------

def test_every_cell_is_trained_scored_on_both_holdouts_and_attached(config_path, small_dataset, tmp_path, monkeypatch):
    attached = []
    monkeypatch.setattr("diffusion.sweep.attach_to_wandb",
                        lambda scores, mode: attached.append((scores["run"], scores["holdout"])))
    run_dirs = go(config_path, small_dataset, tmp_path / "out", SOME)       # two regimes at one seed: never compared
    assert [run_dir.name for run_dir in run_dirs] == NAMES
    assert attached == [(name, holdout) for name in NAMES for holdout in ("random", "block")]
    for run_dir in run_dirs:
        for holdout in ("random", "block"):
            scores = json.loads((run_dir / f"scores_test_{holdout}_0.1.json").read_text())
            assert scores["model_sha256"] == sha256(run_dir / "model.pt")    # the final weights
            assert scores["sampler"]["eval_n_samples"] == 2                 # config_path's protocol
    saved = yaml.safe_load((tmp_path / "out" / "base_block-aware_seed42" / "config.yaml").read_text())
    assert saved["train"]["regime"] == "block-aware" and "sweep" not in saved
    assert json.loads((tmp_path / "out" / "sweep.json").read_text()) == {"tag": "test", "commit": current_commit()}

def test_runs_are_tagged_for_the_store(config_path, small_dataset, tmp_path, monkeypatch):
    inits = []

    class FakeRun:
        def __init__(self, kwargs):
            self.init, self.summary, self.disabled, self.path = kwargs, {}, False, "entity/project/fakeid"

        def log(self, data, step):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *rest):
            pass

    monkeypatch.setattr("diffusion.run.wandb.init", lambda **kwargs: inits.append(kwargs) or FakeRun(kwargs))
    go(config_path, small_dataset, tmp_path / "out", SOME[2:], tag="dry-run")
    assert inits[0]["tags"] == ["base", "block-aware", "seed42", "dry-run"]

def test_the_same_command_again_trains_and_scores_nothing_more(config_path, small_dataset, tmp_path, monkeypatch):
    go(config_path, small_dataset, tmp_path / "out", SOME)
    before = {path: path.read_bytes() for path in (tmp_path / "out").rglob("*") if path.is_file()}
    attached = []

    def must_not_run(*args, **kwargs):
        raise AssertionError("a finished cell was trained or scored again")

    monkeypatch.setattr("diffusion.sweep.run", must_not_run)
    monkeypatch.setattr("diffusion.sweep.score", must_not_run)
    monkeypatch.setattr("diffusion.sweep.attach_to_wandb", lambda scores, mode: attached.append(scores["run"]))
    go(config_path, small_dataset, tmp_path / "out", SOME)
    assert {path: path.read_bytes() for path in (tmp_path / "out").rglob("*") if path.is_file()} == before
    assert len(attached) == 6                       # attached again: a session may have stopped before it was

def test_a_scoring_of_other_weights_is_done_again(config_path, small_dataset, tmp_path):
    [run_dir] = go(config_path, small_dataset, tmp_path / "out", SOME[:1])
    path = run_dir / "scores_test_block_0.1.json"
    scores = json.loads(path.read_text())
    scores["model_sha256"], scores["mae"] = "0" * 64, 99.0                  # an earlier run's weights, say
    path.write_text(json.dumps(scores))
    go(config_path, small_dataset, tmp_path / "out", SOME[:1])
    redone = json.loads(path.read_text())
    assert redone["model_sha256"] == sha256(run_dir / "model.pt") and redone["mae"] != 99.0

def stop_ext1(patch, where):
    """Stop the sweep in ext1_random_seed42's training (after its first epoch) or its block scoring."""
    if where == "training":
        def stopping_train(config, *args, on_checkpoint, **kwargs):
            def checkpoint(state):
                on_checkpoint(state)
                if variant_name(config) == "ext1":
                    raise SessionLost
            return train(config, *args, on_checkpoint=checkpoint, **kwargs)
        patch.setattr("diffusion.run.train", stopping_train)
    else:
        def stopping_score(run_dir, dataset, device, holdout, **kwargs):
            if run_dir.name == "ext1_random_seed42" and holdout == "block":
                raise SessionLost
            return score(run_dir, dataset, device, holdout, **kwargs)
        patch.setattr("diffusion.sweep.score", stopping_score)

@pytest.mark.parametrize("where", ["training", "scoring"])
def test_a_sweep_a_lost_session_stopped_ends_as_one_never_stopped(where, config_path, small_dataset, tmp_path,
                                                                    monkeypatch):
    go(config_path, small_dataset, tmp_path / "whole", SOME)
    with monkeypatch.context() as patch:
        stop_ext1(patch, where)
        with pytest.raises(SessionLost):
            go(config_path, small_dataset, tmp_path / "stopped", SOME)
    stopped = tmp_path / "stopped" / "ext1_random_seed42"
    if where == "training":
        assert torch.load(stopped / "checkpoint.pt")["epoch"] == 0 and not (stopped / "summary.json").exists()
    else:
        assert [path.name for path in stopped.glob("scores_*")] == ["scores_test_random_0.1.json"]
    go(config_path, small_dataset, tmp_path / "stopped", SOME)
    config = yaml.safe_load(config_path.read_text())
    assert compare(tmp_path / "whole", tmp_path / "stopped", config, SOME, log=quiet) == {name: [] for name in NAMES}

def test_a_cell_that_unpairs_its_seed_stops_the_sweep_before_it_is_scored(config_path, small_dataset, tmp_path):
    out = tmp_path / "out"
    go(config_path, small_dataset, out, SOME[:1])
    summary = json.loads((out / "base_random_seed42" / "summary.json").read_text())
    summary["pairing"]["data_order"] = "0" * 64                             # as if it had been shuffled otherwise
    (out / "base_random_seed42" / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="data_order is missing or differs across the random regime's variants "
                                           "at seed 42"):
        go(config_path, small_dataset, out, SOME[:2])
    assert (out / "ext1_random_seed42" / "summary.json").exists()
    assert not list((out / "ext1_random_seed42").glob("scores_*"))

def test_a_finished_run_trained_with_another_config_is_refused(config_path, small_dataset, tmp_path):
    go(config_path, small_dataset, tmp_path / "out", SOME[:1])
    with pytest.raises(ValueError, match="another config"):
        go(config_path, small_dataset, tmp_path / "out", SOME[:1], epochs=3)

def test_a_folder_refuses_another_tag_or_commit(config_path, small_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr("diffusion.sweep.current_commit", lambda: "aaa")
    go(config_path, small_dataset, tmp_path / "out", [], tag="dry-run")
    with pytest.raises(ValueError, match="tagged 'dry-run', not 'sweep'"):
        go(config_path, small_dataset, tmp_path / "out", [], tag="sweep")
    monkeypatch.setattr("diffusion.sweep.current_commit", lambda: "bbb")
    with pytest.raises(ValueError, match="started from commit aaa"):
        go(config_path, small_dataset, tmp_path / "out", [], tag="dry-run")

def test_a_paired_sweep_runs_only_on_the_picked_gpu(config_path, small_dataset, tmp_path, monkeypatch):
    config = yaml.safe_load(config_path.read_text())
    config["protocol"]["paired_gpu"] = "Tesla T4"
    config_path.write_text(yaml.safe_dump(config))
    paired = dict(paired=True, wandb_mode="disabled", config_path=config_path, log=quiet)
    with pytest.raises(RuntimeError, match="not on cpu"):
        sweep(config, small_dataset, tmp_path / "out", "test", "cpu", SOME[:1], ["random"], **paired)
    assert not (tmp_path / "out").exists()                                 # refused before anything is written
    monkeypatch.setattr("diffusion.sweep.uncommitted", lambda: [])         # as a fresh clone, whatever this one holds
    for module in ("diffusion.run", "diffusion.score"):
        monkeypatch.setattr(f"{module}.gpu_name", lambda device: "Tesla T4")        # as on a Colab T4
    scored_paired = []

    def spied_score(*args, **kwargs):
        scored_paired.append(kwargs["paired"])
        return score(*args, **kwargs)

    monkeypatch.setattr("diffusion.sweep.score", spied_score)
    [run_dir] = sweep(config, small_dataset, tmp_path / "out", "test", "cpu", SOME[:1], ["random"], **paired)
    assert json.loads((run_dir / "summary.json").read_text())["gpu"] == "Tesla T4"
    assert json.loads((run_dir / "scores_test_random_0.1.json").read_text())["gpu"] == "Tesla T4"
    assert scored_paired == [True]

def test_a_paired_sweep_refuses_code_its_commit_does_not_hold(config_path, small_dataset, tmp_path, monkeypatch):
    config = yaml.safe_load(config_path.read_text())
    config["protocol"]["paired_gpu"] = "Tesla T4"
    monkeypatch.setattr("diffusion.run.gpu_name", lambda device: "Tesla T4")
    monkeypatch.setattr("diffusion.sweep.uncommitted", lambda: ["diffusion/train.py"])
    with pytest.raises(RuntimeError, match=r"changes its commit .* does not hold: \['diffusion/train.py'\]"):
        sweep(config, small_dataset, tmp_path / "out", "test", "cpu", SOME[:1], ["random"], paired=True,
              wandb_mode="disabled", config_path=config_path, log=quiet)
    assert not (tmp_path / "out").exists()                                 # refused before anything is written
    [run_dir] = go(config_path, small_dataset, tmp_path / "out", SOME[:1])  # unpaired, a check of the driver: runs
    assert (run_dir / "summary.json").exists()

def test_uncommitted_names_the_changed_tracked_files_alone(tmp_path):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    git("init")
    (tmp_path / "train.py").write_text("a\n")
    (tmp_path / "score.py").write_text("a\n")
    git("add", "train.py", "score.py")
    git("commit", "-m", "first")
    assert uncommitted(tmp_path) == []
    (tmp_path / "train.py").write_text("b\n")                              # changed
    (tmp_path / "notes.txt").write_text("not tracked\n")                   # untracked: not counted
    assert uncommitted(tmp_path) == ["train.py"]
    (tmp_path / "score.py").unlink()                                       # deleted
    assert sorted(uncommitted(tmp_path)) == ["score.py", "train.py"]
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".git").write_text("not a repository\n")                    # git fails here: nothing to report
    assert uncommitted(outside) == []

def test_a_scoring_without_its_median_beside_it_is_done_again(config_path, small_dataset, tmp_path):
    [run_dir] = go(config_path, small_dataset, tmp_path / "out", SOME[:1])
    (run_dir / "median_test_block_0.1.npy").unlink()                       # Drive kept the scores, not the median
    before = (run_dir / "scores_test_random_0.1.json").read_bytes()
    go(config_path, small_dataset, tmp_path / "out", SOME[:1])
    assert (run_dir / "median_test_block_0.1.npy").exists()
    assert (run_dir / "scores_test_random_0.1.json").read_bytes() == before   # the other holdout's kept

# --- the rerun check --------------------------------------------------------------------

def test_reproduces_names_what_differs(config_path, small_dataset, tmp_path):
    go(config_path, small_dataset, tmp_path / "first", SOME[:1])
    go(config_path, small_dataset, tmp_path / "rerun", SOME[:1], tag="rerun")
    first, rerun = tmp_path / "first" / NAMES[0], tmp_path / "rerun" / NAMES[0]
    assert reproduces(first, rerun) == []
    curve = json.loads((rerun / "curve.json").read_text())
    curve[0]["seconds"] += 5.0                                              # a slower session: not compared
    (rerun / "curve.json").write_text(json.dumps(curve))
    assert reproduces(first, rerun) == []
    curve[-1]["val_mae"] += 1e-9
    (rerun / "curve.json").write_text(json.dumps(curve))
    summary = json.loads((rerun / "summary.json").read_text())
    summary["wandb_run"] = "entity/project/another"                         # each run has its own: not compared
    summary["final_val_mae"] += 1e-9
    (rerun / "summary.json").write_text(json.dumps(summary))
    scores = json.loads((rerun / "scores_test_block_0.1.json").read_text())
    scores["mae"] += 1e-9
    (rerun / "scores_test_block_0.1.json").write_text(json.dumps(scores))
    lines = (rerun / "history.jsonl").read_text().splitlines()
    record = json.loads(lines[0])
    record["loss"] += 1e-9
    (rerun / "history.jsonl").write_text("\n".join([json.dumps(record), *lines[1:]]) + "\n")
    assert reproduces(first, rerun) == ["history.jsonl", "curve.json", "summary.json final_val_mae",
                                        "scores_test_block_0.1.json mae"]

def test_the_command_line_runs_cells_and_compares_two_folders(config_path, small_dataset, tmp_path):
    save_dataset(small_dataset, tmp_path / "data.npz")
    common = ["--config", str(config_path), "--dataset", str(tmp_path / "data.npz"), "--device", "cpu", "--unpaired",
              "--wandb", "disabled", "--regime", "random", "--variants", "base", "--seeds", "42", "--holdouts", "random"]
    main([*common, "--tag", "dry-run", "--epochs", "1", "--out", str(tmp_path / "first")])
    main([*common, "--tag", "rerun", "--epochs", "1", "--out", str(tmp_path / "rerun")])
    run_dir = tmp_path / "first" / "base_random_seed42"
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["train"]["epochs"] == 1
    assert sorted(path.name for path in run_dir.glob("scores_*")) == ["scores_test_random_0.1.json"]
    assert main([*common, "--out", str(tmp_path / "first"), "--compare", str(tmp_path / "rerun")]) == \
        {"base_random_seed42": []}
    (tmp_path / "rerun" / "base_random_seed42" / "history.jsonl").write_text("")
    with pytest.raises(SystemExit):
        main([*common, "--out", str(tmp_path / "first"), "--compare", str(tmp_path / "rerun")])
    with pytest.raises(SystemExit):
        main([*common, "--out", str(tmp_path / "untagged")])                # the store collects runs by tag
    assert not (tmp_path / "untagged").exists()
