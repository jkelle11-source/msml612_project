"""The paired sweep's results store (M6-2): layout, every refusal, and real runs through it."""
import copy
import json

import pytest
from conftest import CONFIG
from diffusion import results
from diffusion.results import main, problems, record
from diffusion.run import VARIANTS, run, with_variant

T4 = "Tesla T4"

def config_of(variant, seed, regime="random"):
    """A run's wandb config, as ``diffusion.run`` sends it, on a T4 picked by the timing test."""
    config = with_variant({k: v for k, v in copy.deepcopy(CONFIG).items() if k != "tracking"}, variant)
    config["protocol"]["paired_gpu"] = T4
    return {**config, "seed": seed, "variant": variant, "regime": regime, "device": "cuda", "gpu": T4,
            "out_dir": f"/content/msml612_project/outputs/{variant}_{regime}_seed{seed}"}

def pairing_of(variant, seed):
    """Pairing hashes that pair a seed's variants: each one names only its seed."""
    hashes = {"init_shared": f"shared{seed}", "data_order": f"order{seed}", "train_mask": f"masks{seed}"}
    if VARIANTS[variant]["use_mask_embedding"]:
        hashes["init_mask_encoder"] = f"mask_encoder{seed}"
    if VARIANTS[variant]["use_time_encoding"]:
        hashes["init_time_encoder"] = f"time_encoder{seed}"
    return hashes

def fake_run(variant, seed, regime="random", scored=("random/0.1", "block/0.1")):
    """A finished run as ``results.fetch`` gives it, scored under ``scored``."""
    summary = {"variant": variant, "seed": seed, "gpu": T4, "pairing": pairing_of(variant, seed)}
    for i, holdout in enumerate(scored):
        summary |= {f"{holdout}/test/mae": 0.2 + seed / 1000 + i / 100, f"{holdout}/test/rmse": 0.5,
                    f"{holdout}/test/gpu": T4}
    return {"path": f"DATA612/msml612-csdi/{variant}-{regime}-{seed}", "name": f"{variant}_{regime}_seed{seed}",
            "config": config_of(variant, seed, regime), "summary": summary,
            "metadata": {"git": {"commit": "abc1234"}}}

def dry_run():
    """The M6-2 dry run: four variants at two seeds, and base and Ext 1 block-aware at both."""
    return ([fake_run(variant, seed) for variant in VARIANTS for seed in (42, 43)]
            + [fake_run(variant, seed, "block-aware") for variant in ("base", "ext1") for seed in (42, 43)])

def find(runs, variant, seed, regime="random"):
    return next(r for r in runs if (r["config"]["variant"], r["config"]["seed"], r["config"]["regime"])
                == (variant, seed, regime))

# --- the store --------------------------------------------------------------------------

def test_the_store_holds_each_runs_scores_by_regime_variant_and_seed():
    store = record(dry_run(), "dry-run")
    assert list(store["runs"]) == ["block-aware", "random"]
    assert list(store["runs"]["random"]) == list(VARIANTS) and list(store["runs"]["block-aware"]) == ["base", "ext1"]
    assert list(store["runs"]["random"]["ext1+2"]) == ["42", "43"]
    entry = store["runs"]["random"]["ext1+2"]["43"]
    assert entry == {"run": "ext1+2_random_seed43", "wandb_run": "DATA612/msml612-csdi/ext1+2-random-43",
                     "commit": "abc1234", "gpu": T4,
                     "scores": {"block/0.1": {"mae": pytest.approx(0.253), "rmse": 0.5, "gpu": T4},
                                "random/0.1": {"mae": pytest.approx(0.243), "rmse": 0.5, "gpu": T4}}}
    assert store["gpu"] == T4 and store["unscored"] == [] and store["tag"] == "dry-run"

def test_each_regimes_config_is_kept_without_the_toggles_and_labels():
    store = record(dry_run(), "dry-run")
    config = store["configs"]["random"]
    assert config["train"] == CONFIG["train"] and config["protocol"]["paired_gpu"] == T4
    assert not set(config["model"]) & set(VARIANTS["base"])         # the toggles: each run's variant says
    assert not set(config) & {"seed", "variant", "regime", "device", "gpu", "out_dir", "validation"}

def test_a_holdout_not_scored_yet_is_null_and_listed():
    runs = dry_run()
    runs[runs.index(find(runs, "ext2", 43))] = fake_run("ext2", 43, scored=("random/0.1",))
    store = record(runs, "sweep")
    assert store["runs"]["random"]["ext2"]["43"]["scores"]["block/0.1"] is None
    assert store["unscored"] == ["ext2_random_seed43: block/0.1"]

def test_bases_robustness_rates_are_kept_beside_the_headline():
    runs = dry_run()
    runs[runs.index(find(runs, "base", 42))] = fake_run("base", 42, scored=("random/0.1", "block/0.1",
                                                                            "random/0.5", "random/0.9"))
    scores = record(runs, "sweep")["runs"]["random"]["base"]["42"]["scores"]
    assert list(scores) == ["block/0.1", "random/0.1", "random/0.5", "random/0.9"]

# --- what unpairs runs, and what keeps one out ------------------------------------------

def changed(variant, seed, change, regime="random"):
    """``dry_run()`` with ``change(run)`` applied to one of its runs."""
    def apply():
        runs = dry_run()
        change(find(runs, variant, seed, regime))
        return runs
    return apply

def set_pairing(key, value):
    return lambda run: run["summary"]["pairing"].update({key: value})

def another_gpu_for_the_block_aware_regime():
    runs = dry_run()
    for run in runs:
        if run["config"]["regime"] == "block-aware":
            run["config"]["protocol"]["paired_gpu"] = run["summary"]["gpu"] = "NVIDIA L4"
            run["summary"] |= {key: "NVIDIA L4" for key in run["summary"] if key.endswith("/gpu")}
    return runs

def a_rerun_kept_beside_the_run():
    runs = dry_run()
    return runs + [fake_run("ext1", 42) | {"path": "DATA612/msml612-csdi/rerun"}]

UNPAIRED = {
    "batches in another order": (changed("ext1", 42, set_pairing("data_order", "other")), "data_order"),
    "other train masks": (changed("ext2", 43, set_pairing("train_mask", "other")), "train_mask"),
    "shared parts start elsewhere": (changed("ext1+2", 42, set_pairing("init_shared", "other")), "init_shared"),
    "Ext 1's encoder starts elsewhere in ext1+2": (changed("ext1+2", 42, set_pairing("init_mask_encoder", "other")),
                                                    "mask_encoder starts from different weights"),
    "Ext 2's encoder starts elsewhere in ext1+2": (changed("ext1+2", 43, set_pairing("init_time_encoder", "other")),
                                                    "time_encoder starts from different weights"),
    "a block-aware run unpaired": (changed("ext1", 43, set_pairing("data_order", "other"), "block-aware"),
                                   "block-aware regime's variants at seed 43"),
    "a run from before M6-2": (changed("base", 42, lambda run: run["summary"].pop("pairing")), "missing or differs"),
    "another learning rate": (changed("ext2", 42, lambda run: run["config"]["train"].update(lr=1e-4)),
                              "differ in more than their toggles: ['train.lr']"),
    "trained on another GPU": (changed("ext1", 43, lambda run: run["summary"].update(gpu="NVIDIA A100-SXM4-40GB")),
                               "ext1_random_seed43 trained on NVIDIA A100-SXM4-40GB, not a Tesla T4"),
    "scored on another GPU": (changed("base", 43, lambda run: run["summary"].update({"block/0.1/test/gpu": None})),
                              "base_random_seed43's block/0.1 was scored on None, not a Tesla T4"),
    "no GPU picked": (changed("base", 42, lambda run: run["config"]["protocol"].update(paired_gpu=None)),
                      "paired_gpu is not set"),
    "two GPUs picked": (another_gpu_for_the_block_aware_regime, "runs set to different GPUs"),
    "two runs in one cell": (a_rerun_kept_beside_the_run, "two runs of ext1 in the random regime at seed 42"),
}

@pytest.mark.parametrize("make, refusal", UNPAIRED.values(), ids=UNPAIRED.keys())
def test_runs_that_are_not_paired_are_refused(make, refusal):
    with pytest.raises(ValueError, match=refusal.replace("[", r"\[").replace("]", r"\]")):
        record(make(), "dry-run")
    assert any(refusal in problem for problem in problems(make()))

def test_only_whats_wrong_is_listed():
    assert problems(dry_run()) == []
    assert len(problems(changed("ext1", 42, set_pairing("data_order", "other"))())) == 1

def test_what_does_not_touch_training_may_differ():
    runs = dry_run()
    find(runs, "ext1", 42)["config"]["validation"]["every_n_epochs"] = 25     # tracking only
    find(runs, "base", 43)["config"]["out_dir"] = "/content/drive/elsewhere"
    record(runs, "dry-run")

def test_a_pairing_block_wandb_flattened_is_read_the_same():
    runs = dry_run()
    run = find(runs, "ext1", 42)
    run["summary"] |= {f"pairing.{key}": value for key, value in run["summary"].pop("pairing").items()}
    assert record(runs, "dry-run") == record(dry_run(), "dry-run")
    run["summary"]["pairing.data_order"] = "other"
    with pytest.raises(ValueError, match="data_order"):
        record(runs, "dry-run")

def test_no_runs_is_refused():
    with pytest.raises(ValueError, match="no finished run is tagged 'sweep'"):
        record([], "sweep")

# --- real runs --------------------------------------------------------------------------

class Tracker:
    """Stands in for a wandb run; keeps what ``results.fetch`` would read back."""
    def __init__(self, **kwargs):
        self.init, self.summary, self.disabled = kwargs, {}, False
        self.path = f"{kwargs['entity']}/{kwargs['project']}/{kwargs['name']}"

    def log(self, data, step):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

def test_real_runs_are_paired_and_a_mislabeled_one_is_caught(small_dataset, tmp_path, monkeypatch):
    trackers = []
    monkeypatch.setattr("diffusion.run.wandb.init", lambda **kwargs: trackers.append(Tracker(**kwargs)) or trackers[-1])
    monkeypatch.setattr("diffusion.run.gpu_name", lambda device: T4)          # as on a Colab T4
    config = copy.deepcopy(CONFIG)
    config["diffusion"].update(channels=16, layers=1, num_steps=10)
    config["train"].update(epochs=2, batch_size=8)
    config["validation"].update(every_n_epochs=2, n_samples=2, batch_size=8)
    config["protocol"]["paired_gpu"] = T4
    for variant, seed in (("base", 42), ("ext1", 42), ("ext1", 43)):
        run(with_variant(config, variant), small_dataset, seed, "cpu", tmp_path / f"{variant}_{seed}",
            log=lambda line: None, wandb_mode="online", paired=True)
    scored = {f"{holdout}/test/{field}": value for holdout in ("random/0.1", "block/0.1")
              for field, value in (("mae", 0.3), ("rmse", 0.6), ("gpu", T4))}
    runs = [{"path": t.path, "name": t.init["name"], "config": json.loads(json.dumps(t.init["config"])),
             "summary": json.loads(json.dumps(t.summary)) | scored, "metadata": {}} for t in trackers]
    store = record(runs, "dry-run")
    assert set(store["runs"]["random"]) == {"base", "ext1"} and store["gpu"] == T4

    mislabeled = runs[2]                                    # ext1 at seed 43, recorded as seed 42's
    mislabeled["config"]["seed"] = 42
    with pytest.raises(ValueError, match="data_order is missing or differs.*seed 42"):
        record([runs[0], mislabeled], "dry-run")

# --- the command line -------------------------------------------------------------------

def test_the_command_line_writes_the_store_named_for_the_tag(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(results, "RESULTS", tmp_path)
    asked = []
    store = main(["--tag", "dry-run"], fetch=lambda tag: asked.append(tag) or dry_run())
    assert asked == ["dry-run"]
    assert json.loads((tmp_path / "dry-run.json").read_text()) == json.loads(json.dumps(store))
    assert "12 paired runs on Tesla T4, 0 holdout(s) not scored yet" in capsys.readouterr().out
