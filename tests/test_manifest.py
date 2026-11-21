"""The run manifest (M7-1): an entry's fields worked out by hand, every refusal, and the real sweep if it is here."""
import copy
import hashlib
import json

import pytest
import torch
import yaml
from conftest import CONFIG, ROOT
from diffusion.manifest import entry, main, manifest
from diffusion.run import with_variant

GPU = "Some GPU"
PROTOCOL = {"paired_gpu": GPU}
COMMIT = "abc1234"
GATES = {"base": {}, "ext1": {"mask_encoder.gate": 0.25}, "ext2": {"time_encoder.gate": -0.5},
         "ext1+2": {"mask_encoder.gate": 0.25, "time_encoder.gate": -0.5}}
SCORING_SECONDS = {"random/0.1": 7.0, "block/0.1": 8.0, "random/0.5": 9.0}

def make_run(folder, variant="ext1", regime="random", seed=42, gates=None, epochs=(0, 1, 2), gpu=GPU,
             scoring_gpu=GPU, summary_sha=None, scoring_sha=None, holdouts=("random/0.1", "block/0.1")):
    """A finished 3-epoch run's folder: epochs take 1, 2 and 3 s, the last validates in 10 s."""
    run_dir = folder / f"{variant}_{regime}_seed{seed}"
    run_dir.mkdir(parents=True)
    config = with_variant(copy.deepcopy(CONFIG), variant)
    config["train"]["regime"], config["train"]["epochs"] = regime, 3
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    gates = GATES[variant] if gates is None else gates
    torch.save({"w": torch.zeros(2)} | {key: torch.tensor(value) for key, value in gates.items()}, run_dir / "model.pt")
    sha = hashlib.sha256((run_dir / "model.pt").read_bytes()).hexdigest()
    (run_dir / "summary.json").write_text(json.dumps(
        {"variant": variant, "seed": seed, "gpu": gpu, "model_sha256": summary_sha or sha, "wandb_run": "e/p/run1"}))
    rows = [{"epoch": epoch, "seconds": float(i + 1)} for i, epoch in enumerate(epochs)]
    rows[-1]["val_seconds"] = 10.0
    (run_dir / "curve.json").write_text(json.dumps(rows))
    for name in holdouts:
        holdout, fraction = name.split("/")
        (run_dir / f"scores_test_{holdout}_{fraction}.json").write_text(json.dumps(
            {"split": "test", "holdout": holdout, "fraction": float(fraction), "gpu": scoring_gpu,
             "seconds": SCORING_SECONDS[name], "model_sha256": scoring_sha or sha}))
    return run_dir

def test_an_entry_worked_out_by_hand(tmp_path):
    run_dir = make_run(tmp_path, holdouts=("random/0.1", "block/0.1", "random/0.5"))
    found = entry(run_dir, "random", "ext1", 42, PROTOCOL, COMMIT)
    assert found["run"] == "ext1_random_seed42" and found["wandb_run"] == "e/p/run1"
    assert found["seconds"] == {"train": 6.0, "validation": 10.0,
                                "scoring": {"random/0.1": 7.0, "block/0.1": 8.0, "random/0.5": 9.0}}
    assert found["gpu"] == {"train": GPU, "scoring": {"random/0.1": GPU, "block/0.1": GPU, "random/0.5": GPU}}
    assert found["gates"] == {"mask_encoder_gate": 0.25}
    assert found["weights"] == {"path": str((run_dir / "model.pt").resolve()),      # outside the repo: as it is
                                "sha256": hashlib.sha256((run_dir / "model.pt").read_bytes()).hexdigest()}
    assert found["rerun"] == ("git checkout abc1234 && uv run python -m diffusion.sweep --tag rerun "
                              "--out <another folder> --regime random --variants ext1 --seeds 42")
    assert found["config"]["train"]["regime"] == "random" and found["config"]["model"]["use_mask_embedding"]

@pytest.mark.parametrize("variant, gates", [("base", {}), ("ext2", {"time_encoder_gate": -0.5}),
                                            ("ext1+2", {"mask_encoder_gate": 0.25, "time_encoder_gate": -0.5})])
def test_each_variant_has_its_own_gates(tmp_path, variant, gates):
    assert entry(make_run(tmp_path, variant), "random", variant, 42, PROTOCOL, COMMIT)["gates"] == gates

def test_a_path_in_the_repo_is_written_relative_to_it():
    from diffusion.manifest import shown
    assert shown(ROOT / "outputs" / "sweep" / "base_random_seed42" / "model.pt") == "outputs/sweep/base_random_seed42/model.pt"

@pytest.mark.parametrize("built, cell, refusal", [
    ({"epochs": (0, 1, 1)}, {}, "epochs are not 0-2, each once"),
    ({"epochs": (0, 1)}, {}, "epochs are not 0-2, each once"),
    ({"gpu": "Other GPU"}, {}, "training on Other GPU"),
    ({"scoring_gpu": "Other GPU"}, {}, "random/0.1 scoring on Other GPU"),
    ({"summary_sha": "0" * 64}, {}, "not the final weights summary.json records"),
    ({"scoring_sha": "0" * 64}, {}, "scored other weights than model.pt"),
    ({"gates": {}}, {}, "no mask_encoder.gate, though the run trained with use_mask_embedding"),
    ({"variant": "base", "gates": {"mask_encoder.gate": 0.1}}, {"variant": "base"},
     "has mask_encoder.gate, though the run trained without use_mask_embedding"),
    ({"holdouts": ("random/0.1",)}, {}, "no block/0.1 scoring"),
    ({}, {"variant": "ext2"}, "toggles .* are not ext2's"),
    ({}, {"regime": "block-aware"}, "regime is 'random', not 'block-aware'"),
    ({}, {"seed": 43}, "summary.json is ext1 seed 42, not ext1 seed 43"),
])
def test_refusals(tmp_path, built, cell, refusal):
    run_dir = make_run(tmp_path, **built)
    cell = {"regime": "random", "variant": "ext1", "seed": 42} | cell
    with pytest.raises(ValueError, match=refusal):
        entry(run_dir, cell["regime"], cell["variant"], cell["seed"], PROTOCOL, COMMIT)

def test_an_unfinished_run_is_refused(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "summary.json").unlink()
    with pytest.raises(ValueError, match="no summary.json .*has not finished"):
        entry(run_dir, "random", "ext1", 42, PROTOCOL, COMMIT)

def sweep_config():
    """Two cells a seed (base and Ext 1, random regime) at seeds 42 and 43, on ``GPU``."""
    config = copy.deepcopy(CONFIG)
    config["sweep"]["regimes"] = {"random": ["base", "ext1"]}
    config["protocol"]["run_seeds"], config["protocol"]["paired_gpu"] = [42, 43], GPU
    return config

def sweep_folder(tmp_path, tag="t", skip=()):
    folder = tmp_path / "sweep"
    folder.mkdir()
    (folder / "sweep.json").write_text(json.dumps({"tag": tag, "commit": COMMIT}))
    for variant in ("base", "ext1"):
        for seed in (42, 43):
            if (variant, seed) not in skip:
                make_run(folder, variant, seed=seed)
    return folder

def test_the_manifest_is_keyed_as_the_store_is(tmp_path):
    found = manifest(sweep_folder(tmp_path), "t", sweep_config())
    assert (found["tag"], found["commit"], found["gpu"]) == ("t", COMMIT, GPU)
    assert {v: sorted(seeds) for v, seeds in found["runs"]["random"].items()} == {"base": ["42", "43"], "ext1": ["42", "43"]}
    assert found["runs"]["random"]["ext1"]["43"]["rerun"].endswith("--regime random --variants ext1 --seeds 43")
    assert found["runs"]["random"]["base"]["42"]["gates"] == {}

def test_every_problem_is_listed_and_nothing_is_written(tmp_path):
    folder = sweep_folder(tmp_path, skip={("base", 43)})
    (folder / "ext1_random_seed42" / "summary.json").unlink()
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match=r"2 run\(s\) refused") as error:
        main(["--tag", "t", "--out", str(folder), "--manifest", str(out), "--config", str(write_config(tmp_path))])
    assert "ext1_random_seed42: no summary.json" in str(error.value) and "base_random_seed43: no folder" in str(error.value)
    assert not out.exists()

def test_a_folder_started_under_another_tag_is_refused(tmp_path):
    with pytest.raises(ValueError, match="started with tag 'other', not 't'"):
        manifest(sweep_folder(tmp_path, tag="other"), "t", sweep_config())

def write_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(sweep_config()))
    return path

def test_main_writes_the_manifest(tmp_path, capsys):
    out = tmp_path / "manifest.json"
    found = main(["--tag", "t", "--out", str(sweep_folder(tmp_path)), "--manifest", str(out),
                  "--config", str(write_config(tmp_path))])
    assert json.loads(out.read_text()) == found
    # 4 runs of 6 s training + 10 s validation + 7 s + 8 s scoring = 124 s = 0.0344 h
    assert f"4 runs at commit {COMMIT}, 0.0 GPU hours on {GPU}" in capsys.readouterr().out

REAL = ROOT / "outputs" / "sweep" / "ext1+2_random_seed43"

@pytest.mark.skipif(not (REAL / "summary.json").exists(), reason="the sweep's folder is not here (outputs/sweep)")
def test_a_real_sweep_run():
    """Expected values read independently from the folder (curve.json sums, scores files, torch.load)."""
    found = entry(REAL, "random", "ext1+2", 43, CONFIG["protocol"], "20a470d30a393f02e199e1d37e5ba6a216d16279")
    assert found["seconds"] == {"train": 2120.1, "validation": 847.1, "scoring": {"block/0.1": 2324.5, "random/0.1": 2314.0}}
    assert found["gates"] == pytest.approx({"mask_encoder_gate": -0.16509275138378143,
                                            "time_encoder_gate": -0.03555808216333389}, abs=1e-7)
    assert found["weights"] == {"path": "outputs/sweep/ext1+2_random_seed43/model.pt",
                                "sha256": "d0aee4002adb74173eef7db0ea5b9f4893fd563c9787eb72f9f2cdfb747177ac"}
    assert found["gpu"] == {"train": "NVIDIA A100-SXM4-80GB",
                            "scoring": {"block/0.1": "NVIDIA A100-SXM4-80GB", "random/0.1": "NVIDIA A100-SXM4-80GB"}}
