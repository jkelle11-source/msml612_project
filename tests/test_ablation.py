"""M8-1's ablation driver: which cells each study trains, from which config, and where."""
import copy
import json

import numpy as np
import pytest
import torch
import yaml
from conftest import CONFIG
from data.physionet import save_dataset
from diffusion.ablation import (SEEDS, STUDIES, VALIDATE_EVERY, ablate, main, study_cells, study_config,
                                trained_values)
from diffusion.gate import flatten
from diffusion.run import weight_hashes, with_variant
from models.build import build_model
from test_sweep import quick_config

def quiet(line):
    pass

@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(quick_config()))
    return path

# --- which cells ------------------------------------------------------------------------

def test_each_study_trains_its_two_values_besides_the_sweeps():
    assert {study: trained_values(CONFIG, study) for study in STUDIES} == {
        "d_model": [128, 256], "steps": [100, 200], "encoder_layers": [1, 3]}
    assert list(STUDIES["d_model"]["values"]) == CONFIG["model"]["d_model_sweep"]          # §3.6's values

def test_each_value_trains_the_plans_cells_and_scores_the_plans_holdouts():
    # Ext 1+2, the best variant (§3.5), random-trained, at seeds 42-44; for the encoder's depth also Ext 1,
    # trained block-aware, its primary regime (§3.1, §4.1), each scored on both holdouts (M8-1).
    best = [("random", "ext1+2", seed) for seed in (42, 43, 44)]
    assert SEEDS == (42, 43, 44)
    assert {study: study_cells(CONFIG, study) for study in STUDIES} == {
        "d_model": best, "steps": best,
        "encoder_layers": best + [("block-aware", "ext1", seed) for seed in (42, 43, 44)]}
    assert {study: plan["holdouts"] for study, plan in STUDIES.items()} == {
        "d_model": ("random",), "steps": ("random",), "encoder_layers": ("random", "block")}

def test_a_studys_config_moves_its_one_setting_and_how_often_it_validates():
    for study, plan in STUDIES.items():
        for value in trained_values(CONFIG, study):
            moved, sweeps = flatten(study_config(CONFIG, study, value)), flatten(CONFIG)
            assert moved.keys() == sweeps.keys()
            assert {key for key in sweeps if moved[key] != sweeps[key]} == {".".join(plan["setting"]),
                                                                             "validation.every_n_epochs"}
            assert moved[".".join(plan["setting"])] == value and moved["validation.every_n_epochs"] == VALIDATE_EVERY
            assert moved["model.n_heads"] == 8                     # fixed across the d_model study (§3.6)

def test_a_config_whose_setting_is_no_value_of_the_study_is_refused():
    config = copy.deepcopy(CONFIG)
    config["diffusion"]["channels"] = 32
    with pytest.raises(ValueError, match="diffusion.channels is 32, not one of the d_model study's"):
        trained_values(config, "d_model")

def starting_weights(config, variant="ext1+2"):
    torch.manual_seed(0)                    # as RunStreams seeds model_init before the build
    return weight_hashes(build_model(with_variant(config, variant), "cpu").state_dict())

def test_the_steps_and_depth_studies_start_from_the_sweeps_shared_weights():
    # What pairs a value's runs with the sweep's at a seed besides the streams: T leaves every weight
    # as it was, and the encoder's depth all but that encoder's own.
    sweeps = starting_weights(CONFIG)
    for value in trained_values(CONFIG, "steps"):
        assert starting_weights(study_config(CONFIG, "steps", value)) == sweeps
    for value in trained_values(CONFIG, "encoder_layers"):
        deeper = starting_weights(study_config(CONFIG, "encoder_layers", value))
        assert (deeper["shared"], deeper["time_encoder"]) == (sweeps["shared"], sweeps["time_encoder"])
        assert deeper["mask_encoder"] != sweeps["mask_encoder"]

def test_the_steps_study_keeps_upstreams_noise_schedule():
    # T moves diffusion.num_steps alone: beta stays quad from 1e-4 to 0.5, so a longer chain spends more of its
    # steps where the data's signal, sqrt(alpha_bar_t), is under 0.05: 8 of 50, 32 of 100, 91 of 200 (README,
    # The ablation; PROJECT_PLAN §8, Limitations). Counted from the model's own schedule.
    for steps, near_pure_noise in ((50, 8), (100, 32), (200, 91)):
        config = study_config(CONFIG, "steps", steps)
        assert (config["diffusion"]["schedule"], config["diffusion"]["beta_start"], config["diffusion"]["beta_end"]) == (
            "quad", 1.0e-4, 0.5)
        model = build_model(with_variant(config, "ext1+2"), "cpu")
        assert len(model.beta) == steps and model.beta[0] == pytest.approx(1.0e-4) and model.beta[-1] == pytest.approx(0.5)
        assert int((model.alpha ** 0.5 < 0.05).sum()) == near_pure_noise

# --- training them ----------------------------------------------------------------------

class FakeRun:
    """What ``diffusion.run`` uses of a wandb run, so a test can see what each run was started with."""

    def __init__(self, kwargs):
        self.init, self.summary, self.disabled, self.path = kwargs, {}, True, None

    def log(self, data, step):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *rest):
        pass

def test_a_study_trains_each_value_into_a_folder_of_its_own(config_path, small_dataset, tmp_path, monkeypatch):
    inits = []
    monkeypatch.setattr("diffusion.run.wandb.init", lambda **kwargs: inits.append(kwargs) or FakeRun(kwargs))
    out = tmp_path / "ablation"
    run_dirs = ablate(yaml.safe_load(config_path.read_text()), small_dataset, out, "encoder_layers", "test", "cpu",
                      seeds=[42], paired=False, wandb_mode="disabled", config_path=config_path, log=quiet)
    assert [run_dir.relative_to(out).as_posix() for run_dir in run_dirs] == [
        "encoder_layers_1/ext1+2_random_seed42", "encoder_layers_1/ext1_block-aware_seed42",
        "encoder_layers_3/ext1+2_random_seed42", "encoder_layers_3/ext1_block-aware_seed42"]
    regimes = {"ext1+2": ("random", "random"), "ext1": ("block-aware", "block")}     # trained in, validated on
    for run_dir in run_dirs:
        depth = int(run_dir.parent.name.rsplit("_", 1)[1])
        saved = yaml.safe_load((run_dir / "config.yaml").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        assert (saved["train"]["regime"], summary["validation"]["holdout"]) == regimes[summary["variant"]]
        assert saved["model"]["mask_encoder_layers"] == depth
        assert saved["validation"]["every_n_epochs"] == VALIDATE_EVERY
        layers = {key.split(".")[2] for key in torch.load(run_dir / "model.pt") if key.startswith("mask_encoder.temporal.")}
        assert layers == {str(i) for i in range(depth)}
        assert sorted(path.name for path in run_dir.glob("scores_test_*.json")) == [
            "scores_test_block_0.1.json", "scores_test_random_0.1.json"]
    for depth in (1, 3):
        assert json.loads((out / f"encoder_layers_{depth}" / "sweep.json").read_text())["tag"] == "test"
    # On wandb every value's run of a cell has the cell's name, so its tags say which study and value it is
    assert [(init["name"], init["tags"][3:]) for init in inits] == [
        ("ext1+2_random_seed42", ["test", "encoder_layers_1"]), ("ext1_block-aware_seed42", ["test", "encoder_layers_1"]),
        ("ext1+2_random_seed42", ["test", "encoder_layers_3"]), ("ext1_block-aware_seed42", ["test", "encoder_layers_3"])]

def test_the_command_line_passes_its_config_dataset_and_folder_on(small_dataset, tmp_path, monkeypatch):
    config = copy.deepcopy(CONFIG)
    config["protocol"]["eval_n_samples"] = 2                            # not the repo's: --config is read
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    save_dataset(small_dataset, tmp_path / "data.npz")
    calls = []
    monkeypatch.setattr("diffusion.ablation.ablate", lambda *args, **kwargs: calls.append((args, kwargs)))
    main(["--study", "d_model", "--tag", "ablation", "--out", str(tmp_path / "ablation"), "--device", "cpu",
          "--config", str(tmp_path / "config.yaml"), "--dataset", str(tmp_path / "data.npz"), "--wandb", "disabled"])
    [(args, kwargs)] = calls
    assert args[0] == config and args[2:] == (tmp_path / "ablation", "d_model", "ablation", "cpu")
    assert np.array_equal(args[1]["record_ids"], small_dataset["record_ids"])         # --dataset's patients
    assert kwargs["config_path"] == tmp_path / "config.yaml" and kwargs["wandb_mode"] == "disabled"
    assert kwargs.get("paired", True) and kwargs.get("seeds", SEEDS) == SEEDS     # ablate's defaults: paired, 42-44

def test_the_command_line_trains_nothing_off_the_paired_gpu(small_dataset, tmp_path):
    save_dataset(small_dataset, tmp_path / "data.npz")
    with pytest.raises(RuntimeError, match="a paired run goes on a NVIDIA A100-SXM4-80GB, not on cpu"):
        main(["--study", "steps", "--tag", "ablation", "--out", str(tmp_path / "ablation"), "--device", "cpu",
              "--dataset", str(tmp_path / "data.npz"), "--wandb", "disabled"])
    assert not (tmp_path / "ablation").exists()
