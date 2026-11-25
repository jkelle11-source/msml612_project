"""M8-1's ablation driver: which cells each study trains, from which config, and where."""
import copy
import json

import pytest
import torch
import yaml
from conftest import CONFIG
from data.physionet import save_dataset
from diffusion.ablation import STUDIES, VALIDATE_EVERY, ablate, main, study_config, trained_values
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

# --- training them ----------------------------------------------------------------------

def test_a_study_trains_each_value_into_a_folder_of_its_own(config_path, small_dataset, tmp_path):
    out = tmp_path / "ablation"
    run_dirs = ablate(yaml.safe_load(config_path.read_text()), small_dataset, out, "encoder_layers", "test", "cpu",
                      seeds=[42], paired=False, wandb_mode="disabled", config_path=config_path, log=quiet)
    assert [run_dir.relative_to(out).as_posix() for run_dir in run_dirs] == [
        "encoder_layers_1/ext1_random_seed42", "encoder_layers_1/ext1+2_random_seed42",
        "encoder_layers_3/ext1_random_seed42", "encoder_layers_3/ext1+2_random_seed42"]
    for run_dir in run_dirs:
        depth = int(run_dir.parent.name.rsplit("_", 1)[1])
        saved = yaml.safe_load((run_dir / "config.yaml").read_text())
        assert saved["model"]["mask_encoder_layers"] == depth and saved["train"]["regime"] == "random"
        assert saved["validation"]["every_n_epochs"] == VALIDATE_EVERY
        layers = {key.split(".")[2] for key in torch.load(run_dir / "model.pt") if key.startswith("mask_encoder.temporal.")}
        assert layers == {str(i) for i in range(depth)}
        assert sorted(path.name for path in run_dir.glob("scores_test_*.json")) == [
            "scores_test_block_0.1.json", "scores_test_random_0.1.json"]
    for depth in (1, 3):
        assert json.loads((out / f"encoder_layers_{depth}" / "sweep.json").read_text())["tag"] == "test"

def test_the_command_line_trains_nothing_off_the_paired_gpu(small_dataset, tmp_path):
    save_dataset(small_dataset, tmp_path / "data.npz")
    with pytest.raises(RuntimeError, match="a paired run goes on a NVIDIA A100-SXM4-80GB, not on cpu"):
        main(["--study", "steps", "--tag", "ablation", "--out", str(tmp_path / "ablation"), "--device", "cpu",
              "--dataset", str(tmp_path / "data.npz"), "--wandb", "disabled"])
    assert not (tmp_path / "ablation").exists()
