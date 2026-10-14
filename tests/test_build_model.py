import copy

import pytest
import torch
from conftest import CONFIG, csdi_config
from models.build import build_model, config_to_dict
from models.csdi import CSDI_Physio

ACCELERATOR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None

@pytest.fixture
def config() -> dict:
    # A private copy, so a test that changes it (or a build_model that mutates it) cannot
    # leak into the shared CONFIG other tests read.
    return copy.deepcopy(CONFIG)

def heads(model):
    return model.diffmodel.residual_layers[0].time_layer.layers[0].self_attn.num_heads

# --- config.yaml -> upstream's config dict ----------------------------------------------

def test_defaults_are_upstreams_published_physionet_values(config):
    assert config_to_dict(config) == csdi_config("external")

def test_head_count_comes_from_model_n_heads(config):
    config["model"]["n_heads"] = 4
    assert config_to_dict(config)["diffusion"]["nheads"] == 4

def test_converting_leaves_the_loaded_config_unchanged(config):
    before = copy.deepcopy(config)
    config_to_dict(config)
    assert config == before

# --- build_model ------------------------------------------------------------------------

def test_built_model_uses_the_config_values(config):
    config["model"]["n_heads"] = 4
    config["diffusion"].update(channels=32, layers=2, num_steps=10)
    model = build_model(config, "cpu")
    assert isinstance(model, CSDI_Physio)
    assert model.target_strategy == "external"
    assert model.target_dim == config["data"]["n_features"]
    assert model.num_steps == len(model.beta) == 10
    assert model.diffmodel.channels == 32
    assert len(model.diffmodel.residual_layers) == 2
    assert heads(model) == 4

def test_building_leaves_the_loaded_config_unchanged(config):
    # CSDI_base writes side_dim into the diffusion dict it is handed.
    before = copy.deepcopy(config)
    build_model(config, "cpu")
    assert config == before

@pytest.mark.parametrize("toggle", ["use_mask_embedding", "use_time_encoding"])
def test_an_extension_toggle_raises_until_that_extension_exists(config, toggle):
    config["model"][toggle] = True
    with pytest.raises(NotImplementedError):
        build_model(config, "cpu")

def test_initial_weights_follow_the_callers_seed(config):
    # Seeding belongs to RunStreams.seed_model_init(), called right before build_model;
    # build_model must not seed anything itself.
    def weights(seed):
        torch.manual_seed(seed)
        return torch.cat([p.detach().flatten() for p in build_model(config, "cpu").parameters()])

    assert torch.equal(weights(0), weights(0))
    assert not torch.equal(weights(0), weights(1))

@pytest.mark.skipif(ACCELERATOR is None, reason="no GPU available")
def test_weights_and_schedule_are_on_the_requested_device(config):
    # The constructor's device argument moves only the noise schedule; the weights need .to().
    model = build_model(config, ACCELERATOR)
    assert all(p.device.type == ACCELERATOR for p in model.parameters())
    assert model.alpha_torch.device.type == ACCELERATOR
