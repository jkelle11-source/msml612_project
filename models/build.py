import yaml
import torch
from pathlib import Path
from models.csdi import CSDI_Physio
from models.missingness import MissingnessEncoder
from models.time_encoding import ContinuousTimeEncoding

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

def config_to_dict(config):
    diffusion = {**config["diffusion"], "nheads": config["model"]["n_heads"]}
    model = {key: config["model"][key] for key in ("timeemb", "featureemb", "is_unconditional")}
    model["target_strategy"] = "external"   # eval.TrainMasker's masks, never upstream's get_randmask (M2-1)
    return {"diffusion": diffusion, "model": model}

def build_model(conf, device):

    model = CSDI_Physio(config_to_dict(conf),
                        device,
                        target_dim = conf["data"]["n_features"]
                        )

    state = torch.get_rng_state()

    # Extensions are built after the parts every variant shares, so those start
    # from the same weights in every variant of a run seed (README, Random streams).
    # Each is built from the init-RNG state right after them (state), so an encoder
    # starts from the same weights in every variant that has it (PROJECT_PLAN §3.3, M5-2).
    if conf["model"]["use_mask_embedding"]:
        torch.set_rng_state(state)
        model.mask_encoder = MissingnessEncoder(n_features = conf["data"]["n_features"],
                                                n_timesteps = conf["data"]["n_timesteps"],
                                                d_model = conf["diffusion"]["channels"]
                                                )

    if conf["model"]["use_time_encoding"]:
        torch.set_rng_state(state)
        model.time_encoder = ContinuousTimeEncoding(d_k = conf["model"]["time_encoding_dk"],
                                                    d_model = conf["diffusion"]["channels"])

    return model.to(device)
