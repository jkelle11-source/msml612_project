import yaml
from pathlib import Path
from models.csdi import CSDI_Physio
from models.missingness import MissingnessEncoder

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

def config_to_dict(config):
    diffusion = {**config["diffusion"], "nheads": config["model"]["n_heads"]}
    model = {key: config["model"][key] for key in ("timeemb", "featureemb", "is_unconditional")}
    model["target_strategy"] = "external"   # eval.TrainMasker's masks, never upstream's get_randmask (M2-1)
    return {"diffusion": diffusion, "model": model}

def build_model(conf, device):

    if conf["model"]["use_time_encoding"]:
        raise NotImplementedError(
            "model.use_time_encoding is on, but that extension does not exist until M5-2"
            )

    model = CSDI_Physio(config_to_dict(conf),
                        device,
                        target_dim = conf["data"]["n_features"]
                        )

    # Extensions are built after the parts every variant shares, so those start from the
    # same weights in every variant of a run seed (README, Random streams).
    if conf["model"]["use_mask_embedding"]:
        model.mask_encoder = MissingnessEncoder(n_features = conf["data"]["n_features"],
                                                n_timesteps = conf["data"]["n_timesteps"],
                                                d_model = conf["diffusion"]["channels"]
                                                )

    return model.to(device)
