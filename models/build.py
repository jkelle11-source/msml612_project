import yaml
from pathlib import Path
from models.csdi import CSDI_Physio

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

def config_to_dict(config):
    diffusion = {**config["diffusion"], "nheads": config["model"]["n_heads"]}
    model = {key: config["model"][key] for key in ("timeemb", "featureemb", "is_unconditional")}
    model["target_strategy"] = "external"   # eval.TrainMasker's masks, never upstream's get_randmask (M2-1)
    return {"diffusion": diffusion, "model": model}

def build_model(conf, device):
    
    for toggle, issue in (("use_mask_embedding", "M4-2"), ("use_time_encoding", "M5-2")):
        if conf["model"][toggle]:
            raise NotImplementedError(
                f"model.{toggle} is on, but that extension does not exist until {issue}"
                )
            
    model = CSDI_Physio(config_to_dict(conf), 
                        device, 
                        target_dim = conf["data"]["n_features"]
                        )
    
    return model.to(device)