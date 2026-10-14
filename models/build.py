import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

def config_to_dict(path = CONFIG_PATH):
    