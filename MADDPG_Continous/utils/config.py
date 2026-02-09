import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class Config:
    env: Dict[str, Any]
    train: Dict[str, Any]
    log: Dict[str, Any]
    experiment: Dict[str, Any]


def load_config(config_path: str) -> Config:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(
        env=data.get("env", {}),
        train=data.get("train", {}),
        log=data.get("log", {}),
        experiment=data.get("experiment", {}),
    )
