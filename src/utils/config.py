import copy
import logging
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as fp:
        return yaml.safe_load(fp) or {}


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    cfg = load_yaml(path)

    base_key = "_base_"
    if base_key in cfg:
        base_path = path.parent / cfg[base_key]
        base_cfg = load_config(base_path)
        cfg.pop(base_key)
        cfg = merge_dicts(base_cfg, cfg)

    return cfg


def dump_config(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        yaml.safe_dump(config, fp, sort_keys=False)


def log_config(config: Dict[str, Any]) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Configuration snapshot:")
    logger.info(yaml.safe_dump(config, sort_keys=False))
