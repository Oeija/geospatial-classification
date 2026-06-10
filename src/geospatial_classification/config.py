"""Configuration loading utilities."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Union

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML or JSON configuration file.

    Args:
        config_path: Path to the config file.

    Returns:
        Dictionary with configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file format is unsupported.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    with open(config_path, "r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        elif suffix == ".json":
            import json

            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")


def merge_with_env(config: Dict[str, Any], prefix: str = "GSC_") -> Dict[str, Any]:
    """Merge environment variables into a config dict.

    Environment variables starting with ``prefix`` are converted to
    snake_case keys and override values in *config*.
    """
    for key, val in os.environ.items():
        if key.startswith(prefix):
            clean_key = key[len(prefix) :].lower()
            config[clean_key] = val
    return config


def get_device(prefer_cuda: bool = True) -> str:
    """Return the best available compute device as a string."""
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"
