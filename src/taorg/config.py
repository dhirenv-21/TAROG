"""Configuration loading and validation for TAORG experiments.

Uses PyYAML for config file parsing.  Configs are plain dictionaries
with required-field validation.  A resolved config is the fully merged
configuration saved into run artifacts for reproducibility.

Design decision: PyYAML chosen over OmegaConf for simplicity and fewer
dependencies.  Can revisit if interpolation features are needed.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Fields that must be present in a resolved experiment config
REQUIRED_EXPERIMENT_FIELDS = frozenset({
    "dataset",
    "temporal_method",
    "seed",
})


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to a .yaml or .yml config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a YAML mapping, got {type(config).__name__}")

    logger.info("Loaded config from %s", config_path)
    return config


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override into base config (override wins).

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        New merged configuration dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_experiment_config(config: dict[str, Any]) -> list[str]:
    """Validate that a config has required experiment fields.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        List of error messages. Empty if valid.
    """
    errors = []
    for field_name in REQUIRED_EXPERIMENT_FIELDS:
        if field_name not in config:
            errors.append(f"Missing required config field: '{field_name}'")

    seed = config.get("seed")
    if seed is not None and not isinstance(seed, int):
        errors.append(f"'seed' must be an integer, got {type(seed).__name__}")

    return errors
