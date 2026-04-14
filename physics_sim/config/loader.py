"""Hydra/OmegaConf-based config loader.

Replaces the old ``decode_param_json`` JSON parser with a composable
YAML system.  The user points ``--config`` at a root YAML that declares
``defaults:`` pulling in sub-configs from ``physics_sim/conf/``.

Usage from pipeline::

    from physics_sim.config.loader import load_config
    cfg = load_config("experiments/wolf_rigid.yaml")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def load_config(yaml_path: str) -> dict[str, Any]:
    """Load and resolve a Hydra-composable YAML config from an arbitrary path.

    The root YAML should declare::

        hydra:
          searchpath:
            - pkg://physics_sim.conf

    so that ``defaults:`` can reference sub-configs shipped with the package
    (e.g. ``- material: sand``).

    Returns a plain Python dict (fully resolved, no ``${...}`` references).
    """
    yaml_path = os.path.abspath(yaml_path)
    config_dir = os.path.dirname(yaml_path)
    config_name = Path(yaml_path).stem

    # GlobalHydra must be cleared between calls (e.g. in tests).
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg: DictConfig = compose(config_name=config_name)

    resolved: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    return resolved
