"""Load a SimConfig from a Python experiment file.

Usage::

    from physics_sim.config.loader import load_config
    cfg = load_config("experiments/wolf_bread_rigid.py")

The experiment file must define a module-level ``config`` variable of
type :class:`~physics_sim.config.models.SimConfig`.
"""

from __future__ import annotations

import importlib.util
import os
import sys

from physics_sim.config.models import SimConfig


def load_config(path: str) -> SimConfig:
    """Import *path* as a Python module and return its ``config`` attribute."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    spec = importlib.util.spec_from_file_location("_exp_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    # Temporarily add the experiment directory to sys.path so that
    # relative imports within experiment files work if needed.
    exp_dir = os.path.dirname(path)
    added = exp_dir not in sys.path
    if added:
        sys.path.insert(0, exp_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        if added and exp_dir in sys.path:
            sys.path.remove(exp_dir)

    cfg = getattr(module, "config", None)
    if cfg is None:
        raise ValueError(
            f"{path} must define a top-level `config` variable of type SimConfig"
        )
    if not isinstance(cfg, SimConfig):
        raise TypeError(
            f"`config` in {path} must be SimConfig, got {type(cfg).__name__}"
        )
    return cfg
