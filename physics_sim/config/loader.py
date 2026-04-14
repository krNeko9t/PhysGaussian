"""YAML config loader with ``_ref`` fragment resolution.

Resolution rules (apply to each config section: backend, material, time,
preprocess, camera, and per-object material):

1. **String** -- pure reference.  ``camera: orbit`` loads
   ``physics_sim/conf/camera/orbit.yaml``.
2. **Dict with ``_ref``** -- reference + overrides.  The fragment is loaded
   and then deep-merged with the remaining keys.
3. **Dict without ``_ref``** -- inline config, used as-is.

Deep-merge semantics: dicts are merged recursively; all other types
(including lists) are replaced wholesale.

Usage::

    from physics_sim.config.loader import ConfigLoader
    cfg = ConfigLoader().load("experiments/wolf_bread_vbd.yaml")
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from physics_sim.config.schema import (
    CameraConfig,
    PreprocessConfig,
    SimConfig,
    TimeConfig,
)


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into a copy of *base*.

    - dict values are merged recursively.
    - Everything else (scalars, lists) in *overrides* replaces the base value.
    """
    result = copy.deepcopy(base)
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


class ConfigLoader:
    """Load an experiment YAML and resolve all ``_ref`` fragment references."""

    CONF_DIR = Path(__file__).resolve().parent.parent / "conf"

    def __init__(self, conf_dir: Path | str | None = None):
        if conf_dir is not None:
            self.CONF_DIR = Path(conf_dir)

    # -- public API --------------------------------------------------------

    def load(self, yaml_path: str) -> SimConfig:
        """Load *yaml_path* and return a fully-resolved :class:`SimConfig`."""
        yaml_path = os.path.abspath(yaml_path)
        with open(yaml_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        config_dir = os.path.dirname(yaml_path)

        if "backend" not in raw:
            raise ValueError("Config must specify 'backend'.")
        backend_type, backend = self._resolve_backend(raw["backend"])
        material = self._resolve_section("material", raw.get("material", {}))
        time = TimeConfig.from_dict(
            self._resolve_section("time", raw.get("time", "default"))
        )
        preprocess = PreprocessConfig.from_dict(
            self._resolve_section("preprocess", raw.get("preprocess", "default"))
        )
        camera = CameraConfig.from_dict(
            self._resolve_section("camera", raw.get("camera", "orbit"))
        )

        top_filling = self._resolve_optional(
            "particle_filling", raw.get("particle_filling"),
        )

        objects = copy.deepcopy(raw.get("objects", []))
        for obj in objects:
            if "material" in obj:
                obj["material"] = self._resolve_section(
                    "material", obj["material"]
                )
            if "particle_filling" in obj:
                obj["particle_filling"] = self._resolve_optional(
                    "particle_filling", obj["particle_filling"],
                )
            self._normalize_object(obj)

        return SimConfig(
            output=raw.get("output", "output"),
            backend_type=backend_type,
            backend=backend,
            material=material,
            time=time,
            preprocess=preprocess,
            camera=camera,
            objects=objects,
            boundary_conditions=copy.deepcopy(
                raw.get("boundary_conditions", [])
            ),
            particle_filling=top_filling,
        )

    # -- internal helpers --------------------------------------------------

    def _resolve_backend(self, value: Any) -> tuple[str, dict]:
        if isinstance(value, str):
            return value, self._load_fragment("backend", value)
        if isinstance(value, dict):
            value = dict(value)
            ref = value.pop("_ref", None)
            if ref:
                base = self._load_fragment("backend", ref)
                return str(ref), deep_merge(base, value)
            return value.pop("_type", "custom"), value
        raise ValueError(
            f"'backend' must be a string or dict, got {type(value).__name__}"
        )

    @staticmethod
    def _normalize_object(obj: dict) -> None:
        """Allow shorthand: ``ply_path`` at object level => ``source.type=ply``."""
        if "source" not in obj and "ply_path" in obj:
            obj["source"] = {"type": "ply", "ply_path": obj.pop("ply_path")}

    def _resolve_optional(self, group: str, value: Any) -> dict | None:
        """Like ``_resolve_section`` but returns *None* when *value* is absent."""
        if value is None:
            return None
        return self._resolve_section(group, value)

    def _resolve_section(self, group: str, value: Any) -> dict:
        if isinstance(value, str):
            return self._load_fragment(group, value)
        if isinstance(value, dict):
            value = dict(value)
            ref = value.pop("_ref", None)
            if ref:
                base = self._load_fragment(group, str(ref))
                return deep_merge(base, value)
            return value
        return {}

    def _load_fragment(self, group: str, name: str) -> dict:
        path = self.CONF_DIR / group / f"{name}.yaml"
        if not path.exists():
            group_dir = self.CONF_DIR / group
            if group_dir.is_dir():
                available = sorted(p.stem for p in group_dir.glob("*.yaml"))
            else:
                available = []
            raise FileNotFoundError(
                f"Config fragment not found: {path}\n"
                f"  Available in '{group}': {available}"
            )
        with open(path) as f:
            return yaml.safe_load(f) or {}


# Convenience function matching the old API surface.
def load_config(yaml_path: str) -> SimConfig:
    """Shorthand for ``ConfigLoader().load(yaml_path)``."""
    return ConfigLoader().load(yaml_path)
