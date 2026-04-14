#!/usr/bin/env python3
"""
Convert a phys_desc.json (semantic-level physical descriptions produced by a
VLM) into a pipeline-ready YAML config that ``pipeline.py`` can consume.

Usage:
    python phys_desc_to_config.py \
        --phys_desc phys_desc.json \
        --ply_dir   scene_data/plys/ \
        --output    experiments/auto_config.yaml \
        --backend-preference auto \
        [--scene-defaults scene_defaults.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_MATERIAL_LUT = _SCRIPT_DIR / "physics_sim" / "config" / "material_lut.json"
_DEFAULT_GEOMETRY_LUT = _SCRIPT_DIR / "physics_sim" / "config" / "geometry_lut.json"

_BEHAVIOR_TO_MODE = {
    "static": "collider_only",
    "dynamic_rigid": "simulate",
    "dynamic_soft": "simulate",
    "out_of_range": "render_only",
}

_BACKEND_DEFAULTS: dict[str, dict[str, Any]] = {
    "newton_rigid": {
        "substep_dt": 1e-4,
        "collision_geometry": "convex_hull",
        "use_sdf": True,
        "solver": {
            "iterations": 20,
            "contact_relaxation": 0.5,
        },
    },
    "newton_mpm": {
        "substep_dt": 1e-4,
    },
    "newton_vbd": {
        "substep_dt": 1e-4,
        "collision_geometry": "convex_hull",
        "use_sdf": True,
        "solver": {
            "iterations": 20,
            "contact_relaxation": 0.5,
        },
    },
}


# ---------------------------------------------------------------------------
# Material mixing
# ---------------------------------------------------------------------------

def _load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _blend_materials(
    appearance_materials: list[dict],
    physical_priors: dict,
    material_lut: dict,
) -> dict[str, float]:
    """Blend prototype parameters weighted by score, then scale by bin priors."""
    prototypes = material_lut["prototypes"]
    bin_scales = material_lut["bin_scales"]
    bin_targets = material_lut["bin_targets"]

    param_keys = ("density", "E", "nu", "mu")
    blended = {k: 0.0 for k in param_keys}
    total_score = 0.0

    for entry in appearance_materials:
        proto_name = entry.get("prototype", "unknown")
        score = float(entry.get("score", 0))
        proto = prototypes.get(proto_name, prototypes["unknown"])
        for k in param_keys:
            blended[k] += score * float(proto[k])
        total_score += score

    if total_score > 0:
        for k in param_keys:
            blended[k] /= total_score

    for bin_name, target_param in bin_targets.items():
        prior = physical_priors.get(bin_name)
        if prior is None:
            continue
        value = prior.get("value", "unknown")
        confidence = float(prior.get("confidence", 0.5))
        scale_table = bin_scales.get(bin_name, {})
        raw_scale = float(scale_table.get(value, 1.0))
        # Interpolate toward 1.0 when confidence is low.
        effective_scale = 1.0 + (raw_scale - 1.0) * confidence
        if target_param in blended:
            blended[target_param] *= effective_scale

    return blended


# ---------------------------------------------------------------------------
# Collision geometry mapping
# ---------------------------------------------------------------------------

def _map_geometry(geometry_form: str, geometry_lut: dict) -> dict:
    """Return collision-geometry related params for a dynamic body."""
    entry = geometry_lut.get(geometry_form, geometry_lut.get("uncertain", {}))
    return {k: v for k, v in entry.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Static collider auto-configuration
# ---------------------------------------------------------------------------

def _auto_collider_config(
    ply_path: str | None,
    physical_priors: dict,
    material_lut: dict,
) -> dict:
    """Build a collider config block for a static object.

    If ``ply_path`` is available, the PLY is *not* loaded at conversion time —
    the pipeline's existing SVD plane-fitting will handle it at runtime.
    We only need to produce the JSON config that tells the pipeline *how* to
    fit the plane.
    """
    # Friction from priors
    bin_scales = material_lut["bin_scales"]
    friction_prior = physical_priors.get("surface_friction_bin", {})
    friction_value = friction_prior.get("value", "medium")
    friction_confidence = float(friction_prior.get("confidence", 0.5))
    raw_scale = float(bin_scales.get("surface_friction_bin", {}).get(friction_value, 1.0))
    base_friction = 0.5
    friction = base_friction * (1.0 + (raw_scale - 1.0) * friction_confidence)
    friction = max(0.0, min(friction, 1.0))

    # Surface type from stiffness
    stiffness_prior = physical_priors.get("stiffness_bin", {})
    stiffness_value = stiffness_prior.get("value", "medium")
    surface = "slip" if stiffness_value in ("stiff", "very_stiff") else "sticky"

    collider: dict[str, Any] = {
        "type": "plane",
        "space": "world",
        "fit": {
            "method": "svd",
            "sample_max": 200000,
            "seed": 0,
        },
        "surface": surface,
        "friction": round(friction, 3),
    }

    return collider


def _estimate_plane_orientation(ply_path: str) -> list[float] | None:
    """Load a PLY and return the SVD-estimated plane normal, or None on failure.

    This is used at conversion time to set ``prefer_up`` for wall-like surfaces.
    If the PLY is unavailable (or numpy can't read it), we return None and let
    the pipeline's runtime fitting determine the direction.
    """
    try:
        from plyfile import PlyData  # type: ignore
    except ImportError:
        return None

    if ply_path is None or not os.path.exists(ply_path):
        return None

    try:
        ply = PlyData.read(ply_path)
        vertex = ply["vertex"]
        xyz = np.column_stack([
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ])
    except Exception:
        return None

    if len(xyz) < 3:
        return None

    # Subsample for speed
    if len(xyz) > 50000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xyz), 50000, replace=False)
        xyz = xyz[idx]

    centroid = xyz.mean(axis=0)
    _, _, Vt = np.linalg.svd(xyz - centroid, full_matrices=False)
    normal = Vt[2]  # smallest singular value direction

    # Ensure the normal points "outward" (positive z-component preferred)
    if normal[2] < 0:
        normal = -normal

    return normal.tolist()


# ---------------------------------------------------------------------------
# Backend inference
# ---------------------------------------------------------------------------

def _infer_backend(
    objects: list[dict],
    preference: str,
) -> str:
    """Choose the physics backend based on scene composition + user preference."""
    if preference != "auto":
        return preference

    has_rigid = any(
        o.get("_behavior") == "dynamic_rigid" for o in objects
    )
    has_soft = any(
        o.get("_behavior") == "dynamic_soft" for o in objects
    )

    if has_rigid and has_soft:
        return "newton_vbd"
    elif has_soft:
        return "newton_mpm"
    else:
        return "newton_rigid"


# ---------------------------------------------------------------------------
# Backend override generation
# ---------------------------------------------------------------------------

def _build_backend_override(
    backend: str,
    objects: list[dict],
    geometry_lut: dict,
) -> dict[str, Any]:
    """Generate the backend-specific override section for the config JSON."""
    base = dict(_BACKEND_DEFAULTS.get(backend, {}))

    # Determine dominant collision geometry from dynamic objects.
    geo_votes: dict[str, int] = {}
    use_sdf_any = False
    for obj in objects:
        geo = obj.get("_geo_params", {})
        cg = geo.get("collision_geometry")
        if cg:
            geo_votes[cg] = geo_votes.get(cg, 0) + 1
        if geo.get("use_sdf"):
            use_sdf_any = True

    if geo_votes:
        dominant_geo = max(geo_votes, key=geo_votes.get)  # type: ignore[arg-type]
        base["collision_geometry"] = dominant_geo
    base["use_sdf"] = use_sdf_any

    return base


# ---------------------------------------------------------------------------
# Scene defaults
# ---------------------------------------------------------------------------

_SCENE_DEFAULTS: dict[str, Any] = {
    "opacity_threshold": 0.1,
    "axis_permutation": "xz-y",
    "rotation_degree": [0.0],
    "rotation_axis": [0],
    "transform_reference": "shared_ply",

    "substep_dt": 1e-4,
    "frame_dt": 1e-2,
    "frame_num": 120,

    "material": "sand",
    "density": 800,
    "g": [0.0, 0.0, -9.8],
    "mu": 0.5,
    "n_grid": 200,

    "boundary_conditions": [],

    "mpm_space_vertical_upward_axis": [0, 0, 1],
    "default_camera_index": -1,
    "show_hint": False,

    "width": 800,
    "height": 600,
    "fovx_deg": 60.0,
    "fovy_deg": 45.0,

    "init_azimuth": 160.0,
    "init_elevation": 20.0,
    "init_radius": 2.8,
    "move_camera": True,
    "delta_a": -0.6,
    "delta_e": 0.0,
    "delta_r": 0.0,
}

# `3dovs_bench_inst000` -> 0 -> instance_00000.ply
_INST_INDEX_RE = re.compile(r"inst(\d+)", re.IGNORECASE)


def _resolve_ply_path(ply_dir: str, instance_id: str) -> str:
    """Prefer ``{instance_id}.ply``; else ``instance_{idx:05d}.ply`` from ``instNNN`` in id."""
    primary = os.path.join(ply_dir, f"{instance_id}.ply")
    if os.path.exists(primary):
        return primary
    m = _INST_INDEX_RE.search(instance_id)
    if m:
        idx = int(m.group(1))
        secondary = os.path.join(ply_dir, f"instance_{idx:05d}.ply")
        if os.path.exists(secondary):
            return secondary
        return secondary
    return primary


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    phys_desc: list[dict],
    ply_dir: str | None,
    material_lut: dict,
    geometry_lut: dict,
    backend_preference: str = "auto",
    scene_defaults: dict | None = None,
) -> dict:
    """Convert phys_desc entries into a pipeline config dict (new YAML schema).

    The output dict is serialised to YAML by the caller and consumed by
    ``pipeline.py`` via ``physics_sim.config.loader.load_config``.
    """

    config: dict[str, Any] = {}

    # Hydra header: enables ``defaults:`` resolution against the shipped
    # sub-configs in ``physics_sim/conf/``.
    config["hydra"] = {"searchpath": ["pkg://physics_sim.conf"]}
    config["defaults"] = [
        {"material": "sand"},
        {"time": "default"},
        {"preprocess": "default"},
        {"camera": "orbit"},
        # backend default will be overridden below
        "_self_",
    ]

    # Scene-level defaults.
    defaults = dict(_SCENE_DEFAULTS)
    if scene_defaults:
        defaults.update(scene_defaults)

    for k, v in defaults.items():
        if k != "objects":
            config[k] = v

    # Preprocess block
    config["preprocess"] = {
        "opacity_threshold": defaults.get("opacity_threshold", 0.1),
        "axis_permutation": defaults.get("axis_permutation", "xz-y"),
        "rotation_degree": defaults.get("rotation_degree", [0.0]),
        "rotation_axis": defaults.get("rotation_axis", [0]),
        "transform_reference": defaults.get("transform_reference", "shared_ply"),
    }

    # Camera block
    config["camera"] = {
        "camera_mode": "orbit",
        "default_camera_index": defaults.get("default_camera_index", -1),
        "show_hint": defaults.get("show_hint", False),
        "width": defaults.get("width", 800),
        "height": defaults.get("height", 600),
        "fovx_deg": defaults.get("fovx_deg", 60.0),
        "fovy_deg": defaults.get("fovy_deg", 45.0),
        "init_azimuth": defaults.get("init_azimuth", 160.0),
        "init_elevation": defaults.get("init_elevation", 20.0),
        "init_radius": defaults.get("init_radius", 2.8),
        "move_camera": defaults.get("move_camera", True),
        "delta_a": defaults.get("delta_a", -0.6),
        "delta_e": defaults.get("delta_e", 0.0),
        "delta_r": defaults.get("delta_r", 0.0),
    }

    # Build per-object entries (new schema with source blocks).
    objects_internal: list[dict[str, Any]] = []

    for entry in phys_desc:
        response = entry.get("response", entry)
        instance_id = entry.get("instance_id", f"object_{entry.get('id', 0)}")
        behavior = response.get("behavior_template", "out_of_range")
        geometry_form = response.get("geometry_form", "uncertain")
        appearance_materials = response.get("appearance_materials", [])
        physical_priors = response.get("physical_priors", {})

        mode = _BEHAVIOR_TO_MODE.get(behavior, "render_only")

        ply_path = None
        if ply_dir:
            ply_path = _resolve_ply_path(ply_dir, instance_id)

        obj: dict[str, Any] = {
            "name": instance_id,
            "mode": mode,
            "_behavior": behavior,
        }

        # New schema: source block
        if ply_path:
            obj["source"] = {"type": "ply", "ply_path": ply_path}

        if mode == "simulate":
            mat = _blend_materials(appearance_materials, physical_priors, material_lut)
            geo_params = _map_geometry(geometry_form, geometry_lut)
            obj["material"] = {
                "density": round(mat["density"], 1),
                "mu": round(mat["mu"], 3),
                "E": mat["E"],
                "nu": round(mat["nu"], 3),
            }
            if "collision_geometry" in geo_params:
                obj["material"]["collision_geometry"] = geo_params["collision_geometry"]
            obj["_geo_params"] = geo_params

        elif mode == "collider_only":
            collider = _auto_collider_config(ply_path, physical_priors, material_lut)
            if ply_path:
                normal = _estimate_plane_orientation(ply_path)
                if normal is not None:
                    nz_abs = abs(normal[2])
                    if nz_abs < 0.7:
                        collider["prefer_up"] = [
                            round(normal[0], 4),
                            round(normal[1], 4),
                            round(normal[2], 4),
                        ]
            obj["collider"] = collider

        objects_internal.append(obj)

    # Infer backend
    backend = _infer_backend(objects_internal, backend_preference)
    config["backend"] = backend

    # Backend defaults merged to top level (new schema — no nested backend block)
    backend_override = _build_backend_override(backend, objects_internal, geometry_lut)
    for k, v in backend_override.items():
        config[k] = v

    # Add backend default to the defaults list
    config["defaults"].insert(-1, {"backend": backend.replace("_", "_")})

    # Strip internal keys and assemble final objects list
    clean_objects = []
    for obj in objects_internal:
        clean = {k: v for k, v in obj.items() if not k.startswith("_")}
        clean_objects.append(clean)
    config["objects"] = clean_objects

    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert phys_desc.json to PhysGaussian pipeline config YAML."
    )
    parser.add_argument(
        "--phys_desc", type=str, required=True,
        help="Path to phys_desc.json",
    )
    parser.add_argument(
        "--ply_dir", type=str, default=None,
        help="Directory containing per-object PLY files (<instance_id>.ply)",
    )
    parser.add_argument(
        "--output", type=str, default="experiments/auto_config.yaml",
        help="Output path for the generated config YAML",
    )
    parser.add_argument(
        "--backend-preference", type=str, default="auto",
        choices=["auto", "newton_rigid", "newton_mpm", "newton_vbd", "warp_mpm"],
        help="Backend preference (default: auto — inferred from scene composition)",
    )
    parser.add_argument(
        "--scene-defaults", type=str, default=None,
        help="Optional JSON file with scene-level parameter overrides",
    )
    parser.add_argument(
        "--material-lut", type=str, default=None,
        help="Path to material_lut.json (default: physics_sim/config/material_lut.json)",
    )
    parser.add_argument(
        "--geometry-lut", type=str, default=None,
        help="Path to geometry_lut.json (default: physics_sim/config/geometry_lut.json)",
    )
    args = parser.parse_args()

    # Load inputs
    phys_desc = _load_json(args.phys_desc)
    material_lut_path = args.material_lut or _DEFAULT_MATERIAL_LUT
    geometry_lut_path = args.geometry_lut or _DEFAULT_GEOMETRY_LUT
    material_lut = _load_json(material_lut_path)
    geometry_lut = _load_json(geometry_lut_path)

    scene_defaults = None
    if args.scene_defaults:
        scene_defaults = _load_json(args.scene_defaults)

    config = convert(
        phys_desc=phys_desc,
        ply_dir=args.ply_dir,
        material_lut=material_lut,
        geometry_lut=geometry_lut,
        backend_preference=args.backend_preference,
        scene_defaults=scene_defaults,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    try:
        from omegaconf import OmegaConf
        yaml_str = OmegaConf.to_yaml(OmegaConf.create(config))
    except ImportError:
        import yaml  # type: ignore[import-untyped]
        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False,
                             allow_unicode=True)

    with open(args.output, "w") as f:
        f.write(yaml_str)

    # Summary
    n_sim = sum(1 for o in config["objects"] if o["mode"] == "simulate")
    n_col = sum(1 for o in config["objects"] if o["mode"] == "collider_only")
    n_ren = sum(1 for o in config["objects"] if o["mode"] == "render_only")
    print(f"Generated config: {args.output}")
    print(f"  Backend: {config['backend']}")
    print(f"  Objects: {len(config['objects'])} total "
          f"({n_sim} simulate, {n_col} collider_only, {n_ren} render_only)")


if __name__ == "__main__":
    main()
