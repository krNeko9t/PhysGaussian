"""
Configuration parser for physics simulation scenes.

Extracted from PhysGaussian/utils/decode_param.py.
Reads a JSON config and returns structured parameter dictionaries.
"""

import json

# Backend names that can appear as top-level override sections in the config.
_KNOWN_BACKENDS = ("warp_mpm", "newton_mpm")

# Material keys that can be overridden per-object.
_MATERIAL_KEYS = (
    "material", "E", "nu", "density", "friction_angle",
    "yield_stress", "hardening", "rpic_damping", "pic_damping",
    "xi", "plastic_viscosity", "softening",
)


def decode_param_json(json_file: str):
    """Parse a scene configuration JSON file.

    Returns:
        (material_params, bc_params, time_params,
         preprocessing_params, camera_params, backend_overrides,
         scene_objects)

    ``backend_overrides`` — dict keyed by backend name for per-backend
    parameter overrides (time stepping, solver options, etc.).

    ``scene_objects`` — ``None`` for single-object scenes (backward
    compatible), or a list of dicts for multi-object scenes::

        [
            {
                "name": "sand_pile",
                "sim_area": [x0, x1, y0, y1, z0, z1],  # required if no ply_path
                "material": { ... per-object material params ... },
                "particle_filling": { ... } or None,
                "ply_path": "path/to/other.ply" or None,  # dedicated PLY
                "position_offset": [dx, dy, dz] or None,  # in rotated space
            },
            ...
        ]

    Per-object material params inherit from the top-level defaults and
    can override any key in ``_MATERIAL_KEYS``.

    If ``ply_path`` is given, the object's particles come from that PLY
    file (loaded independently).  Otherwise they come from the shared
    ``--ply_path`` and must be selected via ``sim_area``.

    ``position_offset`` shifts the object's particles (in rotated space,
    after the global rotation is applied) before the MPM transform.
    """
    with open(json_file) as f:
        sim_params = json.load(f)

    # ── Material parameters ───────────────────────────────────────────
    material_params = {}
    material_params["material"] = sim_params.get("material", "jelly")
    material_params["grid_lim"] = sim_params.get("grid_lim", 2.0)
    material_params["n_grid"] = sim_params.get("n_grid", 50)

    nu = sim_params.get("nu", 0.4)
    if nu > 0.5 or nu < 0.0:
        raise ValueError("Poisson's ratio should be between 0.0 and 0.5")
    material_params["nu"] = nu
    material_params["E"] = sim_params.get("E", 1e5)

    for key in ("yield_stress", "hardening", "xi", "friction_angle",
                "plastic_viscosity", "softening", "opacity_threshold",
                "grid_v_damping_scale", "rpic_damping", "pic_damping"):
        if key in sim_params:
            material_params[key] = sim_params[key]

    material_params["g"] = sim_params.get("g", 9.8)
    material_params["density"] = sim_params.get("density", 200.0)

    if "additional_material_params" in sim_params:
        additional_params = sim_params["additional_material_params"]
        for p in additional_params:
            if "point" not in p:
                raise TypeError("point is not defined")
            if "size" not in p:
                raise TypeError("size is not defined")
            if "E" not in p:
                raise TypeError("E is not defined")
            if "nu" not in p:
                raise TypeError("nu is not defined")
            p.setdefault("density", material_params["density"])
        material_params["additional_material_params"] = additional_params

    # ── Boundary conditions ───────────────────────────────────────────
    bc_params = sim_params.get("boundary_conditions", {})

    # ── Time parameters ───────────────────────────────────────────────
    time_params = {
        "substep_dt": sim_params.get("substep_dt", 1e-4),
        "frame_dt": sim_params.get("frame_dt", 1e-2),
        "frame_num": sim_params.get("frame_num", 100),
    }

    # ── Preprocessing parameters ──────────────────────────────────────
    preprocessing_params = {
        "opacity_threshold": sim_params.get("opacity_threshold", 0.02),
        "rotation_degree": sim_params.get("rotation_degree", []),
        "rotation_axis": sim_params.get("rotation_axis", []),
        "sim_area": sim_params.get("sim_area", None),
        "scale": sim_params.get("scale", 1.0),
    }

    if "particle_filling" in sim_params:
        filling = sim_params["particle_filling"]
        filling.setdefault("n_grid", material_params["n_grid"] * 4)
        filling.setdefault("density_threshold", 5.0)
        filling.setdefault("search_threshold", 3.0)
        filling.setdefault("max_particles_num", 2000000)
        filling.setdefault("max_partciels_per_cell", 1)
        filling.setdefault("search_exclude_direction", 5)
        filling.setdefault("ray_cast_direction", 4)
        filling.setdefault("boundary", None)
        filling.setdefault("smooth", False)
        filling.setdefault("visualize", False)
        preprocessing_params["particle_filling"] = filling
    else:
        preprocessing_params["particle_filling"] = None

    # ── Camera parameters ─────────────────────────────────────────────
    camera_params = {
        "mpm_space_viewpoint_center": sim_params.get("mpm_space_viewpoint_center", [1.0, 1.0, 1.0]),
        "mpm_space_vertical_upward_axis": sim_params.get("mpm_space_vertical_upward_axis", [0, 0, 1]),
        "default_camera_index": sim_params.get("default_camera_index", 0),
        "show_hint": sim_params.get("show_hint", False),
        "init_azimuthm": sim_params.get("init_azimuthm", None),
        "init_elevation": sim_params.get("init_elevation", None),
        "init_radius": sim_params.get("init_radius", None),
        "delta_a": sim_params.get("delta_a", None),
        "delta_e": sim_params.get("delta_e", None),
        "delta_r": sim_params.get("delta_r", None),
        "move_camera": sim_params.get("move_camera", False),
    }

    # ── Backend-specific overrides ───────────────────────────────────
    # Top-level keys whose name matches a backend (e.g. "newton_mpm")
    # are extracted as override dicts.  The pipeline merges them at runtime.
    backend_overrides: dict[str, dict] = {}
    for backend_name in _KNOWN_BACKENDS:
        if backend_name in sim_params:
            backend_overrides[backend_name] = sim_params[backend_name]

    # ── Multi-object scene definition ────────────────────────────────
    # If "objects" is present, each entry defines a distinct physical
    # object with its own sim_area and (optional) material overrides.
    # Absent → single-object mode (fully backward compatible).
    #
    # Each object can optionally specify ``ply_path`` to load particles
    # from a dedicated PLY file (instead of the shared --ply_path).
    # Objects without ``ply_path`` MUST define ``sim_area`` to select
    # particles from the shared PLY.
    scene_objects = None
    if "objects" in sim_params:
        scene_objects = []
        for i, obj_def in enumerate(sim_params["objects"]):
            has_ply = "ply_path" in obj_def
            has_area = "sim_area" in obj_def
            if not has_ply and not has_area:
                raise ValueError(
                    f"Object {i} ({obj_def.get('name', '?')}) "
                    "must define either 'ply_path' or 'sim_area' (or both)."
                )
            # Build per-object material by inheriting top-level defaults
            obj_material = {}
            for key in _MATERIAL_KEYS:
                if key in obj_def:
                    obj_material[key] = obj_def[key]
                elif key in material_params:
                    obj_material[key] = material_params[key]
            scene_objects.append({
                "name": obj_def.get("name", f"object_{i}"),
                "sim_area": obj_def.get("sim_area", None),
                "material": obj_material,
                "particle_filling": obj_def.get("particle_filling", None),
                "ply_path": obj_def.get("ply_path", None),
                "position_offset": obj_def.get("position_offset", None),
            })

    return (material_params, bc_params, time_params,
            preprocessing_params, camera_params, backend_overrides,
            scene_objects)
