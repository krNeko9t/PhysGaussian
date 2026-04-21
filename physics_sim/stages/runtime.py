"""Stage 0: Runtime initialization (warp, taichi)."""

from __future__ import annotations

import importlib

from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)


def init_runtime(backend_type: str) -> None:
    """Initialize warp (and optionally taichi) for physics backends."""
    if backend_type == "none":
        return

    try:
        import warp as wp  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Python module 'warp' not found. Install warp_lang."
        ) from e

    wp.init()
    if backend_type in ("newton_mpm", "newton_rigid", "newton_vbd"):
        wp.config.verify_cuda = False
    else:
        wp.config.verify_cuda = True

    # Eager-load the Newton backend package while the user still sees early
    # pipeline output (config load / "Assembling scene...").  Otherwise the
    # first `create_backend()` pays a multi-second import stall right after
    # the last [assembler] line (see debug H5 vs H0 in scene_setup).
    _NEWTON_BACKENDS = {
        "newton_vbd": "physics_sim.backend.newton_vbd",
        "newton_rigid": "physics_sim.backend.newton_rigid",
        "newton_mpm": "physics_sim.backend.newton_mpm",
    }
    mod = _NEWTON_BACKENDS.get(backend_type)
    if mod is not None:
        importlib.import_module(mod)

    try:
        import taichi as ti  # type: ignore
        ti.init(arch=ti.cuda, device_memory_GB=8.0)
    except ModuleNotFoundError:
        LOGGER.warning(
            "Taichi is not installed; continuing without taichi runtime support."
        )
