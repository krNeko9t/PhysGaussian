"""Stage 0: Runtime initialization (warp, taichi)."""

from __future__ import annotations


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

    try:
        import taichi as ti  # type: ignore
        ti.init(arch=ti.cuda, device_memory_GB=8.0)
    except ModuleNotFoundError:
        pass
