"""Gravity contract helpers for internal Y-up coordinates."""

from __future__ import annotations

import numpy as np
import torch

E_GRAVITY_SHAPE = "E_GRAVITY_SHAPE"
E_GRAVITY_AXIS = "E_GRAVITY_AXIS"
E_GRAVITY_MISSING = "E_GRAVITY_MISSING"


def gravity_contract_error(
    code: str,
    *,
    backend: str,
    config_path: str,
    detail: str,
    material_name: str | None = None,
    raw_g: object | None = None,
    suggestion: str | None = None,
) -> ValueError:
    """Create a ValueError with consistent gravity-contract context."""
    context = f"backend={backend}, config_path={config_path}"
    if material_name:
        context += f", material={material_name}"
    if raw_g is not None:
        context += f", raw_g={raw_g!r}"

    message = f"[{code}] {detail}. ({context})"
    if suggestion:
        message += f" Suggestion: {suggestion}"
    return ValueError(message)


def normalize_internal_gravity(
    raw_g: object,
    *,
    backend: str,
    config_path: str,
    allow_scalar: bool,
    material_name: str | None = None,
) -> tuple[float, float, float]:
    """Validate/normalize gravity to the internal Y-up contract."""
    if raw_g is None:
        raise gravity_contract_error(
            E_GRAVITY_MISSING,
            backend=backend,
            config_path=config_path,
            material_name=material_name,
            detail="gravity is missing",
            suggestion="provide material.g as [0, -|g|, 0] in internal Y-up space",
        )

    if isinstance(raw_g, (int, float)):
        if not allow_scalar:
            raise gravity_contract_error(
                E_GRAVITY_SHAPE,
                backend=backend,
                config_path=config_path,
                material_name=material_name,
                raw_g=raw_g,
                detail="scalar gravity is not allowed here",
                suggestion="pass vec3 [0, -|g|, 0], or route config through backend_init",
            )
        gy = -abs(float(raw_g))
        return (0.0, gy, 0.0)

    if isinstance(raw_g, np.ndarray):
        raw_g = raw_g.tolist()
    if isinstance(raw_g, torch.Tensor):
        raw_g = raw_g.detach().cpu().tolist()

    if not isinstance(raw_g, (list, tuple)) or len(raw_g) != 3:
        raise gravity_contract_error(
            E_GRAVITY_SHAPE,
            backend=backend,
            config_path=config_path,
            material_name=material_name,
            raw_g=raw_g,
            detail="gravity must be a length-3 vector",
            suggestion="use [0, -|g|, 0]",
        )

    try:
        gx = float(raw_g[0])
        gy = float(raw_g[1])
        gz = float(raw_g[2])
    except (TypeError, ValueError) as exc:
        raise gravity_contract_error(
            E_GRAVITY_SHAPE,
            backend=backend,
            config_path=config_path,
            material_name=material_name,
            raw_g=raw_g,
            detail="gravity vector contains non-numeric values",
            suggestion="use numeric vec3 [0, -|g|, 0]",
        ) from exc

    if not (np.isfinite(gx) and np.isfinite(gy) and np.isfinite(gz)):
        raise gravity_contract_error(
            E_GRAVITY_SHAPE,
            backend=backend,
            config_path=config_path,
            material_name=material_name,
            raw_g=raw_g,
            detail="gravity vector contains non-finite values",
            suggestion="use finite vec3 [0, -|g|, 0]",
        )

    if abs(gx) > 1e-6 or abs(gz) > 1e-6 or gy >= 0.0:
        raise gravity_contract_error(
            E_GRAVITY_AXIS,
            backend=backend,
            config_path=config_path,
            material_name=material_name,
            raw_g=raw_g,
            detail="gravity violates internal Y-up contract (must point to -Y)",
            suggestion="set gravity to [0, -|g|, 0]",
        )

    return (0.0, -abs(gy), 0.0)


def gravity_vector(
    magnitude: float = 9.8,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Gravity in internal coordinates: always [0, -mag, 0]."""
    return torch.tensor([0.0, -magnitude, 0.0], device=device, dtype=torch.float32)
