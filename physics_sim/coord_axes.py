"""Axis parsing, presets, and SourceAxes contract."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import torch

_AXIS_MAP = {
    "+X": [1, 0, 0],
    "-X": [-1, 0, 0],
    "+Y": [0, 1, 0],
    "-Y": [0, -1, 0],
    "+Z": [0, 0, 1],
    "-Z": [0, 0, -1],
}


def parse_axis(s: str) -> np.ndarray:
    """Parse ``+X``/``-Z`` style axis strings into unit vectors."""
    key = s.strip().upper()
    if key in _AXIS_MAP:
        return np.array(_AXIS_MAP[key], dtype=np.float32)
    raise ValueError(
        f"Unknown axis string: '{s}'.  Expected one of {list(_AXIS_MAP.keys())}"
    )


def _normalise_up_string(s: str) -> str:
    """Accept legacy ``Y_UP`` / ``Z_UP`` and canonicalize to ``+Y`` style."""
    s = s.strip().upper().replace("_UP", "").replace("UP", "")
    s = s.replace("_", "").replace(" ", "")
    if s in ("Y", "+Y"):
        return "+Y"
    if s in ("Z", "+Z"):
        return "+Z"
    if s in ("X", "+X"):
        return "+X"
    if s.startswith(("+", "-")) and len(s) == 2 and s[1] in "XYZ":
        return s
    raise ValueError(f"Cannot interpret up-axis string: '{s}'")


class UpAxis(enum.Enum):
    Y_UP = "Y_UP"
    Z_UP = "Z_UP"


PRESETS: dict[str, dict[str, str]] = {
    "OPENGL": dict(up="+Y", front="+Z"),
    "BLENDER": dict(up="+Z", front="-Y"),
    "Z_UP_X_FRONT": dict(up="+Z", front="+X"),
    "Z_UP_Y_FRONT": dict(up="+Z", front="+Y"),
}


@dataclass(frozen=True)
class SourceAxes:
    """Describes mapping from source coordinates to internal Y-up frame."""

    up_str: str
    front_str: str
    A: torch.Tensor
    A_inv: torch.Tensor
    is_identity: bool

    @classmethod
    def from_config(
        cls,
        source_up: str = "+Y",
        source_front: str | None = None,
    ) -> SourceAxes:
        up_s = _normalise_up_string(source_up)
        if source_front is None:
            source_front = _default_front(up_s)
        front_s = source_front.strip().upper()
        if front_s not in _AXIS_MAP:
            raise ValueError(f"Invalid source_front: '{source_front}'")
        return cls._build(up_s, front_s)

    @classmethod
    def from_preset(cls, name: str) -> SourceAxes:
        name = name.strip().upper()
        if name not in PRESETS:
            raise ValueError(
                f"Unknown preset '{name}'.  Available: {list(PRESETS.keys())}"
            )
        p = PRESETS[name]
        return cls._build(p["up"], p["front"])

    @classmethod
    def identity(cls) -> SourceAxes:
        I = torch.eye(3, dtype=torch.float32)
        return cls(up_str="+Y", front_str="+Z", A=I, A_inv=I, is_identity=True)

    @classmethod
    def _build(cls, up_s: str, front_s: str) -> SourceAxes:
        up_vec = parse_axis(up_s)
        front_vec = parse_axis(front_s)

        if abs(np.dot(up_vec, front_vec)) > 1e-6:
            raise ValueError(
                f"source_up='{up_s}' and source_front='{front_s}' are not orthogonal."
            )

        right_vec = np.cross(up_vec, front_vec)
        norm = np.linalg.norm(right_vec)
        if norm < 1e-6:
            raise ValueError(f"Degenerate axes: up={up_s}, front={front_s}")
        right_vec = right_vec / norm

        B_src = np.column_stack([right_vec, up_vec, front_vec])
        A_np = B_src.T.astype(np.float32)
        A = torch.from_numpy(A_np)
        A_inv = A.T.contiguous()
        is_id = bool(torch.allclose(A, torch.eye(3), atol=1e-6))
        return cls(up_str=up_s, front_str=front_s, A=A, A_inv=A_inv, is_identity=is_id)

    def to(self, device: str | torch.device) -> SourceAxes:
        return SourceAxes(
            up_str=self.up_str,
            front_str=self.front_str,
            A=self.A.to(device),
            A_inv=self.A_inv.to(device),
            is_identity=self.is_identity,
        )

    @property
    def up_vector(self) -> np.ndarray:
        return parse_axis(self.up_str)


def _default_front(up_s: str) -> str:
    defaults = {
        "+Y": "+Z",
        "-Y": "-Z",
        "+Z": "-Y",
        "-Z": "+Y",
        "+X": "+Z",
        "-X": "-Z",
    }
    if up_s in defaults:
        return defaults[up_s]
    raise ValueError(f"No default front for up='{up_s}'; specify source_front explicitly.")


def alignment_matrix(
    axes: SourceAxes,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    return axes.A.to(device)


def inverse_alignment_matrix(
    axes: SourceAxes,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    return axes.A_inv.to(device)


def alignment_matrix_np(axes: SourceAxes) -> np.ndarray:
    return axes.A.numpy()

