"""Shared pin-to-body kernel used by both VBD and MPM backends.

A ``PinToBody`` constraint says "these particles follow this rigid body".
Once the particles have been kinematically frozen (mass set to 0 in the
backend's ``apply_constraints``), this kernel re-writes their
``particle_q`` and ``particle_qd`` each substep so they track the body's
current ``body_q`` / ``body_qd``.

The same kernel is used by both backends because the inputs (a body
transform + per-particle local offsets) are backend-agnostic.

Initialization ('bake') note
-----------------------------
The constraint resolver only knows particle world positions at scene
assembly time, before the rigid body has been added to the model.  The
local offsets (in the body's frame) are computed once, on first ``pre_step``
call, by inverting the body's current transform — see
``PinToBodyHook.bake_local_offsets``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


# ── Kernel ──────────────────────────────────────────────────────────

@wp.kernel
def pin_particles_to_body_kernel(
    particle_indices: wp.array(dtype=int),
    local_offsets: wp.array(dtype=wp.vec3),
    body_idx: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    """Overwrite particle q/qd to follow body[body_idx]'s current pose."""
    tid = wp.tid()
    pi = particle_indices[tid]
    offset = local_offsets[tid]

    xform = body_q[body_idx]
    world_pos = wp.transform_point(xform, offset)
    particle_q[pi] = world_pos

    # body_qd is a spatial_vector in Newton's convention: angular (top 3),
    # linear (bottom 3) — both expressed about the COM in world frame.
    twist = body_qd[body_idx]
    omega = wp.vec3(twist[0], twist[1], twist[2])
    v_lin = wp.vec3(twist[3], twist[4], twist[5])
    com = body_com[body_idx]
    r = world_pos - com
    particle_qd[pi] = v_lin + wp.cross(omega, r)


@wp.kernel
def pin_particles_to_world_kernel(
    particle_indices: wp.array(dtype=int),
    world_positions: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    """Overwrite particle q/qd to fixed world positions."""
    tid = wp.tid()
    pi = particle_indices[tid]
    particle_q[pi] = world_positions[tid]
    particle_qd[pi] = wp.vec3(0.0, 0.0, 0.0)


# ── Hook payload ────────────────────────────────────────────────────

@dataclass
class PinToBodyHook:
    """Per-PinToBody-constraint runtime state.

    ``local_offsets`` is None until the first ``pre_step`` call, at which
    point ``bake_local_offsets`` populates it from the body's current
    transform plus the world positions captured at resolution time.
    """
    particle_indices: wp.array       # (N,) int
    body_idx: int                    # body index in model.body_q
    initial_world_positions: np.ndarray  # (N, 3) float64, captured at resolve time
    local_offsets: wp.array | None = None  # (N,) wp.vec3, baked lazily

    def bake_local_offsets(
        self,
        body_q: wp.array,
        device: str,
    ) -> None:
        """Compute body-local offsets once, given the current body transform.

        offset = inv(body_xform) * world_pos
        """
        body_q_np = body_q.numpy()
        xform = body_q_np[self.body_idx]
        # Newton stores transform as (px, py, pz, qx, qy, qz, qw)
        pos = xform[:3]
        quat = xform[3:7]  # xyzw
        qx, qy, qz, qw = quat
        # inverse rotation matrix (transpose)
        # build R from quaternion
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        R = np.array([
            [1 - 2*(yy+zz),     2*(xy-wz),     2*(xz+wy)],
            [    2*(xy+wz), 1 - 2*(xx+zz),     2*(yz-wx)],
            [    2*(xz-wy),     2*(yz+wx), 1 - 2*(xx+yy)],
        ], dtype=np.float64)
        Rt = R.T  # = R^{-1} for orthonormal R
        deltas = self.initial_world_positions - pos[None, :]
        local = deltas @ Rt.T  # (N, 3)
        self.local_offsets = wp.array(local.astype(np.float32), dtype=wp.vec3, device=device)


def launch_pin_hook(
    hook: PinToBodyHook,
    *,
    body_q: wp.array,
    body_qd: wp.array,
    body_com: wp.array,
    particle_q: wp.array,
    particle_qd: wp.array,
    device: str,
) -> None:
    """Launch ``pin_particles_to_body_kernel`` for one hook.

    Bakes local offsets on first call.
    """
    if hook.local_offsets is None:
        hook.bake_local_offsets(body_q, device=device)
    wp.launch(
        kernel=pin_particles_to_body_kernel,
        dim=hook.particle_indices.shape[0],
        inputs=[
            hook.particle_indices,
            hook.local_offsets,
            hook.body_idx,
            body_q,
            body_qd,
            body_com,
            particle_q,
            particle_qd,
        ],
        device=device,
    )


def launch_pin_to_world(
    particle_indices: wp.array,
    world_positions: wp.array,
    *,
    particle_q: wp.array,
    particle_qd: wp.array,
    device: str,
) -> None:
    """Launch a fixed-world-position pin kernel."""
    if particle_indices.shape[0] == 0:
        return
    wp.launch(
        kernel=pin_particles_to_world_kernel,
        dim=particle_indices.shape[0],
        inputs=[
            particle_indices,
            world_positions,
            particle_q,
            particle_qd,
        ],
        device=device,
    )
